import copy
import math
import torch
import logging

import numpy as np
import sentencepiece as spm
import torch.optim as optim

from tqdm import tqdm
from dataclasses import dataclass
from typing import List, Union, Optional, Dict

from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader

from transformers.utils import PaddingStrategy

class PositionalEncoding(nn.Module):
    def __init__(self, d_model=512, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # 初始化位置编码矩阵 (1, max_len, d_model)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = 10000 ** (-torch.arange(0, d_model, 2).float() / d_model)

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # 增加 batch 维度 匹配输入的x

        # 保存为 buffer（不会更新参数，但能随模型保存/迁移）
        self.register_buffer("pe", pe)

    def forward(self, x):
        # x: (batch_size, seq_len, d_model)
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

def attention(query, key, value, mask=None, dropout=None):
    d_k = query.size(-1)
    # 1. QK^T / sqrt(d_k)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

    # 2. 掩码 (mask==0 的位置设为 -inf)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)

    # 3. softmax
    att = F.softmax(scores, dim=-1)

    # 4. dropout（可选）
    if dropout is not None:
        att = dropout(att)

    # 5. 输出 (加权 V) + 注意力权重
    return torch.matmul(att, value), att

class MultiHeadAttention(nn.Module):
    #多头注意力机制
    def __init__(self, h: int, d_model: int , dropout: float=0.1):
        """
        初始化多头注意力层

        :param h: 多头个数（heads）
        :param d_model: 输入的词向量维度（模型维度）
        :param dropout: dropout比例
        """
        super(MultiHeadAttention, self).__init__()
        assert d_model % h == 0, "d_model必须能被h整除"

        # 每个头负责的维度
        self.d_k = d_model // h
        self.h = h

        # Q, K, V, 输出层 的线性变换矩阵
        # 共4个 Linear：前3个分别生成 query/key/value，最后1个用于多头拼接后的线性映射
        self.linears = nn.ModuleList(
            [copy.deepcopy(nn.Linear(d_model, d_model)) for _ in range(4)]
        )

        # 保存注意力权重（可选，用于可视化/调试）
        self.attn = None

        # dropout层，用于防止过拟合
        self.dropout = nn.Dropout(p=dropout)

    def attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor=None,
        dropout: torch.nn.Module=None
    ):
        """
        计算 Scaled Dot-Product Attention

        :param query: 查询向量 [batch_size, h, seq_len, d_k]
        :param key: 键向量 [batch_size, h, seq_len, d_k]
        :param value: 值向量 [batch_size, h, seq_len, d_k]
        :param mask: 掩码 [batch_size, 1, 1, seq_len]，防止某些位置被关注
        :param dropout: dropout模块
        :return: (加权后的value, 注意力权重)
        """
        # 1) 计算注意力得分矩阵: Q * K^T / sqrt(d_k)
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

        # 2) 如果有mask，把不合法的位置替换为 -1e9 (softmax后趋近于0)
        if mask is not None:
            try:
                scores = scores.masked_fill(mask == 0, -1e9)
            except Exception as e:
                logger.error(e.__str__())

        # 3) softmax 得到注意力权重分布
        p_attn = F.softmax(scores, dim=-1)

        # 4) dropout
        if dropout is not None:
            p_attn = dropout(p_attn)

        # 5) 加权求和
        return torch.matmul(p_attn, value), p_attn

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor=None
    ):
        """
        前向传播

        :param query: 查询query [batch_size, seq_len, d_model]
        :param key: 键key [batch_size, seq_len, d_model]
        :param value: 值value [batch_size, seq_len, d_model]
        :param mask: 掩码 [batch_size, 1, seq_len]
        :return: 输出张量 [batch_size, seq_len, d_model]
        """

        # 1) 扩展mask以匹配多头
        if mask is not None:
            mask = mask.unsqueeze(1)  # -> [batch_size, 1, 1, seq_len]

        nbatches = query.size(0)

        if mask is not None:
            logger.debug('mask shape:%s' % str(mask.shape))

        # 2) 通过线性层映射并拆分多头
        # 例如 d_model=512, h=8 -> 每头 d_k=64
        # 映射后再 reshape: [batch, seq_len, d_model] -> [batch, h, seq_len, d_k]
        query, key, value = [
            l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
            for l, x in zip(self.linears, (query, key, value))
        ]

        # 3) 计算多头注意力
        x, self.attn = self.attention(query, key, value, mask=mask, dropout=self.dropout)

        # 4) 拼接多头
        # [batch, h, seq_len, d_k] -> [batch, seq_len, h * d_k = d_model]
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)

        # 5) 最终线性映射，保持输出维度仍为 d_model
        return self.linears[-1](x)
