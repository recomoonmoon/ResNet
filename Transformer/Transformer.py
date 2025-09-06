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


# -------------------------------
# 模块1：位置编码器 (Positional Encoding)
# -------------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model=512, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # 初始化位置编码矩阵 (max_len, d_model)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # [max_len, 1]
        div_term = 10000 ** (-torch.arange(0, d_model, 2).float() / d_model)  # 频率因子

        # 按照公式给偶数/奇数位置赋值
        pe[:, 0::2] = torch.sin(position * div_term)  # 偶数列
        pe[:, 1::2] = torch.cos(position * div_term)  # 奇数列
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]，方便和 batch 输入相加

        # 注册为 buffer：不是参数，不会更新，但保存到模型中
        self.register_buffer("pe", pe)

    def forward(self, x):
        # 输入: x [batch_size, seq_len, d_model]
        # 加上对应位置的编码
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


# -------------------------------
# 模块2：Scaled Dot-Product Attention
# -------------------------------
def attention(query, key, value, mask=None, dropout=None):
    d_k = query.size(-1)

    # 1. QK^T / sqrt(d_k)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

    # 2. 掩码处理：不合法位置设为 -1e9
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)

    # 3. Softmax 转换为权重分布
    att = F.softmax(scores, dim=-1)

    # 4. Dropout（可选）
    if dropout is not None:
        att = dropout(att)

    # 5. 加权求和并返回
    return torch.matmul(att, value), att


# -------------------------------
# 模块3：Multi-Head Attention
# -------------------------------
class MultiHeadAttention(nn.Module):
    def __init__(self, h: int, d_model: int, dropout: float = 0.1):
        """
        多头注意力机制
        :param h: 多头数
        :param d_model: 输入向量维度
        :param dropout: dropout比例
        """
        super(MultiHeadAttention, self).__init__()
        assert d_model % h == 0, "d_model必须能被h整除"

        # 每个头的维度
        self.d_k = d_model // h
        self.h = h

        # 4个线性层：Q, K, V, 输出映射
        self.linears = nn.ModuleList(
            [copy.deepcopy(nn.Linear(d_model, d_model)) for _ in range(4)]
        )

        self.attn = None  # 保存注意力权重
        self.dropout = nn.Dropout(p=dropout)

    def attention(self, query, key, value, mask=None, dropout=None):
        """
        计算单头的 scaled dot-product attention
        """
        d_k = query.size(-1)

        # 1) 注意力分数
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

        # 2) 掩码处理
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        # 3) Softmax
        p_attn = F.softmax(scores, dim=-1)

        # 4) Dropout
        if dropout is not None:
            p_attn = dropout(p_attn)

        # 5) 加权和
        return torch.matmul(p_attn, value), p_attn

    def forward(self, query, key, value, mask=None):
        """
        多头注意力前向传播
        输入: [batch, seq_len, d_model]
        输出: [batch, seq_len, d_model]
        """
        if mask is not None:
            mask = mask.unsqueeze(1)  # 扩展到多头维度

        nbatches = query.size(0)

        # 1) 映射并拆分多头
        query, key, value = [
            l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
            for l, x in zip(self.linears, (query, key, value))
        ]

        # 2) 计算注意力
        x, self.attn = self.attention(query, key, value, mask=mask, dropout=self.dropout)

        # 3) 拼接多头 [batch, h, seq_len, d_k] -> [batch, seq_len, d_model]
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)

        # 4) 输出映射
        return self.linears[-1](x)


# -------------------------------
# 模块4：Layer Normalization
# -------------------------------
class LayerNormalization(nn.Module):
    """
    层标准化 LayerNorm
    """
    def __init__(self, features, eps=1e-6):
        super(LayerNormalization, self).__init__()
        self.gamma = nn.Parameter(torch.ones(features))  # 可学习缩放系数
        self.beta = nn.Parameter(torch.zeros(features))  # 可学习平移系数
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, seq_len, d_model]
        mean = x.mean(dim=-1, keepdim=True)  # 均值
        var = x.var(dim=-1, keepdim=True, unbiased=False)  # 方差

        # 归一化 + 仿射变换
        return self.gamma * (x - mean) / torch.sqrt(var + self.eps) + self.beta


# -------------------------------
# 模块5：Feed Forward Network (FFN)
# -------------------------------
class FFN(nn.Module):
    """
    前馈神经网络（位置独立）
    """
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(FFN, self).__init__()
        self.l1 = nn.Linear(d_model, d_ff)   # 升维
        self.l2 = nn.Linear(d_ff, d_model)   # 降维
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # ReLU 激活 + Dropout
        return self.l2(self.dropout(F.relu(self.l1(x))))


# -------------------------------
# 模块6：Encoder Block One (MHA + Add&Norm)
# -------------------------------
class BlockOne(nn.Module):
    def __init__(self, head_num, d_model, dropout):
        super(BlockOne, self).__init__()
        self.mha = MultiHeadAttention(head_num, d_model, dropout)
        self.ln = LayerNormalization(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        # 1) Multi-head attention
        x_mha = self.mha(query, key, value, mask)

        # 2) 残差连接 + Dropout
        query = query + self.dropout(x_mha)

        # 3) LayerNorm
        query = self.ln(query)

        return query


# -------------------------------
# 模块7：Encoder Block Two (FFN + Add&Norm)
# -------------------------------
class BlockTwo(nn.Module):
    def __init__(self, d_model, d_ff, dropout):
        super(BlockTwo, self).__init__()
        self.ffn = FFN(d_model=d_model, d_ff=d_ff, dropout=dropout)
        self.ln = LayerNormalization(features=d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # 1) FFN
        ffn_x = self.ffn(x)

        # 2) 残差连接 + Dropout
        x = x + self.dropout(ffn_x)

        # 3) LayerNorm
        x = self.ln(x)
        return x


# -------------------------------
# 模块8：完整 Encoder Layer
# -------------------------------
class EncoderLayer(nn.Module):
    def __init__(self, head_num, d_model, d_ff, dropout):
        super(EncoderLayer, self).__init__()
        self.bk1 = BlockOne(head_num=head_num, d_model=d_model, dropout=dropout)
        self.bk2 = BlockTwo(d_model=d_model, d_ff=d_ff, dropout=dropout)

    def forward(self, x, mask=None):
        # Self-Attention + FFN
        x = self.bk1(x, x, x, mask)
        x = self.bk2(x)
        return x


# -------------------------------
# 模块9：完整 Decoder Layer
# -------------------------------
class DecoderLayer(nn.Module):
    def __init__(self, head_num, d_model, d_ff, dropout):
        super(DecoderLayer, self).__init__()
        # 1) masked multi-head self-attention
        self.bk1 = BlockOne(head_num=head_num, d_model=d_model, dropout=dropout)
        # 2) encoder-decoder attention
        self.bk2 = BlockOne(head_num=head_num, d_model=d_model, dropout=dropout)
        # 3) FFN
        self.bk3 = BlockTwo(d_model=d_model, d_ff=d_ff, dropout=dropout)

    def forward(self, query, memory, src_mask=None, tgt_mask=None):
        # 1) masked self-attention
        out = self.bk1.forward(query=query, key=query, value=query, mask=tgt_mask)
        # 2) encoder-decoder attention
        out = self.bk2.forward(query=out, key=memory, value=memory, mask=src_mask)
        # 3) FFN
        out = self.bk3.forward(out)
        return out

class EncoderStack(nn.Module):
    """
    编码器堆栈（由多个 EncoderLayer 顺序堆叠）
    注意：这里的 residual（残差连接）已经在 EncoderLayer 内部实现，
         所以栈本身不需要额外处理残差。
    """

    def __init__(self, layer, layer_num):
        super(EncoderStack, self).__init__()
        # 深拷贝多份同样的 encoder 层
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(layer_num)])
        # 末尾加 LayerNorm（原论文在每个子层后 + 残差前有 norm）
        self.norm = LayerNormalization(layer.d_model)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

class DecoderStack(nn.Module):
    """
    解码器堆叠（由多个 DecoderLayer 顺序堆叠）
    """
    def __init__(self, layer, layer_num):
        super(DecoderStack, self).__init__()
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(layer_num)])
        self.norm = LayerNormalization(layer.d_model)

    def forward(self, x, memory, src_mask, tgt_mask):
        """
        :param x: 目标序列 embedding
        :param memory: 来自 encoder 的输出
        :param src_mask: encoder padding mask
        :param tgt_mask: decoder 自回归 + padding mask
        """
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)

class Generator(nn.Module):
    """
    解码器输出映射层：Linear + LogSoftmax
    """

    def __init__(self, d_model:int, vocab:int):
        super(Generator, self).__init__()
        self.linear = nn.Linear(d_model, vocab)

    def forward(self, x):
        # 使用 log_softmax 而不是 softmax：
        # 1. 数值更稳定（避免极小概率 underflow）
        # 2. 方便与 NLLLoss 搭配使用
        return F.log_softmax(self.linear(x), dim=-1)

class Translate(nn.Module):
    """
    整体翻译模型：Embedding + PositionalEncoding + EncoderStack + DecoderStack + Generator
    """
    def __init__(self, src_vocab_size:int, tgt_vocab_size:int,
                 head_num:int=8, layer_num:int=6,
                 d_model:int=512, d_ff:int=2048, dropout:float=0.1):
        super(Translate, self).__init__()

        # 构建 Encoder / Decoder 层
        encoder_layer = EncoderLayer(head_num, d_model, d_ff, dropout)
        decoder_layer = DecoderLayer(head_num, d_model, d_ff, dropout)

        # 堆叠多个 encoder/decoder 层
        self.encoder_stack = EncoderStack(encoder_layer, layer_num)
        self.decoder_stack = DecoderStack(decoder_layer, layer_num)

        # 位置编码器
        self.pe_encode = PositionalEncoding(d_model, dropout)
        self.pe_decode = PositionalEncoding(d_model, dropout)

        # 词向量
        self.src_embedd = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedd = nn.Embedding(tgt_vocab_size, d_model)

        # 输出层
        self.generator = Generator(d_model, tgt_vocab_size)

    def forward(self, src, tgt, src_mask, tgt_mask):
        """
        :param src: 源序列 [batch_size, src_len]
        :param tgt: 目标序列 [batch_size, tgt_len]
        :param src_mask: 源 mask [batch_size, 1, 1, src_len]
        :param tgt_mask: 目标 mask [batch_size, 1, tgt_len, tgt_len]
        """
        src_embedding = self.pe_encode(self.src_embedd(src))
        tgt_embedding = self.pe_decode(self.tgt_embedd(tgt))  # ⚠️ 原来用 pe_encode，这里改为 pe_decode

        encoder_output = self.encoder_stack(src_embedding, src_mask)
        decoder_output = self.decoder_stack(tgt_embedding, encoder_output, src_mask, tgt_mask)

        softmax_out = self.generator(decoder_output)
        return decoder_output, softmax_out

def get_decoder_mask(data, pad=0):
    """
    Decoder mask：结合 padding mask 和 subsequent mask
    """
    # padding 部分
    tgt_mask = (data != pad).unsqueeze(-2)  # [batch, 1, tgt_len]
    # 自回归屏蔽
    tgt_mask = tgt_mask & subsequent_mask(data.size(-1)).to(data.device)
    return tgt_mask


def subsequent_mask(size):
    """
    生成自回归 mask（上三角为 False，下三角为 True）
    shape: [1, size, size]
    """
    attn_shape = (1, size, size)
    mask = np.triu(np.ones(attn_shape), k=1).astype('bool')
    return torch.from_numpy(~mask)   # True=可见, False=不可见


def get_encoder_mask(data, pad=0):
    """
    Encoder mask：只处理 padding
    """
    return (data != pad).unsqueeze(-2)  # [batch, 1, src_len]
