import math
import torch
import torch.nn as nn
import torch.nn.functional as F

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


