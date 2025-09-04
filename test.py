import math
import torch
import torch.nn as nn
import torch.nn.functional as F

max_length=5000
d_model=512
pe = torch.zeros(max_length, d_model)
position = torch.arange(0, max_length, dtype=torch.float).unsqueeze(1)  # (max_len, 1)
div_term = (10000 * torch.ones(d_model//2)).pow(-torch.arange(0, d_model, 2).float()/d_model)

print(position.shape)
print(div_term.shape)
print((position * div_term).shape)
pe[:, 0::2] = torch.sin(position * div_term)
pe[:, 1::2] = torch.cos(position * div_term)

print(pe.shape)