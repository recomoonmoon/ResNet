import math
import torch
import torch.nn as nn
import torch.nn.functional as F

a = torch.arange(0, 16).float()
a = a.view(4, 4)
print(a)
print(a.mean(dim=-1))