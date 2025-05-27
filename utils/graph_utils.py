import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset


import torch, torch.nn.functional as F

def normalize_A(A: torch.Tensor, sym: bool = False):
    A = F.relu(A)
    if sym:
        A = A + A.T
    d = A.sum(1)
    D = torch.diag_embed((d + 1e-10).pow(-0.5))
    return D @ A @ D

def generate_cheby_adj(L: torch.Tensor, K: int, device=None):
    """生成 Chebyshev 多项式展开的 K 阶邻接张量列表"""
    device = device or L.device
    I = torch.eye(L.shape[0], device=device)
    support = [I, L]
    for _ in range(2, K):
        support.append(2 * L @ support[-1] - support[-2])
    return support[:K]


class eegDataset(Dataset):
    # x_tensor: (sample, channel, datapoint(feature)) type = torch.tensor
    # y_tensor: (sample,) type = torch.tensor

    def __init__(self, x_tensor, y_tensor):
        self.x = x_tensor
        self.y = y_tensor

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return len(self.y)
