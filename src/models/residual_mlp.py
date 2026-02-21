# 构建Residual MLP 模块
import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualMLP(nn.Module):
    """
    x' = x + alpha * MLP(x)
    alpha: fixed float or learnable parameter
    """
    def __init__(self, dim, hidden=512, dropout=0.1, alpha=0.2, learnable_alpha=False):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
        )
        if learnable_alpha:
            self.alpha = nn.Parameter(torch.tensor(float(alpha)))
        else:
            self.register_buffer("alpha", torch.tensor(float(alpha)), persistent=False)

    def forward(self, x):
        return x + self.alpha * self.mlp(x)
