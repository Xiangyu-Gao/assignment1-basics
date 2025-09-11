import torch
import torch.nn as nn

from einops import einsum


class Linear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super(Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.device = device
        self.dtype = dtype
        self.weight = nn.Parameter(torch.randn(out_features, in_features))

    def forward(self, x):
        # y = W x
        # W: (out_features, in_features)
        # x: (batch, channel, in_features)
        # y: (batch, channel, out_features)
        return einsum(self.weight, x, "o i, b c i -> b c o").to(device=self.device, dtype=self.dtype)