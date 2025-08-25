import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super(RMSNorm, self).__init__()
        self.d_model = d_model
        self.eps = eps
        self.device = device
        self.dtype = dtype
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Upcast your input to torch.float32 before performing the normalization to avoid overflow
        x = x.to(dtype=torch.float32)
        # Compute the root mean square of the input tensor
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        # Normalize the input tensor and scale it with the weight
        y = (x / rms) * self.weight.to(device=self.device, dtype=self.dtype)
        # Downcast the output back to the original dtype
        return y.to(dtype=x.dtype)