import torch
import torch.nn as nn


def softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Computes the softmax of the input tensor along the specified dimension.
    
    Args:
        x (torch.Tensor): Input tensor.
        dim (int): Dimension along which to compute the softmax. Default is -1 (last dimension).
    
    Returns:
        torch.Tensor: Softmax of the input tensor.
    """
    # subtract the maximum value for numerical stability
    max_x = torch.max(x, dim=dim, keepdim=True).values
    exp_x = torch.exp(x - max_x)
    sum_exp_x = torch.sum(exp_x, dim=dim, keepdim=True)
    return exp_x / sum_exp_x