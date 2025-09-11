import torch
import torch.nn as nn

from einops import einsum
from cs336_basics.modules.softmax import softmax


def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Compute the scaled dot-product attention.

    Args:
        Q: Queries tensor of shape (..., seq_len_q, d_k)
        K: Keys tensor of shape (..., seq_len_k, d_k)
        V: Values tensor of shape (..., seq_len_v, d_v)
        mask: Optional mask tensor of shape (seq_len_q, seq_len_k)

    Returns:
        output: Attention output tensor of shape (..., seq_len_q, d_v)
    """
    d_k = Q.size(-1)
    
    # Compute scores Q^T K / sqrt(d_k)
    scores = einsum(Q, K, "... i d, ... j d -> ... i j") / torch.sqrt(torch.tensor(d_k, dtype=Q.dtype)) # (..., seq_len_q, seq_len_k)
    
    # Apply the mask (if provided)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    
    # Apply softmax to get the attention weights
    attention_weights = softmax(scores, dim=-1) # (..., seq_len_q, seq_len_k)
    
    # Compute the weighted sum of the values
    output = einsum(attention_weights, V, "... i j, ... j k -> ... i k") # (..., seq_len_q, d_v)
    
    return output