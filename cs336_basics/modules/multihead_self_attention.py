import torch
import torch.nn as nn

from cs336_basics.modules.scaled_dot_product_attention import scaled_dot_product_attention
from cs336_basics.modules.rope import RoPE
from cs336_basics.modules.linear import Linear


class MultiHeadAttention(nn.Module):
    """ Multi-Head Self-Attention Implementation. """
    def __init__(self, d_model: int, num_heads: int, max_seq_len: int = 512, theta: float = 10000.0):
        """
            d_model: int Dimension of the model
            num_heads: int Number of attention heads
        """
        super(MultiHeadAttention, self).__init__()
        
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads # Also d_v = d_v = d_model // num_heads
        self.max_seq_len = max_seq_len
        self.theta = theta
        
        self.W_Q = Linear(d_model, d_model)
        self.W_K = Linear(d_model, d_model)
        self.W_V = Linear(d_model, d_model)
        self.W_O = Linear(d_model, d_model)
        self.rope = RoPE(theta=theta, d_k=self.d_k, max_seq_len=max_seq_len)
        
    def forward(self, x: torch.Tensor, use_rope: bool) -> torch.Tensor:
        x_shape = x.shape  # (..., d_model)
        
        # Linear projections
        Q = self.W_Q(x).view(*x_shape[:-1], self.num_heads, self.d_k).transpose(-2, -3)  # (..., num_heads, seq_len, d_k)
        K = self.W_K(x).view(*x_shape[:-1], self.num_heads, self.d_k).transpose(-2, -3)  # (..., num_heads, seq_len, d_k)
        V = self.W_V(x).view(*x_shape[:-1], self.num_heads, self.d_k).transpose(-2, -3)  # (..., num_heads, seq_len, d_k)

        # Apply Rope to Q and K
        if use_rope:
            Q = self.rope(Q, torch.arange(x_shape[-2]))  # (..., num_heads, seq_len, d_k)
            K = self.rope(K, torch.arange(x_shape[-2]))  # (..., num_heads, seq_len, d_k)

        # Generate mask for multiple heads
        # prevent the model from attending to future tokens in the sequence.
        mask = torch.tril(torch.ones((x_shape[-2], x_shape[-2]), device=x.device)).unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, seq_len)

        # Compute scaled dot-product attention for each head
        attention_output = scaled_dot_product_attention(Q, K, V, mask)  # (..., num_heads, seq_len, d_k)

        # Reshape and combine heads
        attention_output = attention_output.transpose(-2, -3).flatten(-2)  # (..., seq_len, d_model)

        # Final linear projection
        output = self.W_O(attention_output)  # (..., seq_len, d_model)

        return output


        
      