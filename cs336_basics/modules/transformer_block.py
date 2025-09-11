import torch
import torch.nn as nn

from cs336_basics.modules.multihead_self_attention import MultiHeadAttention
from cs336_basics.modules.rmsnorm import RMSNorm
from cs336_basics.modules.positionwise_feedforward import PositionwiseFeedForward


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, max_seq_len: int = 512, theta: float = 10000.0):
        """
        d_model: int, Dimensionality of the Transformer block inputs.
        num_heads: int, Number of heads to use in multi-head self-attention.
        d_ff: int, Dimensionality of the position-wise feed-forward inner layer.
        """
        super(TransformerBlock, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)
        self.ffn = PositionwiseFeedForward(d_model, d_ff)
        self.mha = MultiHeadAttention(d_model, num_heads, max_seq_len, theta)

    def forward(self, x, mask=None):
        # x: (batch_size, seq_len, d_model)
        # Multi-head attention sub-layer
        x = x + self.mha(self.norm1(x), use_rope=True)

        # Feed-forward sub-layer
        x = x + self.ffn(self.norm2(x))

        return x
