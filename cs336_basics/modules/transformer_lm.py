import torch
import torch.nn as nn

from cs336_basics.modules.transformer_block import TransformerBlock
from cs336_basics.modules.embedding import Embedding
from cs336_basics.modules.linear import Linear
from cs336_basics.modules.rmsnorm import RMSNorm
from cs336_basics.modules.softmax import softmax


class TransformerLM(nn.Module):
    """ Transformer Language Model Implementation. """
    def __init__(self, vocab_size: int, context_length: int, num_layers: int, d_model: int, num_heads: int, d_ff: int, theta: float = 10000.0):
        """
            vocab_size: int Size of the vocabulary
            context_length: int The maximum context length, necessary for determining the dimensionality of the position embedding matrix.
            num_layers: int The number of Transformer blocks to use.
            d_model: int Dimension of the model
            num_heads: int Number of attention heads
            d_ff: int Dimension of the feedforward network
            theta: float Θ value for the RoPE
        """
        super(TransformerLM, self).__init__()
        
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.theta = theta
        
        self.embedding = Embedding(vocab_size, d_model)
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, context_length, theta) for _ in range(num_layers)
        ])
        self.rmsnorm = RMSNorm(d_model)
        self.linear = Linear(d_model, vocab_size)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: Long[Tensor, " ..., seq_len"]: Input tensor containing token indices
        """
        
        x = self.embedding(x)   # (batch_size, seq_len, d_model)
        
        # Pass through each transformer block
        for block in self.transformer_blocks:
            x = block(x)  # (batch_size, seq_len, d_model)
        
        x = self.rmsnorm(x)  # (..., seq_len, d_model)

        x = self.linear(x)  # (..., seq_len, vocab_size)
        
        return x  # note that softmax is applied in the loss function