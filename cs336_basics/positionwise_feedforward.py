import torch
import torch.nn as nn

from cs336_basics.linear import Linear


def silu(x):
    """Sigmoid Linear Unit (SiLU) activation function, also known as Swish."""
    # SiLU(x) = x * sigmoid(x)
    return x * torch.sigmoid(x)


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(PositionwiseFeedForward, self).__init__()
        self.linear1 = Linear(d_model, d_ff)
        self.linear2 = Linear(d_ff, d_model)
        self.linear3 = Linear(d_model, d_ff)

    def forward(self, x):
        # FFN(x) = SwiGLU(x, W1 , W2 , W3 ) = W2 (SiLU(W1 x) ⊙ W3 x)        
        return self.linear2(silu(self.linear1(x)) * self.linear3(x))
