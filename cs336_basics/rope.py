import torch
import torch.nn as nn


class RoPE(nn.Module):
    """ Rotary Positional Embedding Implementation. """
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        """
            theta: float Θ value for the RoPE
            d_k: int dimension of query and key vectors
            max_seq_len: int Maximum sequence length that will be inputted
            device: torch.device | None = None Device to store the buffer on
        """
        super(RoPE, self).__init__()
        
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self.device = device if device is not None else torch.device("cpu")

        self._rope_init() # Initialize the RoPE parameters

    def _rope_init(self):
        """ Initialize the RoPE parameters. """
        inv_freq = 1.0 / (self.theta ** (torch.arange(0, self.d_k, 2).float() / self.d_k))   # inv_freq: (d_k // 2,)
        seq_idx = torch.arange(self.max_seq_len, device=self.device).float()    # seq_idx: (max_seq_len,)
        freqs = torch.einsum('i,j->ij', seq_idx, inv_freq)  # freqs: (max_seq_len, d_k // 2)
        cache = torch.stack([torch.cos(freqs), torch.sin(freqs)], dim=-1)   # cache: (max_seq_len, d_k // 2, 2)
        self.register_buffer('inv_freq', inv_freq, persistent=False)
        self.register_buffer('cache', cache, persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        """
        x: Float[Tensor, " ... seq_len d_k"]: Input tensor to apply RoPE on
        token_positions: Long[Tensor, " seq_len"]: Positions of the tokens in the sequence
        """
        x_shaped = x.view(*x.shape[:-1], -1, 2) # Reshape to (, seq_len, d_k // 2, 2)

        # extract the cache for the given token positions
        rope_cache = self.cache[token_positions]   # (seq_len, d_k // 2, 2)

        # compute the real and imaginary parts of rope
        # real_part = x_real * cos(freqs) - x_imag * sin(freqs)
        # imag_part = x_real * sin(freqs) + x_imag * cos(freqs)
        real_part = x_shaped[..., 0] * rope_cache[..., 0] - x_shaped[..., 1] * rope_cache[..., 1]   # (, seq_len, d_k // 2)
        imag_part = x_shaped[..., 0] * rope_cache[..., 1] + x_shaped[..., 1] * rope_cache[..., 0]   # (, seq_len, d_k // 2)

        # concatenate the real and imaginary parts to get the final output
        rope_output = torch.cat([real_part.unsqueeze(-1), imag_part.unsqueeze(-1)], dim=-1)  # (, seq_len, d_k // 2, 2)
        
        return rope_output.flatten(-2)  # Flatten to (, seq_len, d_k)