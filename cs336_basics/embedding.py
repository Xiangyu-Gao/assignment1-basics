import torch
import torch.nn as nn

class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super(Embedding, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.device = device
        self.dtype = dtype
        self.weight = nn.Parameter(torch.randn(num_embeddings, embedding_dim))

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        # token_ids: (batch, sequence_length)
        # weight: (num_embeddings, embedding_dim)
        # output: (batch, sequence_length, embedding_dim)
        return self.weight[token_ids].to(device=self.device, dtype=self.dtype)
