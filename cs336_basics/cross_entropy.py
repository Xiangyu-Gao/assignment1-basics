import torch
import torch.nn as nn



def cross_entropy_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    Compute the cross-entropy loss between logits and labels.

    Args:
        logits (torch.Tensor): The predicted logits with shape (N, C) where N is the batch size and C is the number of classes.
        labels (torch.Tensor): The true labels with shape (N,) containing class indices.

    Returns:
        torch.Tensor: The computed cross-entropy loss.
    """
    # Subtract the maximum logit for numerical stability
    max_logits = torch.max(logits, dim=-1, keepdim=True).values
    stabilized_logits = logits - max_logits
    
    # Calculate the sum of exponentials for each row
    sum_exp = torch.exp(stabilized_logits).sum(dim=-1, keepdim=True)

    # Calculate the cross entropy loss with log and exp cancel outling
    log_probs = stabilized_logits - torch.log(sum_exp)
    loss = -log_probs[torch.arange(logits.size(0)), labels].mean()

    return loss