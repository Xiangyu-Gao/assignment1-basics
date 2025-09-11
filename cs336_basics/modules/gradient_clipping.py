import torch

from typing import Iterable


def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> None:
    """
    Clips the gradients of the given parameters to have a maximum L2 norm of `max_norm`.

    Args:
        parameters (Iterable[torch.nn.Parameter]): An iterable of model parameters whose gradients will be clipped.
        max_l2_norm (float): The maximum allowed L2 norm for the gradients.

    Returns:
        None: The function modifies the gradients in-place.
    """
    total_norm = 0.0
    for p in parameters:
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5

    clip_coef = max_l2_norm / (total_norm + 1e-6)
    if clip_coef < 1:
        for p in parameters:
            if p.grad is not None:
                p.grad.data.mul_(clip_coef)