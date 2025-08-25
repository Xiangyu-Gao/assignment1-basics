import torch
import math
import torch.nn as nn

from typing import Callable, Optional


class AdamW(torch.optim.Optimizer):
    """Implements the AdamW optimization algorithm."""

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.95), eps=1e-8, weight_decay=0):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        
        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
            }
        
        super().__init__(params, defaults)
    
    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        
        for group in self.param_groups:
            lr = group["lr"] # Get the learning rate.
            eps = group["eps"]
            beta1, beta2 = group["betas"]
            weight_decay = group["weight_decay"]
            
            for p in group["params"]:
                if p.grad is None:
                    continue
                
                state = self.state[p] # Get state associated with p.
                t = state.get("t", 0) # Get iteration number from the state, or initial value.
                if t == 0:
                    state["m"] = torch.zeros_like(p.data) # Initialize first moment vector.
                    state["v"] = torch.zeros_like(p.data) # Initialize second moment vector.
                
                m, v = state["m"], state["v"]
                grad = p.grad.data # Get the gradient of loss with respect to p.
                
                # Update moment vectors in-place
                # m = beta1 * m + (1 - beta1) * grad
                # v = beta2 * v + (1 - beta2) * (grad * grad)
                m.mul_(beta1).add_(grad, alpha=1 - beta1) # Update biased first moment estimate.
                v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2) # Update biased second moment estimate.

                # Compute adjusted learning rate for iteration t
                lr_t = lr * (math.sqrt(1 - beta2 ** (t + 1)) / (1 - beta1 ** (t + 1))) # t + 1 because we want it starting from 1, not 0.s
                
                # Update parameters
                p.data -= lr_t * m / (torch.sqrt(v) + eps)

                # Apply weight decay
                p.data -= lr * weight_decay * p.data
                
                # Increment iteration number.
                state["t"] = t + 1 

        return loss