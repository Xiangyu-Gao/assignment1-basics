import math


def lr_cosine_schedule(it: int, max_learning_rate: float, min_learning_rate: float, warmup_iters: int, cosine_cycle_iters: int,) -> float:
    """
    Compute the learning rate at iteration `it` using a cosine schedule with linear warmup.

    Args:
        it (int): Current iteration number.
        max_learning_rate (float): Maximum learning rate after warmup.
        min_learning_rate (float): Minimum learning rate at the end of the cosine cycle.
        warmup_iters (int): Number of iterations for linear warmup.
        cosine_cycle_iters (int): Number of iterations for one cosine cycle.

    Returns:
        float: The computed learning rate for the current iteration.
    """
    if it < warmup_iters:
        # Linear warmup
        lr = max_learning_rate * (it / warmup_iters)
    elif it <= cosine_cycle_iters:
        # Cosine learning rate decay
        cosine_decay = 0.5 * (1 + math.cos(math.pi * (it -warmup_iters) / (cosine_cycle_iters - warmup_iters)))
        lr = min_learning_rate + (max_learning_rate - min_learning_rate) * cosine_decay
    else:
        # After the cosine cycles, keep the learning rate at min_learning_rate
        lr = min_learning_rate

    return lr