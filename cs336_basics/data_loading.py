import torch
import torch.nn as nn
import numpy as np


def data_loading(x: np.ndarray, batch_size: int, context_length: int, device: str = 'cpu') -> tuple[torch.Tensor, torch.Tensor]:
    """
    Given a dataset (a 1D numpy array of integers) and a desired batch size and
    context length, sample language modeling input sequences and their corresponding
    labels from the dataset.

    Args:
        dataset (np.array): 1D numpy array of integer token IDs in the dataset.
        batch_size (int): Desired batch size to sample.
        context_length (int): Desired context length of each sampled example.
        device (str): PyTorch device string (e.g., 'cpu' or 'cuda:0') indicating the device
            to place the sampled input sequences and labels on.

    Returns:
        Tuple of torch.LongTensors of shape (batch_size, context_length). The first tuple item
        is the sampled input sequences, and the second tuple item is the corresponding
        language modeling labels.
    """
    # randomly select index from [0, len(dataset) - context_length]
    if len(x) < context_length + 1:
        raise ValueError("Dataset is too small for the given context length.")
    selected_idx = np.random.choice(len(x) - context_length, size=batch_size, replace=True)
    
    # prepare batches
    x_batch = np.zeros((batch_size, context_length), dtype=np.int64)
    y_batch = np.zeros((batch_size, context_length), dtype=np.int64)
    for i, start_idx in enumerate(selected_idx):
        x_batch[i] = x[start_idx:start_idx + context_length]
        y_batch[i] = x[start_idx + 1:start_idx + context_length + 1]
    
    # convert to torch tensors and move to device
    x_tensor = torch.tensor(x_batch, dtype=torch.long).to(device)
    y_tensor = torch.tensor(y_batch, dtype=torch.long).to(device)
    
    return (x_tensor, y_tensor)

