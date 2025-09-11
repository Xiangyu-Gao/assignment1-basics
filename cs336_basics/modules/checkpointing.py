import torch
import os

from typing import IO, Any, BinaryIO
from collections import defaultdict


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
):
    all_params = {
        "model": {},
        "optimizer": {},
    }
    
    # gather model parameters
    for param in model.state_dict():
        all_params["model"][param] = model.state_dict()[param].cpu()
    
    # gather optimizer parameters
    for param in optimizer.state_dict():
        all_params["optimizer"][param] = optimizer.state_dict()[param]
    
    # gather iteration
    all_params["iteration"] = iteration

    # save to disk
    torch.save(all_params, out)


def load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
):  
    # load from disk
    checkpoint = torch.load(src, map_location="cpu")

    # load model state dict
    model.load_state_dict(checkpoint["model"])

    # load optimizer state dict
    optimizer.load_state_dict(checkpoint["optimizer"])

    # return iteration
    return checkpoint["iteration"]