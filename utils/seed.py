"""
MASOS - Reproducibility utilities.
Sets seeds for Python, NumPy, and PyTorch.
"""
import os
import random
import numpy as np
import torch


def set_seed(seed: int = 42, deterministic: bool = True):
    """
    Set random seeds for reproducibility.

    Args:
        seed: Random seed value
        deterministic: If True, set CUDA deterministic mode
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ["PYTHONHASHSEED"] = str(seed)

    print(f"[Seed] Set all random seeds to {seed}")
