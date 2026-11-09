import os
import random
import numpy as np
import torch


def set_seed(seed: int) -> None:
    """
    Set the seed for all possible sources of randomness to ensure reproducibility.

    Args:
        seed (int): Seed value for all random number generators.
    """
    # Optional but improves reproducibility in some edge cases
    os.environ["PYTHONHASHSEED"] = str(seed)

    # Python
    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch CPU
    torch.manual_seed(seed)

    # PyTorch GPU
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        # Deterministic behavior (reproducibility)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
