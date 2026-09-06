"""Deterministic seeding utilities to ensure reproducible training and evaluation."""

import os
import random
import numpy as np
import os
import random
import numpy as np


def set_seed(seed: int = 42) -> None:
    """Enforce strict deterministic execution across all random number generators.

    Args:
        seed: Integer random seed (default: 42).
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
