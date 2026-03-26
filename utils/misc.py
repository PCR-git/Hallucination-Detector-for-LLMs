import torch
import numpy as np
import random
import os

def set_seed(seed_value=42):
    """Set seeds for reproducibility."""
    # Set seed for Python's built-in random number generator
    random.seed(seed_value)
    # Set a fixed value for the hash seed
    os.environ['PYTHONHASHSEED'] = str(seed_value)

    # Set seed for NumPy's random number generator
    np.random.seed(seed_value)

    # Set seed for PyTorch's random number generators (both CPU and CUDA)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value) # For multi-GPU models

    # Configure PyTorch to use deterministic algorithms (affects GPU performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False # Disabling benchmarking ensures deterministic selection of convolution algorithms
    