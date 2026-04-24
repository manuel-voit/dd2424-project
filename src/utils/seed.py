import random
import numpy as np
import torch

def set_seed(seed: int = 42):
    """
    Ensures complete reproducibility across Python, NumPy and PyTorch.
    """
    # Set standard Python and NumPy seeds
    random.seed(seed)
    np.random.seed(seed)
    
    # Set PyTorch CPU and GPU seeds
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        
    # Force CuDNN to be deterministic
    # Note: This might slightly slow down training, but guarantees exact reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
