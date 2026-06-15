import random
import numpy as np
import os

def set_seed(seed=42):
    """Sets random seeds for reproducibility across all libraries."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
