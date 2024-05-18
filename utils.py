import os
import random
import numpy as np
import torch

def init_seed(seed, is_determine=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    if is_determine:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False