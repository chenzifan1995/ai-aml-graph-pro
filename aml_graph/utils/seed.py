import os, random, numpy as np, torch

def seed_all(seed=42):
    random.seed(seed); np.random.seed(seed)
    os.environ['PYTHONHASHSEED']=str(seed)
    try:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass
