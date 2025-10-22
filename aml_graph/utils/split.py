import numpy as np

def temporal_split(timestamps, train_frac=0.7, val_frac=0.15):
    ts = np.asarray(timestamps)
    thr1 = np.quantile(ts, train_frac)
    thr2 = np.quantile(ts, train_frac+val_frac)
    idx_train = np.where(ts <= thr1)[0]
    idx_val = np.where((ts > thr1) & (ts <= thr2))[0]
    idx_test = np.where(ts > thr2)[0]
    return idx_train, idx_val, idx_test
