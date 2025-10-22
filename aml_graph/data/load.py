import json, os, numpy as np, torch
from torch_geometric.data import Data

def load_graph(root: str):
    acc = [json.loads(l) for l in open(os.path.join(root, 'accounts.jsonl'))]
    tx  = [json.loads(l) for l in open(os.path.join(root, 'transactions.jsonl'))]

    # Node features
    X = np.array([[a['region'], a['kyc_score'], a['activity']] for a in acc], dtype=np.float32)

    # Edge structure & labels
    edge_index = np.array([[t['src'], t['dst']] for t in tx], dtype=np.int64).T
    y_edge = np.array([t['label'] for t in tx], dtype=np.float32)
    ts = np.array([t['ts'] for t in tx], dtype=np.int64)

    # Edge features
    amounts = np.array([t['amount'] for t in tx], dtype=np.float32)
    geo_mismatch = np.array([t['geo_mismatch'] for t in tx], dtype=np.float32)
    # normalize amount by per-src z-score
    z = np.zeros_like(amounts, dtype=np.float32)
    for s in np.unique(edge_index[0]):
        idx = np.where(edge_index[0]==s)[0]
        if len(idx) > 1:
            mu, sd = amounts[idx].mean(), amounts[idx].std()+1e-6
            z[idx] = (amounts[idx]-mu)/sd
        else:
            z[idx] = 0.0

    edge_attr = np.stack([amounts, z, geo_mismatch], axis=1).astype(np.float32)

    data = Data(
        x=torch.tensor(X),
        edge_index=torch.tensor(edge_index),
        edge_attr=torch.tensor(edge_attr),
        edge_label=torch.tensor(y_edge),
        edge_ts=torch.tensor(ts)
    )
    return data
