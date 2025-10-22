import argparse
from aml_graph.data.load import load_graph
from aml_graph.utils.metrics import classification_report
import numpy as np

def score_edges(data):
    amounts = data.edge_attr[:,0].numpy()
    z = data.edge_attr[:,1].numpy()
    geo = data.edge_attr[:,2].numpy()
    # simple heuristic: large positive z, large amount, and geo mismatch
    s = 0.5*(z>2).astype(float) + 0.3*(amounts>50000).astype(float) + 0.2*geo
    # map to [0,1]
    s = (s - s.min())/(s.max()-s.min()+1e-9)
    return s

if __name__=='__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_root', default='data/simulated')
    args = ap.parse_args()
    data = load_graph(args.data_root)
    prob = score_edges(data)
    rep = classification_report(data.edge_label.numpy(), prob)
    print(rep)
