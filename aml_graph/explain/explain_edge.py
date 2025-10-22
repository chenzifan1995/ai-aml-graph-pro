import argparse, torch
from torch_geometric.explain import GNNExplainer
from aml_graph.data.load import load_graph
from aml_graph.models.edge_gat import EdgeGAT

if __name__=='__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_root', default='data/simulated')
    ap.add_argument('--k', type=int, default=3)
    args = ap.parse_args()
    data = load_graph(args.data_root)
    model = EdgeGAT(data.x.size(1), data.edge_attr.size(1))
    model.eval()
    explainer = GNNExplainer(model, epochs=100)
    # Pick top-k edges by simple heuristic (amount)
    topk = data.edge_attr[:,0].topk(k=args.k).indices
    for idx in topk.tolist():
        edge_mask = explainer.explain_graph(data.x, data.edge_index, edge_attr=data.edge_attr)[0]
        print(f"Edge {idx} explanation mask sum:", float(edge_mask.sum()))
