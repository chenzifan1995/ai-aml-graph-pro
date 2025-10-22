import torch
from torch import nn
from torch_geometric.nn import GATv2Conv

class EdgeGAT(nn.Module):
    def __init__(self, in_dim_node, in_dim_edge, hidden_dim=96, heads=2, dropout=0.2):
        super().__init__()
        self.gat1 = GATv2Conv(in_dim_node, hidden_dim, heads=heads, dropout=dropout)
        self.gat2 = GATv2Conv(hidden_dim*heads, hidden_dim, heads=1, dropout=dropout)
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim*2 + in_dim_edge, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )
        self.act = nn.ReLU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x, edge_index, edge_attr):
        h = self.act(self.gat1(x, edge_index))
        h = self.drop(h)
        h = self.act(self.gat2(h, edge_index))
        src, dst = edge_index
        h_edge = torch.cat([h[src], h[dst], edge_attr], dim=1)
        logits = self.edge_mlp(h_edge).squeeze(-1)
        return logits
