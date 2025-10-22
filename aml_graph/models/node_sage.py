import torch
from torch import nn
from torch_geometric.nn import SAGEConv

class NodeSAGE(nn.Module):
    def __init__(self, in_dim, hidden_dim=64, num_layers=2, dropout=0.2):
        super().__init__()
        self.layers = nn.ModuleList()
        dims = [in_dim] + [hidden_dim]*num_layers
        for i in range(num_layers):
            self.layers.append(SAGEConv(dims[i], dims[i+1]))
        self.head = nn.Linear(hidden_dim, 1)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x, edge_index):
        h = x
        for conv in self.layers:
            h = self.act(conv(h, edge_index))
            h = self.drop(h)
        return self.head(h).squeeze(-1)
