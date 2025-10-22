import pytorch_lightning as pl, torch, argparse, numpy as np
from aml_graph.data.load import load_graph
from aml_graph.models.node_sage import NodeSAGE
from aml_graph.utils.seed import seed_all

class NodeModule(pl.LightningModule):
    def __init__(self, data_root='data/simulated', hidden_dim=64, lr=1e-3, max_epochs=5):
        super().__init__()
        self.save_hyperparameters()
        self.data = load_graph(data_root)
        self.model = NodeSAGE(self.data.x.size(1), hidden_dim)
        # create a simple undirected structure from transactions
        ei = self.data.edge_index
        rev = torch.stack([ei[1], ei[0]])
        self.edge_index = torch.cat([ei, rev], dim=1)
        # Construct pseudo node labels: suspicious if any outgoing edge is suspicious
        y = torch.zeros(self.data.x.size(0))
        y.index_put_((ei[0],), self.data.edge_label, accumulate=True)
        self.y = (y>0).float()
        self.idx = torch.arange(self.data.x.size(0))

    def training_step(self, batch, bx):
        logits = self.model(self.data.x, self.edge_index)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, self.y)
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

if __name__=='__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_root', default='data/simulated')
    ap.add_argument('--hidden_dim', type=int, default=64)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--max_epochs', type=int, default=5)
    args = ap.parse_args()
    seed_all(42)
    pl.Trainer(max_epochs=args.max_epochs, enable_checkpointing=False, logger=False).fit(NodeModule(args.data_root, args.hidden_dim, args.lr, args.max_epochs))
