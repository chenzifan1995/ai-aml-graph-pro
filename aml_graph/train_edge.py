import pytorch_lightning as pl, torch, argparse, numpy as np
from torchmetrics.classification import AUROC, AveragePrecision
from aml_graph.data.load import load_graph
from aml_graph.models.edge_gat import EdgeGAT
from aml_graph.utils.seed import seed_all
from aml_graph.utils.metrics import classification_report
from aml_graph.utils.split import temporal_split

class EdgeModule(pl.LightningModule):
    def __init__(self, data_root: str = 'data/simulated', hidden_dim: int = 96, lr: float = 5e-4, max_epochs: int = 5):
        super().__init__()
        self.save_hyperparameters()
        self.data = load_graph(data_root)
        self.model = EdgeGAT(self.data.x.size(1), self.data.edge_attr.size(1), hidden_dim)
        # temporal split on edges
        ts = self.data.edge_ts.numpy()
        tr, va, te = temporal_split(ts)
        self.idx_train, self.idx_val, self.idx_test = map(torch.tensor, (tr, va, te))
        pos = float(self.data.edge_label[self.idx_train].mean().item())
        self.pos_weight = torch.tensor((1.0 - pos) / max(1e-6, pos))

        self.auroc = AUROC(task='binary')
        self.auprc = AveragePrecision(task='binary')

    def training_step(self, batch, bx):
        logits = self.model(self.data.x, self.data.edge_index, self.data.edge_attr)
        logits = logits[self.idx_train]
        y = self.data.edge_label[self.idx_train]
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, y, pos_weight=self.pos_weight)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, bx):
        with torch.no_grad():
            logits = self.model(self.data.x, self.data.edge_index, self.data.edge_attr)[self.idx_val]
            y = self.data.edge_label[self.idx_val]
            prob = torch.sigmoid(logits)
            self.log('val_auroc', self.auroc(prob, y.int()))
            self.log('val_auprc', self.auprc(prob, y.int()))

    def test_step(self, batch, bx):
        with torch.no_grad():
            logits = self.model(self.data.x, self.data.edge_index, self.data.edge_attr)[self.idx_test]
            y = self.data.edge_label[self.idx_test]
            prob = torch.sigmoid(logits).cpu().numpy()
            y_np = y.cpu().numpy()
            rep = classification_report(y_np, prob)
            for k,v in rep.items(): self.log(f'test_{k}', v)

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.hparams.lr)

if __name__=='__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_root', default='data/simulated')
    ap.add_argument('--hidden_dim', type=int, default=96)
    ap.add_argument('--lr', type=float, default=5e-4)
    ap.add_argument('--max_epochs', type=int, default=5)
    args = ap.parse_args()
    seed_all(42)
    trainer = pl.Trainer(max_epochs=args.max_epochs, enable_checkpointing=False, logger=False)
    module = EdgeModule(args.data_root, args.hidden_dim, args.lr, args.max_epochs)
    trainer.fit(module); trainer.test(module)
