# AI for Anti-Money Laundering: A Graph-Based Approach to Suspicious Transaction Detection (Pro)

This repository provides a **research-grade** and **production-friendly** framework for
graph-based AML using **Graph Neural Networks (GNNs)** with **edge-level** suspicious transaction
detection and **node-level** suspicious account scoring.

## What's New vs. Basic Version
- **Temporal graph generator** with realistic AML patterns (smurfing, layering, circular flows).
- **Edge features** (amount z-scores, inter-event time, geo/region mismatch) and **node features**.
- **Hydra-configured** experiments (`configs/`) for clean overrides and sweeps.
- **Rule-based baseline** and **tabular ML baseline** for comparisons.
- **Imbalance-aware training** with `pos_weight` and AUPRC focus.
- **Explainability** with `GNNExplainer` for edge-level predictions.
- **CI pipeline**, **unit tests**, **Dockerfile**, **Makefile**, and **pyproject.toml**.

## Quickstart
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1) Generate synthetic data
python -m aml_graph.data.generate --out data/simulated --n_accounts 5000 --n_edges 20000 --seed 42

# 2) Train GNN (edge-level AML)
python -m aml_graph.train_edge task.max_epochs=5 task.hidden_dim=96 task.lr=5e-4

# 3) Train node scorer (optional)
python -m aml_graph.train_node task.max_epochs=5

# 4) Run rule-based baseline
python -m aml_graph.baselines.rules --data_root data/simulated

# 5) Explain top suspicious edges
python -m aml_graph.explain.explain_edge --data_root data/simulated --k 5
```

## Results & Metrics
Primary: **AUPRC**, Secondary: **AUROC**, **F1**. Class imbalance handled via `BCEWithLogitsLoss(pos_weight=...)`.

## Structure
```
aml_graph/
  data/           # generation, loading, feature engineering
  models/         # GAT edge model, MLP node scorer
  baselines/      # heuristics and tabular baseline
  explain/        # GNNExplainer utilities
  utils/          # metrics, seeding, splits
configs/
tests/
.github/workflows/ci.yml
Dockerfile
Makefile
pyproject.toml
requirements.txt
```
