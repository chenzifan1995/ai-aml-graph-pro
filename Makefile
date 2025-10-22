.PHONY: data edge node rules test lint

data:
	python -m aml_graph.data.generate --out data/simulated --n_accounts 3000 --n_edges 12000 --seed 7

edge:
	python -m aml_graph.train_edge

node:
	python -m aml_graph.train_node

rules:
	python -m aml_graph.baselines.rules --data_root data/simulated

test:
	pytest -q

lint:
	python -m pip install flake8 && flake8 aml_graph
