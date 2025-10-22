from aml_graph.data.generate import generate
from aml_graph.data.load import load_graph
import tempfile, os

def test_generate_and_load():
    with tempfile.TemporaryDirectory() as d:
        out = os.path.join(d, 'sim')
        generate(out, n_accounts=200, n_edges=800, seed=1)
        data = load_graph(out)
        assert data.x.shape[0] == 200
        assert data.edge_index.shape[1] > 0
        assert data.edge_attr.shape[0] == data.edge_index.shape[1]
