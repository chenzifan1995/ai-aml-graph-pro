import os, json, argparse, numpy as np

# Patterns: smurfing (many small deposits), layering (chains), circular flows (A->B->C->A)

def generate(out_dir="data/simulated", n_accounts=5000, n_edges=20000, seed=42):
    rng = np.random.default_rng(seed)
    os.makedirs(out_dir, exist_ok=True)

    accounts = []
    for i in range(n_accounts):
        region = int(rng.integers(0, 20))
        kyc_score = float(np.clip(rng.normal(0.8, 0.1), 0, 1))
        activity = float(np.clip(rng.lognormal(0.0, 1.0), 0.1, 50.0))
        accounts.append({'account_id': i, 'region': region, 'kyc_score': kyc_score, 'activity': activity})

    edges = []
    timestamps = rng.integers(1_600_000_000, 1_700_000_000, size=n_edges)
    timestamps.sort()
    # baseline random edges
    for j in range(n_edges):
        src = int(rng.integers(0, n_accounts))
        dst = int(rng.integers(0, n_accounts))
        while dst == src:
            dst = int(rng.integers(0, n_accounts))
        amt = float(np.clip(rng.lognormal(3, 1), 1, 1e6))
        geo_mismatch = int(accounts[src]['region'] != accounts[dst]['region'])
        # base fraud probability
        p = 0.01 + 0.05*geo_mismatch + 0.1*max(0, 0.6-accounts[src]['kyc_score'])
        edges.append({'src': src, 'dst': dst, 'amount': amt, 'ts': int(timestamps[j]), 'label': 0, 'geo_mismatch': geo_mismatch})

    # Inject patterns
    # 1) Smurfing: pick hubs that receive many small incoming transfers
    for _ in range(n_accounts//50):
        hub = int(rng.integers(0, n_accounts))
        for k in range(int(rng.integers(10, 40))):
            src = int(rng.integers(0, n_accounts))
            amt = float(np.clip(rng.lognormal(2, 0.3), 1, 500))  # many small
            t = int(rng.integers(1_650_000_000, 1_670_000_000))
            edges.append({'src': src, 'dst': hub, 'amount': amt, 'ts': t, 'label': 1, 'geo_mismatch': int(accounts[src]['region']!=accounts[hub]['region'])})

    # 2) Layering chains: sequences with similar amounts
    for _ in range(n_accounts//100):
        length = int(rng.integers(4, 8))
        nodes = rng.choice(n_accounts, size=length, replace=False)
        base_amt = float(np.clip(rng.lognormal(5, 0.2), 5000, 100000))
        start_t = int(rng.integers(1_660_000_000, 1_680_000_000))
        for i in range(length-1):
            edges.append({'src': int(nodes[i]), 'dst': int(nodes[i+1]), 'amount': float(base_amt*(0.8+0.4*rng.random())),
                          'ts': start_t + i*3600, 'label': 1,
                          'geo_mismatch': int(nodes[i]%20 != nodes[i+1]%20)})

    # 3) Circular flows
    for _ in range(n_accounts//150):
        a,b,c = rng.choice(n_accounts, size=3, replace=False)
        t0 = int(rng.integers(1_665_000_000, 1_675_000_000))
        amt = float(np.clip(rng.lognormal(4, 0.5), 1000, 50000))
        edges += [
            {'src': int(a), 'dst': int(b), 'amount': amt, 'ts': t0, 'label': 1, 'geo_mismatch': int(a%20 != b%20)},
            {'src': int(b), 'dst': int(c), 'amount': amt*0.95, 'ts': t0+1800, 'label': 1, 'geo_mismatch': int(b%20 != c%20)},
            {'src': int(c), 'dst': int(a), 'amount': amt*0.9, 'ts': t0+3600, 'label': 1, 'geo_mismatch': int(c%20 != a%20)},
        ]

    # Save
    with open(os.path.join(out_dir, "accounts.jsonl"), "w") as f:
        for a in accounts: f.write(json.dumps(a) + "\n")
    with open(os.path.join(out_dir, "transactions.jsonl"), "w") as f:
        for e in edges: f.write(json.dumps(e) + "\n")
    meta = {'n_accounts': n_accounts, 'n_edges': len(edges)}
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Generated accounts={n_accounts}, edges={len(edges)} at {out_dir}")
