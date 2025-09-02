#!/usr/bin/env python3
import argparse, csv, json, os, time, math, random
import numpy as np
import torch as th

def set_seed(s=42):
    random.seed(s); np.random.seed(s); th.manual_seed(s)

def synthetic_batch(n=20000, dim=256):
    # Create a synthetic "database" of embeddings (n x dim) and a single query (dim)
    db = th.randn(n, dim)
    q  = th.randn(dim)
    return db, q

@th.no_grad()
def compute_metric(db: th.Tensor, q: th.Tensor, topk=10):
    # Cosine similarity; take mean of top-k as a simple quality proxy
    qn = q / (q.norm() + 1e-9)
    dbn = db / (db.norm(dim=1, keepdim=True) + 1e-9)
    sims = (dbn @ qn)
    vals, _ = th.topk(sims, k=topk)
    return float(vals.mean().item())

def run_once(n, dim, topk):
    db, q = synthetic_batch(n=n, dim=dim)
    start = time.perf_counter()
    metric = compute_metric(db, q, topk=topk)
    elapsed = time.perf_counter() - start
    max_mem_mb = th.cuda.max_memory_allocated() / (1024*1024) if th.cuda.is_available() else 0.0
    return {"metric": metric, "runtime_s": elapsed, "max_mem_mb": max_mem_mb}

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", type=int, default=3)
    p.add_argument("--n", type=int, default=20000)
    p.add_argument("--dim", type=int, default=256)
    p.add_argument("--topk", type=int, default=10)
    p.add_argument("--csv", type=str, default="bench.csv")
    args = p.parse_args()

    set_seed(42)
    results=[]
    for _ in range(args.runs):
        r = run_once(args.n, args.dim, args.topk)
        results.append(r)

    runtime_s = float(np.mean([r["runtime_s"] for r in results]))
    metric    = float(np.mean([r["metric"] for r in results]))
    correct = (not math.isnan(metric)) and (-1.0 <= metric <= 1.0)

    row = {
        "commit": os.environ.get("GITHUB_SHA", "local"),
        "runs": args.runs,
        "n": args.n,
        "dim": args.dim,
        "topk": args.topk,
        "runtime_s": round(runtime_s, 6),
        "metric": round(metric, 6),
        "correct": int(correct),
        "device": "cpu"
    }

    # Append CSV (create with header if missing)
    write_header = not os.path.exists(args.csv)
    with open(args.csv, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header: w.writeheader()
        w.writerow(row)

    print(json.dumps(row, indent=2))

if __name__ == "__main__":
    main()
