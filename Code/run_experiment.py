#!/usr/bin/env python3
"""
run_experiment.py -- headless runner for the main factorial.

The grid is large enough that it is worth splitting across processes and
machines.  Datasets are assigned to shards by a greedy longest-processing-time
rule on an n x p cost proxy, so the shards finish at roughly the same time.
Each dataset is written to the shard's CSV as soon as it completes, so a run can
be interrupted and resumed.

    python run_experiment.py --shard 0 --nshards 4 --seeds 10
"""
from __future__ import annotations
import argparse, sys, time
from pathlib import Path
import pandas as pd
import leakage_lib as L


def shard_of(man: pd.DataFrame, nshards: int) -> dict[str, int]:
    cost = (man.n * man.n_features).astype(float)
    order = man.assign(cost=cost).sort_values("cost", ascending=False)
    load = [0.0] * nshards
    out = {}
    for _, r in order.iterrows():
        k = load.index(min(load))
        out[r["name"]] = k
        load[k] += r["cost"]
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()

    man = L.manifest()
    assign = shard_of(man, a.nshards)
    mine = [n for n in man.name if assign[n] == a.shard]
    out = L.RES / f"raw_shard{a.shard}{a.tag}.csv"

    done: set[str] = set()
    if out.exists():
        done = set(pd.read_csv(out).dataset.unique())

    print(f"shard {a.shard}/{a.nshards}: {len(mine)} datasets, {len(done)} already done", flush=True)
    t0 = time.time()
    for name in mine:
        if name in done:
            continue
        t = time.time()
        df = L.run_dataset(name, n_seeds=a.seeds)
        df.to_csv(out, mode="a", header=not out.exists(), index=False)
        print(f"{name:26s} {len(df):5d} rows  {time.time()-t:7.1f}s"
              f"  [{(time.time()-t0)/60:6.1f} min elapsed]", flush=True)
    print("SHARD COMPLETE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
