#!/usr/bin/env python3
"""
run_simulation.py -- the controlled simulation.

Real datasets confound imbalance ratio with sample size, dimensionality and
class overlap, because in practice those move together.  A synthetic grid pulls
them apart.  For every cell we fit the same classifier under the leaked and the
correct ordering, so the cell yields a premium whose causes are known by
construction.

    python run_simulation.py --seeds 3
"""
from __future__ import annotations
import argparse, sys, time
import numpy as np, pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import leakage_lib as L

IRS = [2, 4, 8, 16, 32]
DIMS = [10, 30, 80]
SEPS = [0.6, 1.0, 1.6]
NS = [800, 3000]
CLFS = ["LogReg", "DecTree", "5NN", "RandForest", "XGBoost"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    a = ap.parse_args()
    out = L.RES / "simulation.csv"
    done = set()
    if out.exists():
        prev = pd.read_csv(out)
        done = set(map(tuple, prev[["ir", "dim", "sep", "n", "seed"]].to_numpy().tolist()))

    t0 = time.time()
    for ir in IRS:
        for dim in DIMS:
            for sep in SEPS:
                for n in NS:
                    rows = []
                    for seed in range(a.seeds):
                        key = (ir, dim, sep, n, seed)
                        if key in done:
                            continue
                        w = 1.0 / (1.0 + ir)
                        X, y = make_classification(
                            n_samples=n, n_features=dim,
                            n_informative=max(3, dim // 3), n_redundant=max(1, dim // 10),
                            weights=[1 - w, w], flip_y=0.01, class_sep=sep,
                            random_state=seed)
                        if y.sum() < L.MIN_MIN_SIM:
                            continue
                        Xtr, Xte, ytr, yte = train_test_split(
                            X, y, test_size=L.TEST_SIZE, random_state=seed, stratify=y)
                        try:
                            Xr, yr = L.make_samplers(seed)["SMOTE"].fit_resample(X, y)
                            Ltr, Lte, ltr, lte = train_test_split(
                                Xr, yr, test_size=L.TEST_SIZE, random_state=seed, stratify=yr)
                        except Exception:
                            continue
                        cont = L.contamination(X, y, L.make_samplers(seed)["SMOTE"], seed)
                        for c in CLFS:
                            rec = dict(ir=ir, dim=dim, sep=sep, n=n, seed=seed,
                                       clf=c, contam=cont)
                            try:
                                rec["leaked"] = L.fit_eval(L.make_clfs(seed)[c], Ltr, ltr,
                                                           Lte, lte, None)["f1"]
                                rec["correct"] = L.fit_eval(L.make_clfs(seed)[c], Xtr, ytr,
                                                            Xte, yte,
                                                            L.make_samplers(seed)["SMOTE"])["f1"]
                                rec["none"] = L.fit_eval(L.make_clfs(seed)[c], Xtr, ytr,
                                                         Xte, yte, None)["f1"]
                            except Exception as e:
                                rec["error"] = f"{type(e).__name__}"[:40]
                            rows.append(rec)
                    if rows:
                        df = pd.DataFrame(rows)
                        df.to_csv(out, mode="a", header=not out.exists(), index=False)
                        print(f"ir={ir:2d} dim={dim:2d} sep={sep} n={n:5d}  "
                              f"{len(df):4d} rows  [{(time.time()-t0)/60:6.1f} min]", flush=True)
    print("SIMULATION COMPLETE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
