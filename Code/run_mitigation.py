#!/usr/bin/env python3
"""
run_mitigation.py -- does partial oversampling shrink the artefact?

If the premium comes from synthetic minority points crossing the split, then
resampling to less than full balance should shrink it, because fewer synthetic
points exist to cross.  We sweep SMOTE's sampling_strategy over the whole
corpus and record the premium at each level.

    python run_mitigation.py --seeds 3
"""
from __future__ import annotations
import argparse, sys, time
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
import leakage_lib as L

STRATEGIES = [0.15, 0.25, 0.40, 0.60, 0.80, 1.00]
CLFS = ["LogReg", "DecTree", "RandForest"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    a = ap.parse_args()
    man = L.manifest()
    out = L.RES / "mitigation.csv"
    done = set()
    if out.exists():
        done = set(pd.read_csv(out).dataset.unique())

    t0 = time.time()
    for _, meta in man.iterrows():
        name = meta["name"]
        if name in done:
            continue
        X, y, _ = L.load(name)
        base = y.mean() / (1 - y.mean())          # current minority:majority ratio
        rows = []
        for seed in range(a.seeds):
            Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=L.TEST_SIZE,
                                                  random_state=seed, stratify=y)
            for ss in STRATEGIES:
                if ss <= base:                     # already at or above that prevalence
                    continue
                try:
                    smp = L.make_samplers(seed, sampling_strategy=ss)["SMOTE"]
                    Xr, yr = smp.fit_resample(X, y)
                    Ltr, Lte, ltr, lte = train_test_split(Xr, yr, test_size=L.TEST_SIZE,
                                                          random_state=seed, stratify=yr)
                except Exception:
                    continue
                cont = L.contamination(X, y, L.make_samplers(seed, sampling_strategy=ss)["SMOTE"],
                                       seed)
                for c in CLFS:
                    rec = dict(dataset=name, clf=c, seed=seed, sampling_strategy=ss,
                               contam=cont, imbalance_ratio=meta["imbalance_ratio"],
                               domain=meta["domain"])
                    try:
                        rec["leaked"] = L.fit_eval(L.make_clfs(seed)[c], Ltr, ltr,
                                                   Lte, lte, None)["f1"]
                        rec["correct"] = L.fit_eval(
                            L.make_clfs(seed)[c], Xtr, ytr, Xte, yte,
                            L.make_samplers(seed, sampling_strategy=ss)["SMOTE"])["f1"]
                    except Exception as e:
                        rec["error"] = f"{type(e).__name__}"[:40]
                    rows.append(rec)
        if rows:
            pd.DataFrame(rows).to_csv(out, mode="a", header=not out.exists(), index=False)
            print(f"{name:26s} {len(rows):4d} rows  [{(time.time()-t0)/60:6.1f} min]", flush=True)
    print("MITIGATION COMPLETE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
