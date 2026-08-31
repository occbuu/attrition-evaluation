#!/usr/bin/env python3
"""
04_export_model.py -- fit the shipped screening model and export it to JSON.

The model that ships inside the leakcheck package is fitted here, on the same
paired corpus the manuscript analyses, and written out as plain JSON so that
installing the tool pulls in nothing but NumPy.  Regression fixtures are stored
alongside it so `leakcheck.selftest()` can prove that the pure-NumPy evaluator
reproduces scikit-learn's predictions exactly.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import roc_auc_score

import leakage_lib as L

# Import the tree-walker module directly rather than through the package, because
# importing the package loads model.json -- which is the file this script writes.
import importlib.util
_spec = importlib.util.spec_from_file_location(
    "_forest", L.ROOT / "Paper2B" / "leakcheck" / "src" / "leakcheck" / "_forest.py")
_forest = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_forest)
export_sklearn, Forest = _forest.export_sklearn, _forest.Forest

OUT = L.ROOT / "Paper2B" / "leakcheck" / "src" / "leakcheck" / "model.json"


def build_frame():
    R = pd.read_csv(L.RES / "raw_results.csv")
    if "error" in R.columns:
        R = R[R["error"].isna()]
    piv = L.paired(R, L.manifest())
    diag = pd.concat([piv.assign(F1=piv.leaked, lab=1),
                      piv.assign(F1=piv.correct, lab=0)], ignore_index=True)
    diag["log_ir"] = np.log10(diag.imbalance_ratio)
    diag["log_nmin"] = np.log10(diag.n_minority)
    diag["log_p"] = np.log10(diag.n_features)
    X = pd.concat([diag[["F1", "log_ir", "log_nmin", "log_p"]],
                   pd.get_dummies(diag.clf, prefix="clf").astype(float)], axis=1)
    return X, diag.lab.to_numpy(), diag.dataset.to_numpy(), piv


def main() -> int:
    X, y, groups, piv = build_frame()
    mdl = GradientBoostingClassifier(random_state=0)

    oof = np.zeros(len(y))
    for tr, te in LeaveOneGroupOut().split(X, y, groups):
        m = GradientBoostingClassifier(random_state=0).fit(X.iloc[tr], y[tr])
        oof[te] = m.predict_proba(X.iloc[te])[:, 1]
    auc = float(roc_auc_score(y, oof))
    print(f"leave-one-dataset-out AUC = {auc:.4f}")

    final = mdl.fit(X, y)

    # honest reference curve: mean correct F1 as a function of log10 imbalance
    slope, intercept = np.polyfit(np.log10(piv.imbalance_ratio), piv.correct, 1)

    meta = dict(
        trained_on="leakage-premium corpus",
        n_datasets=int(len(np.unique(groups))),
        n_examples=int(len(y)),
        lodo_auc=round(auc, 4),
        ir_range=[float(piv.imbalance_ratio.min()), float(piv.imbalance_ratio.max())],
        nmin_range=[int(piv.n_minority.min()), int(piv.n_minority.max())],
        p_range=[int(piv.n_features.min()), int(piv.n_features.max())],
        classifiers=sorted(piv.clf.unique().tolist()),
        resamplers=sorted(piv.sampler.unique().tolist()),
        honest_curve=dict(slope=round(float(slope), 5), intercept=round(float(intercept), 5)),
    )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    export_sklearn(final, list(X.columns), meta, OUT)

    # verify the pure-NumPy evaluator against scikit-learn, then store fixtures
    f = Forest(json.load(open(OUT)))
    got = f.predict_proba(X.to_numpy(dtype=float))
    want = final.predict_proba(X)[:, 1]
    err = float(np.abs(got - want).max())
    print(f"max |numpy - sklearn| = {err:.2e}")
    assert err < 1e-9, "JSON export does not reproduce the fitted model"

    fam_col = {c: c[4:] for c in X.columns if c.startswith("clf_")}
    fixtures = []
    rng = np.random.default_rng(0)
    for i in rng.choice(len(X), 12, replace=False):
        row = X.iloc[int(i)]
        fam = next((v for k, v in fam_col.items() if row[k] == 1.0), "RandForest")
        fixtures.append(dict(
            f1=round(float(row["F1"]), 6),
            imbalance_ratio=round(float(10 ** row["log_ir"]), 6),
            n_minority=int(round(10 ** row["log_nmin"])),
            n_features=int(round(10 ** row["log_p"])),
            clf=fam,
            probability=None))
    spec = json.load(open(OUT))
    forest = Forest(spec)

    def row_vec(fx):
        x = {c: 0.0 for c in spec["columns"]}
        x["F1"] = fx["f1"]; x["log_ir"] = np.log10(fx["imbalance_ratio"])
        x["log_nmin"] = np.log10(fx["n_minority"]); x["log_p"] = np.log10(fx["n_features"])
        k = f"clf_{fx['clf']}"
        if k in x:
            x[k] = 1.0
        return np.array([x[c] for c in spec["columns"]], dtype=float)

    for fx in fixtures:
        fx["probability"] = float(forest.predict_proba(row_vec(fx)[None, :])[0])
    spec["metadata"]["fixtures"] = fixtures
    json.dump(spec, open(OUT, "w"))
    print(f"wrote {OUT}  ({OUT.stat().st_size/1024:.0f} kB, {len(spec['trees'])} trees)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
