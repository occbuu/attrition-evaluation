#!/usr/bin/env python3
"""
01_fetch_datasets.py -- build the dataset corpus for the leakage-premium study.

Two groups:

  A. 35 imbalanced binary benchmarks derived from the Penn Machine Learning
     Benchmarks (PMLB, Olson et al. 2017; Romano et al. 2021).  Downloaded
     automatically over HTTPS and cached in data/pmlb_cache/.  Multi-class
     sources are binarised one-vs-rest on the rarest class with at least
     MIN_MINORITY cases -- the same construction imbalanced-learn used for its
     Zenodo benchmark collection, which is unreachable from this network.

  B. 6 HR / employee-turnover datasets already present under data/.  Each has a
     dedicated loader; the script reports which are missing and skips them, so
     it is safe to re-run as files arrive.

Everything is harmonised to X (float matrix, no NaN) and y (1 = minority class),
plus a metadata row.  Output: data/processed/<name>.npz and manifest.csv.

Usage:
    python 01_fetch_datasets.py
    python 01_fetch_datasets.py --group A --max-rows 5000
"""
from __future__ import annotations
import argparse, sys, os, urllib.request, warnings
from pathlib import Path
import numpy as np, pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
PROC = DATA / "processed"
PMLB = DATA / "pmlb_cache"
PROC.mkdir(parents=True, exist_ok=True); PMLB.mkdir(parents=True, exist_ok=True)

SEED = 42
MIN_MINORITY = 30           # below this a 70:30 split leaves too few test positives
DEFAULT_MAX_ROWS = 5000     # stratified cap; keeps the 41-dataset factorial tractable

# PMLB stores its data files in git-lfs, so the media endpoint is required.
PMLB_URL = ("https://media.githubusercontent.com/media/EpistasisLab/pmlb/"
            "master/datasets/{n}/{n}.tsv.gz")

# name -> positive (minority) class label in the PMLB `target` column.
# Chosen as the rarest class with >= MIN_MINORITY cases; for genuinely binary
# sources this is simply the minority label.
GROUP_A: dict[str, int] = {
    "bupa": 0, "sonar": 1, "credit_approval_australia": 0, "spambase": 1,
    "ionosphere": 0, "magic": 1, "tic_tac_toe": 0, "titanic": 1,
    "credit_approval_germany": 1, "phoneme": 1, "haberman": 2, "dna": 2,
    "adult": 0, "contraceptive_method": 1, "clean2": 1, "segmentation": 3,
    "churn": 1, "dermatology": 4, "ecoli": 4, "mfeat_karhunen": 9,
    "optdigits": 8, "satimage": 4, "pendigits": 3, "vowel": 7,
    "analcatdata_authorship": 2, "coil2000": 1, "hypothyroid": 0,
    "car_evaluation": 3, "letter": 26, "wine_quality_white": 4,
    "ann_thyroid": 1, "allhyper": 2, "yeast": 7, "dis": 0, "allrep": 1,
}


# ---------------------------------------------------------------------------
# harmonisation
# ---------------------------------------------------------------------------
def encode(df: pd.DataFrame, target: str, positive) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """One-hot nominal columns, median-impute numerics, return X, y, categorical indices."""
    y = (df[target].isin(positive) if isinstance(positive, (list, set, tuple))
         else df[target] == positive).astype(int).to_numpy()
    X = df.drop(columns=[target])
    X = X.loc[:, X.nunique(dropna=False) > 1]                      # drop constants
    X = X.loc[:, [not (X[c].dtype == object and X[c].nunique() > 50) for c in X.columns]]
    num = X.select_dtypes(include=[np.number]).columns
    cat = [c for c in X.columns if c not in num]
    for c in num:
        X[c] = pd.to_numeric(X[c], errors="coerce").fillna(pd.to_numeric(X[c], errors="coerce").median())
    for c in cat:
        X[c] = X[c].astype(str).fillna("missing")
    X = pd.get_dummies(X, columns=list(cat), drop_first=True).astype(float)
    X = X.replace([np.inf, -np.inf], 0.0).fillna(0.0)
    cat_idx = [i for i, c in enumerate(X.columns) if set(np.unique(X[c].to_numpy())) <= {0.0, 1.0}]
    if y.mean() > 0.5:                                             # ensure 1 = minority
        y = 1 - y
    return X.to_numpy(dtype=float), y, cat_idx


def save(name, X, y, cat_idx, domain, source, note, max_rows, rows) -> None:
    if len(X) > max_rows:                                          # stratified subsample
        from sklearn.model_selection import train_test_split
        idx, _ = train_test_split(np.arange(len(X)), train_size=max_rows,
                                  random_state=SEED, stratify=y)
        X, y = X[idx], y[idx]
    n_min = int(y.sum())
    if n_min < MIN_MINORITY or n_min == len(y):
        print(f"  ! {name}: only {n_min} minority cases -- skipped"); return
    np.savez_compressed(PROC / f"{name}.npz", X=X, y=y, cat_idx=np.array(cat_idx, dtype=int))
    ir = (len(y) - n_min) / n_min
    rows.append(dict(name=name, domain=domain, n=len(X), n_features=X.shape[1],
                     n_minority=n_min, imbalance_ratio=round(ir, 2),
                     n_categorical=len(cat_idx), source=source, construction=note))
    print(f"  ok {name:26s} n={len(X):6d}  p={X.shape[1]:4d}  min={n_min:5d}  IR={ir:7.1f}")


# ---------------------------------------------------------------------------
# Group A -- PMLB
# ---------------------------------------------------------------------------
def _pmlb(name: str) -> pd.DataFrame:
    p = PMLB / f"{name}.tsv.gz"
    stale = (not p.exists()) or (p.stat().st_size < 1000 and p.read_bytes()[:7] == b"version")
    if stale:
        with urllib.request.urlopen(PMLB_URL.format(n=name), timeout=120) as r:
            blob = r.read()
        p.write_bytes(blob)
    return pd.read_csv(p, sep="\t")


def group_a(max_rows: int, rows: list) -> None:
    print(f"\n[Group A] {len(GROUP_A)} PMLB benchmarks (automatic download -> data/pmlb_cache/)")
    for name, pos in GROUP_A.items():
        try:
            df = _pmlb(name)
        except Exception as e:
            print(f"  ! {name}: download failed ({type(e).__name__}: {e})"); continue
        k = df["target"].nunique()
        note = "binary" if k == 2 else f"one-vs-rest on class {pos} of {k}"
        y = (df["target"] == pos).astype(int).to_numpy()
        X = df.drop(columns=["target"]).astype(float)
        X = X.loc[:, X.nunique() > 1]
        X = X.replace([np.inf, -np.inf], np.nan)
        if X.isna().any().any():                       # PMLB keeps a few NaNs
            X = X.fillna(X.median(numeric_only=True)).fillna(0.0)
        cat_idx = [i for i, c in enumerate(X.columns)
                   if set(np.unique(X[c].to_numpy())) <= {0.0, 1.0}]
        save(name, X.to_numpy(float), y, cat_idx, "benchmark",
             "PMLB (Olson et al. 2017; Romano et al. 2021)", note, max_rows, rows)


# ---------------------------------------------------------------------------
# Group B -- HR datasets already downloaded under data/
# ---------------------------------------------------------------------------
def _ibm(p):
    df = pd.read_csv(p).drop(columns=["EmployeeCount", "StandardHours", "Over18",
                                      "EmployeeNumber"], errors="ignore")
    return encode(df, "Attrition", "Yes")

def _jobchange(p):
    df = pd.read_csv(p).drop(columns=["enrollee_id"], errors="ignore")
    df = df.dropna(subset=["target"])
    return encode(df, "target", 1.0)

def _casestudy(p):
    df = pd.read_csv(p).drop(columns=["EmployeeID", "EmployeeCount", "StandardHours",
                                      "Over18"], errors="ignore")
    return encode(df, "Attrition", "Yes")

def _turnover_ru(p):
    df = pd.read_csv(p, encoding="latin-1")
    return encode(df, "event", 1)

def _healthcare(p):
    df = pd.read_csv(p).drop(columns=["EmployeeID", "EmployeeCount", "StandardHours",
                                      "Over18"], errors="ignore")
    return encode(df, "Attrition", "Yes")

def _saudi(p):
    """Mendeley 6z2hty8php. Survey instrument: every field is an ordinal or nominal
    string, including Age and salary bands, so everything but the target is nominal."""
    df = pd.read_excel(p).drop(columns=["ID"], errors="ignore")
    df.columns = [c.replace("\xa0", " ").strip() for c in df.columns]
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].astype(str).str.replace("\xa0", " ").str.strip()
    return encode(df, "Attrition", "Yes")


HR = [
    ("ibm_hr",          ["HR-Employee-Attrition.csv", "WA_Fn-UseC_-HR-Employee-Attrition.csv"],
     _ibm,        "Kaggle pavansubhasht/ibm-hr-analytics-attrition-dataset"),
    ("saudi_attrition", ["Original Dataset of Employee Attrition.xlsx"],
     _saudi,      "Mendeley Data 10.17632/6z2hty8php.1 (Alsaleh & Alsaleh 2025)"),
    ("job_change_ds",   ["aug_train.csv"],
     _jobchange,  "Kaggle arashnic/hr-analytics-job-change-of-data-scientists"),
    ("hr_case_study",   ["general_data.csv"],
     _casestudy,  "Kaggle vjchoudhary7/hr-analytics-case-study"),
    ("turnover_russia", ["turnover.csv"],
     _turnover_ru, "Kaggle davinwijaya/employee-turnover"),
    ("healthcare_attr", ["watson_healthcare_modified.csv"],
     _healthcare, "Kaggle jpmiller/employee-attrition-for-healthcare"),
]

SEARCH_ROOTS = [DATA, ROOT.parent / "Paper1" / "Code"]
EXCLUDE = {"DataOld", "processed", "pmlb_cache"}


def find(patterns: list[str]) -> Path | None:
    files: list[Path] = []
    for root in SEARCH_ROOTS:
        if not root.exists():
            continue
        for f in root.rglob("*"):
            if f.is_file() and f.suffix.lower() in {".csv", ".xlsx", ".xls"} \
               and not (EXCLUDE & set(f.relative_to(root).parts)):
                files.append(f)
    for pat in patterns:
        for f in files:
            if pat.lower() in f.name.lower():
                return f
    return None


def group_b(max_rows: int, rows: list) -> None:
    print("\n[Group B] HR / employee-turnover datasets")
    missing = []
    for name, pats, loader, src in HR:
        p = find(pats)
        if p is None:
            missing.append((name, pats[0], src)); continue
        try:
            X, y, ci = loader(p)
            save(name, X, y, ci, "HR", src, f"file: {p.name}", max_rows, rows)
        except Exception as e:
            print(f"  ! {name}: {type(e).__name__}: {e}")
    if missing:
        print("\n  Not found -- add these under data/ and re-run:")
        for name, fn, src in missing:
            print(f"    - {name:18s} expects a file matching '{fn}'\n        {src}")


# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", choices=["A", "B", "all"], default="all")
    ap.add_argument("--max-rows", type=int, default=DEFAULT_MAX_ROWS)
    a = ap.parse_args()
    rows: list = []
    if a.group in ("A", "all"): group_a(a.max_rows, rows)
    if a.group in ("B", "all"): group_b(a.max_rows, rows)
    if not rows:
        print("\nNothing processed."); return 1
    m = pd.DataFrame(rows).sort_values(["domain", "imbalance_ratio"])
    if a.group == "all":
        m.to_csv(PROC / "manifest.csv", index=False)
    print(f"\n{'='*78}\n{len(m)} datasets in data/processed/  (manifest.csv written)")
    print(f"  imbalance ratio : {m.imbalance_ratio.min():.2f} - {m.imbalance_ratio.max():.1f}")
    print(f"  sample size     : {m.n.min()} - {m.n.max()}")
    print(f"  minority cases  : {m.n_minority.min()} - {m.n_minority.max()}")
    print(f"  features        : {m.n_features.min()} - {m.n_features.max()}")
    print(f"  by domain       : {dict(m.domain.value_counts())}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
