#!/usr/bin/env python3
"""
hrmj_extras.py -- robustness for the HRMJ recast, using files already on disk.

Does not re-run the 41-dataset factorial.

1. Recompute the protocol premium after dropping job_change_ds (labour-market
   mobility, not internal attrition) and after restricting to employer-style tables.
2. If the unused survey and attendance files in data/hr_case_study are present,
   join them onto general_data and re-evaluate three families with SMOTE only,
   ten seeds, three protocols.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

HERE = Path(__file__).resolve().parent
CODE = HERE if (HERE / "leakage_lib.py").exists() else HERE.parent
sys.path.insert(0, str(CODE))

import leakage_lib as L
from leakage_lib import TEST_SIZE
import importlib

fetch = importlib.import_module("01_fetch_datasets")

RES, PROC = L.RES, L.PROC
DATA = L.ROOT / "data"
CASE_DIR = DATA / "hr_case_study"
INTERNAL = ["ibm_hr", "healthcare_attr", "hr_case_study", "saudi_attrition", "turnover_russia"]
IBM_LIKE = ["ibm_hr", "healthcare_attr", "hr_case_study"]
CLFS = ["LogReg", "RandForest", "XGBoost"]
SEEDS = 10


def _r(x, k=4):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return None if not np.isfinite(v) else round(v, k)


def _boot_ci(a, n=2000, seed=0):
    a = np.asarray(a, float)
    a = a[np.isfinite(a)]
    rng = np.random.default_rng(seed)
    means = [a[rng.integers(0, len(a), len(a))].mean() for _ in range(n)]
    lo, hi = np.quantile(means, [0.025, 0.975])
    return _r(lo), _r(hi)


def subset_from_factorial() -> tuple[dict, pd.DataFrame]:
    R = pd.read_csv(RES / "raw_results.csv")
    if "error" in R.columns:
        R = R[R.error.isna()]
    man = L.manifest()
    piv = L.paired(R, man)
    H = piv[piv.domain == "HR"]
    keys = {}
    rows = []

    def pack(name, label, sub):
        n_ds = int(sub.dataset.nunique())
        prem = _r(sub.premium.mean())
        lo, hi = _boot_ci(sub.premium.to_numpy())
        keys[f"{name}_n"] = n_ds
        keys[f"{name}_premium"] = prem
        keys[f"{name}_premium_lo"] = lo
        keys[f"{name}_premium_hi"] = hi
        keys[f"{name}_correct"] = _r(sub.correct.mean())
        keys[f"{name}_leaked"] = _r(sub.leaked.mean())
        ir = sub.groupby("dataset").imbalance_ratio.first()
        pds = sub.groupby("dataset").premium.mean()
        if len(ir) >= 3:
            keys[f"{name}_corr_ir"] = _r(np.corrcoef(np.log10(ir), pds)[0, 1])
        rows.append(dict(
            subset=label, n_datasets=n_ds,
            correct=_r(sub.correct.mean()), leaked=_r(sub.leaked.mean()),
            premium=prem, ci_low=lo, ci_high=hi,
        ))

    pack("all6", "All six public tables", H)
    pack("internal", "Employer-style tables (drop job_change_ds)", H[H.dataset.isin(INTERNAL)])
    pack("ibm_like", "IBM-like schema (ibm_hr, healthcare_attr, hr_case_study)",
         H[H.dataset.isin(IBM_LIKE)])
    job = H[H.dataset == "job_change_ds"]
    keys["job_premium"] = _r(job.premium.mean())
    keys["job_correct"] = _r(job.correct.mean())
    keys["job_leaked"] = _r(job.leaked.mean())
    rows.append(dict(
        subset="job_change_ds only (labour-market mobility)", n_datasets=1,
        correct=keys["job_correct"], leaked=keys["job_leaked"],
        premium=keys["job_premium"], ci_low=_boot_ci(job.premium.to_numpy())[0],
        ci_high=_boot_ci(job.premium.to_numpy())[1],
    ))
    tab = pd.DataFrame(rows)
    tab.to_csv(RES / "table9_construct.csv", index=False)
    return keys, tab


def _attendance_summary(path: Path) -> pd.DataFrame:
    d = pd.read_csv(path)
    idcol = d.columns[0]
    days = d.drop(columns=[idcol])
    return pd.DataFrame({
        "EmployeeID": pd.to_numeric(d[idcol], errors="coerce"),
        path.stem + "_missing_frac": days.isna().mean(axis=1).to_numpy(),
        path.stem + "_n_recorded": days.notna().sum(axis=1).to_numpy(),
    })


def case_files_present() -> bool:
    needed = ["general_data.csv", "employee_survey_data.csv",
              "manager_survey_data.csv", "in_time.csv", "out_time.csv"]
    return CASE_DIR.is_dir() and all((CASE_DIR / f).exists() for f in needed)


def build_case_enriched() -> tuple[np.ndarray, np.ndarray, dict]:
    g = pd.read_csv(CASE_DIR / "general_data.csv")
    es = pd.read_csv(CASE_DIR / "employee_survey_data.csv")
    ms = pd.read_csv(CASE_DIR / "manager_survey_data.csv")
    inn = _attendance_summary(CASE_DIR / "in_time.csv")
    out = _attendance_summary(CASE_DIR / "out_time.csv")
    d = (g.merge(es, on="EmployeeID", how="left")
          .merge(ms, on="EmployeeID", how="left")
          .merge(inn, on="EmployeeID", how="left")
          .merge(out, on="EmployeeID", how="left"))
    n0 = len(d)
    d = d.drop(columns=["EmployeeID", "EmployeeCount", "StandardHours", "Over18"],
               errors="ignore")
    X, y, cat_idx = fetch.encode(d, "Attrition", "Yes")
    meta = dict(
        n=int(len(X)), n_features=int(X.shape[1]), n_minority=int(y.sum()),
        imbalance_ratio=_r((len(y) - y.sum()) / y.sum(), 2),
        n_before_encode=int(n0),
    )
    np.savez_compressed(PROC / "hr_case_study_survey.npz", X=X, y=y,
                        cat_idx=np.array(cat_idx, dtype=int))
    return X, y, meta


def _premium_ci(d: pd.DataFrame) -> tuple[float | None, float | None, float | None]:
    piv = (d[d.condition.isin(["leaked", "correct"])]
           .pivot_table(index=["clf", "seed"], columns="condition", values="f1")
           .dropna())
    prem = (piv["leaked"] - piv["correct"]).to_numpy()
    lo, hi = _boot_ci(prem)
    return _r(prem.mean()), lo, hi


def _case_keys(d: pd.DataFrame) -> dict:
    keys = {}
    for cond in ("none", "correct", "leaked"):
        keys[f"case_survey_{cond}_f1"] = _r(d[d.condition == cond].f1.mean())
        keys[f"case_survey_{cond}_mcc"] = _r(d[d.condition == cond].mcc.mean())
    prem, lo, hi = _premium_ci(d)
    keys["case_survey_premium"] = prem
    keys["case_survey_premium_lo"] = lo
    keys["case_survey_premium_hi"] = hi
    R = pd.read_csv(RES / "raw_results.csv")
    if "error" in R.columns:
        R = R[R.error.isna()]
    orig = R[(R.dataset == "hr_case_study") & (R.clf.isin(CLFS))
             & ((R.sampler == "SMOTE") | (R.condition == "none"))]
    for cond in ("none", "correct", "leaked"):
        keys[f"case_orig_{cond}_f1"] = _r(orig[orig.condition == cond].f1.mean())
        keys[f"case_orig_{cond}_mcc"] = _r(orig[orig.condition == cond].mcc.mean())
    prem, lo, hi = _premium_ci(orig)
    keys["case_orig_premium"] = prem
    keys["case_orig_premium_lo"] = lo
    keys["case_orig_premium_hi"] = hi
    return keys


def forensic_case_study() -> dict:
    """Univariate IBM replica + within-file duplicate structure. Not a row-wise IBM copy."""
    if not case_files_present():
        return {}
    g = pd.read_csv(CASE_DIR / "general_data.csv")
    es = pd.read_csv(CASE_DIR / "employee_survey_data.csv")
    ms = pd.read_csv(CASE_DIR / "manager_survey_data.csv")
    drop = ["EmployeeID", "EmployeeCount", "StandardHours", "Over18"]
    sub = g.drop(columns=[c for c in drop if c in g.columns])
    vc = sub.fillna("__NA__").value_counts()
    keys = dict(
        case_n_rows=int(len(g)),
        case_n_ids=int(g.EmployeeID.nunique()),
        case_n_unique_profiles=int(vc.shape[0]),
        case_n_triple=int((vc == 3).sum()),
        case_n_double=int((vc == 2).sum()),
        case_n_single=int((vc == 1).sum()),
        case_n_leavers=int((g.Attrition == "Yes").sum()),
    )
    ibm_path = DATA / "DataOld" / "raw" / "HR-Employee-Attrition.csv"
    if ibm_path.exists():
        ibm = pd.read_csv(ibm_path)
        shared = [c for c in g.columns if c in ibm.columns and c not in drop]
        ibm_t = set(map(tuple, ibm[shared].fillna(-999).astype(str).to_numpy()))
        case_t = set(map(tuple, g[shared].fillna(-999).astype(str).drop_duplicates().to_numpy()))
        keys["ibm_case_shared_overlap"] = int(len(ibm_t & case_t))
        keys["ibm_n"] = int(len(ibm))
        keys["ibm_n_leavers"] = int((ibm.Attrition == "Yes").sum())
        g2 = g[shared].copy()
        if "MonthlyIncome" in g2.columns:
            g2["MonthlyIncome"] = g2["MonthlyIncome"] / 10.0
            ibm2 = ibm[shared].copy()
            ibm_s = set(map(tuple, ibm2.fillna(-999).astype(str).to_numpy()))
            case_s = set(map(tuple, g2.fillna(-999).astype(str).drop_duplicates().to_numpy()))
            keys["ibm_case_overlap_income_div10"] = int(len(ibm_s & case_s))
        items = ["JobSatisfaction", "EnvironmentSatisfaction", "RelationshipSatisfaction",
                 "WorkLifeBalance", "JobInvolvement", "PerformanceRating"]
        y = (ibm["Attrition"] == "Yes").astype(int)
        rs = []
        for c in items:
            if c not in ibm.columns:
                continue
            r = float(np.corrcoef(pd.to_numeric(ibm[c], errors="coerce"), y)[0, 1])
            keys[f"ibm_r_{c}"] = _r(r, 3)
            if c != "PerformanceRating":
                rs.append(r)
        keys["ibm_r_attitudes_min"] = _r(min(rs), 3)
        keys["ibm_r_attitudes_max"] = _r(max(rs), 3)
    d = g.merge(es, on="EmployeeID").merge(ms, on="EmployeeID")
    d["_k"] = sub.fillna("__NA__").astype(str).agg("|".join, axis=1)
    for c in ["EnvironmentSatisfaction", "JobSatisfaction", "WorkLifeBalance",
              "JobInvolvement", "PerformanceRating"]:
        nuniq = d.groupby("_k")[c].nunique(dropna=False)
        keys[f"survey_groups_disagree_{c}"] = int((nuniq > 1).sum())
    return keys


def clf_spread_keys() -> dict:
    R = pd.read_csv(RES / "raw_results.csv")
    if "error" in R.columns:
        R = R[R.error.isna()]
    man = L.manifest()
    H = L.paired(R, man)
    H = H[H.domain == "HR"]
    hclf = H.groupby("clf").correct.mean().sort_values()
    return dict(
        hr_clf_correct_min=_r(hclf.min()),
        hr_clf_correct_min_name=str(hclf.index[0]),
        hr_clf_correct_max=_r(hclf.max()),
        hr_clf_correct_max_name=str(hclf.index[-1]),
        hr_clf_spread_correct=_r(float(hclf.max() - hclf.min())),
    )


def run_case_enriched(force: bool = False) -> dict:
    cache = RES / "hr_case_study_survey_results.csv"
    meta_path = RES / "hr_case_study_survey_meta.json"
    if not force and cache.exists():
        d = pd.read_csv(cache)
        keys = _case_keys(d)
        if meta_path.exists():
            keys.update({f"case_enriched_{k}": v
                         for k, v in json.loads(meta_path.read_text(encoding="utf-8")).items()})
        return keys

    if not case_files_present():
        print("  skip case-study join: raw hr_case_study files not on disk")
        return {}

    X, y, meta = build_case_enriched()
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    rows = []
    for seed in range(SEEDS):
        Xtr, Xte, ytr, yte = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=seed, stratify=y)
        for cname in CLFS:
            r = L.fit_eval(L.make_clfs(seed)[cname], Xtr, ytr, Xte, yte, None)
            rows.append(dict(dataset="hr_case_study_survey", clf=cname, sampler="none",
                             condition="none", seed=seed, **r))
        for cname in CLFS:
            Xr, yr = L.make_samplers(seed)["SMOTE"].fit_resample(X, y)
            Ltr, Lte, ltr, lte = train_test_split(
                Xr, yr, test_size=TEST_SIZE, random_state=seed, stratify=yr)
            r = L.fit_eval(L.make_clfs(seed)[cname], Ltr, ltr, Lte, lte, None)
            rows.append(dict(dataset="hr_case_study_survey", clf=cname, sampler="SMOTE",
                             condition="leaked", seed=seed, **r))
            r = L.fit_eval(L.make_clfs(seed)[cname], Xtr, ytr, Xte, yte,
                           L.make_samplers(seed)["SMOTE"])
            rows.append(dict(dataset="hr_case_study_survey", clf=cname, sampler="SMOTE",
                             condition="correct", seed=seed, **r))
        print(f"  case-survey seed {seed} done", flush=True)
    d = pd.DataFrame(rows)
    d.to_csv(cache, index=False)
    keys = _case_keys(d)
    keys.update({f"case_enriched_{k}": v for k, v in meta.items()})
    return keys


def run_case_dedup(force: bool = False) -> dict:
    """Deduplicate on every non-identifier column (including income and tenure).

    An earlier key that dropped MonthlyIncome / tenure was used only to force
    n=1,470; that is not a valid unique-employee definition and is not used here.
    """
    cache = RES / "hr_case_study_dedup_results.csv"
    if not force and cache.exists():
        df = pd.read_csv(cache)
        meta = {}
        if (RES / "hr_case_study_dedup_meta.json").exists():
            meta = json.loads((RES / "hr_case_study_dedup_meta.json").read_text(encoding="utf-8"))
        # refuse the invalid 1,470-row forced key
        if meta.get("n_unique") == 1470 or (not meta and force is False and False):
            pass
    else:
        df = None
        meta = {}

    need_run = force or df is None
    if not need_run and (RES / "hr_case_study_dedup_meta.json").exists():
        meta = json.loads((RES / "hr_case_study_dedup_meta.json").read_text(encoding="utf-8"))
        need_run = int(meta.get("n_unique", 0)) == 1470

    if need_run:
        if not case_files_present():
            return {}
        g = pd.read_csv(CASE_DIR / "general_data.csv")
        es = pd.read_csv(CASE_DIR / "employee_survey_data.csv")
        ms = pd.read_csv(CASE_DIR / "manager_survey_data.csv")
        inn = _attendance_summary(CASE_DIR / "in_time.csv")
        out = _attendance_summary(CASE_DIR / "out_time.csv")
        d = (g.merge(es, on="EmployeeID", how="left")
              .merge(ms, on="EmployeeID", how="left")
              .merge(inn, on="EmployeeID", how="left")
              .merge(out, on="EmployeeID", how="left"))
        core_cols = [c for c in g.columns
                     if c not in ["EmployeeID", "EmployeeCount", "StandardHours", "Over18"]]
        d_dedup = d.drop_duplicates(subset=core_cols, keep="first").reset_index(drop=True)
        g_dedup = d_dedup[list(g.columns)].drop(
            columns=["EmployeeID", "EmployeeCount", "StandardHours", "Over18"], errors="ignore")
        d_dedup_clean = d_dedup.drop(
            columns=["EmployeeID", "EmployeeCount", "StandardHours", "Over18"], errors="ignore")
        X_33, y_33, _ = fetch.encode(g_dedup, "Attrition", "Yes")
        X_42, y_42, _ = fetch.encode(d_dedup_clean, "Attrition", "Yes")
        meta = dict(n_unique=int(len(X_33)), n_minority=int(y_33.sum()),
                    n_features_33=int(X_33.shape[1]), n_features_42=int(X_42.shape[1]),
                    key="all_non_id_columns")
        (RES / "hr_case_study_dedup_meta.json").write_text(
            json.dumps(meta, indent=2), encoding="utf-8")
        rows = []
        for label, X, y in [("dedup_33", X_33, y_33), ("dedup_42", X_42, y_42)]:
            for seed in range(SEEDS):
                Xtr, Xte, ytr, yte = train_test_split(
                    X, y, test_size=TEST_SIZE, random_state=seed, stratify=y)
                for cname in CLFS:
                    r = L.fit_eval(L.make_clfs(seed)[cname], Xtr, ytr, Xte, yte, None)
                    rows.append(dict(dataset=label, clf=cname, sampler="none",
                                     condition="none", seed=seed, **r))
                    Xr, yr = L.make_samplers(seed)["SMOTE"].fit_resample(X, y)
                    Ltr, Lte, ltr, lte = train_test_split(
                        Xr, yr, test_size=TEST_SIZE, random_state=seed, stratify=yr)
                    r_lk = L.fit_eval(L.make_clfs(seed)[cname], Ltr, ltr, Lte, lte, None)
                    rows.append(dict(dataset=label, clf=cname, sampler="SMOTE",
                                     condition="leaked", seed=seed, **r_lk))
                    r_co = L.fit_eval(L.make_clfs(seed)[cname], Xtr, ytr, Xte, yte,
                                      L.make_samplers(seed)["SMOTE"])
                    rows.append(dict(dataset=label, clf=cname, sampler="SMOTE",
                                     condition="correct", seed=seed, **r_co))
                print(f"  case-dedup seed {seed} {label} done", flush=True)
        df = pd.DataFrame(rows)
        df.to_csv(cache, index=False)
    else:
        meta = json.loads((RES / "hr_case_study_dedup_meta.json").read_text(encoding="utf-8"))

    keys = {"case_dedup_n": int(meta.get("n_unique", len(df))),
            "case_dedup_n_minority": int(meta.get("n_minority", 0)),
            "case_dedup_key": meta.get("key", "all_non_id_columns")}
    for label in ("dedup_33", "dedup_42"):
        sub = df[df.dataset == label]
        for cond in ("none", "correct", "leaked"):
            keys[f"case_{label}_{cond}_f1"] = _r(sub[sub.condition == cond].f1.mean())
            keys[f"case_{label}_{cond}_mcc"] = _r(sub[sub.condition == cond].mcc.mean())
        co = sub[sub.condition == "correct"].set_index(["clf", "seed"]).f1
        lk = sub[sub.condition == "leaked"].set_index(["clf", "seed"]).f1
        prem = (lk - co).to_numpy()
        keys[f"case_{label}_premium"] = _r(prem.mean())
        lo, hi = _boot_ci(prem)
        keys[f"case_{label}_premium_lo"] = lo
        keys[f"case_{label}_premium_hi"] = hi
    return keys


def make_all(force: bool = False) -> dict:
    keys, _ = subset_from_factorial()
    keys.update(clf_spread_keys())
    keys.update(forensic_case_study())
    keys.update(run_case_enriched(force=force))
    keys.update(run_case_dedup(force=force))
    (RES / "summary_hrmj.json").write_text(
        json.dumps(keys, indent=2, sort_keys=True), encoding="utf-8")
    print(f"wrote {RES / 'summary_hrmj.json'} ({len(keys)} keys)")
    return keys


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--force", action="store_true")
    args = p.parse_args()
    make_all(force=args.force)
