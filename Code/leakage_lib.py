"""
leakage_lib.py -- shared machinery for the leakage-premium study.

Everything the notebooks and the headless runner need lives here, so that the
manuscripts, the notebooks and the batch run all execute identical code.

Three conditions, differing only in where the resampler sits relative to the
train/test split:

    leaked   resample the WHOLE dataset, then split 70:30
    correct  split 70:30, then resample the TRAINING partition only
    none     split 70:30, no resampling
"""
from __future__ import annotations
import time, warnings
from pathlib import Path
import numpy as np, pandas as pd

warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (f1_score, roc_auc_score, average_precision_score,
                             precision_score, recall_score, matthews_corrcoef)
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, ADASYN, RandomOverSampler
from imblearn.pipeline import Pipeline as ImbPipeline

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

SEED = 42
TEST_SIZE = 0.30
MIN_MIN_SIM = 20   # minimum minority cases for a simulation cell to be usable

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "data" / "processed"
RES = ROOT / "results"; RES.mkdir(exist_ok=True)
FIG = ROOT / "figures"; FIG.mkdir(exist_ok=True)

BLUE, RED, GREY, GREEN = "#2f5d8c", "#b5451b", "#7a7a7a", "#3c7a5e"


# ---------------------------------------------------------------------------
def load(name: str):
    z = np.load(PROC / f"{name}.npz")
    return z["X"], z["y"], list(z["cat_idx"])


def manifest() -> pd.DataFrame:
    return pd.read_csv(PROC / "manifest.csv")


def make_clfs(seed: int = SEED) -> dict:
    d = {
        "LogReg":     LogisticRegression(max_iter=3000, random_state=seed),
        "DecTree":    DecisionTreeClassifier(max_depth=6, random_state=seed),
        "5NN":        KNeighborsClassifier(5),
        "SVM-RBF":    SVC(C=1.0, random_state=seed, cache_size=400),
        "RandForest": RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=1),
        "GradBoost":  GradientBoostingClassifier(random_state=seed),
    }
    if HAS_XGB:
        d["XGBoost"] = XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.1,
                                     eval_metric="logloss", random_state=seed,
                                     n_jobs=1, verbosity=0)
    return d


def make_samplers(seed: int, sampling_strategy="auto") -> dict:
    return {
        "SMOTE":      SMOTE(random_state=seed, sampling_strategy=sampling_strategy),
        "Borderline": BorderlineSMOTE(random_state=seed, sampling_strategy=sampling_strategy),
        "ADASYN":     ADASYN(random_state=seed, sampling_strategy=sampling_strategy),
        "RandomOS":   RandomOverSampler(random_state=seed, sampling_strategy=sampling_strategy),
    }


CLF_NAMES = list(make_clfs())
SAMP_NAMES = list(make_samplers(0))


# ---------------------------------------------------------------------------
def scores(ytrue, ypred, yscore) -> dict:
    return dict(
        f1=f1_score(ytrue, ypred, zero_division=0),
        precision=precision_score(ytrue, ypred, zero_division=0),
        recall=recall_score(ytrue, ypred, zero_division=0),
        mcc=matthews_corrcoef(ytrue, ypred),
        roc_auc=roc_auc_score(ytrue, yscore) if len(set(ytrue)) > 1 else np.nan,
        pr_auc=average_precision_score(ytrue, yscore) if len(set(ytrue)) > 1 else np.nan,
    )


def fit_eval(clf, Xtr, ytr, Xte, yte, sampler=None) -> dict:
    steps = [("sc", StandardScaler())]
    if sampler is not None:
        steps.append(("rs", sampler))
    steps.append(("clf", clf))
    pipe = ImbPipeline(steps).fit(Xtr, ytr)
    pred = pipe.predict(Xte)
    try:
        sc = pipe.predict_proba(Xte)[:, 1]
    except Exception:
        sc = pipe.decision_function(Xte)
    return scores(yte, pred, sc)


def contamination(X, y, sampler, seed, p_test=TEST_SIZE) -> float:
    """Fraction of held-out minority points whose nearest synthetic training point
    is closer than half the natural spacing between real minority points.

    This is the mediator: it measures, in the units of the data itself, how much
    of the test set the leaked training partition has effectively already seen.
    """
    n0 = len(X)
    if not np.isfinite(X).all():
        return np.nan
    try:
        Xr, yr = sampler.fit_resample(X, y)
    except Exception:
        return np.nan
    if not np.isfinite(Xr).all():
        return np.nan
    if len(Xr) <= n0:
        return 0.0
    idx = np.arange(len(Xr))
    itr, ite = train_test_split(idx, test_size=p_test, random_state=seed, stratify=yr)
    te_min = [i for i in ite if i < n0 and y[i] == 1]
    tr_syn = [i for i in itr if i >= n0]
    if len(te_min) < 5 or len(tr_syn) < 5:
        return np.nan
    sc = StandardScaler().fit(X)
    Z = sc.transform(Xr)
    d, _ = NearestNeighbors(n_neighbors=1).fit(Z[tr_syn]).kneighbors(Z[te_min])
    Zm = sc.transform(X[y == 1])
    if len(Zm) < 2:
        return np.nan
    dm, _ = NearestNeighbors(n_neighbors=2).fit(Zm).kneighbors(Zm)
    ref = float(np.median(dm[:, 1]))
    return float((d[:, 0] < 0.5 * ref).mean()) if ref > 0 else np.nan


# ---------------------------------------------------------------------------
def run_dataset(name: str, clf_set=None, samp_set=None, n_seeds: int = 10,
                verbose: bool = False) -> pd.DataFrame:
    """The full three-condition factorial for one dataset."""
    clf_set = clf_set or CLF_NAMES
    samp_set = samp_set or SAMP_NAMES
    X, y, _ = load(name)
    rows = []
    for seed in range(n_seeds):
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=TEST_SIZE,
                                              random_state=seed, stratify=y)
        # ---- none -------------------------------------------------------
        for cname in clf_set:
            try:
                r = fit_eval(make_clfs(seed)[cname], Xtr, ytr, Xte, yte, None)
                rows.append(dict(dataset=name, clf=cname, sampler="none",
                                 condition="none", seed=seed, contam=np.nan, **r))
            except Exception as e:
                rows.append(dict(dataset=name, clf=cname, sampler="none",
                                 condition="none", seed=seed, error=f"{type(e).__name__}: {e}"[:90]))
        for sname in samp_set:
            # ---- leaked: resample everything, then split -----------------
            try:
                Xr, yr = make_samplers(seed)[sname].fit_resample(X, y)
                Ltr, Lte, ltr, lte = train_test_split(Xr, yr, test_size=TEST_SIZE,
                                                      random_state=seed, stratify=yr)
                ok_leak = True
            except Exception:
                ok_leak = False
            try:
                cont = contamination(X, y, make_samplers(seed)[sname], seed)
            except Exception:
                cont = np.nan
            for cname in clf_set:
                if ok_leak:
                    try:
                        r = fit_eval(make_clfs(seed)[cname], Ltr, ltr, Lte, lte, None)
                        rows.append(dict(dataset=name, clf=cname, sampler=sname,
                                         condition="leaked", seed=seed, contam=cont, **r))
                    except Exception as e:
                        rows.append(dict(dataset=name, clf=cname, sampler=sname,
                                         condition="leaked", seed=seed,
                                         error=f"{type(e).__name__}: {e}"[:90]))
                # ---- correct: split, then resample the training half -----
                try:
                    r = fit_eval(make_clfs(seed)[cname], Xtr, ytr, Xte, yte,
                                 make_samplers(seed)[sname])
                    rows.append(dict(dataset=name, clf=cname, sampler=sname,
                                     condition="correct", seed=seed, contam=cont, **r))
                except Exception as e:
                    rows.append(dict(dataset=name, clf=cname, sampler=sname,
                                     condition="correct", seed=seed,
                                     error=f"{type(e).__name__}: {e}"[:90]))
        if verbose:
            print(f"    {name} seed {seed} done", flush=True)
    return pd.DataFrame(rows).reindex(columns=RESULT_COLUMNS)


RESULT_COLUMNS = ["dataset", "clf", "sampler", "condition", "seed", "contam",
                  "f1", "precision", "recall", "mcc", "roc_auc", "pr_auc", "error"]


# ---------------------------------------------------------------------------
def paired(R: pd.DataFrame, man: pd.DataFrame) -> pd.DataFrame:
    """One row per (dataset, clf, sampler, seed) carrying leaked, correct and premium."""
    piv = (R[R.condition.isin(["leaked", "correct"])]
           .pivot_table(index=["dataset", "clf", "sampler", "seed"],
                        columns="condition", values="f1").dropna().reset_index())
    piv["premium"] = piv["leaked"] - piv["correct"]
    cont = (R[R.condition == "leaked"]
            .groupby(["dataset", "clf", "sampler", "seed"])["contam"].first().reset_index())
    piv = piv.merge(cont, on=["dataset", "clf", "sampler", "seed"], how="left")
    piv = piv.merge(man[["name", "domain", "n", "n_features", "n_minority", "imbalance_ratio"]],
                    left_on="dataset", right_on="name", how="left").drop(columns="name")
    return piv
