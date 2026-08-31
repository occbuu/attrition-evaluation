"""
leakcheck -- screen a reported classification result for the signature of
resampling applied before the train/test split.

The premise is narrow and worth stating plainly.  When synthetic over-sampling
is applied to a whole dataset and the result is then split into training and
test partitions, interpolated copies of test-set minority points end up in the
training partition.  The reported minority-class F1 stops describing the
problem and starts describing the classifier's capacity to memorise near
duplicates -- so it lands in a narrow band regardless of how hard the problem
actually is.  That band is what this package recognises.

    >>> import leakcheck
    >>> r = leakcheck.check(f1=0.92, imbalance_ratio=5.2, n_minority=237,
    ...                     n_features=44, clf="RandomForest")
    >>> round(r.probability, 2)          # doctest: +SKIP
    0.97
    >>> print(r.verdict)                 # doctest: +SKIP
    likely leaked

What this is not.  A high probability is not evidence of misconduct and not a
finding of error.  It says the reported number sits where leaked results sit
and honest results on a problem that hard rarely do.  It is a prompt to ask the
authors where the resampler sits in their pipeline -- nothing more.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence
import math

import numpy as np

from ._forest import Forest, load_forest, export_sklearn

__version__ = "1.0.0"
__all__ = ["check", "check_many", "Result", "MODEL", "model_info", "selftest",
           "Forest", "load_forest", "export_sklearn"]

MODEL: Forest = load_forest()

# Model families the screening model was trained on.  Anything else is mapped
# to the closest family by memorisation capacity, which is the property that
# matters: a model that can memorise interpolated duplicates shows the effect
# most strongly.
_FAMILY = {
    "logreg": "LogReg", "logisticregression": "LogReg", "logistic": "LogReg",
    "lr": "LogReg", "linear": "LogReg", "lda": "LogReg", "naivebayes": "LogReg",
    "gnb": "LogReg", "ridge": "LogReg",
    "dectree": "DecTree", "decisiontree": "DecTree", "cart": "DecTree",
    "tree": "DecTree", "j48": "DecTree",
    "5nn": "5NN", "knn": "5NN", "kneighbors": "5NN", "nearestneighbors": "5NN",
    "svm-rbf": "SVM-RBF", "svm": "SVM-RBF", "svc": "SVM-RBF", "rbf": "SVM-RBF",
    "randforest": "RandForest", "randomforest": "RandForest", "rf": "RandForest",
    "extratrees": "RandForest", "bagging": "RandForest",
    "gradboost": "GradBoost", "gradientboosting": "GradBoost", "gbm": "GradBoost",
    "adaboost": "GradBoost", "catboost": "GradBoost",
    "xgboost": "XGBoost", "xgb": "XGBoost", "lightgbm": "XGBoost", "lgbm": "XGBoost",
}
_DEFAULT_FAMILY = "RandForest"

THRESHOLDS = {"unremarkable": 0.35, "worth asking about": 0.65, "likely leaked": 1.01}


@dataclass
class Result:
    """The outcome of one screen."""
    probability: float
    verdict: str
    f1: float
    imbalance_ratio: float
    n_minority: int
    n_features: int
    clf: str
    family: str
    expected_honest_f1: float | None = None
    notes: list[str] = field(default_factory=list)

    def __repr__(self) -> str:
        return (f"Result(probability={self.probability:.3f}, "
                f"verdict={self.verdict!r}, f1={self.f1}, "
                f"imbalance_ratio={self.imbalance_ratio})")

    def explain(self) -> str:
        """A paragraph a reviewer can paste into a report."""
        lines = [
            f"Reported minority-class F1 of {self.f1:.3f} on a problem with "
            f"imbalance ratio {self.imbalance_ratio:g}, {self.n_minority} minority "
            f"cases and {self.n_features} features, using {self.clf} "
            f"(screened as the {self.family} family).",
            f"Estimated probability that this result was produced by resampling "
            f"before the train/test split: {self.probability:.2f} -- {self.verdict}.",
        ]
        if self.expected_honest_f1 is not None:
            lines.append(
                f"On this corpus, correctly evaluated results at that difficulty "
                f"average about F1 {self.expected_honest_f1:.2f}.")
        lines += self.notes
        lines.append("This is a screen, not a finding. The appropriate response is to "
                     "ask where the resampler sits relative to the split.")
        return " ".join(lines)


# ---------------------------------------------------------------------------
def _family(clf: str | None) -> str:
    if not clf:
        return _DEFAULT_FAMILY
    key = str(clf).strip().lower().replace(" ", "").replace("_", "")
    return _FAMILY.get(key, _DEFAULT_FAMILY)


def _row(f1: float, imbalance_ratio: float, n_minority: int, n_features: int,
         family: str) -> np.ndarray:
    x = {c: 0.0 for c in MODEL.columns}
    x["F1"] = float(f1)
    x["log_ir"] = math.log10(float(imbalance_ratio))
    x["log_nmin"] = math.log10(float(n_minority))
    x["log_p"] = math.log10(float(n_features))
    key = f"clf_{family}"
    if key in x:
        x[key] = 1.0
    return np.array([x[c] for c in MODEL.columns], dtype=float)


def _validate(f1, imbalance_ratio, n_minority, n_features) -> list[str]:
    notes = []
    if not 0.0 <= f1 <= 1.0:
        raise ValueError("f1 must be between 0 and 1")
    if imbalance_ratio <= 0:
        raise ValueError("imbalance_ratio must be positive")
    if n_minority < 1 or n_features < 1:
        raise ValueError("n_minority and n_features must be at least 1")
    m = MODEL.metadata
    lo, hi = m.get("ir_range", (1.0, 110.0))
    if not lo <= imbalance_ratio <= hi:
        notes.append(f"Imbalance ratio {imbalance_ratio:g} is outside the "
                     f"{lo:g}-{hi:g} range the model was fitted on, so the "
                     f"probability is an extrapolation.")
    lo, hi = m.get("nmin_range", (30, 2000))
    if not lo <= n_minority <= hi:
        notes.append(f"Minority count {n_minority} is outside the fitted "
                     f"{lo:g}-{hi:g} range.")
    return notes


def _verdict(p: float) -> str:
    for label, cut in THRESHOLDS.items():
        if p < cut:
            return label
    return "likely leaked"


def check(f1: float, imbalance_ratio: float, n_minority: int, n_features: int,
          clf: str | None = None) -> Result:
    """Screen a single reported result.

    Parameters
    ----------
    f1
        The reported minority-class F1.
    imbalance_ratio
        Majority cases divided by minority cases in the dataset as analysed.
    n_minority
        Number of minority cases before any resampling.
    n_features
        Number of predictors entering the model, after encoding.
    clf
        Model family, in whatever spelling the paper used.  Unrecognised names
        fall back to the tree-ensemble family, which is the common case.

    Returns
    -------
    Result
    """
    notes = _validate(f1, imbalance_ratio, n_minority, n_features)
    fam = _family(clf)
    x = _row(f1, imbalance_ratio, n_minority, n_features, fam)
    p = float(MODEL.predict_proba(x[None, :])[0])
    exp = MODEL.metadata.get("honest_curve")
    e = None
    if exp:
        li = math.log10(float(imbalance_ratio))
        e = float(np.clip(exp["intercept"] + exp["slope"] * li, 0.0, 1.0))
    return Result(probability=p, verdict=_verdict(p), f1=float(f1),
                  imbalance_ratio=float(imbalance_ratio), n_minority=int(n_minority),
                  n_features=int(n_features), clf=str(clf or "unspecified"),
                  family=fam, expected_honest_f1=e, notes=notes)


def check_many(records: Iterable[dict] | "Sequence[dict]"):
    """Screen many results at once.

    Accepts an iterable of dicts with the same keys as :func:`check`.  Returns a
    pandas DataFrame if pandas is importable, otherwise a list of Result.
    """
    out = [check(**r) for r in records]
    try:
        import pandas as pd
    except ImportError:
        return out
    return pd.DataFrame([{
        "f1": r.f1, "imbalance_ratio": r.imbalance_ratio, "n_minority": r.n_minority,
        "n_features": r.n_features, "clf": r.clf, "family": r.family,
        "probability": r.probability, "verdict": r.verdict,
        "expected_honest_f1": r.expected_honest_f1} for r in out])


def model_info() -> dict:
    """Provenance of the shipped screening model."""
    d = dict(MODEL.metadata)
    d["n_trees"] = len(MODEL.trees)
    d["features"] = list(MODEL.columns)
    d["version"] = __version__
    return d


def selftest() -> bool:
    """Check the shipped model against its stored regression fixtures."""
    fx = MODEL.metadata.get("fixtures", [])
    if not fx:
        raise RuntimeError("this model.json carries no fixtures")
    ok = True
    for row in fx:
        got = check(**{k: row[k] for k in
                       ("f1", "imbalance_ratio", "n_minority", "n_features", "clf")}).probability
        if abs(got - row["probability"]) > 1e-6:
            ok = False
            print(f"MISMATCH {row} -> {got:.8f}")
    return ok
