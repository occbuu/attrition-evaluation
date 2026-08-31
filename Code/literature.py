"""
literature.py -- reported results from the published attrition literature.

Every row was read from the publisher's own version of record while writing this
manuscript.  `protocol` records what the paper itself says about where the
resampler sits relative to the train/test split, in three levels:

    "train-only"    the paper states resampling was confined to training data
    "before-split"  the paper states resampling was applied before splitting
    "not stated"    the paper does not say

The last category is the point.  We classify studies by what they report, never
by what our screening model guesses, and we make no claim about any individual
study's correctness.

`f1` is the headline minority-class F1 as reported.  Where a study reports
several models we take the one the authors nominate as best, because that is the
number a reader carries away.
"""
from __future__ import annotations
import pandas as pd

ROWS = [
    dict(key="hamja2025", study="Hamja et al. (2025)", venue="Annals of Data Science",
         dataset="IBM HR", n=1470, n_features=35, n_minority=237, imbalance_ratio=5.2,
         model="GradientBoosting", clf="GradBoost", f1=0.99, protocol="not stated"),
    dict(key="nassreddine2026", study="Nassreddine et al. (2026)", venue="Computers",
         dataset="IBM HR", n=1470, n_features=35, n_minority=237, imbalance_ratio=5.2,
         model="Ensemble (ROS)", clf="RandForest", f1=0.9774, protocol="not stated"),
    dict(key="baydili2025", study="Baydili & Tasci (2025)", venue="Systems",
         dataset="IBM HR", n=1470, n_features=35, n_minority=237, imbalance_ratio=5.2,
         model="GAN-Transformer", clf="GradBoost", f1=0.9158, protocol="before-split"),
    dict(key="alali2026", study="AL-Ali et al. (2026)", venue="Scientific Reports",
         dataset="IBM HR", n=1470, n_features=35, n_minority=237, imbalance_ratio=5.2,
         model="AdaBoost", clf="GradBoost", f1=0.7969, protocol="not stated"),
    dict(key="alali2026", study="AL-Ali et al. (2026)", venue="Scientific Reports",
         dataset="Job change", n=19158, n_features=13, n_minority=4777,
         imbalance_ratio=3.0, model="HistGradientBoosting", clf="GradBoost",
         f1=0.6594, protocol="not stated"),
    dict(key="alyousef2026", study="Alyousef et al. (2026)", venue="Information",
         dataset="IBM HR", n=1470, n_features=35, n_minority=237, imbalance_ratio=5.2,
         model="LogisticRegression", clf="LogReg", f1=0.5042, protocol="train-only"),
    dict(key="cavescu2026", study="Cavescu & Popescu (2026)", venue="Information",
         dataset="IBM HR", n=1470, n_features=35, n_minority=237, imbalance_ratio=5.2,
         model="LogisticRegression", clf="LogReg", f1=0.439, protocol="train-only"),
]


def frame() -> pd.DataFrame:
    return pd.DataFrame(ROWS)


def screened() -> pd.DataFrame:
    """Attach the leakcheck probability to each reported result."""
    import sys
    from pathlib import Path
    root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(root / "Paper2B" / "leakcheck" / "src"))
    import leakcheck
    d = frame()
    d["p_leaked"] = [
        leakcheck.check(f1=r.f1, imbalance_ratio=r.imbalance_ratio,
                        n_minority=r.n_minority, n_features=r.n_features,
                        clf=r.clf).probability
        for r in d.itertuples()]
    return d
