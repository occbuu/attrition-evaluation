"""
Pure-NumPy inference for a gradient-boosted tree ensemble.

The screening model is trained with scikit-learn but shipped as plain JSON and
evaluated here, so that installing leakcheck pulls in nothing but NumPy and a
result read today is bit-identical to the same result read in five years.
Unpickling a fitted estimator would tie every future user to one scikit-learn
version; a few hundred lines of thresholds do not.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

__all__ = ["Forest", "load_forest"]


class Forest:
    """A gradient-boosted ensemble of regression trees with a logistic link."""

    def __init__(self, spec: dict):
        self.columns: list[str] = spec["columns"]
        self.init: float = float(spec["init"])
        self.learning_rate: float = float(spec["learning_rate"])
        self.trees: list[dict] = spec["trees"]
        self.metadata: dict = spec.get("metadata", {})
        self._trees = [
            (np.asarray(t["feature"], dtype=np.int64),
             np.asarray(t["threshold"], dtype=np.float64),
             np.asarray(t["left"], dtype=np.int64),
             np.asarray(t["right"], dtype=np.int64),
             np.asarray(t["value"], dtype=np.float64))
            for t in self.trees
        ]

    # -- one tree, all rows at once ----------------------------------------
    @staticmethod
    def _apply(tree, X: np.ndarray) -> np.ndarray:
        feature, threshold, left, right, value = tree
        node = np.zeros(len(X), dtype=np.int64)
        active = feature[node] >= 0
        while active.any():
            idx = np.flatnonzero(active)
            n = node[idx]
            go_left = X[idx, feature[n]] <= threshold[n]
            node[idx] = np.where(go_left, left[n], right[n])
            active = np.zeros(len(X), dtype=bool)
            active[idx] = feature[node[idx]] >= 0
        return value[node]

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X[None, :]
        out = np.full(len(X), self.init, dtype=np.float64)
        for t in self._trees:
            out += self.learning_rate * self._apply(t, X)
        return out

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        z = self.decision_function(X)
        return 1.0 / (1.0 + np.exp(-z))


def load_forest(path: str | Path | None = None) -> Forest:
    path = Path(path) if path else Path(__file__).with_name("model.json")
    with open(path, "r", encoding="utf-8") as f:
        return Forest(json.load(f))


# ---------------------------------------------------------------------------
def export_sklearn(model, columns: list[str], metadata: dict, path: str | Path) -> dict:
    """Serialise a fitted sklearn GradientBoostingClassifier to the JSON above.

    Kept in the installed package rather than in the study repository so that
    anyone can regenerate model.json from their own corpus and check that this
    conversion is faithful -- `leakcheck.selftest` does exactly that.
    """
    spec = {
        "columns": list(columns),
        "init": float(model.init_.class_prior_[1] if hasattr(model, "init_") and
                      hasattr(model.init_, "class_prior_") else 0.0),
        "learning_rate": float(model.learning_rate),
        "trees": [],
        "metadata": metadata,
    }
    # sklearn stores the raw initial prediction in _raw_predict_init
    z0 = model._raw_predict_init(np.zeros((1, len(columns))))
    spec["init"] = float(np.ravel(z0)[0])
    for stage in model.estimators_:
        t = stage[0].tree_
        spec["trees"].append({
            "feature": [int(v) for v in t.feature],
            "threshold": [float(v) for v in t.threshold],
            "left": [int(v) for v in t.children_left],
            "right": [int(v) for v in t.children_right],
            "value": [float(v) for v in t.value.ravel()],
        })
    with open(path, "w", encoding="utf-8") as f:
        json.dump(spec, f)
    return spec
