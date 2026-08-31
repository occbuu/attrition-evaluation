import json
import math

import numpy as np
import pytest

import leakcheck


def test_model_loads():
    info = leakcheck.model_info()
    assert info["n_trees"] > 0
    assert "F1" in info["features"]
    assert info["lodo_auc"] > 0.5


def test_selftest_fixtures():
    assert leakcheck.selftest() is True


def test_basic_check():
    r = leakcheck.check(f1=0.92, imbalance_ratio=5.2, n_minority=237,
                        n_features=44, clf="RandomForest")
    assert 0.0 <= r.probability <= 1.0
    assert r.family == "RandForest"
    assert r.verdict in leakcheck.THRESHOLDS
    assert "F1" in r.explain() or "F1" in r.explain().upper()


def test_high_f1_on_hard_problem_scores_higher_than_low_f1():
    """The whole premise: on a hard problem, a very high score is the suspicious one."""
    hard_high = leakcheck.check(f1=0.95, imbalance_ratio=40, n_minority=100,
                                n_features=30, clf="RandForest").probability
    hard_low = leakcheck.check(f1=0.25, imbalance_ratio=40, n_minority=100,
                               n_features=30, clf="RandForest").probability
    assert hard_high > hard_low


def test_probability_increases_with_f1():
    ps = [leakcheck.check(f1=f, imbalance_ratio=20, n_minority=150, n_features=30,
                          clf="DecisionTree").probability
          for f in (0.3, 0.5, 0.7, 0.9)]
    assert ps == sorted(ps) or ps[-1] > ps[0]


def test_family_aliases():
    for name in ["xgboost", "XGB", "LightGBM", "lgbm"]:
        assert leakcheck.check(f1=.9, imbalance_ratio=5, n_minority=100,
                               n_features=20, clf=name).family == "XGBoost"
    for name in ["logistic regression", "LogReg", "LR"]:
        assert leakcheck.check(f1=.9, imbalance_ratio=5, n_minority=100,
                               n_features=20, clf=name).family == "LogReg"
    # an unknown name must not raise; it falls back to a documented default
    assert leakcheck.check(f1=.9, imbalance_ratio=5, n_minority=100,
                           n_features=20, clf="TabPFN").family == "RandForest"


def test_out_of_range_is_flagged_not_refused():
    r = leakcheck.check(f1=0.9, imbalance_ratio=5000, n_minority=40,
                        n_features=20, clf="RandomForest")
    assert r.notes, "an extrapolated input should carry a note"
    assert 0.0 <= r.probability <= 1.0


@pytest.mark.parametrize("kwargs", [
    dict(f1=1.4, imbalance_ratio=5, n_minority=100, n_features=20),
    dict(f1=0.9, imbalance_ratio=0, n_minority=100, n_features=20),
    dict(f1=0.9, imbalance_ratio=5, n_minority=0, n_features=20),
    dict(f1=0.9, imbalance_ratio=5, n_minority=100, n_features=0),
])
def test_invalid_inputs_raise(kwargs):
    with pytest.raises(ValueError):
        leakcheck.check(**kwargs)


def test_check_many():
    recs = [dict(f1=0.92, imbalance_ratio=5.2, n_minority=237, n_features=44, clf="RF"),
            dict(f1=0.44, imbalance_ratio=5.2, n_minority=237, n_features=44, clf="LogReg")]
    out = leakcheck.check_many(recs)
    assert len(out) == 2


def test_forest_matches_batch_and_single():
    from leakcheck._forest import load_forest
    f = load_forest()
    X = np.zeros((3, len(f.columns)))
    X[:, f.columns.index("F1")] = [0.2, 0.6, 0.95]
    X[:, f.columns.index("log_ir")] = math.log10(10)
    X[:, f.columns.index("log_nmin")] = math.log10(200)
    X[:, f.columns.index("log_p")] = math.log10(30)
    batch = f.predict_proba(X)
    single = np.array([f.predict_proba(x[None, :])[0] for x in X])
    assert np.allclose(batch, single)


def test_cli(capsys):
    from leakcheck.cli import main
    assert main(["--f1", "0.92", "--ir", "5.2", "--n-minority", "237",
                 "--n-features", "44", "--clf", "RandomForest"]) == 0
    out = capsys.readouterr().out
    assert "P(leaked)" in out
