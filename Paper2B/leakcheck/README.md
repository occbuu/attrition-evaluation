# leakcheck

Screen a reported classification result for the signature of resampling applied
**before** the train/test split.

[![tests](https://img.shields.io/badge/tests-pytest-blue)](tests/)
[![license](https://img.shields.io/badge/license-BSD--3--Clause-green)](LICENSE)

## Why

Applying SMOTE or a related over-sampler to a whole dataset and only then
splitting it puts interpolated copies of test points into the training set. The
reported minority-class F1 then measures the model's capacity to memorise near
duplicates, not its ability to generalise -- and it stops tracking how hard the
problem is. A score that is high *for a problem that difficult* is the signal
this package looks for.

## Install

```bash
pip install leakcheck            # runtime dependency: numpy only
pip install 'leakcheck[table]'   # adds pandas, for screening a CSV
```

## Use

```python
import leakcheck

r = leakcheck.check(f1=0.92, imbalance_ratio=5.2, n_minority=237,
                    n_features=44, clf="RandomForest")

r.probability      # e.g. 0.97
r.verdict          # 'likely leaked'
print(r.explain()) # a paragraph you can paste into a review
```

Screen many at once:

```python
import pandas as pd, leakcheck
leakcheck.check_many(pd.read_csv("reported.csv").to_dict("records"))
```

Or from the shell:

```bash
leakcheck --f1 0.92 --ir 5.2 --n-minority 237 --n-features 44 --clf RandomForest
leakcheck --file reported.csv --format csv
leakcheck --info          # provenance of the shipped model
```

## Inputs

| Argument | Meaning |
|---|---|
| `f1` | reported minority-class F1 |
| `imbalance_ratio` | majority cases / minority cases |
| `n_minority` | minority cases **before** any resampling |
| `n_features` | predictors entering the model, after encoding |
| `clf` | model family, in any spelling; unknown names fall back to the tree-ensemble family |

## What the answer means

The probability is *P(a result of this size, on a problem of this difficulty,
came from the leaked ordering)*. It is **not** evidence that a paper is wrong,
and the package never says that it is. Innocent explanations produce the same
signature: a macro-averaged F1 reported as if it were minority-class, unusual
preprocessing, a genuinely separable dataset. The right response to a high
probability is a question -- *where does the resampler sit relative to the
split?* -- not a conclusion.

Inputs outside the ranges the model was fitted on come back with a note saying
the answer is an extrapolation.

## How the model was built

A controlled study fitted the same models on the same datasets both ways, so the
correct label is known by construction, and a gradient-boosted classifier was
trained to tell the two apart from reported quantities alone. Validation is
leave-one-dataset-out: the model never sees the dataset it is scored on. See
`leakcheck.model_info()` for the fitted ranges, the classifier families covered
and the validated AUC.

The trained model ships as plain JSON and is evaluated by a small pure-NumPy tree
walker, so the package has no scikit-learn dependency and results do not change
when scikit-learn does. `leakcheck.selftest()` verifies the shipped model against
stored fixtures.

## Development

```bash
pip install -e '.[dev]'
pytest -q
```

## Citation

If you use this tool, please cite the software paper and the companion
methodological study; `CITATION.cff` carries the machine-readable form.

## License

BSD 3-Clause.
