#!/usr/bin/env python3
"""
build_paper2B.py -- the software paper.

Target venue: Journal of Open Source Software.  JOSS wants a short paper (a
statement of need, a summary, a note on the state of the field) plus a working,
tested, documented repository; the repository is the submission and the paper is
the front door.  We therefore also write out paper.md and paper.bib in the
format JOSS expects, alongside the .docx for the authors' own records.
"""
from __future__ import annotations
import sys, shutil
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from paperkit import Numbers, Paper, table_csv, fmt, ROOT
from refs import Bib

PKG = ROOT / "Paper2B" / "leakcheck"
OUT = ROOT / "Paper2B" / "Paper2B_leakcheck_JOSS.docx"


# ---------------------------------------------------------------------------
JOSS_MD = """---
title: 'leakcheck: screening reported classification results for the resampling-before-splitting artefact'
tags:
  - Python
  - data leakage
  - class imbalance
  - SMOTE
  - reproducibility
  - peer review
authors:
  - name: Author names withheld for review
    affiliation: 1
affiliations:
  - name: Affiliation withheld for review
    index: 1
date: {date}
bibliography: paper.bib
---

# Summary

`leakcheck` estimates the probability that a published classification result was
produced by applying a synthetic over-sampler to a dataset *before* splitting it
into training and test partitions, rather than to the training partition alone.
It takes only quantities that papers already report -- the headline
minority-class F1, the imbalance ratio, the number of minority cases, the number
of features and the model family -- and returns a calibrated probability with a
short written explanation. It runs from Python or from the command line, on one
result or on a table of them.

The screen is a gradient-boosted classifier trained on {n_examples} labelled
results from a controlled study of {n_datasets} public datasets, in which the
same model was fitted both ways on the same data so that the correct label is
known by construction. It is validated leave-one-dataset-out, giving an AUC of
{auc}. The trained model ships as plain JSON and is evaluated by a
small pure-NumPy tree walker, so installing `leakcheck` pulls in nothing but
NumPy and a result computed today is reproducible years from now, independent of
any scikit-learn version.

# Statement of need

Applying SMOTE [@chawla2002] or a related over-sampler to a whole dataset and
only then splitting it is one of the most common forms of data leakage in
applied machine learning [@kaufman2012; @kapoor2023; @vandewiele2021]. The
resulting score describes the model's ability to recognise interpolated copies of
its own training points rather than its ability to generalise.

Reviewers and editors are rarely in a position to run an author's code, and
often cannot see it. The advice available to them is qualitative -- *watch out
for leakage* -- and does not help with a specific number in a specific
manuscript. `leakcheck` converts that advice into an answer about the case in
hand: given what this paper reports, how unusual is this number?

The empirical basis is that the artefact has a stable signature. Correctly
evaluated scores track how hard a problem is; leaked scores do not, collapsing
into a narrow band determined mostly by the classifier's capacity to memorise
near-duplicates. A reported score that is high *for a problem that difficult* is
therefore informative in a way that a high score alone is not.

`leakcheck` is intended for three uses:

1. **Peer review.** Decide whether to ask the authors where the resampler sits
   in their pipeline. The tool prints a paragraph suitable for pasting into a
   report, which states plainly that the output is a screen and not a finding.
2. **Self-audit.** Check one's own results before submission.
3. **Meta-research.** Screen a corpus of published results to estimate how
   widespread the pattern is in a field. `leakcheck --file results.csv` does this
   in one call.

# A worked example

```python
import leakcheck

r = leakcheck.check(f1=0.92, imbalance_ratio=5.2, n_minority=237,
                    n_features=44, clf="RandomForest")
print(r.probability, r.verdict)
print(r.explain())
```

From the command line:

```console
$ leakcheck --f1 0.92 --ir 5.2 --n-minority 237 --n-features 44 \\
            --clf RandomForest
```

# What the tool deliberately does not do

`leakcheck` returns a probability that a number of that size, on a problem of
that difficulty, came from the leaked ordering. It cannot observe a pipeline, so
it cannot establish that one is wrong. Several innocent explanations produce the
same signature: a macro-averaged rather than minority-class F1, a different
preprocessing path, an unusually separable dataset. The package documentation and
the `explain()` output both say so, and no verdict string asserts that an error
occurred. Inputs outside the fitted ranges are answered with an explicit note
that the result is an extrapolation, rather than silently.

# State of the field

Tooling for leakage detection is thin, and what exists operates on code or data
rather than on reported results. `deepchecks` and similar validation libraries
detect train/test contamination when they can see both partitions.
Reproducibility checklists [@kapoor2023] ask authors to attest to correct
practice but give a reader no way to check. Imbalance libraries such as
`imbalanced-learn` [@lemaitre2017] provide pipeline objects that make the correct
ordering easy, which prevents the error prospectively but says nothing about the
existing literature. `leakcheck` occupies the remaining position: it works from
the outside, on results that have already been published, using only what was
published.

# Quality control

The package has no dependency beyond NumPy at run time. The test suite covers
input validation, model-family alias resolution, monotonicity of the probability
in the reported score, agreement between batched and single-record evaluation,
the command line interface, and a set of stored regression fixtures verified to
reproduce the original scikit-learn predictions to within 1e-9. `leakcheck.selftest()`
runs the fixture check on an installed copy.

# Acknowledgements

The corpus on which the screening model is trained is described in the companion
methodological study [@companion].

# References
"""

JOSS_BIB = """@article{chawla2002,
  title   = {{SMOTE}: Synthetic Minority Over-sampling Technique},
  author  = {Chawla, N. V. and Bowyer, K. W. and Hall, L. O. and Kegelmeyer, W. P.},
  journal = {Journal of Artificial Intelligence Research},
  volume  = {16},
  pages   = {321--357},
  year    = {2002},
  doi     = {10.1613/jair.953}
}

@article{kaufman2012,
  title   = {Leakage in data mining: formulation, detection, and avoidance},
  author  = {Kaufman, Shachar and Rosset, Saharon and Perlich, Claudia and Stitelman, Ori},
  journal = {ACM Transactions on Knowledge Discovery from Data},
  volume  = {6},
  number  = {4},
  pages   = {15:1--15:21},
  year    = {2012},
  doi     = {10.1145/2382577.2382579}
}

@article{kapoor2023,
  title   = {Leakage and the reproducibility crisis in machine-learning-based science},
  author  = {Kapoor, Sayash and Narayanan, Arvind},
  journal = {Patterns},
  volume  = {4},
  number  = {9},
  pages   = {100804},
  year    = {2023},
  doi     = {10.1016/j.patter.2023.100804}
}

@article{vandewiele2021,
  title   = {Overly optimistic prediction results on imbalanced data: a case study of flaws and benefits when applying over-sampling},
  author  = {Vandewiele, Gilles and Dehaene, Isabelle and Kov{\\'a}cs, Gy{\\"o}rgy and others},
  journal = {Artificial Intelligence in Medicine},
  volume  = {111},
  pages   = {101987},
  year    = {2021},
  doi     = {10.1016/j.artmed.2020.101987}
}

@article{lemaitre2017,
  title   = {Imbalanced-learn: a {P}ython toolbox to tackle the curse of imbalanced datasets in machine learning},
  author  = {Lema{\\^i}tre, Guillaume and Nogueira, Fernando and Aridas, Christos K.},
  journal = {Journal of Machine Learning Research},
  volume  = {18},
  number  = {17},
  pages   = {1--5},
  year    = {2017}
}

@misc{companion,
  title  = {Leakage makes every problem look equally easy: a multi-dataset measurement of the resampling-before-splitting artefact},
  author = {Author names withheld for review},
  year   = {2026},
  note   = {Companion methodological study, under review}
}
"""

README = """# leakcheck

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
"""


def build() -> Path:
    N = Numbers()
    B = Bib()

    # ---- files that make up the JOSS submission ------------------------
    info = {
        "date": "2026",
        "n_examples": f"{N.raw('diag_n'):,}",
        "n_datasets": N.raw("n_datasets"),
        "auc": f"{N.raw('diag_auc'):.3f}",
    }
    (PKG / "paper.md").write_text(JOSS_MD.format(**info), encoding="utf-8")
    (PKG / "paper.bib").write_text(JOSS_BIB, encoding="utf-8")
    (PKG / "README.md").write_text(README, encoding="utf-8")
    (PKG / "CITATION.cff").write_text(
        "cff-version: 1.2.0\n"
        "message: \"If you use this software, please cite it as below.\"\n"
        "title: leakcheck\n"
        "version: 1.0.0\n"
        "abstract: >-\n"
        "  Screen a reported classification result for the signature of resampling\n"
        "  applied before the train/test split.\n"
        "authors:\n"
        "  - name: \"Author names withheld for review\"\n"
        "license: BSD-3-Clause\n"
        "date-released: 2026-01-01\n", encoding="utf-8")

    # ---- the .docx version --------------------------------------------
    d = Paper(N)
    d.title("leakcheck: screening reported classification results for the "
            "resampling-before-splitting artefact")
    d.authors("Author names withheld for review", "Affiliations withheld for review")

    d.abstract(
        """leakcheck estimates the probability that a published classification result was
        produced by applying a synthetic over-sampler to a dataset before splitting it into
        training and test partitions, rather than to the training partition alone. It takes only
        quantities that papers already report - the headline minority-class F1, the imbalance
        ratio, the number of minority cases, the number of features and the model family - and
        returns a calibrated probability together with a short written explanation. The
        screening model is a gradient-boosted classifier trained on {diag_n} labelled results
        from a controlled study of {n_datasets} public datasets in which the same models were
        fitted both ways on the same data, so the correct label is known by construction;
        leave-one-dataset-out validation gives an AUC of {diag_auc}. The trained model ships as
        plain JSON and is evaluated by a pure-NumPy tree walker, so the package depends on
        nothing but NumPy at run time and a result computed today does not change when a
        modelling library does.""")

    d.keywords("data leakage; class imbalance; SMOTE; peer review; research software; "
               "reproducibility")

    d.h1("1  Summary")
    d.p(f"""Applying SMOTE {B.c('chawla2002')} or a related over-sampler to a whole dataset and
        only then splitting it into training and test partitions is one of the most frequently
        reported forms of data leakage in applied machine learning
        {B.c('kaufman2012', 'kapoor2023', 'vandewiele2021')}. The score that results describes
        the model's ability to recognise interpolated copies of its own training points, not its
        ability to generalise to new cases.""")
    d.p("""leakcheck is a small Python package that screens a reported result for that error's
        signature. It is designed for readers rather than authors: it needs no access to code,
        no access to data, and nothing beyond what a methods section and a results table already
        contain.""")

    d.h1("2  Statement of need")
    d.p("""Reviewers and editors are rarely able to run an author's code and often cannot see it.
        The guidance available to them is qualitative and does not help with a specific number in
        a specific manuscript. leakcheck turns the general warning into a question about the case
        in hand: given what this paper reports, how unusual is this number?""")

    d.p("""The empirical basis is a regularity established in the companion study. Correctly
        evaluated scores track how hard a problem is; leaked scores do not, collapsing into a
        narrow band determined mostly by the classifier's capacity to memorise near-duplicates.
        In that corpus the honest score carries {variance_ratio} times the variance of the
        leaked score. A reported score that is high for a problem that difficult is therefore
        informative in a way that a high score on its own is not.""")

    d.bullets([
        """Peer review. Decide whether to ask the authors where the resampler sits in their
        pipeline. The tool prints a paragraph suitable for pasting into a report, which states
        that the output is a screen rather than a finding.""",
        "Self-audit. Check one's own results before submission.",
        """Meta-research. Screen a corpus of published results to estimate how widespread the
        pattern is in a field; a single command accepts a table.""",
    ])

    d.h1("3  How the screening model was built")
    d.p("""Each matched pair in the companion corpus yields two labelled examples: a leaked
        result and an honest one, identical in every respect a paper would report except the
        number itself. A gradient-boosted classifier is trained to separate them from the
        reported F1, the log imbalance ratio, the log minority count, the log feature count and
        the model family. Validation is leave-one-dataset-out, so the model is never scored on a
        dataset it has seen; that is the validation that would expose a screen which had merely
        memorised the corpus.""")

    d.p("""The reported AUC is {diag_auc}, with accuracy {diag_acc} at the natural threshold and
        a Brier score of {diag_brier}. Per-dataset AUC has median
        {diag_auc_perdataset_median} and minimum {diag_auc_perdataset_min}. An ablation that
        removes the reported F1 and leaves only the dataset descriptors falls to
        {diag_auc_without_f1}, close to chance, confirming that the signal lies in the
        relationship between the score and the difficulty of the problem rather than in the
        descriptors themselves.""")

    d.h1("4  Design decisions worth stating")
    d.h2("4.1  The model ships as JSON, not as a pickle")
    d.p("""A fitted scikit-learn estimator serialised with pickle binds every future user to one
        library version and offers no way to inspect what was stored. The screening model is
        instead exported as plain JSON - thresholds, split features and leaf values - and
        evaluated by a short pure-NumPy tree walker included in the package. The package
        therefore has one run-time dependency, the stored model is human-readable, and a
        probability computed today is reproducible indefinitely. A stored fixture set verifies
        that the NumPy evaluator reproduces the original scikit-learn predictions to within
        1e-9, and leakcheck.selftest() runs that check on an installed copy.""")

    d.h2("4.2  Unknown model families resolve rather than fail")
    d.p("""Papers name their models in many ways. The package maps around thirty spellings onto
        the seven families in the training corpus, and an unrecognised name falls back to the
        tree-ensemble family with the fallback documented, rather than raising. The purpose is
        a tool that gives a usable answer on real bibliographic input.""")

    d.h2("4.3  Extrapolation is announced")
    d.p("""Inputs outside the fitted ranges - imbalance ratios beyond {ir_min} to {ir_max}, or
        minority counts outside {minority_min} to {minority_max} - still return a probability,
        but the result carries an explicit note that the answer is an extrapolation. Silently
        returning a confident number outside the region where the model was fitted would be the
        same category of error the package exists to detect.""")

    d.h2("4.4  The output is a question, not a verdict")
    d.p("""No verdict string asserts that an error occurred; the strongest is 'likely leaked',
        and every explanation ends by saying that the appropriate response is to ask where the
        resampler sits. Several innocent explanations produce the same signature: a
        macro-averaged F1 reported as a minority-class F1, unusual preprocessing, a genuinely
        separable dataset. A tool of this kind that overstated its own authority would do more
        harm than the error it screens for.""")

    d.h1("5  State of the field")
    d.p(f"""Existing tooling for leakage operates on code or on data. Validation libraries detect
        train/test contamination when they can see both partitions. Reproducibility checklists
        {B.c('kapoor2023')} ask authors to attest to correct practice but give a reader no means
        of checking. Pipeline objects in imbalanced-learn {B.c('lemaitre2017')} make the correct
        ordering easy and so prevent the error prospectively, but say nothing about the
        literature that already exists. leakcheck occupies the remaining position: it works from
        the outside, on results that have already been published, using only what was
        published.""")

    d.h1("6  Quality control")
    d.p("""The test suite covers input validation, model-family alias resolution, monotonicity of
        the probability in the reported score, agreement between batched and single-record
        evaluation, the command line interface, and the stored regression fixtures. The package
        is installable from source with a standard build backend and carries a machine-readable
        citation file.""")

    d.h1("Availability")
    d.p("""Source, tests, documentation and the exported screening model are released under the
        BSD 3-Clause licence. Repository and archival DOI to be inserted on acceptance. The
        corpus on which the screening model was trained is described in the companion
        methodological study, released alongside it.""")

    d.references(B.list())
    p = d.save(OUT)
    if N.missing:
        raise SystemExit(f"unresolved placeholders: {sorted(N.missing)}")
    print(f"wrote {p}")
    print(f"wrote {PKG/'paper.md'}, {PKG/'paper.bib'}, {PKG/'README.md'}, "
          f"{PKG/'CITATION.cff'}")
    return p


if __name__ == "__main__":
    build()
