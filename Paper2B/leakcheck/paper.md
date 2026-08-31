---
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
date: 2026
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

The screen is a gradient-boosted classifier trained on 22,400 labelled
results from a controlled study of 41 public datasets, in which the
same model was fitted both ways on the same data so that the correct label is
known by construction. It is validated leave-one-dataset-out, giving an AUC of
0.795. The trained model ships as plain JSON and is evaluated by a
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
$ leakcheck --f1 0.92 --ir 5.2 --n-minority 237 --n-features 44 \
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
