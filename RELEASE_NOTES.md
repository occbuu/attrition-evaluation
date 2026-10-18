# Release notes

## v1.1.1 — Claim audit (local; not yet pushed)

Re-checked every headline number in the upgraded `_v1` manuscript against
`raw_results.csv`, `summary.json`, `summary_2c.json`, and a re-run of
`hrmj_extras.py`. Factorial leakage numbers are unchanged. Two upgraded
claims were corrected:

- **Duplicate profiles, not a row-wise IBM clone.** `hr_case_study` has
  4,410 distinct IDs; 1,498 unique complete-attribute profiles (1,442
  appear three times, 28 twice, 28 once). Row and leaver counts are
  exactly $3\times$ IBM HR and univariate distributions match, but
  row-wise overlap with IBM is zero. Dedup now uses every non-identifier
  column (not a key that dropped income/tenure to force $n=1470$).
- **Classifier spread endpoints.** Correct-protocol F1 ranges from 0.512
  (5NN) to 0.641 (random forest / XGBoost), spread 0.129 — not
  0.590–0.719.

Deduplicated three-family SMOTE F1: 0.322 (33 features) and 0.331 (42
features); premiums 0.511 and 0.516.

---

## v1.1.0 — HRMJ extras (local; not yet pushed)

Adds the construct-subset reanalysis and case-study join used in the HRMJ `_v1`
manuscript. Does **not** re-run the 41-dataset factorial. Headline leakage
numbers on the six public tables are unchanged.

### What's new

- `Code/hrmj_extras.py` rebuilds `results/summary_hrmj.json` and
  `results/table9_construct.csv` from `raw_results.csv`.
- If the raw `hr_case_study` CSVs are present locally, it joins employee
  survey, manager survey and in/out missingness onto `general_data`, writes
  `data/processed/hr_case_study_survey.npz`, and re-evaluates logistic
  regression, Random Forest and XGBoost with SMOTE (ten seeds, three
  protocols). The archived results of that run are in
  `results/hr_case_study_survey_results.csv`.
- `paper/` holds `Paper2C_hr_attrition_evaluation_v1.tex`, the bibliography
  and the four manuscript figures.

### Headline extras (from the archived factorial unless noted)

| Quantity | Value |
|---|---|
| Six-table leakage premium (minority-class F1) | 0.235 (95% CI 0.226–0.244) |
| Five employer-style tables (drop `job_change_ds`) | 0.231 (0.221–0.241) |
| Three IBM-like schemas | 0.350 (0.342–0.359) |
| `job_change_ds` only (labour-market mobility) | 0.256 |
| Case-study join, three families, SMOTE: original 33 features | correct 0.708, leaked 0.892, premium 0.184 |
| Same employees after survey/attendance join, 42 features | correct 0.707, leaked 0.900, premium 0.193 |
| Deduplicated unique profiles ($n=1498$), 33 features | correct 0.322, leaked 0.833, premium 0.511 |
| Deduplicated unique profiles ($n=1498$), 42 features | correct 0.331, leaked 0.847, premium 0.516 |

Joining unused HR-looking columns does not close the protocol gap. Unique
complete-attribute profiles, not a byte-for-byte IBM copy, are what the
dedup check evaluates.

### How to verify without re-running the factorial

```bash
cd Code
pip install -r requirements.txt
python 03_analysis.py
python 05_verify_numbers.py
python hrmj_extras.py
```

`hrmj_extras.py` will skip the case-study join if the raw CSVs are absent and
will reuse `hr_case_study_survey_results.csv` when it is already in `results/`.

---

## v1.0.0 — Initial public release

Replication archive for *What an attrition model is worth: evaluation protocol, not algorithm choice*.

This release is the version intended for archival (Zenodo DOI) and for reviewers who need to regenerate the HR results from the experimental record.

## What's included

- **Data.** Six public employee-turnover datasets, already harmonised, plus the benchmark problems used as a reference distribution (`data/processed/*.npz`, `manifest.csv`). Raw Kaggle / Mendeley files are not redistributed; filenames and sources are documented in `Code/01_fetch_datasets.py`.
- **Experiment.** Three evaluation protocols that differ only in where the over-sampler sits relative to a 70:30 split (`leaked` / `correct` / `none`), seven classifiers, four over-samplers, ten seeds.
- **Analysis.** `Code/03_analysis.py` rebuilds `results/summary.json`, Table 8 (`results/table8_hr.csv`), and `figures/fig01_corpus.png`, `fig06_hr.png`, `fig07_hr_vs_benchmark.png`. `Code/05_verify_numbers.py` recomputes headline quantities from `raw_results.csv`.
- **Notebook.** `Code/analysis.ipynb` walks through the six turnover datasets.

## Headline numbers in this archive (HR datasets)

| Quantity | Value |
|---|---|
| Turnover datasets | 6 |
| Mean leakage premium (minority-class F1) | 0.235 (95% CI 0.226–0.244) |
| Correct-protocol mean F1 | 0.591 |
| Leaked-protocol mean F1 | 0.826 |
| Premium on IBM HR | 0.443 |
| Premium on the near-balanced Russian turnover set | 0.003 |

On these data, correctly ordered resampling still beats the no-resampling baseline; the recommendation is to report that baseline, not to stop resampling.

## How to verify without re-running the factorial

```bash
cd Code
pip install -r requirements.txt
python 03_analysis.py
python 05_verify_numbers.py
```

## Requirements

Python 3.9+; see `Code/requirements.txt`. Versions used for the reported run are recorded in `results/summary.json` under `versions`.

## License

BSD 3-Clause.
