# Paper 2 — the leakage premium

Code for three papers built on one experiment:

| | Paper | Venue aimed at | Folder |
|---|---|---|---|
| **2A** | *Leakage makes every problem look equally easy* — the measurement, the invariance, the mechanism | Data Mining and Knowledge Discovery | `Paper2A/` |
| **2B** | `leakcheck` — the screening tool | Journal of Open Source Software | `Paper2B/` |
| **2C** | *What an attrition model is worth* — the HR-facing paper | HR analytics / IS journal | `Paper2C/` |

All three read the same `results/summary.json`. No number is typed into any manuscript by
hand; `05_verify_numbers.py` enforces that.

## Layout

```
Paper2/
  Code/
    leakage_lib.py            the three conditions, the metrics, the contamination measure
    01_fetch_datasets.py      builds data/processed/*.npz + manifest.csv
    run_experiment.py         the main factorial, shardable and resumable
    run_simulation.py         the controlled synthetic grid
    run_mitigation.py         the sampling_strategy sweep
    03_analysis.py            -> summary.json, table1-8.csv, fig01-07.png
    04_export_model.py        fits the screen and exports leakcheck/model.json
    05_verify_numbers.py      three independent checks on the manuscripts
    paperkit.py, refs.py, literature.py, make_notebooks.py
    02_leakage_experiment.ipynb      master pipeline notebook
    Paper2A/  Paper2B/  Paper2C/     per-paper build scripts and notebooks
  data/
    processed/                harmonised .npz + manifest.csv
    pmlb_cache/               downloaded benchmark sources
    <hr datasets>/            the six employee-turnover datasets
  results/                    raw_results.csv, summary.json, table*.csv
  figures/                    fig01-fig07.png
  Paper2A/ Paper2B/ Paper2C/  the manuscripts and the leakcheck package
```

## Running it

```bash
pip install -r requirements.txt

# 1. corpus  (Group A downloads itself; Group B is already under data/)
python 01_fetch_datasets.py

# 2. the main factorial -- one process per core, resumable
for s in 0 1; do
    OMP_NUM_THREADS=1 python run_experiment.py --shard $s --nshards 2 --seeds 10 &
done; wait

# 3. supporting experiments (independent of each other and of step 2)
python run_simulation.py --seeds 3
python run_mitigation.py --seeds 3

# 4. everything downstream
python 03_analysis.py
python 04_export_model.py
python Paper2A/build_paper2A.py
python Paper2B/build_paper2B.py
python Paper2C/build_paper2C.py
python 05_verify_numbers.py
```

`run_experiment.py` writes one CSV per shard and skips datasets already present in it, so
it can be stopped and restarted freely. `03_analysis.py` concatenates the shards into
`results/raw_results.csv`.

## The corpus

41 binary classification problems: 35 benchmarks derived from PMLB and 6 employee-turnover
datasets. Multi-class benchmark sources are binarised one-versus-rest on the rarest class
with at least 30 cases — the construction used to build imbalanced benchmark suites from
multi-class sources.

Two thresholds are choices rather than facts and are exposed as parameters: a dataset needs
at least 30 minority cases to enter, and datasets over `--max-rows` (default 5,000) are
stratified-subsampled so the factorial stays tractable. Both are stated in the manuscripts.

The original design called for the imbalanced-learn Zenodo benchmark collection. Zenodo is
unreachable from the network this study was run on; PMLB serves the same purpose, is
independently citable, and covers the same underlying UCI sources.

## The discipline rule

Every quantity in every manuscript is a placeholder resolved at build time from
`results/summary.json` or a `results/table*.csv`. `05_verify_numbers.py` runs three checks:

- **A** — no numeric literal in manuscript prose except an explicit allow-list of structural
  constants;
- **B** — no unresolved placeholder survives into a built `.docx`;
- **C** — headline quantities in `summary.json` are recomputed from `raw_results.csv` by a
  separate code path and must agree.

Run it before every submission, and re-run the whole pipeline once immediately before
submitting so that `summary.json['versions']` records the libraries actually used.
