# Attrition evaluation — replication code

Code, processed datasets and experimental record for:

> What an attrition model is worth: evaluation protocol, not algorithm choice

## Layout

| Path | Contents |
|---|---|
| `Code/` | Harmonisation, factorial, analysis |
| `data/processed/` | Six turnover datasets plus the benchmark corpus |
| `results/` | Experimental record; Table 8 is derived from here |
| `figures/` | `fig01_corpus.png`, `fig06_hr.png`, `fig07_hr_vs_benchmark.png` |

Raw Kaggle / Mendeley files are not redistributed; `Code/01_fetch_datasets.py`
documents the filenames.

## Reproduce from the archived results

```bash
cd Code
pip install -r requirements.txt
python 03_analysis.py
python 05_verify_numbers.py
```

Notebook: `Code/analysis.ipynb`.

## GitHub + Zenodo DOI

```bash
git init
git add -A
git commit -m "Initial public release"
git branch -M main
git remote add origin https://github.com/<you>/<repo>.git
git push -u origin main
```

Then zenodo.org → GitHub → enable this repository → GitHub Release `v1.0.0`.
Put the DOI in the paper and in `CITATION.cff`.

## License

BSD 3-Clause. See `LICENSE`.
