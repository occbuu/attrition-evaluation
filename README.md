# Paper_Attrition_Code

Public code for **Paper 2C**

> What an attrition model is worth: evaluation protocol, not algorithm choice

This repository is what should go on GitHub + Zenodo for the HR-facing paper.

## What is public here (and why)

| Path | Why it is in the archive |
|---|---|
| `Code/` | Same experiment as Paper 2A; `Paper2C/build_paper2C.py` builds this manuscript |
| `data/processed/` | Six HR datasets + 35 benchmarks, already harmonised |
| `results/` | Experimental record; Table 8 and the HR figures are derived from here |
| `figures/` | Includes `fig06_hr.png` and `fig07_hr_vs_benchmark.png` |
| `Paper2C/` | Built manuscript |

The six HR sources are public (IBM, Saudi/Mendeley, four Kaggle sets). Raw
files are not redistributed; `Code/01_fetch_datasets.py` documents the
filenames and URLs.

## Reproduce the paper from the archived results

```bash
cd Code
pip install -r requirements.txt
python 03_analysis.py
python Paper2C/build_paper2C.py
```

Notebook: `Code/Paper2C/Paper2C_hr_analysis.ipynb`.

Companion archives: `Paper_Leakage_Code` (measurement paper) and
`Paper_Leakcheck_Code` (software).

## Publish on GitHub and mint a Zenodo DOI

Three papers need **three repositories** and **three DOIs**. Do not put all
three folders in one GitHub repo: Zenodo versions the whole repository, and
JOSS (Paper 2B) expects the software to sit at the repository root.

1. Create an empty **public** GitHub repository (one per folder).
2. From this folder:

```bash
git init
git add -A
git commit -m "Initial public release"
git branch -M main
git remote add origin https://github.com/<you>/<repo>.git
git push -u origin main
```

3. At [zenodo.org](https://zenodo.org) → GitHub → enable the repository.
4. On GitHub: **Releases → Create a new release** (tag `v1.0.0`).
5. Zenodo mints a DOI within a minute. Put that DOI in the paper's
   *Data and code availability* section and in `CITATION.cff`.
6. For a later correction, cut a new release (`v1.0.1`); the concept DOI
   always resolves to the latest version.

Do **not** upload `data/pmlb_cache/`, Kaggle zips, or `__pycache__`.
Processed `.npz` files and `results/` are already here so a reviewer can
rebuild the manuscript without re-running the factorial.

## License

BSD 3-Clause. See `LICENSE`.
