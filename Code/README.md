# Code

| Script | Role |
|---|---|
| `01_fetch_datasets.py` | Build `data/processed/*.npz` |
| `run_experiment.py` | Main factorial (shardable) |
| `run_simulation.py` | Controlled synthetic grid |
| `run_mitigation.py` | Partial-oversampling sweep |
| `03_analysis.py` | `summary.json`, tables, figures |
| `05_verify_numbers.py` | Recompute headline numbers from `raw_results.csv` |
| `analysis.ipynb` | Interactive walkthrough of the six turnover datasets |

```bash
pip install -r requirements.txt
python 03_analysis.py
python 05_verify_numbers.py
```
