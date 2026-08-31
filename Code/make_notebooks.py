#!/usr/bin/env python3
"""
make_notebooks.py -- generate the notebooks from a single description.

The notebooks are the readable front end to the study; the modules in Code/ are
where the logic lives.  Keeping the notebooks generated rather than hand-edited
means they can never drift from the code they call, which is the failure mode
that produces a notebook nobody can run six months later.

    python make_notebooks.py
"""
from __future__ import annotations
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CODE = ROOT / "Code"


def nb(cells: list[tuple[str, str]]) -> dict:
    out = []
    for kind, src in cells:
        lines = src.strip("\n").split("\n")
        lines = [l + "\n" for l in lines[:-1]] + [lines[-1]]
        if kind == "md":
            out.append({"cell_type": "markdown", "metadata": {}, "source": lines})
        else:
            out.append({"cell_type": "code", "execution_count": None, "metadata": {},
                        "outputs": [], "source": lines})
    return {"cells": out, "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"}},
        "nbformat": 4, "nbformat_minor": 5}


BOOT = """
import sys, json, warnings
from pathlib import Path
warnings.filterwarnings('ignore')

CODE = Path.cwd()
while CODE.name != 'Code' and CODE.parent != CODE:
    CODE = CODE.parent
sys.path.insert(0, str(CODE))

import numpy as np, pandas as pd
import leakage_lib as L
pd.set_option('display.width', 200, 'display.max_columns', 60)
print('corpus  :', L.PROC)
print('results :', L.RES)
"""


# ===========================================================================
MASTER = [
    ("md", """
# The leakage premium — master pipeline

Companion code for *"Leakage makes every problem look equally easy: a multi-dataset
measurement of the resampling-before-splitting artefact"* and its two companion papers.

This notebook runs the whole study end to end. Each step is a thin call into a module in
`Code/`, so the notebook and the batch scripts execute identical code.

| Step | Script | Output |
|---|---|---|
| 1. Build the corpus | `01_fetch_datasets.py` | `data/processed/*.npz`, `manifest.csv` |
| 2. Main factorial | `run_experiment.py` | `results/raw_shard*.csv` |
| 3. Controlled simulation | `run_simulation.py` | `results/simulation.csv` |
| 4. Mitigation sweep | `run_mitigation.py` | `results/mitigation.csv` |
| 5. Analysis | `03_analysis.py` | `results/summary.json`, tables, figures |
| 6. Export the screening model | `04_export_model.py` | `leakcheck/model.json` |
| 7. Build the manuscripts | `Paper2*/build_paper2*.py` | three `.docx` files |
| 8. Verify every number | `05_verify_numbers.py` | pass/fail |

**Runtime.** Step 2 is the expensive one. On two cores it takes a few hours; it shards
across processes and checkpoints after every dataset, so it can be interrupted and resumed.
"""),
    ("code", BOOT),
    ("md", "## 1. The corpus"),
    ("code", """
# Safe to re-run: datasets already present are re-derived deterministically.
!python "{CODE}/01_fetch_datasets.py"
"""),
    ("code", """
man = L.manifest()
print(f'{len(man)} datasets   IR {man.imbalance_ratio.min():.2f} - {man.imbalance_ratio.max():.1f}'
      f'   n {man.n.min()} - {man.n.max()}   p {man.n_features.min()} - {man.n_features.max()}')
man.groupby('domain').agg(datasets=('name','size'), median_IR=('imbalance_ratio','median'),
                          total_rows=('n','sum'))
"""),
    ("code", """
man[['name','domain','n','n_features','n_minority','imbalance_ratio','construction']]
"""),
    ("md", """
## 2. The main factorial

Three conditions that differ only in where the resampler sits:

| Condition | Procedure |
|---|---|
| `leaked`  | resample the **whole** dataset, then split 70:30 |
| `correct` | split 70:30, then resample the **training partition only** |
| `none`    | split 70:30, no resampling |

Run the shards from a terminal so they proceed in parallel — two shards on a two-core
machine, four on four cores. Each writes its own CSV and skips datasets already done.

```bash
for s in 0 1; do
    OMP_NUM_THREADS=1 python run_experiment.py --shard $s --nshards 2 --seeds 10 &
done
```

The cell below runs a single-process version instead, which is fine for a quick check with
fewer seeds but slow for the full grid.
"""),
    ("code", """
QUICK = True          # set False only if you intend to run the full grid in-notebook

if QUICK:
    demo = L.run_dataset('ibm_hr', clf_set=['LogReg','DecTree','RandForest'],
                         samp_set=['SMOTE'], n_seeds=2)
    print(demo.groupby('condition').f1.agg(['mean','std','size']).round(4))
else:
    import subprocess
    subprocess.run([sys.executable, str(CODE/'run_experiment.py'),
                    '--shard','0','--nshards','1','--seeds','10'], check=True)
"""),
    ("md", "## 3-4. Simulation and mitigation sweep"),
    ("code", """
# Both are independent of the main factorial and can run while it is still going.
# !python "{CODE}/run_simulation.py" --seeds 3
# !python "{CODE}/run_mitigation.py" --seeds 3
print('run these from a terminal; they take 30-90 minutes each')
"""),
    ("md", "## 5. Analysis — produces every table, figure and number in the papers"),
    ("code", """
!python "{CODE}/03_analysis.py"
"""),
    ("code", """
S = json.load(open(L.RES/'summary.json'))
for k in ['n_datasets','n_paired','leaked_f1_mean','correct_f1_mean','variance_ratio',
          'mean_premium','law_r2','metareg_r2_correct','metareg_r2_leaked','diag_auc']:
    print(f'{k:24s} {S.get(k)}')
"""),
    ("md", "## 6-8. Screening model, manuscripts, verification"),
    ("code", """
!python "{CODE}/04_export_model.py"
!python "{CODE}/Paper2A/build_paper2A.py"
!python "{CODE}/Paper2B/build_paper2B.py"
!python "{CODE}/Paper2C/build_paper2C.py"
!python "{CODE}/05_verify_numbers.py"
"""),
    ("md", """
---
### Checklist before submission

- [ ] the full grid completed on every dataset in the manifest
- [ ] `summary.json` regenerated from that single run
- [ ] `05_verify_numbers.py` passes all three checks
- [ ] library versions in `summary.json['versions']` match the run being reported
- [ ] `raw_results.csv`, `manifest.csv` and the figures archived with a DOI
"""),
]


# ===========================================================================
A = [
    ("md", """
# Paper 2A — the invariance law

Analysis notebook for the methodological paper. It assumes `run_experiment.py`,
`run_simulation.py` and `run_mitigation.py` have completed and reproduces, interactively,
the results reported in Sections 4.2 to 4.7.
"""),
    ("code", BOOT + """
R = pd.read_csv(L.RES/'raw_results.csv')
if 'error' in R.columns:
    R = R[R.error.isna()]
man = L.manifest()
piv = L.paired(R, man)
print(f'{len(R):,} fits   {len(piv):,} matched leaked/correct pairs')
"""),
    ("md", "## 4.2  The invariance\\n\\nThe central claim: under the leaked ordering the score stops tracking difficulty."),
    ("code", """
print(f"leaked   mean {piv.leaked.mean():.4f}   sd {piv.leaked.std():.4f}")
print(f"correct  mean {piv.correct.mean():.4f}   sd {piv.correct.std():.4f}")
print(f"variance ratio (correct/leaked): {piv.correct.var()/piv.leaked.var():.2f}")
print(f"mean premium: {piv.premium.mean():.4f}   positive in {(piv.premium>0).mean():.1%} of pairs")
"""),
    ("code", """
b, a = np.polyfit(piv.correct, piv.premium, 1)
r2 = np.corrcoef(piv.correct, piv.premium)[0,1]**2
print(f"premium = {a:.4f} + {b:.4f} * F1_correct      R^2 = {r2:.4f}")

li = np.log10(piv.imbalance_ratio)
print(f"corr(log IR, leaked)  = {np.corrcoef(li, piv.leaked)[0,1]:+.4f}")
print(f"corr(log IR, correct) = {np.corrcoef(li, piv.correct)[0,1]:+.4f}")
"""),
    ("code", """
# per-dataset means: the level at which datasets are exchangeable
byds = piv.groupby('dataset')[['leaked','correct','premium']].mean().join(
    man.set_index('name')[['domain','imbalance_ratio','n_minority']])
byds.sort_values('imbalance_ratio').round(4)
"""),
    ("code", """
from IPython.display import Image, display
for f in ['fig01_corpus.png','fig02_invariance.png']:
    display(Image(str(L.FIG/f)))
"""),
    ("md", "## 4.3  Meta-regression — the problem explains the honest score, the model explains the leaked one"),
    ("code", """
import statsmodels.api as sm
M = piv.copy()
M['log_ir']=np.log10(M.imbalance_ratio); M['log_nmin']=np.log10(M.n_minority)
M['log_p']=np.log10(M.n_features);       M['log_n']=np.log10(M.n)
D = pd.get_dummies(M[['clf','sampler']], drop_first=True).astype(float)
base = ['log_ir','log_nmin','log_p','log_n']
for label, X in [('data only', M[base]), ('algorithm only', D),
                 ('both', pd.concat([M[base], D], axis=1))]:
    Xc = sm.add_constant(X).to_numpy(dtype=float)
    r_c = sm.OLS(M.correct, Xc).fit(cov_type='cluster', cov_kwds={'groups':M.dataset}).rsquared
    r_l = sm.OLS(M.leaked,  Xc).fit(cov_type='cluster', cov_kwds={'groups':M.dataset}).rsquared
    print(f'{label:16s}  R2 correct {r_c:.3f}   R2 leaked {r_l:.3f}')
"""),
    ("md", "## 4.4-4.6  Simulation, mechanism, mitigation"),
    ("code", """
for f in ['fig03_simulation.png','fig04_mechanism_mitigation.png']:
    display(Image(str(L.FIG/f)))
pd.read_csv(L.RES/'table4_mitigation.csv').round(4)
"""),
    ("code", """
d = piv.dropna(subset=['contam'])
print(f'contamination: mean {d.contam.mean():.4f}   corr with premium {np.corrcoef(d.contam,d.premium)[0,1]:+.4f}')
d.groupby('sampler')[['contam','premium','leaked','correct']].mean().round(4)
"""),
    ("md", "## 4.7  The screening model, validated leave-one-dataset-out"),
    ("code", """
display(Image(str(L.FIG/'fig05_diagnostic.png')))
S = json.load(open(L.RES/'summary.json'))
{k: S[k] for k in S if k.startswith('diag_')}
"""),
    ("md", "## Build the manuscript"),
    ("code", """
!python "{CODE}/Paper2A/build_paper2A.py"
"""),
]


# ===========================================================================
Bnb = [
    ("md", """
# Paper 2B — the `leakcheck` package

Screen a reported F1 for the signature of resampling **before** the train/test split.
The shipped model is plain JSON, evaluated by a pure-NumPy tree walker (runtime
dependency: NumPy only).

| What this notebook shows | Source |
|---|---|
| Leave-one-dataset-out ROC and the screening curves | `figures/fig05_diagnostic.png` |
| Model card (AUC, corpus, fitted ranges) | `leakcheck.model_info()` |
| Worked example on IBM HR (F1 = 0.92) | `leakcheck.check(...)` |
| Heatmap: same F1, different imbalance | live grid |
| Seven published attrition results | `literature.py` |
"""),
    ("code", BOOT + """
import matplotlib.pyplot as plt
from IPython.display import Image, display, Markdown

plt.rcParams.update({'figure.dpi': 120, 'savefig.bbox': 'tight',
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'font.size': 10, 'figure.facecolor': 'white'})
BLUE, RED, GREY, GREEN = L.BLUE, L.RED, L.GREY, L.GREEN

PKG = L.ROOT/'Paper2B'/'leakcheck'
sys.path.insert(0, str(PKG/'src'))
print('package :', PKG)
print('figure  :', L.FIG/'fig05_diagnostic.png')
"""),
    ("md", """
## 1. The screen, as published

Panel (a) is leave-one-dataset-out detection on the 41-dataset corpus.
Panel (b) is what a reviewer sees: P(leaked) as a function of the reported F1,
at four imbalance ratios.
"""),
    ("code", """
fig05 = L.FIG / 'fig05_diagnostic.png'
if fig05.exists():
    display(Image(str(fig05), width=860))
else:
    print('fig05_diagnostic.png missing — run Code/03_analysis.py first')

S = json.load(open(L.RES/'summary.json'))
pd.DataFrame([
    ('Leave-one-dataset-out AUC', f"{S['diag_auc']:.3f}"),
    ('Accuracy at 0.5', f"{S['diag_acc']:.3f}"),
    ('Brier score', f"{S['diag_brier']:.3f}"),
    ('Datasets scored', S['diag_n_datasets_scored']),
    ('Datasets with AUC > 0.9', S['diag_n_datasets_above_09']),
    ('Paired examples', S['diag_n']),
], columns=['Diagnostic', 'Value'])
"""),
    ("md", "## 2. The shipped model"),
    ("code", """
# Re-export only if the JSON is missing. The factorial is already done.
model_json = PKG/'src'/'leakcheck'/'model.json'
if not model_json.exists():
    !python "{CODE}/04_export_model.py"
else:
    print('using existing', model_json)

import importlib, leakcheck
importlib.reload(leakcheck)
info = leakcheck.model_info()
card = pd.DataFrame([
    ('Trees', info['n_trees']),
    ('LODO AUC', f"{info['lodo_auc']:.3f}"),
    ('Datasets', info['n_datasets']),
    ('Training examples', info['n_examples']),
    ('Imbalance range', f"{info['ir_range'][0]:.2f} – {info['ir_range'][1]:.1f}"),
    ('Minority-count range', f"{info['nmin_range'][0]} – {info['nmin_range'][1]}"),
    ('Classifier families', ', '.join(info['classifiers'])),
], columns=['Model card', 'Value'])
display(card)
assert leakcheck.selftest(), 'shipped fixtures drifted'
print('selftest: shipped JSON matches stored fixtures')
"""),
    ("md", """
## 3. Worked example — IBM HR, reported F1 = 0.92

A headline score that is common in the attrition literature, on a problem with
237 leavers and imbalance 5.2 : 1.
"""),
    ("code", """
r = leakcheck.check(f1=0.92, imbalance_ratio=5.2, n_minority=237,
                    n_features=44, clf='RandomForest')
print(r.explain())

cuts = list(leakcheck.THRESHOLDS.items())
fig, ax = plt.subplots(figsize=(7.2, 1.6))
ax.barh([0], [r.probability], color=RED, height=0.45)
ax.set_xlim(0, 1)
ax.set_yticks([])
ax.set_xlabel('P(leaked)')
for label, cut in cuts:
    ax.axvline(cut if cut <= 1 else 1, color=GREY, lw=0.8, ls=':')
ax.axvline(0.35, color=GREY, lw=0.8)
ax.axvline(0.65, color=GREY, lw=0.8)
ax.text(0.17, 0.7, 'unremarkable', ha='center', color=GREY, fontsize=8)
ax.text(0.50, 0.7, 'worth asking', ha='center', color=GREY, fontsize=8)
ax.text(0.83, 0.7, 'likely leaked', ha='center', color=RED, fontsize=8)
ax.set_title(f'IBM HR · F1 0.92 · P(leaked) = {r.probability:.2f} · {r.verdict}')
fig.tight_layout()
plt.show()
"""),
    ("md", """
## 4. The same F1 is ordinary on an easy problem and alarming on a hard one
"""),
    ("code", """
rows = [dict(f1=f, imbalance_ratio=ir, n_minority=200, n_features=30, clf='RandomForest')
        for f in (0.45, 0.70, 0.92) for ir in (2, 10, 50)]
grid = leakcheck.check_many(rows).round(3)
display(grid)

piv = grid.pivot(index='imbalance_ratio', columns='f1', values='probability')
fig, ax = plt.subplots(figsize=(5.6, 2.8))
im = ax.imshow(piv.to_numpy(), cmap='RdYlBu_r', vmin=0, vmax=1, aspect='auto')
ax.set_xticks(range(len(piv.columns)), [f'{c:.2f}' for c in piv.columns])
ax.set_yticks(range(len(piv.index)), [f'IR {int(i)}' for i in piv.index])
ax.set_xlabel('Reported minority-class F1')
ax.set_title('P(leaked) — RandomForest, 200 minority cases, 30 features')
for i in range(piv.shape[0]):
    for j in range(piv.shape[1]):
        v = piv.to_numpy()[i, j]
        ax.text(j, i, f'{v:.2f}', ha='center', va='center',
                color='white' if v > 0.55 or v < 0.15 else 'black', fontsize=9)
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='P(leaked)')
fig.tight_layout()
plt.show()
"""),
    ("code", """
f1s = np.linspace(0.35, 0.98, 40)
fig, ax = plt.subplots(figsize=(6.4, 3.4))
for ir, c in zip([2, 5, 20, 80], [GREY, GREEN, BLUE, RED]):
    ps = [leakcheck.check(f1=f, imbalance_ratio=ir, n_minority=200,
                          n_features=30, clf='RandomForest').probability
          for f in f1s]
    ax.plot(f1s, ps, color=c, lw=1.8, label=f'IR = {ir}')
ax.axhline(0.35, color=GREY, lw=0.7, ls=':')
ax.axhline(0.65, color=GREY, lw=0.7, ls=':')
ax.set(xlabel='Reported minority-class F1', ylabel='P(leaked)',
       ylim=(0, 1), title='Screening curves from the shipped JSON model')
ax.legend(frameon=False, loc='upper left')
fig.tight_layout()
plt.show()
"""),
    ("md", "## 5. Screen the published attrition literature"),
    ("code", """
import literature
lit = literature.screened().copy()
show = lit[['study', 'dataset', 'model', 'f1', 'protocol', 'p_leaked']].round(3)
show = show.rename(columns={'p_leaked': 'P(leaked)'})
display(show.sort_values('P(leaked)', ascending=False))

fig, ax = plt.subplots(figsize=(7.4, 3.6))
d = lit.sort_values('p_leaked')
col = d.protocol.map({'before-split': RED, 'train-only': GREEN, 'not stated': GREY})
ax.barh(range(len(d)), d.p_leaked, color=col, height=0.65)
ax.set_yticks(range(len(d)), [f"{s} · {ds}" for s, ds in zip(d.study, d.dataset)])
ax.set_xlabel('P(leaked)')
ax.set_xlim(0, 1)
ax.axvline(0.35, color=GREY, lw=0.7, ls=':')
ax.axvline(0.65, color=GREY, lw=0.7, ls=':')
ax.set_title('Published headline F1, screened — colour = what the paper said about the split')
from matplotlib.patches import Patch
ax.legend(handles=[
    Patch(color=RED, label='paper says before-split'),
    Patch(color=GREY, label='protocol not stated'),
    Patch(color=GREEN, label='paper says train-only'),
], frameon=False, loc='lower right', fontsize=8)
fig.tight_layout()
plt.show()
"""),
    ("md", "## 6. Tests, then build the paper"),
    ("code", """
!cd "{PKG}" && python -m pytest -q
"""),
    ("code", """
!python "{CODE}/Paper2B/build_paper2B.py"
"""),
]


# ===========================================================================
C = [
    ("md", """
# Paper 2C — employee attrition, honestly evaluated

The HR-facing analysis: what the six employee-turnover datasets look like under each of the
three protocols, and what an HR analytics team should conclude.
"""),
    ("code", BOOT + """
R = pd.read_csv(L.RES/'raw_results.csv')
if 'error' in R.columns:
    R = R[R.error.isna()]
man = L.manifest()
piv = L.paired(R, man)
H = piv[piv.domain=='HR']
print(f'{H.dataset.nunique()} HR datasets, {len(H):,} matched comparisons')
"""),
    ("md", "## The datasets"),
    ("code", """
man[man.domain=='HR'][['name','n','n_features','n_minority','imbalance_ratio','source']]
"""),
    ("md", "## How much the protocol is worth"),
    ("code", """
print(f'correct  {H.correct.mean():.4f}')
print(f'leaked   {H.leaked.mean():.4f}')
print(f'premium  {H.premium.mean():.4f}')
pd.read_csv(L.RES/'table8_hr.csv').round(4)
"""),
    ("code", """
from IPython.display import Image, display
display(Image(str(L.FIG/'fig06_hr.png')))
display(Image(str(L.FIG/'fig07_hr_vs_benchmark.png')))
"""),
    ("md", """
## Protocol versus algorithm

The comparison that matters for the paper's claim: moving the resampler across the split
changes the reported score by more than switching between any two model families.
"""),
    ("code", """
spread_protocol  = H.leaked.mean() - H.correct.mean()
byclf = H.groupby('clf')[['correct']].mean()
spread_algorithm = float(byclf.correct.max() - byclf.correct.min())
print(f'protocol effect  {spread_protocol:.4f}')
print(f'algorithm spread {spread_algorithm:.4f}  (best minus worst family, correct protocol)')
byclf.round(4)
"""),
    ("md", "## Is resampling worth doing at all?"),
    ("code", """
none = R[(R.condition=='none') & (R.dataset.isin(H.dataset.unique()))]
n_ = none.groupby(['dataset','clf','seed']).f1.mean()
c_ = H.groupby(['dataset','clf','seed']).correct.mean()
m = pd.concat({'none': n_, 'correct': c_}, axis=1).dropna()
print(f'no-resampling beats correctly-ordered resampling in {(m.none>m.correct).mean():.1%} '
      f'of matched HR comparisons')
m.groupby(level='dataset').mean().round(4)
"""),
    ("md", "## Build the manuscript"),
    ("code", """
!python "{CODE}/Paper2C/build_paper2C.py"
"""),
]


def main() -> int:
    targets = [
        (CODE / "02_leakage_experiment.ipynb", MASTER),
        (CODE / "Paper2A" / "Paper2A_analysis.ipynb", A),
        (CODE / "Paper2B" / "Paper2B_leakcheck.ipynb", Bnb),
        (CODE / "Paper2C" / "Paper2C_hr_analysis.ipynb", C),
    ]
    for path, cells in targets:
        path.parent.mkdir(parents=True, exist_ok=True)
        cooked = [(k, s.replace("{CODE}", str(CODE)).replace(
            "{PKG}", str(ROOT / "Paper2B" / "leakcheck"))) for k, s in cells]
        with open(path, "w", encoding="utf-8") as f:
            json.dump(nb(cooked), f, indent=1)
        print(f"wrote {path.relative_to(ROOT)}  ({len(cells)} cells)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
