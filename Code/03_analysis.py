#!/usr/bin/env python3
"""
03_analysis.py -- turn the raw experimental output into every number, table and
figure quoted in the manuscripts.

Reads   results/raw_shard*.csv, results/simulation.csv, results/mitigation.csv
Writes  results/raw_results.csv      concatenated raw output
        results/summary.json         THE single source of truth for the papers
        results/table1..table7.csv   manuscript tables
        figures/fig01..fig07.png     manuscript figures

Nothing in the manuscripts may be typed by hand; everything comes from here.
"""
from __future__ import annotations
import json, sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import leakage_lib as L

warnings.filterwarnings("ignore")
plt.rcParams.update({"savefig.dpi": 300, "font.size": 8.5, "savefig.bbox": "tight",
                     "axes.spines.top": False, "axes.spines.right": False,
                     "figure.facecolor": "white", "font.family": "DejaVu Sans"})
BLUE, RED, GREY, GREEN = L.BLUE, L.RED, L.GREY, L.GREEN
RES, FIG = L.RES, L.FIG
S: dict = {}


def r(x, k=4):
    """Round for the summary, keeping json clean of numpy types and NaN."""
    if x is None:
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return x
    return None if not np.isfinite(v) else round(v, k)


def ci95(a) -> tuple[float, float]:
    a = np.asarray(a, dtype=float)
    a = a[np.isfinite(a)]
    if len(a) < 2:
        return (np.nan, np.nan)
    se = a.std(ddof=1) / np.sqrt(len(a))
    return (a.mean() - 1.96 * se, a.mean() + 1.96 * se)


def boot_ci(a, stat=np.mean, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    a = np.asarray(a, dtype=float); a = a[np.isfinite(a)]
    if len(a) < 3:
        return (np.nan, np.nan)
    d = [stat(rng.choice(a, len(a), replace=True)) for _ in range(n)]
    return tuple(np.percentile(d, [2.5, 97.5]))


# ===========================================================================
# 1. load
# ===========================================================================
def load_raw() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    shards = sorted(RES.glob("raw_shard*.csv"))
    if not shards:
        sys.exit("no raw_shard*.csv found -- run run_experiment.py first")
    R = pd.concat([pd.read_csv(f) for f in shards], ignore_index=True)
    R.to_csv(RES / "raw_results.csv", index=False)
    man = L.manifest()
    n_err = int(R["error"].notna().sum()) if "error" in R.columns else 0
    if "error" in R.columns:
        R = R[R["error"].isna()].drop(columns=["error"])
    S["n_fits_attempted"] = int(len(R) + n_err)
    S["n_fits_failed"] = n_err
    S["n_results"] = int(len(R))
    piv = L.paired(R, man)
    S["n_paired"] = int(len(piv))
    return R, piv, man


# ===========================================================================
# 2. corpus  (Table 1, Figure 1)
# ===========================================================================
def corpus(man: pd.DataFrame, R: pd.DataFrame) -> None:
    have = set(R.dataset.unique())
    man = man[man.name.isin(have)].copy()
    t1 = man[["name", "domain", "n", "n_features", "n_minority", "imbalance_ratio",
              "construction", "source"]].sort_values(["domain", "imbalance_ratio"])
    t1.to_csv(RES / "table1_corpus.csv", index=False)
    S.update(
        n_datasets=int(len(man)),
        n_benchmark=int((man.domain == "benchmark").sum()),
        n_hr=int((man.domain == "HR").sum()),
        ir_min=r(man.imbalance_ratio.min(), 2), ir_max=r(man.imbalance_ratio.max(), 1),
        n_min_rows=int(man.n.min()), n_max_rows=int(man.n.max()),
        p_min=int(man.n_features.min()), p_max=int(man.n_features.max()),
        minority_min=int(man.n_minority.min()), minority_max=int(man.n_minority.max()),
        hr_ir_min=r(man[man.domain == "HR"].imbalance_ratio.min(), 2),
        hr_ir_max=r(man[man.domain == "HR"].imbalance_ratio.max(), 2),
    )

    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    for dom, c, mk, lab in [("benchmark", GREY, "o", "Benchmark"), ("HR", RED, "D", "HR / turnover")]:
        d = man[man.domain == dom]
        ax.scatter(d.imbalance_ratio, d.n_minority, s=34, c=c, marker=mk,
                   edgecolor="white", linewidth=.6, label=lab, zorder=3 if dom == "HR" else 2)
    for _, row in man[man.domain == "HR"].iterrows():
        ax.annotate(row["name"], (row.imbalance_ratio, row.n_minority),
                    textcoords="offset points", xytext=(5, 4), fontsize=6, color=RED)
    ax.set(xscale="log", yscale="log", xlabel="Imbalance ratio (majority : minority)",
           ylabel="Minority cases")
    ax.legend(frameon=False, fontsize=7.5, loc="upper right")
    ax.grid(alpha=.25, lw=.5)
    fig.savefig(FIG / "fig01_corpus.png"); plt.close(fig)


# ===========================================================================
# 3. the invariance law  (Table 2, Figure 2)
# ===========================================================================
def invariance(piv: pd.DataFrame, R: pd.DataFrame) -> None:
    S.update(
        leaked_f1_mean=r(piv.leaked.mean()), leaked_f1_sd=r(piv.leaked.std()),
        correct_f1_mean=r(piv.correct.mean()), correct_f1_sd=r(piv.correct.std()),
        leaked_f1_iqr=r(piv.leaked.quantile(.75) - piv.leaked.quantile(.25)),
        correct_f1_iqr=r(piv.correct.quantile(.75) - piv.correct.quantile(.25)),
        variance_ratio=r(piv.correct.var() / piv.leaked.var(), 2),
        sd_ratio=r(piv.correct.std() / piv.leaked.std(), 2),
        mean_premium=r(piv.premium.mean()),
        median_premium=r(piv.premium.median()),
        premium_sd=r(piv.premium.std()),
        pct_premium_positive=r(100 * (piv.premium > 0).mean(), 1),
        corr_leaked_correct=r(np.corrcoef(piv.leaked, piv.correct)[0, 1]),
    )
    lo, hi = boot_ci(piv.premium.to_numpy())
    S["premium_ci_lo"], S["premium_ci_hi"] = r(lo), r(hi)

    # per-dataset means, the unit at which datasets are exchangeable
    byds = piv.groupby("dataset")[["leaked", "correct", "premium"]].mean()
    S["variance_ratio_bydataset"] = r(byds.correct.var() / byds.leaked.var(), 2)
    S["leaked_range_bydataset"] = r(byds.leaked.max() - byds.leaked.min())
    S["correct_range_bydataset"] = r(byds.correct.max() - byds.correct.min())

    # the law: premium as a function of honest difficulty
    x, y = piv.correct.to_numpy(), piv.premium.to_numpy()
    b, a = np.polyfit(x, y, 1)
    S["law_slope"], S["law_intercept"] = r(b), r(a)
    S["law_slope_abs"] = r(abs(b))
    S["law_r2"] = r(np.corrcoef(x, y)[0, 1] ** 2)
    xb, yb = byds.correct.to_numpy(), byds.premium.to_numpy()
    bb, ab = np.polyfit(xb, yb, 1)
    S["law_slope_bydataset"], S["law_intercept_bydataset"] = r(bb), r(ab)
    S["law_r2_bydataset"] = r(np.corrcoef(xb, yb)[0, 1] ** 2)

    # sensitivity of each condition to the imbalance ratio
    li = np.log10(piv.imbalance_ratio.to_numpy())
    S["corr_leaked_logir"] = r(np.corrcoef(li, piv.leaked)[0, 1])
    S["corr_correct_logir"] = r(np.corrcoef(li, piv.correct)[0, 1])
    S["slope_leaked_logir"] = r(np.polyfit(li, piv.leaked, 1)[0])
    S["slope_correct_logir_abs"] = r(abs(np.polyfit(li, piv.correct, 1)[0]))
    S["slope_correct_logir"] = r(np.polyfit(li, piv.correct, 1)[0])

    # Table 2 -- invariance across the design factors
    rows = []
    q = pd.qcut(piv.imbalance_ratio, 4, duplicates="drop")
    for lab, g in piv.groupby(q, observed=True):
        rows.append(dict(factor="Imbalance ratio", level=f"{lab.left:.2f}-{lab.right:.1f}",
                         n=len(g), leaked=g.leaked.mean(), correct=g.correct.mean(),
                         none=np.nan, premium=g.premium.mean()))
    for fac, col in [("Classifier", "clf"), ("Resampler", "sampler"), ("Domain", "domain")]:
        for lab, g in piv.groupby(col, observed=True):
            rows.append(dict(factor=fac, level=str(lab), n=len(g),
                             leaked=g.leaked.mean(), correct=g.correct.mean(),
                             none=np.nan, premium=g.premium.mean()))
    t2 = pd.DataFrame(rows)
    none = R[R.condition == "none"]
    nb = none.groupby("clf").f1.mean()
    t2.loc[t2.factor == "Classifier", "none"] = t2.loc[t2.factor == "Classifier", "level"].map(nb)
    for c in ["leaked", "correct", "none", "premium"]:
        t2[c] = t2[c].round(4)
    t2.to_csv(RES / "table2_invariance.csv", index=False)

    for fac in ["Imbalance ratio", "Classifier", "Resampler"]:
        d = t2[t2.factor == fac]
        S[f"spread_leaked_{fac.split()[0].lower()}"] = r(d.leaked.max() - d.leaked.min())
        S[f"spread_correct_{fac.split()[0].lower()}"] = r(d.correct.max() - d.correct.min())

    byclf = piv.groupby("clf")[["leaked", "correct", "premium"]].mean()
    S["leaked_by_clf"] = {k: r(v) for k, v in byclf.leaked.items()}
    S["correct_by_clf"] = {k: r(v) for k, v in byclf.correct.items()}
    S["premium_by_clf"] = {k: r(v) for k, v in byclf.premium.items()}
    S["leaked_clf_max_name"] = str(byclf.leaked.idxmax())
    S["leaked_clf_min_name"] = str(byclf.leaked.idxmin())
    S["premium_clf_max_name"] = str(byclf.premium.idxmax())
    S["premium_clf_min_name"] = str(byclf.premium.idxmin())
    byir = piv.assign(b=pd.qcut(piv.imbalance_ratio, 4, duplicates="drop")).groupby("b", observed=True)
    gi = byir[["leaked", "correct"]].mean()
    S["ir_q1_leaked"] = r(float(gi.leaked.iloc[0])); S["ir_q4_leaked"] = r(float(gi.leaked.iloc[-1]))
    S["ir_q1_correct"] = r(float(gi.correct.iloc[0])); S["ir_q4_correct"] = r(float(gi.correct.iloc[-1]))

    # the 'none' baseline -- Paper 1's finding, re-checked at scale
    nn = R[R.condition == "none"].groupby(["dataset", "clf", "seed"]).f1.mean().reset_index()
    cc = piv.groupby(["dataset", "clf", "seed"]).correct.mean().reset_index()
    mm = nn.merge(cc, on=["dataset", "clf", "seed"])
    S["none_f1_mean"] = r(mm.f1.mean())
    S["none_minus_correct"] = r((mm.f1 - mm.correct).mean())
    S["pct_none_beats_correct"] = r(100 * (mm.f1 > mm.correct).mean(), 1)

    # other metrics -- is the law an artefact of F1?  (Table 6)
    rows = []
    for metric in ["f1", "mcc", "pr_auc", "roc_auc", "precision", "recall"]:
        p = (R[R.condition.isin(["leaked", "correct"])]
             .pivot_table(index=["dataset", "clf", "sampler", "seed"],
                          columns="condition", values=metric).dropna())
        if p.empty:
            continue
        rows.append(dict(metric=metric, leaked_mean=p.leaked.mean(), leaked_sd=p.leaked.std(),
                         correct_mean=p.correct.mean(), correct_sd=p.correct.std(),
                         premium=(p.leaked - p.correct).mean(),
                         variance_ratio=p.correct.var() / p.leaked.var()))
    t6 = pd.DataFrame(rows).round(4)
    t6.to_csv(RES / "table6_metrics.csv", index=False)
    S["variance_ratio_mcc"] = r(float(t6.loc[t6.metric == "mcc", "variance_ratio"].iloc[0]), 2)
    S["variance_ratio_prauc"] = r(float(t6.loc[t6.metric == "pr_auc", "variance_ratio"].iloc[0]), 2)
    S["variance_ratio_rocauc"] = r(float(t6.loc[t6.metric == "roc_auc", "variance_ratio"].iloc[0]), 2)
    S["premium_mcc"] = r(float(t6.loc[t6.metric == "mcc", "premium"].iloc[0]))
    S["premium_rocauc"] = r(float(t6.loc[t6.metric == "roc_auc", "premium"].iloc[0]))
    S["premium_prauc"] = r(float(t6.loc[t6.metric == "pr_auc", "premium"].iloc[0]))

    # ---- Figure 2 ----------------------------------------------------------
    fig, ax = plt.subplots(1, 3, figsize=(11.4, 3.4))

    ax[0].scatter(piv.correct, piv.leaked, s=7, alpha=.18, c=BLUE, edgecolor="none")
    ax[0].plot([0, 1], [0, 1], "k--", lw=.9, label="no leakage effect")
    ax[0].axhline(piv.leaked.mean(), color=RED, lw=1.4,
                  label=f"mean leaked F1 = {piv.leaked.mean():.3f}")
    ax[0].set(xlim=(0, 1), ylim=(0, 1), xlabel="F1, correct ordering",
              ylabel="F1, leaked ordering", title="(a) A band, not a shifted diagonal")
    ax[0].legend(frameon=False, fontsize=7, loc="lower right")

    b1 = piv.assign(bin=pd.qcut(np.log10(piv.imbalance_ratio), 8, duplicates="drop"))
    g = b1.groupby("bin", observed=True).agg(
        ir=("imbalance_ratio", "median"),
        lm=("leaked", "mean"), ls=("leaked", "std"),
        cm=("correct", "mean"), cs=("correct", "std"))
    ax[1].errorbar(g.ir, g.lm, yerr=g.ls, fmt="o-", c=RED, ms=4, lw=1.4, capsize=3, label="leaked")
    ax[1].errorbar(g.ir, g.cm, yerr=g.cs, fmt="s--", c=BLUE, ms=4, lw=1.4, capsize=3, label="correct")
    ax[1].set(xscale="log", xlabel="Imbalance ratio", ylabel="Minority-class F1",
              ylim=(0, 1), title="(b) The honest score falls; the leaked score rises")
    ax[1].legend(frameon=False, fontsize=7.5)

    ax[2].scatter(piv.correct, piv.premium, s=7, alpha=.18, c=GREY, edgecolor="none")
    xs = np.linspace(0, 1, 50)
    ax[2].plot(xs, a + b * xs, c=RED, lw=1.8,
               label=f"premium = {a:.3f} {'-' if b < 0 else '+'} {abs(b):.3f}·F1$_{{correct}}$\n$R^2$ = {S['law_r2']:.3f}")
    ax[2].axhline(0, color="k", lw=.7)
    ax[2].set(xlabel="F1, correct ordering", ylabel="Leakage premium",
              title="(c) The premium is difficulty, inverted")
    ax[2].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / "fig02_invariance.png"); plt.close(fig)


# ===========================================================================
# 4. meta-regression  (Table 3)
# ===========================================================================
def meta_regression(piv: pd.DataFrame) -> None:
    import statsmodels.api as sm
    M = piv.copy()
    M["log_ir"] = np.log10(M.imbalance_ratio)
    M["log_nmin"] = np.log10(M.n_minority)
    M["log_p"] = np.log10(M.n_features)
    M["log_n"] = np.log10(M.n)
    D = pd.get_dummies(M[["clf", "sampler"]], drop_first=True).astype(float)
    base = ["log_ir", "log_nmin", "log_p", "log_n"]
    Xd = sm.add_constant(pd.concat([M[base], D], axis=1))
    rows, fits = [], {}
    for dep in ["correct", "leaked", "premium"]:
        m = sm.OLS(M[dep].to_numpy(), Xd.to_numpy(dtype=float)).fit(
            cov_type="cluster", cov_kwds={"groups": M.dataset})
        fits[dep] = m
        rows.append(dict(term="R2", correct=np.nan, leaked=np.nan, premium=np.nan))
    names = ["const"] + base + list(D.columns)
    out = pd.DataFrame({"term": names})
    for dep in ["correct", "leaked", "premium"]:
        m = fits[dep]
        out[f"{dep}_coef"] = np.round(m.params, 4)
        out[f"{dep}_se"] = np.round(m.bse, 4)
        out[f"{dep}_p"] = np.round(m.pvalues, 4)
        S[f"metareg_r2_{dep}"] = r(m.rsquared)
    out.to_csv(RES / "table3_metaregression.csv", index=False)
    S["metareg_r2_ratio"] = r(S["metareg_r2_correct"] / max(S["metareg_r2_leaked"], 1e-9), 2)

    # data-only model: what the data alone explains, ignoring the algorithm
    Xo = sm.add_constant(M[base])
    for dep in ["correct", "leaked"]:
        m = sm.OLS(M[dep].to_numpy(), Xo.to_numpy(dtype=float)).fit(
            cov_type="cluster", cov_kwds={"groups": M.dataset})
        S[f"dataonly_r2_{dep}"] = r(m.rsquared)
    # algorithm-only model: what the classifier and resampler alone explain
    Xa = sm.add_constant(D)
    for dep in ["correct", "leaked"]:
        m = sm.OLS(M[dep].to_numpy(), Xa.to_numpy(dtype=float)).fit(
            cov_type="cluster", cov_kwds={"groups": M.dataset})
        S[f"algonly_r2_{dep}"] = r(m.rsquared)


# ===========================================================================
# 5. mechanism -- contamination  (Figure 4a)
# ===========================================================================
def mechanism(piv: pd.DataFrame) -> None:
    d = piv.dropna(subset=["contam"])
    if d.empty:
        return
    S["contam_mean"] = r(d.contam.mean())
    S["contam_min"] = r(d.contam.min())
    S["contam_max"] = r(d.contam.max())
    S["corr_contam_premium"] = r(np.corrcoef(d.contam, d.premium)[0, 1])
    S["corr_contam_leaked"] = r(np.corrcoef(d.contam, d.leaked)[0, 1])
    S["corr_contam_logir"] = r(np.corrcoef(np.log10(d.imbalance_ratio), d.contam)[0, 1])
    # by resampler -- RandomOverSampler duplicates rather than interpolates
    g = d.groupby("sampler").agg(contam=("contam", "mean"), premium=("premium", "mean")).round(4)
    g.to_csv(RES / "table7_mechanism.csv")
    S["contam_by_sampler"] = {k: r(v) for k, v in g.contam.items()}
    S["premium_by_sampler"] = {k: r(v) for k, v in g.premium.items()}
    for k, v in g.contam.items():
        S[f"contam_{k}"] = r(v)
    for k, v in g.premium.items():
        S[f"premium_{k}"] = r(v)
    # mediation: does contamination carry the effect of difficulty on the premium?
    import statsmodels.api as sm
    X1 = sm.add_constant(d[["correct"]].to_numpy())
    tot = sm.OLS(d.premium.to_numpy(), X1).fit().params[1]
    X2 = sm.add_constant(d[["correct", "contam"]].to_numpy())
    dir_ = sm.OLS(d.premium.to_numpy(), X2).fit().params[1]
    S["mediation_total"] = r(tot)
    S["mediation_direct"] = r(dir_)
    S["mediation_share"] = r(1 - dir_ / tot, 3)
    S["mediation_share_pct"] = r(100 * (1 - dir_ / tot), 1)


# ===========================================================================
# 6. simulation  (Figure 3)
# ===========================================================================
def simulation() -> None:
    f = RES / "simulation.csv"
    if not f.exists():
        print("  (no simulation.csv -- skipping)"); return
    G = pd.read_csv(f)
    if "error" in G.columns:
        G = G[G["error"].isna()]
    G = G.dropna(subset=["leaked", "correct"])
    G["premium"] = G.leaked - G.correct
    S["sim_cells"] = int(len(G))
    S["sim_leaked_mean"] = r(G.leaked.mean()); S["sim_leaked_sd"] = r(G.leaked.std())
    S["sim_correct_mean"] = r(G.correct.mean()); S["sim_correct_sd"] = r(G.correct.std())
    S["sim_variance_ratio"] = r(G.correct.var() / G.leaked.var(), 2)
    for fac in ["ir", "dim", "sep", "n"]:
        g = G.groupby(fac)[["leaked", "correct"]].mean()
        S[f"sim_range_leaked_{fac}"] = r(g.leaked.max() - g.leaked.min())
        S[f"sim_range_correct_{fac}"] = r(g.correct.max() - g.correct.min())
    g = G.groupby("ir")[["leaked", "correct", "premium"]].mean().round(4)
    g.to_csv(RES / "table5_simulation.csv")
    S["sim_ir2_correct"] = r(float(g.correct.iloc[0]))
    S["sim_irmax_correct"] = r(float(g.correct.iloc[-1]))
    S["sim_ir2_leaked"] = r(float(g.leaked.iloc[0]))
    S["sim_irmax_leaked"] = r(float(g.leaked.iloc[-1]))

    fig, ax = plt.subplots(1, 3, figsize=(11.4, 3.3))
    for a_, (fac, lab, logx) in zip(ax, [("ir", "Imbalance ratio", True),
                                         ("dim", "Number of features", False),
                                         ("sep", "Class separation", False)]):
        g = G.groupby(fac)[["leaked", "correct"]].agg(["mean", "std"])
        a_.errorbar(g.index, g[("leaked", "mean")], yerr=g[("leaked", "std")],
                    fmt="o-", c=RED, ms=4, lw=1.4, capsize=3, label="leaked")
        a_.errorbar(g.index, g[("correct", "mean")], yerr=g[("correct", "std")],
                    fmt="s--", c=BLUE, ms=4, lw=1.4, capsize=3, label="correct")
        if logx:
            a_.set_xscale("log")
        a_.set(xlabel=lab, ylim=(0, 1))
        a_.grid(alpha=.2, lw=.5)
    ax[0].set_ylabel("Minority-class F1")
    ax[0].legend(frameon=False, fontsize=7.5)
    ax[0].set_title("(a) varying imbalance", fontsize=8.5)
    ax[1].set_title("(b) varying dimensionality", fontsize=8.5)
    ax[2].set_title("(c) varying class overlap", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(FIG / "fig03_simulation.png"); plt.close(fig)


# ===========================================================================
# 7. mitigation  (Table 4, Figure 4)
# ===========================================================================
def mitigation(piv: pd.DataFrame) -> None:
    f = RES / "mitigation.csv"
    if not f.exists():
        print("  (no mitigation.csv -- skipping)"); return
    M = pd.read_csv(f)
    if "error" in M.columns:
        M = M[M["error"].isna()]
    M = M.dropna(subset=["leaked", "correct"])
    M["premium"] = M.leaked - M.correct
    t4 = M.groupby("sampling_strategy").agg(
        n=("premium", "size"), premium=("premium", "mean"), premium_sd=("premium", "std"),
        leaked=("leaked", "mean"), correct=("correct", "mean"),
        contamination=("contam", "mean")).round(4)
    lo_hi = M.groupby("sampling_strategy").premium.apply(lambda s: pd.Series(boot_ci(s.to_numpy()),
                                                                            index=["lo", "hi"]))
    t4 = t4.join(lo_hi.unstack().round(4))
    t4.to_csv(RES / "table4_mitigation.csv")
    S["mitigation_premium"] = {str(k): r(v) for k, v in t4.premium.items()}
    S["mitigation_contam"] = {str(k): r(v) for k, v in t4.contamination.items()}
    for k, v in t4.premium.items():
        S[f"premium_at_{int(round(k*100)):03d}"] = r(v)
    for k, v in t4.contamination.items():
        S[f"contam_at_{int(round(k*100)):03d}"] = r(v)
    lows = [k for k in t4.index if k <= 0.25]
    if lows:
        S["premium_at_025"] = r(float(t4.loc[max(lows), "premium"]))
        S["premium_at_015"] = r(float(t4.loc[min(t4.index), "premium"]))
    S["premium_at_100"] = r(float(t4.loc[t4.index.max(), "premium"]))
    S["mitigation_reduction_pct"] = r(
        100 * (1 - S["premium_at_025"] / S["premium_at_100"]), 1) if S.get("premium_at_025") else None

    fig, ax = plt.subplots(1, 2, figsize=(8.4, 3.3))
    d = piv.dropna(subset=["contam"])
    if not d.empty:
        b = d.assign(bin=pd.qcut(d.contam, 10, duplicates="drop")).groupby("bin", observed=True)
        gg = b.agg(c=("contam", "mean"), p=("premium", "mean"), s=("premium", "std"))
        ax[0].errorbar(gg.c, gg.p, yerr=gg.s, fmt="o-", c=GREEN, ms=4, lw=1.4, capsize=3)
        ax[0].set(xlabel="Contamination rate", ylabel="Leakage premium",
                  title=f"(a) Mechanism   r = {S.get('corr_contam_premium', float('nan')):.3f}")
        ax[0].grid(alpha=.2, lw=.5)
    ax[1].errorbar(t4.index, t4.premium,
                   yerr=[t4.premium - t4["lo"], t4["hi"] - t4.premium],
                   fmt="o-", c=RED, ms=5, lw=1.6, capsize=3, label="premium")
    ax[1].axhline(0, color="k", lw=.7)
    a2 = ax[1].twinx()
    a2.plot(t4.index, t4.contamination, "s--", c=GREY, ms=4, lw=1.2, label="contamination")
    a2.set_ylabel("Contamination rate", color=GREY)
    a2.spines["top"].set_visible(False)
    ax[1].set(xlabel="SMOTE sampling_strategy (target minority prevalence)",
              ylabel="Leakage premium", title="(b) Partial resampling as a safeguard")
    h1, l1 = ax[1].get_legend_handles_labels(); h2, l2 = a2.get_legend_handles_labels()
    ax[1].legend(h1 + h2, l1 + l2, frameon=False, fontsize=7.5, loc="upper left")
    fig.tight_layout()
    fig.savefig(FIG / "fig04_mechanism_mitigation.png"); plt.close(fig)


# ===========================================================================
# 8. the diagnostic  (Figure 5)
# ===========================================================================
def diagnostic(piv: pd.DataFrame) -> None:
    from sklearn.model_selection import LeaveOneGroupOut
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score, roc_curve, brier_score_loss

    diag = pd.concat([piv.assign(F1=piv.leaked, lab=1),
                      piv.assign(F1=piv.correct, lab=0)], ignore_index=True)
    diag["log_ir"] = np.log10(diag.imbalance_ratio)
    diag["log_nmin"] = np.log10(diag.n_minority)
    diag["log_p"] = np.log10(diag.n_features)
    feats = ["F1", "log_ir", "log_nmin", "log_p"]
    Xd = pd.concat([diag[feats], pd.get_dummies(diag.clf, prefix="clf").astype(float)], axis=1)
    yd = diag.lab.to_numpy()
    groups = diag.dataset.to_numpy()

    # GradBoost is the shipped model: it is what 04_export_model.py exports, and a
    # linear alternative is fitted alongside only as a robustness check.  Choosing the
    # reported model by its own validation score would be a small selection effect.
    models = {"GradBoost": GradientBoostingClassifier(random_state=0),
              "LogReg": LogisticRegression(max_iter=2000)}
    oofs = {}
    for nm, mdl in models.items():
        oof = np.zeros(len(yd))
        for tr, te in LeaveOneGroupOut().split(Xd, yd, groups):
            m = mdl.__class__(**mdl.get_params()).fit(Xd.iloc[tr], yd[tr])
            oof[te] = m.predict_proba(Xd.iloc[te])[:, 1]
        oofs[nm] = oof
        S[f"diag_auc_{nm}"] = r(roc_auc_score(yd, oof))
    best = "GradBoost"
    oof_best = oofs[best]
    auc_best = roc_auc_score(yd, oof_best)
    S["diag_model"] = best
    S["diag_auc"] = r(auc_best)
    S["diag_brier"] = r(brier_score_loss(yd, oof_best))
    S["diag_acc"] = r(((oof_best > .5) == yd).mean())
    S["diag_n"] = int(len(yd))
    ds_auc = []
    for ds in np.unique(groups):
        m = groups == ds
        if len(set(yd[m])) > 1:
            ds_auc.append(roc_auc_score(yd[m], oof_best[m]))
    S["diag_auc_perdataset_median"] = r(float(np.median(ds_auc)))
    S["diag_auc_perdataset_min"] = r(float(np.min(ds_auc)))
    S["diag_n_datasets_scored"] = int(len(ds_auc))
    S["diag_n_datasets_below_chance"] = int(sum(a < 0.5 for a in ds_auc))
    S["diag_n_datasets_above_09"] = int(sum(a > 0.9 for a in ds_auc))
    # a features-only ablation: no F1, just the data descriptors
    fo = Xd.drop(columns=["F1"])
    oof2 = np.zeros(len(yd))
    for tr, te in LeaveOneGroupOut().split(fo, yd, groups):
        m = GradientBoostingClassifier(random_state=0).fit(fo.iloc[tr], yd[tr])
        oof2[te] = m.predict_proba(fo.iloc[te])[:, 1]
    S["diag_auc_without_f1"] = r(roc_auc_score(yd, oof2))

    # train the shipped model on everything
    final = models[best].__class__(**models[best].get_params()).fit(Xd, yd)
    import pickle
    with open(RES / "diagnostic_model.pkl", "wb") as fh:
        pickle.dump({"model": final, "columns": list(Xd.columns), "feats": feats}, fh)

    fig, ax = plt.subplots(1, 2, figsize=(8.6, 3.3))
    fpr, tpr, _ = roc_curve(yd, oof_best)
    ax[0].plot(fpr, tpr, c=BLUE, lw=1.8, label=f"AUC = {auc_best:.3f}")
    ax[0].plot([0, 1], [0, 1], "k--", lw=.8)
    ax[0].set(xlabel="False positive rate", ylabel="True positive rate",
              title="(a) Leave-one-dataset-out detection")
    ax[0].legend(frameon=False, fontsize=7.5)

    grid_f1 = np.linspace(.3, .99, 60)
    for ir, c in zip([2, 5, 20, 80], [GREY, GREEN, BLUE, RED]):
        proto = pd.DataFrame({"F1": grid_f1})
        proto["log_ir"] = np.log10(ir)
        proto["log_nmin"] = np.log10(float(diag.n_minority.median()))
        proto["log_p"] = np.log10(float(diag.n_features.median()))
        for c2 in Xd.columns:
            if c2.startswith("clf_"):
                proto[c2] = 1.0 if c2 == "clf_RandForest" else 0.0
        p = final.predict_proba(proto[Xd.columns])[:, 1]
        ax[1].plot(grid_f1, p, c=c, lw=1.6, label=f"IR = {ir}")
    ax[1].axhline(.5, color="k", lw=.7, ls=":")
    ax[1].set(xlabel="Reported minority-class F1", ylabel="P(result came from leaked ordering)",
              ylim=(0, 1), title="(b) A reported score, screened")
    ax[1].legend(frameon=False, fontsize=7.5)
    fig.tight_layout()
    fig.savefig(FIG / "fig05_diagnostic.png"); plt.close(fig)


# ===========================================================================
# 9. HR-specific analysis for the applied paper  (Figure 6, Table 8)
# ===========================================================================
def hr_focus(piv: pd.DataFrame, R: pd.DataFrame, man: pd.DataFrame) -> None:
    H = piv[piv.domain == "HR"]
    B = piv[piv.domain == "benchmark"]
    if H.empty:
        return
    S["hr_leaked_mean"] = r(H.leaked.mean()); S["hr_correct_mean"] = r(H.correct.mean())
    S["hr_premium"] = r(H.premium.mean())
    lo, hi = boot_ci(H.premium.to_numpy())
    S["hr_premium_ci_lo"], S["hr_premium_ci_hi"] = r(lo), r(hi)
    S["bench_premium"] = r(B.premium.mean())
    lo, hi = boot_ci(B.premium.to_numpy())
    S["bench_premium_ci_lo"], S["bench_premium_ci_hi"] = r(lo), r(hi)
    S["hr_variance_ratio"] = r(H.correct.var() / H.leaked.var(), 2)

    none = R[(R.condition == "none")].merge(man[["name", "domain"]], left_on="dataset",
                                            right_on="name")
    nh = none[none.domain == "HR"]
    rows = []
    for ds, g in H.groupby("dataset"):
        m = man[man.name == ds].iloc[0]
        nb = nh[nh.dataset == ds].f1
        lo, hi = boot_ci(g.premium.to_numpy())
        rows.append(dict(dataset=ds, n=int(m.n), n_features=int(m.n_features),
                         n_minority=int(m.n_minority), imbalance_ratio=m.imbalance_ratio,
                         leaked=g.leaked.mean(), correct=g.correct.mean(),
                         none=nb.mean() if len(nb) else np.nan,
                         premium=g.premium.mean(), premium_lo=lo, premium_hi=hi))
    t8 = pd.DataFrame(rows).sort_values("imbalance_ratio").round(4)
    t8.to_csv(RES / "table8_hr.csv", index=False)
    S["hr_best_none_datasets"] = int((t8["none"] > t8["correct"]).sum())
    S["hr_n_datasets"] = int(len(t8))
    nh2 = nh.groupby(["dataset", "clf", "seed"]).f1.mean()
    ch2 = H.groupby(["dataset", "clf", "seed"]).correct.mean()
    mm2 = pd.concat({"none": nh2, "correct": ch2}, axis=1).dropna()
    S["hr_pct_none_beats_correct"] = r(100 * (mm2.none > mm2.correct).mean(), 1)
    S["hr_none_minus_correct"] = r(float((mm2.none - mm2.correct).mean()))
    S["hr_premium_min"] = r(t8.premium.min()); S["hr_premium_max"] = r(t8.premium.max())
    S["hr_premium_min_dataset"] = str(t8.loc[t8.premium.idxmin(), "dataset"])
    S["hr_premium_max_dataset"] = str(t8.loc[t8.premium.idxmax(), "dataset"])
    S["hr_corr_ir_premium"] = r(np.corrcoef(np.log10(t8.imbalance_ratio), t8.premium)[0, 1])
    S["hr_leaked_range"] = r(t8.leaked.max() - t8.leaked.min())
    S["hr_correct_range"] = r(t8.correct.max() - t8.correct.min())
    S["hr_by_dataset"] = {
        row["dataset"]: {k: r(row[k]) for k in ("leaked", "correct", "none", "premium",
                                                "premium_lo", "premium_hi")}
        for _, row in t8.iterrows()}
    S["hr_leaked_sd"] = r(H.leaked.std())
    S["hr_correct_sd"] = r(H.correct.std())
    S["hr_correct_min"] = r(t8.correct.min()); S["hr_correct_max"] = r(t8.correct.max())
    S["hr_leaked_min"] = r(t8.leaked.min()); S["hr_leaked_max"] = r(t8.leaked.max())
    S["hr_law_r2"] = r(np.corrcoef(H.correct, H.premium)[0, 1] ** 2)
    hclf = H.groupby("clf")[["leaked", "correct"]].mean()
    S["hr_clf_spread_correct"] = r(hclf.correct.max() - hclf.correct.min())
    S["hr_clf_spread_leaked"] = r(hclf.leaked.max() - hclf.leaked.min())
    S["hr_best_clf"] = str(hclf.correct.idxmax())
    # precision and recall under the correct protocol, for the 'what to expect' section
    hrids = set(man[man.domain == "HR"].name)
    cH = R[(R.condition == "correct") & (R.dataset.isin(hrids))]
    S["hr_correct_recall"] = r(cH.recall.mean())
    S["hr_correct_precision"] = r(cH.precision.mean())
    S["hr_correct_recall_min"] = r(cH.groupby("dataset").recall.mean().min())
    S["hr_correct_recall_max"] = r(cH.groupby("dataset").recall.mean().max())
    S["hr_correct_prauc"] = r(cH.pr_auc.mean())
    S["hr_correct_rocauc"] = r(cH.roc_auc.mean())
    lH = R[(R.condition == "leaked") & (R.dataset.isin(hrids))]
    S["hr_leaked_precision"] = r(lH.precision.mean())
    S["hr_leaked_recall"] = r(lH.recall.mean())
    if "ibm_hr" in set(t8.dataset):
        row = t8[t8.dataset == "ibm_hr"].iloc[0]
        S["ibm_ratio"] = r(row["leaked"] / row["correct"], 2)

    fig, ax = plt.subplots(figsize=(7.4, 3.4))
    x = np.arange(len(t8)); w = .26
    ax.bar(x - w, t8.leaked, w, color=RED, label="leaked")
    ax.bar(x, t8.correct, w, color=BLUE, label="correct")
    ax.bar(x + w, t8["none"], w, color=GREY, label="no resampling")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{n}\nIR {ir:.1f}" for n, ir in zip(t8.dataset, t8.imbalance_ratio)],
                       fontsize=6.8)
    ax.set(ylabel="Minority-class F1", ylim=(0, 1))
    ax.legend(frameon=False, fontsize=7.5, ncol=3)
    ax.grid(axis="y", alpha=.2, lw=.5)
    fig.tight_layout()
    fig.savefig(FIG / "fig06_hr.png"); plt.close(fig)

    # Figure 7 -- what the leaked band would predict for HR-scale problems
    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    ax.scatter(B.imbalance_ratio, B.premium, s=6, alpha=.12, c=GREY, label="benchmark")
    ax.scatter(H.imbalance_ratio, H.premium, s=10, alpha=.35, c=RED, label="HR")
    ax.set(xscale="log", xlabel="Imbalance ratio", ylabel="Leakage premium")
    ax.axhline(0, color="k", lw=.7)
    ax.legend(frameon=False, fontsize=7.5)
    ax.grid(alpha=.2, lw=.5)
    fig.tight_layout()
    fig.savefig(FIG / "fig07_hr_vs_benchmark.png"); plt.close(fig)


# ===========================================================================
def main() -> int:
    R, piv, man = load_raw()
    print(f"{S['n_results']} usable fits, {S['n_paired']} paired comparisons")
    corpus(man, R)
    invariance(piv, R)
    meta_regression(piv)
    mechanism(piv)
    simulation()
    mitigation(piv)
    diagnostic(piv)
    hr_focus(piv, R, man)

    import sklearn, imblearn, scipy, sys as _s
    S["versions"] = dict(python=_s.version.split()[0], numpy=np.__version__,
                         pandas=pd.__version__, scipy=scipy.__version__,
                         scikit_learn=sklearn.__version__, imbalanced_learn=imblearn.__version__)
    try:
        import xgboost; S["versions"]["xgboost"] = xgboost.__version__
    except ImportError:
        pass
    with open(RES / "summary.json", "w") as f:
        json.dump(S, f, indent=2, sort_keys=True)
    print(f"\nwrote {RES/'summary.json'} with {len(S)} keys")
    for k in ["n_datasets", "variance_ratio", "mean_premium", "law_r2",
              "metareg_r2_correct", "metareg_r2_leaked", "diag_auc"]:
        print(f"  {k:24s} {S.get(k)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
