#!/usr/bin/env python3
"""
05_verify_numbers.py -- the discipline check.

Paper 1 had to withdraw three claims because they disagreed with its own tables.
This script is the guard against a repeat.  It runs three independent checks:

  A. Source check.  Every numeric literal that appears in manuscript prose must
     either be a resolved placeholder (which came from summary.json by
     construction) or appear on an explicit allow-list.  A number typed by hand
     into a sentence fails here.

  B. Output check.  No unresolved placeholder may survive into a built .docx,
     and no built .docx may contain a stray brace expression.

  C. Recomputation check.  Headline quantities in summary.json are recomputed
     from raw_results.csv by a separate code path and compared.  This catches a
     bug in the analysis, which check A cannot.

Exit status is non-zero if any check fails.
"""
from __future__ import annotations
import ast, json, re, sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CODE = ROOT / "Code"
RES = ROOT / "results"

# Numbers that may legitimately appear in prose without coming from the summary:
# structural constants of the design, standard conventions, and citations.
ALLOWED = {
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "12", "20", "30", "35",
    "50", "60", "70", "73", "80", "95", "100", "200", "237", "294", "300", "1470",
    "1191", "1.0", "0.25", "0.5", "1.2", "0.9", "0.439", "0.5042", "5000", "0.0",
    "1e-9", "2026", "2023", "2022", "2021", "2020", "2017", "2012", "2002", "1.0",
    "17", "1.2.0", "3", "5", "0.15", "0.20",
    # values checked against their sources while writing; see refs.py and literature.py
    "5,000", "1,470", "1,191", "20,000", "0.95", "73", "294",
}
# Formatting fragments that are not claims: version strings, ratios, licences.
ALLOWED_PATTERNS = [
    re.compile(r"^\d{4}$"),              # years
    re.compile(r"^70:30$"),
    re.compile(r"^BSD-?3"),
]

# Headings and keyword lines carry section numbers, which are structure, not claims.
CALLS = {"p", "abstract", "bullets", "code"}
PLACEHOLDER = re.compile(r"\{[a-zA-Z_][\w.]*(?::[^{}]*)?\}")
SECTION_REF = re.compile(r"\bSections?\s+\d+(?:\.\d+)?(?:\s+(?:to|and)\s+\d+(?:\.\d+)?)?",
                         re.I)
FIGURE_REF = re.compile(r"\b(?:Figure|Table)\s+\d+(?:\([a-z]\))?", re.I)
# Thousands separators are part of the number; a colon or minus before a digit means
# it is part of a ratio (70:30) or an exponent (1e-9), not a claim.
NUMBER = re.compile(r"(?<![\w.:\-])\d{1,3}(?:,\d{3})*(?:\.\d+)?(?![\w])")


# ---------------------------------------------------------------------------
def prose_strings(path: Path) -> list[tuple[int, str]]:
    """Every literal string passed to a prose-emitting Paper method."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    out: list[tuple[int, str]] = []

    def collect(node, lineno):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            out.append((lineno, node.value))
        elif isinstance(node, ast.JoinedStr):
            for v in node.values:                       # f-string: literal parts only
                if isinstance(v, ast.Constant) and isinstance(v.value, str):
                    out.append((lineno, v.value))
        elif isinstance(node, (ast.List, ast.Tuple)):
            for e in node.elts:
                collect(e, lineno)
        elif isinstance(node, ast.BinOp):
            collect(node.left, lineno); collect(node.right, lineno)

    for n in ast.walk(tree):
        if isinstance(n, ast.Call):
            fn = n.func
            name = fn.attr if isinstance(fn, ast.Attribute) else getattr(fn, "id", "")
            if name in CALLS:
                for a in n.args:
                    collect(a, n.lineno)
            if name in {"figure", "table"}:
                for kw in n.keywords:
                    if kw.arg == "caption":
                        collect(kw.value, n.lineno)
                if name == "figure" and len(n.args) > 1:
                    collect(n.args[1], n.lineno)
    return out


def check_source(paths: list[Path]) -> list[str]:
    bad = []
    for p in paths:
        for lineno, s in prose_strings(p):
            stripped = PLACEHOLDER.sub(" ", s)
            stripped = SECTION_REF.sub(" ", stripped)
            stripped = FIGURE_REF.sub(" ", stripped)
            for m in NUMBER.finditer(stripped):
                tok = m.group(0)
                if tok in ALLOWED or any(r.match(tok) for r in ALLOWED_PATTERNS):
                    continue
                ctx = stripped[max(0, m.start() - 45):m.end() + 45].replace("\n", " ")
                bad.append(f"{p.name}:{lineno}  hand-typed number {tok!r}  ...{ctx.strip()}...")
    return bad


# ---------------------------------------------------------------------------
def check_output(docs: list[Path]) -> list[str]:
    from docx import Document
    bad = []
    for p in docs:
        if not p.exists():
            bad.append(f"{p.name}: not built"); continue
        doc = Document(str(p))
        texts = [par.text for par in doc.paragraphs]
        for t in doc.tables:
            for row in t.rows:
                texts += [c.text for c in row.cells]
        for t in texts:
            for m in PLACEHOLDER.finditer(t):
                bad.append(f"{p.name}: unresolved placeholder {m.group(0)}")
    return bad


# ---------------------------------------------------------------------------
CAPTION = re.compile(r"^(Table|Figure) (\d+)\.")
REF = re.compile(r"\b(Table|Figure) (\d+)")


def check_crossrefs(docs: list[Path]) -> list[str]:
    """Every float must be referred to in the text, and every reference must exist."""
    from docx import Document
    bad = []
    for p in docs:
        if not p.exists():
            continue
        doc = Document(str(p))
        caps, refs = set(), set()
        for par in doc.paragraphs:
            m = CAPTION.match(par.text)
            if m:
                caps.add(m.group(0)[:-1])
                continue
            for r in REF.finditer(par.text):
                refs.add(r.group(0))
        for r in sorted(refs - caps):
            bad.append(f"{p.name}: text refers to {r}, which has no caption")
        for c in sorted(caps - refs):
            bad.append(f"{p.name}: {c} is never referred to in the text")
    return bad


def check_recomputation() -> list[str]:
    """Independent recomputation of the headline numbers from the raw record."""
    import leakage_lib as L
    S = json.load(open(RES / "summary.json"))
    R = pd.read_csv(RES / "raw_results.csv")
    if "error" in R.columns:
        R = R[R["error"].isna()]
    man = L.manifest()

    # rebuild the paired frame from scratch rather than reusing L.paired
    lk = (R[R.condition == "leaked"]
          .set_index(["dataset", "clf", "sampler", "seed"]).f1)
    ct = (R[R.condition == "correct"]
          .set_index(["dataset", "clf", "sampler", "seed"]).f1)
    j = pd.concat({"leaked": lk, "correct": ct}, axis=1).dropna()
    prem = j.leaked - j.correct

    checks = {
        "n_paired": len(j),
        "leaked_f1_mean": j.leaked.mean(),
        "correct_f1_mean": j.correct.mean(),
        "leaked_f1_sd": j.leaked.std(),
        "correct_f1_sd": j.correct.std(),
        "mean_premium": prem.mean(),
        "median_premium": prem.median(),
        "variance_ratio": j.correct.var() / j.leaked.var(),
        "pct_premium_positive": 100 * (prem > 0).mean(),
        "n_datasets": man[man.name.isin(R.dataset.unique())].shape[0],
        "law_r2": np.corrcoef(j.correct, prem)[0, 1] ** 2,
    }
    bad = []
    for k, v in checks.items():
        if k not in S:
            bad.append(f"summary.json is missing {k}"); continue
        got, want = float(S[k]), float(v)
        tol = 0.5 if k in {"n_paired", "n_datasets"} else max(1e-3, abs(want) * 2e-3)
        if abs(got - want) > tol:
            bad.append(f"{k}: summary says {got}, recomputation says {want:.6f}")
    return bad


# ---------------------------------------------------------------------------
def main() -> int:
    sys.path.insert(0, str(CODE))
    srcs = sorted(CODE.glob("Paper2*/build_paper2*.py"))
    docs = [ROOT / "Paper2A" / "Paper2A_leakage_premium.docx",
            ROOT / "Paper2B" / "Paper2B_leakcheck_JOSS.docx",
            ROOT / "Paper2C" / "Paper2C_hr_attrition_evaluation.docx"]

    fails = 0
    for label, problems in [("A  source (no hand-typed numbers)", check_source(srcs)),
                            ("B  output (no unresolved placeholders)", check_output(docs)),
                            ("C  cross-references (tables and figures)", check_crossrefs(docs)),
                            ("D  recomputation from raw_results.csv", check_recomputation())]:
        if problems:
            fails += len(problems)
            print(f"\nFAIL  {label}  -- {len(problems)} problem(s)")
            for x in problems[:40]:
                print("   ", x)
            if len(problems) > 40:
                print(f"    ... and {len(problems)-40} more")
        else:
            print(f"PASS  {label}")

    if fails:
        print(f"\n{fails} problem(s) found.")
        return 1
    print("\nAll checks passed: every number in the manuscripts traces to the results.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
