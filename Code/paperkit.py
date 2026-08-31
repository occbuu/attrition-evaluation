"""
paperkit.py -- build the manuscripts as .docx with no hand-typed numbers.

Paper 1 had to withdraw three claims because they did not match its own tables.
The rule adopted for Paper 2 is that every quantity in the prose is a
placeholder resolved at build time from results/summary.json or a results table.
A placeholder that does not resolve stops the build; a number typed directly
into the prose is caught by 05_verify_numbers.py.

Placeholder syntax inside any text string:

    {variance_ratio}            value from summary.json, default formatting
    {mean_premium:.3f}          explicit format spec
    {premium_ci_lo:.3f}-{premium_ci_hi:.3f}
    {pct:pct_premium_positive}  as a percentage with one decimal
"""
from __future__ import annotations
import json, re
from pathlib import Path

import pandas as pd
from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
FIG = ROOT / "figures"

_PLACEHOLDER = re.compile(r"\{([a-zA-Z_][\w.]*)(?::([^{}]*))?\}")


def default_format(v) -> str:
    """House style: scores in [0,1) to three decimals, ratios and counts read naturally.

    '4.86 times the variance' reads; '4.860 times the variance' does not, and a
    manuscript full of spurious trailing zeros invites a reviewer to wonder what
    else was left on a default.
    """
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return str(v)
    if isinstance(v, int):
        return f"{v:,}" if abs(v) >= 10000 else str(v)
    if abs(v) >= 1:
        return f"{v:.2f}".rstrip("0").rstrip(".")
    return f"{v:.3f}"


class Numbers:
    """Read-only view of summary.json that records which keys were used."""

    def __init__(self, path: Path | None = None):
        self.path = path or (RES / "summary.json")
        with open(self.path, "r", encoding="utf-8") as f:
            self.d = json.load(f)
        self.used: set[str] = set()
        self.missing: set[str] = set()

    def raw(self, key: str):
        cur = self.d
        for part in key.split("."):
            if isinstance(cur, dict) and part in cur:
                cur = cur[part]
            else:
                self.missing.add(key)
                raise KeyError(f"summary.json has no key {key!r}")
        self.used.add(key)
        return cur

    def fill(self, text: str) -> str:
        def sub(m: re.Match) -> str:
            key, spec = m.group(1), m.group(2)
            if key == "pct":
                key, spec = spec, ".1f"
                v = self.raw(key)
                return format(float(v), spec)
            v = self.raw(key)
            if spec:
                return format(float(v), spec) if isinstance(v, (int, float)) else str(v)
            return default_format(v)
        return _PLACEHOLDER.sub(sub, text)


# ---------------------------------------------------------------------------
class Paper:
    """A thin, opinionated wrapper over python-docx for journal manuscripts."""

    def __init__(self, numbers: Numbers, font: str = "Times New Roman", size: float = 11):
        self.N = numbers
        self.doc = Document()
        st = self.doc.styles["Normal"]
        st.font.name = font
        st.font.size = Pt(size)
        st.element.rPr.rFonts.set(qn("w:eastAsia"), font)
        st.paragraph_format.space_after = Pt(6)
        st.paragraph_format.line_spacing = 1.15
        for i, sz, it in [(1, 14, False), (2, 12, False), (3, 11, True)]:
            h = self.doc.styles[f"Heading {i}"]
            h.font.name = font
            h.font.size = Pt(sz)
            h.font.bold = True
            h.font.italic = it
            h.font.color.rgb = RGBColor(0, 0, 0)
            h.paragraph_format.space_before = Pt(12)
            h.paragraph_format.space_after = Pt(4)
        sec = self.doc.sections[0]
        sec.left_margin = sec.right_margin = Inches(1.0)
        sec.top_margin = sec.bottom_margin = Inches(1.0)
        self._caption_counts = {"Table": 0, "Figure": 0}

    # -- text ---------------------------------------------------------------
    def title(self, text: str, subtitle: str | None = None):
        p = self.doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = p.add_run(self.N.fill(text)); r.bold = True; r.font.size = Pt(16)
        if subtitle:
            q = self.doc.add_paragraph(); q.alignment = WD_ALIGN_PARAGRAPH.CENTER
            s = q.add_run(self.N.fill(subtitle)); s.italic = True; s.font.size = Pt(11)

    def authors(self, line: str, affil: str | None = None):
        p = self.doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.add_run(line).font.size = Pt(11)
        if affil:
            q = self.doc.add_paragraph(); q.alignment = WD_ALIGN_PARAGRAPH.CENTER
            r = q.add_run(affil); r.italic = True; r.font.size = Pt(9.5)

    def h1(self, text: str):
        self.doc.add_heading(self.N.fill(text), level=1)

    def h2(self, text: str):
        self.doc.add_heading(self.N.fill(text), level=2)

    def h3(self, text: str):
        self.doc.add_heading(self.N.fill(text), level=3)

    def p(self, text: str, italic: bool = False, justify: bool = True):
        par = self.doc.add_paragraph()
        r = par.add_run(self.N.fill(" ".join(text.split())))
        r.italic = italic
        if justify:
            par.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        return par

    def abstract(self, text: str):
        self.h1("Abstract")
        p = self.p(text)
        p.paragraph_format.left_indent = Inches(0.25)
        p.paragraph_format.right_indent = Inches(0.25)

    def keywords(self, words: str):
        p = self.doc.add_paragraph()
        r = p.add_run("Keywords: "); r.bold = True
        p.add_run(words)

    def bullets(self, items: list[str], numbered: bool = False):
        style = "List Number" if numbered else "List Bullet"
        for it in items:
            self.doc.add_paragraph(self.N.fill(" ".join(it.split())), style=style)

    def code(self, text: str):
        p = self.doc.add_paragraph()
        r = p.add_run(text)
        r.font.name = "Consolas"; r.font.size = Pt(9)
        p.paragraph_format.left_indent = Inches(0.3)
        p.paragraph_format.space_after = Pt(8)

    def pagebreak(self):
        self.doc.add_page_break()

    # -- floats -------------------------------------------------------------
    def _caption(self, kind: str, text: str, above: bool, numbered: bool = True):
        if numbered:
            self._caption_counts[kind] += 1
            n = self._caption_counts[kind]
            lead = f"{kind} {n}. "
        else:
            n = None
            lead = ""
        p = self.doc.add_paragraph()
        r = p.add_run(lead); r.bold = True; r.font.size = Pt(9.5)
        s = p.add_run(self.N.fill(" ".join(text.split()))); s.font.size = Pt(9.5)
        p.paragraph_format.space_before = Pt(4 if above else 2)
        p.paragraph_format.space_after = Pt(6 if above else 10)
        return n

    def figure(self, filename: str, caption: str, width: float = 6.2):
        path = FIG / filename
        if not path.exists():
            raise FileNotFoundError(path)
        p = self.doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.add_run().add_picture(str(path), width=Inches(width))
        self._caption("Figure", caption, above=False)

    def table(self, df: pd.DataFrame, caption: str, headers: list[str] | None = None,
              fontsize: float = 8.5, widths: list[float] | None = None,
              align_right_from: int = 1, numbered: bool = True):
        self._caption("Table", caption, above=True, numbered=numbered)
        cols = headers or list(df.columns)
        t = self.doc.add_table(rows=1, cols=len(cols))
        t.style = "Table Grid"
        t.alignment = WD_TABLE_ALIGNMENT.CENTER
        for i, c in enumerate(cols):
            cell = t.rows[0].cells[i]
            cell.text = ""
            r = cell.paragraphs[0].add_run(str(c)); r.bold = True; r.font.size = Pt(fontsize)
        for _, row in df.iterrows():
            cells = t.add_row().cells
            for i, v in enumerate(row):
                cells[i].text = ""
                par = cells[i].paragraphs[0]
                if i >= align_right_from:
                    par.alignment = WD_ALIGN_PARAGRAPH.RIGHT
                par.add_run("" if pd.isna(v) else str(v)).font.size = Pt(fontsize)
        if widths:
            for r_ in t.rows:
                for i, w in enumerate(widths):
                    r_.cells[i].width = Inches(w)
        self.doc.add_paragraph().paragraph_format.space_after = Pt(4)

    def references(self, refs: list[str]):
        self.h1("References")
        for i, r in enumerate(refs, 1):
            p = self.doc.add_paragraph()
            p.paragraph_format.left_indent = Inches(0.3)
            p.paragraph_format.first_line_indent = Inches(-0.3)
            p.paragraph_format.space_after = Pt(4)
            run = p.add_run(f"[{i}] {r}")
            run.font.size = Pt(9.5)

    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.doc.save(str(path))
        return path


# ---------------------------------------------------------------------------
def table_csv(name: str) -> pd.DataFrame:
    return pd.read_csv(RES / name)


def fmt(df: pd.DataFrame, dec: dict[str, int] | None = None,
        rename: dict[str, str] | None = None,
        keep: list[str] | None = None) -> pd.DataFrame:
    d = df.copy()
    if keep:
        d = d[keep]
    for c, k in (dec or {}).items():
        if c in d.columns:
            d[c] = d[c].map(lambda v: "" if pd.isna(v) else f"{v:.{k}f}")
    if rename:
        d = d.rename(columns=rename)
    return d
