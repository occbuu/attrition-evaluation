"""Command line front end:  `leakcheck --f1 0.92 --ir 5.2 ...`"""
from __future__ import annotations
import argparse, json, sys

from . import check, check_many, model_info, __version__


def _screen_file(path: str, fmt: str) -> int:
    try:
        import pandas as pd
    except ImportError:
        print("reading a table needs pandas: pip install 'leakcheck[table]'", file=sys.stderr)
        return 2
    df = pd.read_csv(path)
    need = {"f1", "imbalance_ratio", "n_minority", "n_features"}
    missing = need - set(df.columns)
    if missing:
        print(f"missing columns: {sorted(missing)}", file=sys.stderr)
        return 2
    if "clf" not in df.columns:
        df["clf"] = None
    out = check_many(df[["f1", "imbalance_ratio", "n_minority", "n_features",
                         "clf"]].to_dict("records"))
    if fmt == "json":
        print(out.to_json(orient="records", indent=2))
    else:
        print(out.to_csv(index=False))
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="leakcheck",
        description="Screen a reported classification result for the signature "
                    "of resampling applied before the train/test split.")
    p.add_argument("--version", action="version", version=f"leakcheck {__version__}")
    p.add_argument("--f1", type=float, help="reported minority-class F1")
    p.add_argument("--ir", "--imbalance-ratio", dest="ir", type=float,
                   help="majority : minority ratio")
    p.add_argument("--n-minority", type=int, help="minority cases before resampling")
    p.add_argument("--n-features", type=int, help="predictors after encoding")
    p.add_argument("--clf", default=None, help="model family, any spelling")
    p.add_argument("--file", help="CSV with one row per reported result")
    p.add_argument("--format", choices=["text", "json", "csv"], default="text")
    p.add_argument("--info", action="store_true", help="print model provenance and exit")
    a = p.parse_args(argv)

    if a.info:
        print(json.dumps(model_info(), indent=2)); return 0
    if a.file:
        return _screen_file(a.file, "json" if a.format == "json" else "csv")
    if None in (a.f1, a.ir, a.n_minority, a.n_features):
        p.error("give --f1, --ir, --n-minority and --n-features, or --file")

    r = check(f1=a.f1, imbalance_ratio=a.ir, n_minority=a.n_minority,
              n_features=a.n_features, clf=a.clf)
    if a.format == "json":
        print(json.dumps(r.__dict__, indent=2))
    elif a.format == "csv":
        print("probability,verdict"); print(f"{r.probability:.4f},{r.verdict}")
    else:
        print(f"P(leaked) = {r.probability:.2f}   [{r.verdict}]")
        print()
        print(r.explain())
    return 0


if __name__ == "__main__":
    sys.exit(main())
