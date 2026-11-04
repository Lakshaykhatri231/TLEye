"""Pack the cleaned dataset (code TEXT + constraints text + labels + splits) for a PRIVATE Kaggle dataset.

    python -I -m tleye.data.kaggle_pack
Writes data/processed/kaggle_pack/{tleye_data.parquet, dataset-metadata.json}. Code is read as bytes and
decoded as text only; nothing is executed. No credentials or local paths are written.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd

from tleye.constraints.parser import parse_statement

ROOT = Path("data/raw/extracted/Project_CodeNet")
OUT = Path("data/processed/kaggle_pack")
MAX_CHARS = 24_000
DATASET_ID = "khatrilakshay2004/tleye-data"


def fmt(v: float) -> str:
    if v >= 1e300:
        return "huge"
    e = math.floor(math.log10(v)) if v > 0 else 0
    return str(int(v)) if v < 1000 else f"{v / 10 ** e:g}e{e}"


def constraint_text_from_html(html: str) -> str:
    """The `constraints:` line of the LoRA prompt. SHARED by training-data packing and inference (tleye.models.lora)."""
    c = parse_statement(html)
    items = sorted(c.upper.items(), key=lambda kv: -kv[1])[:8]
    return ", ".join(f"{k}<={fmt(v)}" for k, v in items) if items else "none"


def constraint_text(pid: str) -> str:
    return constraint_text_from_html((ROOT / "problem_descriptions" / f"{pid}.html").read_text(errors="replace"))


def main() -> None:
    from tleye.models.baselines import add_est  # same estimate B2 uses -> identical information
    from tleye.models.baselines import load as load_main
    d = load_main()
    d = add_est(d, float(d[d.split == "train"]["log10_primary"].median()))
    d["sym"] = [
        f"loop_depth={int(r.max_loop_depth)} loops={int(r.n_loops)} recursion={'yes' if r.recursive else 'no'} "
        f"sorts={int(r.n_sort)} sort_in_loop={int(r.sort_in_loop)} linear_ops_in_loop={int(r.linear_in_loop)} "
        f"est_log10_ops={r.est_log_ops:.1f} budget_score={r.b1_score:.1f}"
        for r in d.itertuples()
    ]
    ctext = {p: constraint_text(p) for p in d["problem_id"].unique()}
    d["constraints"] = d["problem_id"].map(ctext)
    d["code"] = [Path(p).read_bytes().decode("utf8", "replace")[:MAX_CHARS] for p in d["path"]]
    d["rt"] = d["runtime"].where(~d["runtime"].str.startswith("PyPy"), "PyPy")
    # fixed 400-item test sample for the prompted-LLM baseline B3 (stratified by language, seeded)
    te = d[d.split == "test"]
    pick = pd.concat([g.sample(min(len(g), n), random_state=0)
                      for (lang, n) in [("C++", 140), ("Java", 120), ("Python", 140)]
                      for g in [te[te.language == lang]]])
    d["b3_sample"] = d["submission_id"].isin(pick["submission_id"])
    cols = ["submission_id", "problem_id", "split", "language", "rt", "y_tle", "time_limit_ms",
            "constraints", "sym", "code", "b3_sample"]
    OUT.mkdir(parents=True, exist_ok=True)
    d[cols].to_parquet(OUT / "tleye_data.parquet", compression="zstd", index=False)
    (OUT / "dataset-metadata.json").write_text(json.dumps({
        "title": "TLEye data", "id": DATASET_ID, "licenses": [{"name": "other"}],
        "description": "Derived from IBM Project CodeNet (AtCoder subset): cleaned AC/TLE submissions with problem-disjoint splits. Private.",
    }, indent=1))
    print(d[cols].groupby("split").size().to_dict(), "b3:", int(d["b3_sample"].sum()))
    print("parquet MB:", round((OUT / "tleye_data.parquet").stat().st_size / 1e6, 1))
    print("example constraints:", list(ctext.values())[:3])


if __name__ == "__main__":
    main()
