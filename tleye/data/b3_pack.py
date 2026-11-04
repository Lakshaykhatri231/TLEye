"""Pack the fixed 400-item B3 sample (+ plain-text problem statements) for a small PRIVATE Kaggle dataset.
    python -I -m tleye.data.b3_pack      -> data/processed/kaggle_b3/{b3_items.parquet, dataset-metadata.json}"""
import html
import json
import re
from pathlib import Path

import pandas as pd

ROOT = Path("data/raw/extracted/Project_CodeNet/problem_descriptions")
OUT = Path("data/processed/kaggle_b3")


def statement_text(pid: str, limit: int = 6000) -> str:
    h = re.sub(r"(?i)</(?:li|p|pre|div|h\d|ul|ol|section)>|<br\s*/?>", "\n", (ROOT / f"{pid}.html").read_text(errors="replace"))
    t = html.unescape(re.sub(r"<[^>]+>", " ", h))
    return re.sub(r"\n\s*\n+", "\n", re.sub(r"[ \t]+", " ", t)).strip()[:limit]


def main() -> None:
    d = pd.read_parquet("data/processed/kaggle_pack/tleye_data.parquet")
    d = d[d.b3_sample].reset_index(drop=True)
    assert len(d) == 400 and (d.split == "test").all()
    d["statement"] = d.problem_id.map({p: statement_text(p) for p in d.problem_id.unique()})
    OUT.mkdir(parents=True, exist_ok=True)
    d[["submission_id", "problem_id", "language", "rt", "y_tle", "time_limit_ms", "constraints", "code", "statement"]].to_parquet(OUT / "b3_items.parquet", index=False)
    (OUT / "dataset-metadata.json").write_text(json.dumps({"title": "TLEye B3 items", "id": "khatrilakshay2004/tleye-b3", "licenses": [{"name": "other"}],
        "description": "400 fixed test-split items (derived from IBM Project CodeNet, AtCoder) with problem statements, for a prompted-LLM baseline. Private."}, indent=1))
    print(d.groupby("language").size().to_dict(), "| TLE share %.3f" % d.y_tle.mean(), "| problems", d.problem_id.nunique(),
          "| median statement chars", int(d.statement.str.len().median()), "| MB", round((OUT / "b3_items.parquet").stat().st_size / 1e6, 2))


if __name__ == "__main__":
    main()
