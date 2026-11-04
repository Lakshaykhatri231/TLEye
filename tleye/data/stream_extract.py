"""Single-pass streaming extraction of the CodeNet tarball.

Reads the archive as text only; never executes anything from it.
Pass 1 (`--listing`): record the directory layout of the first N members so we
verify real paths before hard-coding patterns.
Pass 2 (default): extract metadata CSVs and problem descriptions, and source
files only for the languages/problems requested.

Usage:
    python -I -m tleye.data.stream_extract --tar data/raw/Project_CodeNet.tar.gz --listing 200
    python -I -m tleye.data.stream_extract --tar ... --out data/raw/extracted --meta-only
"""
from __future__ import annotations

import argparse
import tarfile
from pathlib import Path

LANG_DIRS = {"C++", "Java", "Python"}


def _safe_target(out: Path, name: str) -> Path | None:
    """Reject absolute paths / traversal in tar member names."""
    target = (out / name).resolve()
    return target if str(target).startswith(str(out.resolve())) else None


def listing(tar_path: Path, n: int) -> None:
    with tarfile.open(tar_path, "r|gz") as tf:
        for i, m in enumerate(tf):
            print(m.name, m.size)
            if i + 1 >= n:
                break


def extract(tar_path: Path, out: Path, meta_only: bool, sel_ids: set[str] | None) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    counts = {"metadata": 0, "descriptions": 0, "sources": 0, "bytes": 0}
    with tarfile.open(tar_path, "r|gz") as tf:
        for m in tf:
            if not m.isfile():
                continue
            parts = Path(m.name).parts
            is_meta = "metadata" in parts and m.name.endswith(".csv")
            is_desc = "problem_descriptions" in parts and m.name.endswith(".html")
            # layout verified: Project_CodeNet/data/<problem>/<LangDir>/<submission_id>.<ext>
            is_src = (
                not meta_only
                and sel_ids is not None
                and len(parts) == 5
                and parts[1] == "data"
                and parts[-2] in LANG_DIRS
                and parts[-1].rsplit(".", 1)[0] in sel_ids
            )
            if not (is_meta or is_desc or is_src):
                continue
            target = _safe_target(out, m.name)
            if target is None:
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            f = tf.extractfile(m)
            if f is None:
                continue
            target.write_bytes(f.read())
            counts["bytes"] += m.size
            counts["metadata" if is_meta else "descriptions" if is_desc else "sources"] += 1
    return counts


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tar", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("data/raw/extracted"))
    ap.add_argument("--listing", type=int, default=0)
    ap.add_argument("--meta-only", action="store_true")
    ap.add_argument("--selected", type=Path, help="parquet with a submission_id column (see tleye.data.select)")
    a = ap.parse_args()
    if a.listing:
        listing(a.tar, a.listing)
    else:
        ids = None
        if a.selected:
            import pandas as pd
            ids = set(pd.read_parquet(a.selected, columns=["submission_id"])["submission_id"])
        print(extract(a.tar, a.out, a.meta_only, ids))
