"""Data version manifest: SHA-256, size and row count of the CodeNet archive and every processed data file the results were computed from.
    python -I -m tleye.data.manifest            -> ci/data_manifest.json (commit it with the results it belongs to)
    python -I -m tleye.data.manifest --check    -> exit 1 and list every file that differs from the committed manifest

The data is too large for git (7.8 GB archive, ~200 MB processed). It is rebuilt by `make data` from one fixed archive; this manifest pins
exactly which bytes the reported numbers came from. A rebuild with other library versions can change parquet bytes without changing
content, so row counts are recorded too, and a mismatch is reported, never silently accepted.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ARCHIVE = Path("data/raw/Project_CodeNet.tar.gz")
SOURCE_URL = "https://codait-cos-dax.s3.us.cloud-object-storage.appdomain.cloud/dax-project-codenet/1.0.0/Project_CodeNet.tar.gz"
PROCESSED = Path("data/processed")
OUT = Path("ci/data_manifest.json")
SKIP = {"kaggle_adapters"}          # copies of model adapters, not data


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def rows(p: Path) -> int | None:
    if p.suffix == ".parquet":
        import pyarrow.parquet as pq
        return pq.ParquetFile(p).metadata.num_rows
    if p.suffix == ".jsonl":
        with open(p, "rb") as f:
            return sum(1 for _ in f)
    return None


def files() -> list[Path]:
    processed = [p for p in sorted(PROCESSED.rglob("*")) if p.is_file() and not set(p.relative_to(PROCESSED).parts) & SKIP]
    return ([ARCHIVE] if ARCHIVE.exists() else []) + processed


def build() -> dict:
    entries = {}
    for p in files():
        entries[str(p)] = {"sha256": sha256(p), "bytes": p.stat().st_size, **({"rows": r} if (r := rows(p)) is not None else {})}
    return {"source": SOURCE_URL, "files": entries}


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--check", action="store_true"); a = ap.parse_args()
    now = build()
    if not a.check:
        OUT.write_text(json.dumps(now, indent=1) + "\n"); print(f"wrote {OUT}: {len(now['files'])} files"); return 0
    ref = json.loads(OUT.read_text())["files"]
    bad = [f"missing  {k}" for k in ref if k not in now["files"]]
    bad += [f"changed  {k}" + (f" (rows {ref[k].get('rows')} -> {v.get('rows')})" if ref[k].get("rows") != v.get("rows") else " (same rows)")
            for k, v in now["files"].items() if k in ref and v["sha256"] != ref[k]["sha256"]]
    bad += [f"new      {k}" for k in now["files"] if k not in ref]
    print("\n".join(bad) if bad else f"data matches {OUT} ({len(ref)} files)")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
