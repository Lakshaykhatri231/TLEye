"""Download the verdict model's LoRA adapter from Hugging Face at a PINNED revision and check every file's SHA-256.
    python -I -m tleye.models.fetch [--dest models/lora]        (or: make weights)

The revision and checksums are the ones the reported results came from (reports/kaggle_confirm/cfgN, seed 0).
A file that is already present with the right checksum is not downloaded again; a wrong one is replaced, never trusted.
"""
from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
from pathlib import Path

REPO = "LakshayKhatri231/tleye-lora-0.5b"
REVISION = "cd628e9712f4ba9c1db8e7b69096c951e38862b5"
FILES = {
    "adapter_model.safetensors": "5e51fa121e0151384935f6cbdba82eea54bf7e75225c5d27bd7f89ac52535976",
    "adapter_config.json": "07099e0b2e167c3f73799179238556da4060a9fa438bec1ec27ffe31f18afc89",
    "meta.json": "8830bc49325daac28ae6559b89a2b92255f3d09a55c8b2c38914845da9efd20e",
}


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def ok(p: Path, want: str) -> bool:
    return p.exists() and sha256(p) == want


def fetch(dest: Path = Path("models/lora")) -> list[str]:
    """Returns the files that were downloaded. Raises if a downloaded file does not match its pinned checksum."""
    from huggingface_hub import hf_hub_download
    dest.mkdir(parents=True, exist_ok=True)
    got = []
    for name, want in FILES.items():
        target = dest / name
        if ok(target, want):
            continue
        src = Path(hf_hub_download(REPO, name, revision=REVISION))
        if sha256(src) != want:
            raise RuntimeError(f"{name} from {REPO}@{REVISION[:12]} does not match its pinned checksum")
        shutil.copyfile(src, target); got.append(name)
    return got


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--dest", type=Path, default=Path("models/lora")); a = ap.parse_args()
    got = fetch(a.dest)
    print(f"{REPO}@{REVISION[:12]}: " + (f"downloaded {', '.join(got)}" if got else "all files already present") + "; checksums verified")
    return 0


if __name__ == "__main__":
    sys.exit(main())
