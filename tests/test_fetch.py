from pathlib import Path

import pytest

from tleye.models import fetch


def test_present_files_with_the_pinned_checksum_are_accepted_and_others_are_not(tmp_path):
    p = tmp_path / "f"
    p.write_bytes(b"abc")
    good = fetch.sha256(p)
    assert fetch.ok(p, good) and not fetch.ok(p, "0" * 64) and not fetch.ok(tmp_path / "missing", good)


def test_pins_are_a_full_revision_and_full_checksums():
    assert len(fetch.REVISION) == 40 and all(len(v) == 64 for v in fetch.FILES.values())


@pytest.mark.skipif(not Path("models/lora/adapter_model.safetensors").exists(), reason="local adapter not present")
def test_the_local_adapter_is_the_pinned_one():
    assert all(fetch.ok(Path("models/lora") / n, h) for n, h in fetch.FILES.items())
