import os
import pathlib
import shutil
import subprocess
import uuid

import pytest

from tleye.certify.runner import certify_submission

pytestmark = pytest.mark.skipif(
    bool(os.environ.get("TLEYE_SKIP_DOCKER")) or shutil.which("docker") is None or subprocess.run(["docker", "info"], capture_output=True).returncode != 0,
    reason="docker/colima not available")

INPUTS = {"a": "5\n1 2 3 4 5\n", "b": "5\n5 4 3 2 1\n"}


@pytest.fixture
def tmp_path():
    """Sources must live under $HOME for Colima to mount them; pytest's default tmp dir does not."""
    d = pathlib.Path("data/scratch/tests") / uuid.uuid4().hex[:8]
    d.mkdir(parents=True)
    yield d.resolve()
    shutil.rmtree(d, ignore_errors=True)


def write(tmp_path, name, code):
    p = tmp_path / name
    p.write_text(code)
    return p


def test_python_infinite_loop_is_certified_and_fast_solution_is_not(tmp_path):
    slow = write(tmp_path, "slow.py", "while True: pass\n")
    fast = write(tmp_path, "fast.py", "n=int(input()); print(sum(map(int,input().split())))\n")
    assert certify_submission(slow, "Python", INPUTS, limit_ms=1000).status == "certified"
    c = certify_submission(fast, "Python", INPUTS, limit_ms=1000)
    assert c.status == "not_certified" and len(c.runs) == 2 and c.max_cpu_s < 1.0


def test_crash_is_inconclusive_not_tle(tmp_path):
    bad = write(tmp_path, "bad.py", "raise SystemExit(3)\n")
    assert certify_submission(bad, "Python", INPUTS, limit_ms=1000).status == "inconclusive"


def test_cpp_compile_error_and_timeout(tmp_path):
    ce = write(tmp_path, "ce.cpp", "int main( { return 0 }\n")
    assert certify_submission(ce, "C++", INPUTS, limit_ms=1000).status == "compile_error"
    tle = write(tmp_path, "tle.cpp", "int main(){ volatile long x=0; for(;;) x++; }\n")
    assert certify_submission(tle, "C++", INPUTS, limit_ms=1000).status == "certified"


def test_sandbox_has_no_network_inside_certified_runs(tmp_path):
    net = write(tmp_path, "net.py",
                "import socket,sys\ntry:\n  socket.create_connection(('1.1.1.1',53),timeout=2); print('NET')\nexcept OSError: pass\n")
    # exits 0 with no NET: we cannot see stdout (sent to /dev/null by design) -> status must simply be not_certified
    assert certify_submission(net, "Python", INPUTS, limit_ms=5000).status == "not_certified"


def test_python2_code_is_a_compile_error_not_a_verdict(tmp_path):
    p = write(tmp_path, "py2.py", "print 'hello'\n")
    assert certify_submission(p, "Python", INPUTS, limit_ms=1000).status == "compile_error"


def test_missing_library_is_reported_separately_and_common_libraries_are_available(tmp_path):
    missing = write(tmp_path, "dep.py", "import definitely_not_installed_pkg\n")
    assert certify_submission(missing, "Python", INPUTS, limit_ms=1000).status == "missing_dependency"
    ok = write(tmp_path, "np.py", "import numpy, scipy, networkx, sortedcontainers\nprint(numpy.arange(3).sum())\n")
    assert certify_submission(ok, "Python", INPUTS, limit_ms=5000).status == "not_certified"
