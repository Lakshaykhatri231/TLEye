"""Run ONE submission against an input battery inside the sandbox and decide whether it is CERTIFIED TLE.

Certified = on at least one input the program either hit the kill cap (timeout) or finished with CPU time >= the
problem's limit, measured on OUR hardware (arm64 container), NOT on the contest judge. A crash (RE) on generated
inputs is inconclusive (our input may be invalid), never counted as TLE. Compile once, run all inputs in one container.
"""
from __future__ import annotations

import re
import shutil
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from tleye.certify.sandbox import run_in_sandbox

IMAGES = {"C++": "gcc:13", "Java": "eclipse-temurin:21-jdk", "Python": "tleye-python:3.11", "PyPy": "pypy:3.10-slim"}
PREP = {
    "C++": ("cp /src/sol /w/main.cpp && g++ -O2 -std=gnu++17 -o /w/a /w/main.cpp 2>/w/cerr", "/w/a"),
    "Java": ("cp /src/sol /w/Main.java && javac -J-Xmx512m -d /w /w/Main.java 2>/w/cerr",
             "java -Xss256m -Xmx768m -cp /w Main"),
    "Python": ("cp /src/sol /w/main.py && python3 -m py_compile /w/main.py", "python3 /w/main.py"),   # py_compile: Python 2 code -> compile error
    "PyPy": ("cp /src/sol /w/main.py && pypy3 -m py_compile /w/main.py", "pypy3 /w/main.py"),
}
SCRATCH = Path("data/scratch/certify")      # inside $HOME so Colima can bind-mount it (read-only into the container)
RUN_RE = re.compile(r"RUN (\S+) rc=(\d+) t=([\d.]*)")
ERR_RE = re.compile(r"^ERR (\S+) (.*)$", re.M)


@dataclass
class Cert:
    status: str                       # certified | not_certified | inconclusive | compile_error | missing_dependency | infra_error
    runs: list[dict] = field(default_factory=list)
    max_cpu_s: float = 0.0
    wall_s: float = 0.0


def _script(runtime: str, cap: float, limit_s: float) -> str:
    prep, cmd = PREP[runtime]
    return f"""
ulimit -s unlimited 2>/dev/null || ulimit -s 1048576 2>/dev/null
cd /w
[ -f /src/sol ] || {{ echo MOUNT_ERROR; exit 0; }}
if ! ( {prep} ); then echo COMPILE_ERROR; exit 0; fi
for f in /in/*.txt; do
  ( TIMEFORMAT='%U %S'; time timeout -k 1 {cap} {cmd} < "$f" > /dev/null 2> /w/err ) 2> /w/t
  rc=$?
  t=$(awk '{{print $1+$2}}' /w/t | tail -1)
  echo "RUN $(basename "$f") rc=$rc t=$t"
  if [ "$rc" != 0 ] && [ "$rc" != 124 ]; then echo "ERR $(basename "$f") $(tail -c 240 /w/err | tr '\n\r' '  ')"; fi
  if [ "$rc" = 124 ]; then break; fi
  if awk -v t="$t" -v l={limit_s} 'BEGIN{{exit !(t>=l)}}'; then break; fi
done
"""


def certify_submission(src: str | Path, runtime: str, inputs: dict[str, str], limit_ms: float,
                       cap_factor: float = 2.0) -> Cert:
    limit_s = limit_ms / 1000.0
    cap = round(min(max(cap_factor * limit_s, limit_s + 2.0), 15.0), 2)
    d = (SCRATCH / uuid.uuid4().hex[:12]).resolve()
    (d / "in").mkdir(parents=True)
    try:
        for name, txt in inputs.items():
            (d / "in" / f"{name}.txt").write_text(txt)
        r = run_in_sandbox(
            IMAGES[runtime], ["bash", "-c", _script(runtime, cap, limit_s)],
            timeout_s=90 + len(inputs) * (cap + 5), memory="1g", pids=256,
            mounts=[(str(Path(src).resolve()), "/src/sol"), (str(d / "in"), "/in")],
            tmpfs=("/w:rw,size=256m,exec", "/tmp:rw,size=64m,exec"),
        )
    finally:
        shutil.rmtree(d, ignore_errors=True)
    if r.timed_out or r.exit_code not in (0,):
        return Cert("infra_error", wall_s=r.wall_s)
    if "MOUNT_ERROR" in r.stdout:
        return Cert("infra_error", wall_s=r.wall_s)       # source not visible in the VM (outside $HOME?) - never a verdict
    if "COMPILE_ERROR" in r.stdout:
        return Cert("compile_error", wall_s=r.wall_s)
    errs = {m[0]: m[1].strip()[:240] for m in ERR_RE.findall(r.stdout)}            # stderr tail of crashed runs (text only)
    runs = [{"input": m[0], "rc": int(m[1]), "cpu_s": float(m[2] or 0), **({"err": errs[m[0]]} if m[0] in errs else {})} for m in RUN_RE.findall(r.stdout)]
    if not runs:
        return Cert("infra_error", wall_s=r.wall_s)
    # Judges measure CPU time: require cpu >= limit. A wall-clock kill with less CPU than the limit means the
    # process was starved/blocked (not provably slow) -> inconclusive, never a certificate.
    certified = any(x["rc"] in (0, 124) and x["cpu_s"] >= limit_s for x in runs)
    crashed = any(x["rc"] not in (0, 124) or (x["rc"] == 124 and x["cpu_s"] < limit_s) for x in runs)
    dep = any("No module named" in x.get("err", "") or "ModuleNotFoundError" in x.get("err", "") for x in runs)
    status = "certified" if certified else ("missing_dependency" if dep else ("inconclusive" if crashed else "not_certified"))
    return Cert(status, runs, max(x["cpu_s"] for x in runs), r.wall_s)
