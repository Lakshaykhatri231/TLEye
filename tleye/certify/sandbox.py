"""Run untrusted code in a locked-down Docker container.

Never execute dataset code on the host. Each run gets: no network, CPU/memory/pids
caps, read-only root, non-root user, and a wall-clock kill that removes the
CONTAINER by name (killing only the `docker run` client leaves it running).
"""
from __future__ import annotations

import subprocess
import time
import uuid
from dataclasses import dataclass


@dataclass
class RunResult:
    exit_code: int | None
    timed_out: bool
    oom_killed: bool
    wall_s: float
    stdout: str
    stderr: str


def run_in_sandbox(
    image: str,
    cmd: list[str],
    *,
    stdin: str = "",
    timeout_s: float = 10.0,
    memory: str = "512m",
    cpus: float = 1.0,
    pids: int = 64,
    mounts: list[tuple[str, str]] | None = None,  # (host_path, container_path), mounted read-only
    tmpfs: tuple[str, ...] = ("/tmp:rw,size=64m,exec",),  # writable scratch (only these paths are writable)
    max_output: int = 1_000_000,
) -> RunResult:
    name = f"tleye-{uuid.uuid4().hex[:12]}"
    args = [
        "docker", "run", "--rm", "-i", "--name", name,
        "--network", "none", "--cpus", str(cpus),
        "--memory", memory, "--memory-swap", memory,
        "--pids-limit", str(pids), "--read-only",
        "--user", "65534:65534", "--cap-drop", "ALL",
        "--security-opt", "no-new-privileges",
    ]
    for t in tmpfs:
        args += ["--tmpfs", t]
    for host, ctr in mounts or []:
        args += ["-v", f"{host}:{ctr}:ro"]
    args += [image, *cmd]

    t0 = time.monotonic()
    timed_out = False
    try:
        p = subprocess.run(
            args, input=stdin, capture_output=True, text=True, timeout=timeout_s
        )
        code, out, err = p.returncode, p.stdout, p.stderr
    except subprocess.TimeoutExpired as e:
        timed_out = True
        subprocess.run(["docker", "kill", name], capture_output=True)
        subprocess.run(["docker", "rm", "-f", name], capture_output=True)
        code = None
        out = e.stdout.decode() if isinstance(e.stdout, bytes) else (e.stdout or "")
        err = e.stderr.decode() if isinstance(e.stderr, bytes) else (e.stderr or "")
    wall = time.monotonic() - t0
    return RunResult(code, timed_out, code == 137 and not timed_out, wall,
                     out[:max_output], err[:max_output])
