"""Scores a candidate program by running it in an isolated subprocess."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field

_WORKER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_eval_worker.py")

# The API requires a number for every metric and sum_of_radii is maximised, so a
# large negative sentinel keeps failed candidates from ever being selected.
FAILURE_SCORE = -1e12


@dataclass
class EvalResult:
    valid: bool
    score: float
    insight: str
    centers: list = field(default_factory=list)
    radii: list = field(default_factory=list)


# numpy/scipy link a multithreaded BLAS by default, and RLIMIT_CPU in the worker
# sums CPU across every thread. Pinning to one thread keeps the CPU backstop
# roughly equal to wall clock, so a candidate that budgets its own wall time is
# not killed for using several cores.
_SINGLE_THREAD_ENV = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
}


def evaluate_program(source: str, n: int = 26, timeout: float = 60.0) -> EvalResult:
    """Run `source` and score its packing.

    Returns a sentinel score plus an insight message on any failure; the insight
    is what steers the next generation, so failures still carry information.
    """
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as fh:
        fh.write(source)
        path = fh.name

    try:
        proc = subprocess.run(
            [sys.executable, _WORKER, path, str(n)],
            capture_output=True,
            text=True,
            timeout=timeout,
            env={**os.environ, **_SINGLE_THREAD_ENV},
        )
    except subprocess.TimeoutExpired:
        return EvalResult(False, FAILURE_SCORE,
                          f"the program did not finish within {timeout:.0f}s — it must be "
                          "fast enough to evaluate; bound any iterative refinement.")
    finally:
        os.unlink(path)

    marker = "<<<RESULT>>>"
    if marker not in proc.stdout:
        # A negative return code is a signal. SIGKILL here means the worker's
        # RLIMIT_CPU or RLIMIT_AS backstop fired, which the model can act on;
        # a bare "exit code -9" tells it nothing.
        if proc.returncode < 0:
            reason = ("the process was killed for exceeding the resource limits "
                      "(50s CPU or 4GB memory) — reduce the time budget your "
                      "search loop allows itself, and avoid allocating large "
                      "dense arrays")
        else:
            detail = [line for line in (proc.stderr or "").strip().splitlines()
                      if "Warning" not in line]
            reason = detail[-1] if detail else f"exit code {proc.returncode}"
            reason = f"the program crashed without producing a result: {reason}"
        return EvalResult(False, FAILURE_SCORE, reason)

    payload = json.loads(proc.stdout.split(marker, 1)[1])
    if not payload["valid"]:
        return EvalResult(False, FAILURE_SCORE, payload["insight"])

    return EvalResult(
        valid=True,
        score=payload["score"],
        insight=payload["insight"],
        centers=payload.get("centers", []),
        radii=payload.get("radii", []),
    )
