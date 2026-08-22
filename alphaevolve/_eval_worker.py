"""Subprocess worker: run one candidate program and report a diagnosis.

Runs untrusted, model-generated code, so it is launched as a separate process
with CPU and address-space limits and is expected to be killed by a wall-clock
timeout from the parent. Emits a single JSON object on stdout.
"""

from __future__ import annotations

import json
import resource
import sys
import traceback

# Backstops only — a well-behaved candidate should finish inside the budget the
# prompt advertises. RLIMIT_CPU sums CPU across threads, so the parent pins BLAS
# to a single thread to keep this comparable to wall clock.
CPU_SECONDS = 50
ADDRESS_SPACE_BYTES = 4 * 1024**3


def _apply_limits() -> None:
    resource.setrlimit(resource.RLIMIT_CPU, (CPU_SECONDS, CPU_SECONDS))
    resource.setrlimit(resource.RLIMIT_AS, (ADDRESS_SPACE_BYTES, ADDRESS_SPACE_BYTES))
    # No core dumps from a crashing candidate.
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))


def _emit(payload: dict) -> None:
    sys.stdout.write("<<<RESULT>>>" + json.dumps(payload))
    sys.stdout.flush()


def _diagnose(centers, radii, n, np) -> tuple[bool, str]:
    """Check the packing constraints and describe the worst violation.

    The insight text is fed back into the next prompt, so it names the specific
    circles and the magnitude of the violation rather than just failing.
    """
    if getattr(centers, "shape", None) != (n, 2):
        return False, f"centers must have shape ({n}, 2), got {getattr(centers, 'shape', type(centers))}."
    if getattr(radii, "shape", None) != (n,):
        return False, f"radii must have shape ({n},), got {getattr(radii, 'shape', type(radii))}."
    if not np.isfinite(centers).all():
        return False, "centers contains NaN or infinite values."
    if not np.isfinite(radii).all():
        return False, "radii contains NaN or infinite values."
    if (radii < 0).any():
        worst = int(np.argmin(radii))
        return False, f"radii must be non-negative; circle {worst} has radius {radii[worst]:.6g}."

    # Containment: every circle must lie inside the unit square.
    slack = np.minimum.reduce([
        centers[:, 0] - radii,
        centers[:, 1] - radii,
        1.0 - centers[:, 0] - radii,
        1.0 - centers[:, 1] - radii,
    ])
    if slack.min() < -1e-9:
        i = int(np.argmin(slack))
        return False, (
            f"circle {i} at ({centers[i, 0]:.4f}, {centers[i, 1]:.4f}) with radius "
            f"{radii[i]:.4f} sticks out of the unit square by {-slack.min():.3e}."
        )

    # Overlap: for every pair, r_i + r_j must not exceed the centre distance.
    diff = centers[:, None, :] - centers[None, :, :]
    dist = np.sqrt((diff**2).sum(-1))
    overlap = radii[:, None] + radii[None, :] - dist
    np.fill_diagonal(overlap, -np.inf)
    worst = float(overlap.max())
    if worst > 1e-9:
        i, j = np.unravel_index(int(np.argmax(overlap)), overlap.shape)
        return False, (
            f"circles {int(i)} and {int(j)} overlap by {worst:.3e} "
            f"(radii {radii[i]:.4f} + {radii[j]:.4f} exceed centre distance {dist[i, j]:.4f}). "
            "Shrink radii to the largest non-overlapping values before returning."
        )

    return True, f"valid packing; tightest pair gap {-worst:.3e}, tightest wall gap {slack.min():.3e}."


def main() -> int:
    _apply_limits()
    program_path, n = sys.argv[1], int(sys.argv[2])

    with open(program_path) as fh:
        source = fh.read()

    import numpy as np

    namespace: dict = {"__name__": "candidate"}
    try:
        exec(compile(source, "candidate.py", "exec"), namespace)
    except BaseException:
        _emit({
            "valid": False,
            "score": None,
            "insight": "the program failed to import: "
                       + traceback.format_exc(limit=2).strip().splitlines()[-1],
        })
        return 0

    construct = namespace.get("construct_packing")
    if not callable(construct):
        _emit({"valid": False, "score": None,
               "insight": "the program must define construct_packing(n, random_seed)."})
        return 0

    try:
        result = construct(n, random_seed=42)
    except BaseException:
        _emit({
            "valid": False,
            "score": None,
            "insight": "construct_packing raised: "
                       + traceback.format_exc(limit=2).strip().splitlines()[-1],
        })
        return 0

    try:
        centers, radii = np.asarray(result[0], dtype=float), np.asarray(result[1], dtype=float)
    except BaseException:
        _emit({"valid": False, "score": None,
               "insight": "construct_packing must return (centers, radii, sum_of_radii)."})
        return 0

    ok, insight = _diagnose(centers, radii, n, np)
    if not ok:
        _emit({"valid": False, "score": None, "insight": insight})
        return 0

    _emit({
        "valid": True,
        "score": float(radii.sum()),
        "insight": insight,
        "centers": centers.tolist(),
        "radii": radii.tolist(),
    })
    return 0


if __name__ == "__main__":
    sys.exit(main())
