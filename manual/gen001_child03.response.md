Uniform rings force uniform radii. A mixed-size layout does better: a coarse
grid of large circles with smaller ones dropped into the gaps between them. Seed
that pattern, then hill-climb the centres one at a time, re-solving the radii LP
after each move and keeping only the moves that help.

```python
"""Mixed-size interstitial layout, refined by per-circle hill climbing."""
import time

import numpy as np
from scipy.optimize import linprog


def _wall_limits(centers):
    return np.minimum.reduce([
        centers[:, 0], centers[:, 1], 1.0 - centers[:, 0], 1.0 - centers[:, 1]
    ])


def _pair_distances(centers):
    diff = centers[:, None, :] - centers[None, :, :]
    return np.sqrt((diff ** 2).sum(-1))


def _max_radii(centers):
    """Largest feasible radii for fixed centres, via a linear program."""
    n = centers.shape[0]
    dist = _pair_distances(centers)
    iu, ju = np.triu_indices(n, k=1)

    rows = np.arange(len(iu))
    a_ub = np.zeros((len(iu), n))
    a_ub[rows, iu] = 1.0
    a_ub[rows, ju] = 1.0

    walls = np.maximum(_wall_limits(centers), 0.0)
    result = linprog(-np.ones(n), A_ub=a_ub, b_ub=dist[iu, ju],
                     bounds=list(zip(np.zeros(n), walls)), method="highs")
    if not result.success:
        return np.zeros(n)
    return np.maximum(result.x, 0.0)


def _repair(centers, radii):
    """Scale down until the packing satisfies every constraint exactly."""
    radii = np.clip(radii, 0.0, np.maximum(_wall_limits(centers), 0.0))
    dist = _pair_distances(centers)
    np.fill_diagonal(dist, np.inf)
    worst = ((radii[:, None] + radii[None, :]) / dist).max()
    if worst > 1.0:
        radii = radii / worst
    return radii * (1.0 - 1e-12)


def _interstitial_layout(n, coarse):
    """A `coarse` x `coarse` grid plus a shifted grid in the gaps between it."""
    large = [((i + 0.5) / coarse, (j + 0.5) / coarse)
             for i in range(coarse) for j in range(coarse)]
    small = [((i + 1.0) / coarse, (j + 1.0) / coarse)
             for i in range(coarse - 1) for j in range(coarse - 1)]

    centers = large + small
    # Top up along the edges if the two grids do not reach n.
    edge = 0
    while len(centers) < n:
        t = (edge + 0.5) / max(n - len(large) - len(small), 1)
        centers.append((min(max(t, 0.02), 0.98), 0.02 if edge % 2 else 0.98))
        edge += 1

    return np.clip(np.array(centers[:n], dtype=float), 0.01, 0.99)


def _hill_climb(centers, deadline, rng, scale=0.08):
    """Move one circle at a time, keeping only moves that raise the sum."""
    n = centers.shape[0]
    centers = centers.copy()
    best = float(_max_radii(centers).sum())

    while time.monotonic() < deadline:
        index = rng.integers(n)
        saved = centers[index].copy()

        proposal = saved + rng.normal(0.0, scale, size=2)
        centers[index] = np.clip(proposal, 0.005, 0.995)

        total = float(_max_radii(centers).sum())
        if total > best:
            best = total
        else:
            centers[index] = saved
            # Shrink the step as improvements get harder to find.
            scale = max(scale * 0.995, 0.004)

    return centers


def construct_packing(n, random_seed: int):
    """Try a few interstitial layouts, refine each, and keep the best."""
    rng = np.random.default_rng(random_seed)
    budget = 18.0

    candidates = [_interstitial_layout(n, coarse) for coarse in (4, 5)]
    candidates.append(rng.uniform(0.05, 0.95, size=(n, 2)))

    slice_seconds = budget / len(candidates)
    best_centers = None
    best_radii = None
    best_sum = -1.0

    for start in candidates:
        refined = _hill_climb(start, time.monotonic() + slice_seconds, rng)
        radii = _repair(refined, _max_radii(refined))
        total = float(radii.sum())
        if total > best_sum:
            best_sum, best_centers, best_radii = total, refined, radii

    return best_centers, best_radii, best_sum
```
