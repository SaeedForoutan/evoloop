The ring layout wastes the corners of the square. Rather than guessing a
pattern, let the centres find their own spacing: repel every pair, confine them
to the square, and iterate to a relaxed configuration. Then solve the radii
exactly for whatever centres the relaxation settled on.

```python
"""Repulsion-relaxed centres, with radii solved as a linear program."""
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
    """Maximise sum(r) for fixed centres — linear in r, so use an LP."""
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
    return _repair(centers, np.maximum(result.x, 0.0))


def _repair(centers, radii):
    """Scale the radii down until every constraint holds exactly."""
    radii = np.clip(radii, 0.0, np.maximum(_wall_limits(centers), 0.0))
    dist = _pair_distances(centers)
    np.fill_diagonal(dist, np.inf)
    worst = ((radii[:, None] + radii[None, :]) / dist).max()
    if worst > 1.0:
        radii = radii / worst
    return radii * (1.0 - 1e-12)


def _relax(centers, steps=400, step_size=0.01):
    """Push centres apart, keeping them inside the square.

    Each pair repels with a force falling off as 1/d^2, and walls repel too so
    the configuration spreads into the corners instead of bunching centrally.
    """
    n = centers.shape[0]
    centers = centers.copy()

    for step in range(steps):
        diff = centers[:, None, :] - centers[None, :, :]
        dist = np.sqrt((diff ** 2).sum(-1))
        np.fill_diagonal(dist, np.inf)

        # Pairwise repulsion, normalised so no single close pair dominates.
        direction = diff / dist[:, :, None]
        magnitude = 1.0 / np.maximum(dist, 1e-6) ** 2
        force = (direction * magnitude[:, :, None]).sum(axis=1)

        # Wall repulsion, strongest as a centre approaches an edge.
        wall = np.zeros_like(centers)
        wall[:, 0] += 1.0 / np.maximum(centers[:, 0], 1e-6) ** 2
        wall[:, 0] -= 1.0 / np.maximum(1.0 - centers[:, 0], 1e-6) ** 2
        wall[:, 1] += 1.0 / np.maximum(centers[:, 1], 1e-6) ** 2
        wall[:, 1] -= 1.0 / np.maximum(1.0 - centers[:, 1], 1e-6) ** 2

        total = force + 0.35 * wall
        norm = np.linalg.norm(total, axis=1, keepdims=True)
        total = total / np.maximum(norm, 1e-12)

        # Cooling schedule: large moves early, fine adjustment late.
        rate = step_size * (1.0 - step / steps)
        centers = np.clip(centers + rate * total, 1e-4, 1.0 - 1e-4)

    return centers


def construct_packing(n, random_seed: int):
    """Relax several random starts and keep the best packing found."""
    rng = np.random.default_rng(random_seed)

    best_centers = None
    best_radii = None
    best_sum = -1.0

    for _ in range(6):
        start = rng.uniform(0.05, 0.95, size=(n, 2))
        centers = _relax(start)
        radii = _max_radii(centers)
        total = float(radii.sum())
        if total > best_sum:
            best_sum, best_centers, best_radii = total, centers, radii

    return best_centers, best_radii, best_sum
```
