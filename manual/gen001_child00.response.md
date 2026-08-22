The seed spends all its freedom on ring geometry and then computes radii with a
greedy pairwise scale-down, which throws away most of the achievable sum. Two
changes: pick centres from structured grid layouts instead of rings, and solve
for the radii exactly — for fixed centres, maximising the sum of radii is a
linear program.

```python
"""Structured grid centres with radii solved exactly as a linear program."""
import numpy as np
from scipy.optimize import linprog


def _wall_limits(centers):
    """Largest radius each centre allows before leaving the unit square."""
    return np.minimum.reduce([
        centers[:, 0], centers[:, 1], 1.0 - centers[:, 0], 1.0 - centers[:, 1]
    ])


def _pair_distances(centers):
    diff = centers[:, None, :] - centers[None, :, :]
    return np.sqrt((diff ** 2).sum(-1))


def _max_radii(centers):
    """For fixed centres, maximise sum(r) subject to r_i + r_j <= d_ij.

    This is a linear program: the objective and every constraint are linear in r.
    """
    n = centers.shape[0]
    dist = _pair_distances(centers)
    iu, ju = np.triu_indices(n, k=1)

    rows = np.arange(len(iu))
    a_ub = np.zeros((len(iu), n))
    a_ub[rows, iu] = 1.0
    a_ub[rows, ju] = 1.0
    b_ub = dist[iu, ju]

    walls = np.maximum(_wall_limits(centers), 0.0)
    result = linprog(-np.ones(n), A_ub=a_ub, b_ub=b_ub,
                     bounds=list(zip(np.zeros(n), walls)), method="highs")
    if not result.success:
        return np.zeros(n)
    return _repair(centers, np.maximum(result.x, 0.0))


def _repair(centers, radii):
    """Make the radii exactly feasible.

    The LP is solved to a tolerance, so it can leave overlaps of ~1e-9. Clipping
    to the walls and then scaling every radius by the worst overlap ratio
    restores feasibility without changing the shape of the solution.
    """
    radii = np.clip(radii, 0.0, np.maximum(_wall_limits(centers), 0.0))
    dist = _pair_distances(centers)
    np.fill_diagonal(dist, np.inf)
    ratio = (radii[:, None] + radii[None, :]) / dist
    worst = ratio.max()
    if worst > 1.0:
        radii = radii / worst
    return radii * (1.0 - 1e-12)


def _grid_layout(rows):
    """Centres for a row-structured layout, each row evenly spread."""
    centers = []
    for row_index, count in enumerate(rows):
        y = (row_index + 0.5) / len(rows)
        for col in range(count):
            x = (col + 0.5) / count
            centers.append((x, y))
    return np.array(centers, dtype=float)


def _candidate_layouts(n, rng):
    """Row decompositions of n, plus jittered variants."""
    layouts = []
    for row_count in range(3, 8):
        base, extra = divmod(n, row_count)
        if base == 0:
            continue
        rows = [base + (1 if i < extra else 0) for i in range(row_count)]
        layouts.append(rows)
        # Stagger long/short rows, which is what a hexagonal packing does.
        if extra:
            layouts.append(rows[::-1])
        if row_count > 1 and base > 1:
            alt = [base + 1 if i % 2 == 0 else base - 1 for i in range(row_count)]
            if sum(alt) == n and min(alt) > 0:
                layouts.append(alt)

    centers = [_grid_layout(rows) for rows in layouts]

    # A few jittered copies of the best-shaped layouts add variety cheaply.
    for rows in layouts[:4]:
        base = _grid_layout(rows)
        for _ in range(3):
            jitter = base + rng.normal(0.0, 0.02, base.shape)
            centers.append(np.clip(jitter, 0.001, 0.999))

    return centers


def construct_packing(n, random_seed: int):
    """Try several structured layouts and keep the one with the largest sum."""
    rng = np.random.default_rng(random_seed)

    best_centers = None
    best_radii = None
    best_sum = -1.0

    for centers in _candidate_layouts(n, rng):
        if centers.shape[0] != n:
            continue
        radii = _max_radii(centers)
        total = float(radii.sum())
        if total > best_sum:
            best_sum, best_centers, best_radii = total, centers, radii

    if best_centers is None:
        best_centers = _grid_layout([5, 5, 5, 5, 6])
        best_radii = _max_radii(best_centers)
        best_sum = float(best_radii.sum())

    return best_centers, best_radii, best_sum
```
