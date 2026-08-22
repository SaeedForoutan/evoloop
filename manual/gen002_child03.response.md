Hill climbing one circle at a time is a slow way to explore. The interstitial
layout idea is sound but it should be enumerated rather than climbed: generate
many structured starts — different coarse grids, different row decompositions,
mirrored variants — and give each a joint solve over centres and radii.

```python
"""Enumerated structured layouts, each finished with a joint solve."""
import time

import numpy as np
from scipy.optimize import linprog, minimize


def _wall_limits(centers):
    return np.minimum.reduce([
        centers[:, 0], centers[:, 1], 1.0 - centers[:, 0], 1.0 - centers[:, 1]
    ])


def _pair_distances(centers):
    diff = centers[:, None, :] - centers[None, :, :]
    return np.sqrt((diff ** 2).sum(-1))


def _max_radii(centers):
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
    return np.maximum(result.x, 0.0) if result.success else np.zeros(n)


def _repair(centers, radii):
    radii = np.clip(radii, 0.0, np.maximum(_wall_limits(centers), 0.0))
    dist = _pair_distances(centers)
    np.fill_diagonal(dist, np.inf)
    worst = ((radii[:, None] + radii[None, :]) / dist).max()
    if worst > 1.0:
        radii = radii / worst
    return radii * (1.0 - 1e-12)


def _solve(centers, n, maxiter=200):
    iu, ju = np.triu_indices(n, k=1)
    m = len(iu)
    grad = np.concatenate([np.zeros(2 * n), -np.ones(n)])

    def cons(z):
        x, y, r = z[:n], z[n:2 * n], z[2 * n:]
        dist = np.sqrt((x[iu] - x[ju]) ** 2 + (y[iu] - y[ju]) ** 2)
        return np.concatenate([dist - r[iu] - r[ju], x - r, y - r,
                               1.0 - x - r, 1.0 - y - r])

    def cons_jac(z):
        x, y = z[:n], z[n:2 * n]
        dx, dy = x[iu] - x[ju], y[iu] - y[ju]
        dist = np.maximum(np.sqrt(dx ** 2 + dy ** 2), 1e-12)
        jac = np.zeros((m + 4 * n, 3 * n))
        rows = np.arange(m)
        jac[rows, iu], jac[rows, ju] = dx / dist, -dx / dist
        jac[rows, n + iu], jac[rows, n + ju] = dy / dist, -dy / dist
        jac[rows, 2 * n + iu] = jac[rows, 2 * n + ju] = -1.0
        idx = np.arange(n)
        for k, (offset, sign) in enumerate([(0, 1.0), (n, 1.0), (0, -1.0), (n, -1.0)]):
            block = m + k * n + idx
            jac[block, offset + idx] = sign
            jac[block, 2 * n + idx] = -1.0
        return jac

    z0 = np.concatenate([centers[:, 0], centers[:, 1], _max_radii(centers)])
    result = minimize(lambda z: -z[2 * n:].sum(), z0, jac=lambda _: grad,
                      bounds=[(0.0, 1.0)] * (2 * n) + [(0.0, 0.5)] * n,
                      method="SLSQP",
                      constraints=[{"type": "ineq", "fun": cons, "jac": cons_jac}],
                      options={"maxiter": maxiter, "ftol": 1e-11})
    z = result.x
    out = np.clip(np.column_stack([z[:n], z[n:2 * n]]), 0.0, 1.0)
    return out, _repair(out, z[2 * n:])


def _row_layout(n, rows, stagger):
    centers = []
    for row_index, count in enumerate(rows):
        y = (row_index + 0.5) / len(rows)
        shift = stagger / max(count, 1) if row_index % 2 else 0.0
        for col in range(count):
            centers.append(((col + 0.5) / count + shift, y))
    return np.clip(np.array(centers[:n], dtype=float), 0.01, 0.99)


def _interstitial(n, coarse):
    large = [((i + 0.5) / coarse, (j + 0.5) / coarse)
             for i in range(coarse) for j in range(coarse)]
    small = [((i + 1.0) / coarse, (j + 1.0) / coarse)
             for i in range(coarse - 1) for j in range(coarse - 1)]
    centers = (large + small)[:n]
    while len(centers) < n:
        centers.append((0.5, 0.5))
    return np.clip(np.array(centers, dtype=float), 0.01, 0.99)


def _layouts(n):
    """Every structured start worth a solve, cheapest-looking first."""
    out = []
    for row_count in range(4, 8):
        base, extra = divmod(n, row_count)
        if base < 2:
            continue
        rows = [base + (1 if i < extra else 0) for i in range(row_count)]
        for stagger in (0.0, 0.25, 0.5):
            out.append(_row_layout(n, rows, stagger))
            if extra:
                out.append(_row_layout(n, rows[::-1], stagger))
    for coarse in (4, 5, 6):
        out.append(_interstitial(n, coarse))
    return [c for c in out if c.shape[0] == n]


def construct_packing(n, random_seed: int):
    """Solve from each structured layout within the time budget."""
    rng = np.random.default_rng(random_seed)
    deadline = time.monotonic() + 19.0

    best_centers = best_radii = None
    best_sum = -1.0

    for start in _layouts(n):
        if time.monotonic() > deadline:
            break
        try:
            centers, radii = _solve(start, n)
        except (ValueError, np.linalg.LinAlgError):
            continue
        total = float(radii.sum())
        if total > best_sum:
            best_sum, best_centers, best_radii = total, centers, radii

    # Spend anything left over on random restarts.
    while time.monotonic() < deadline:
        try:
            centers, radii = _solve(rng.uniform(0.05, 0.95, size=(n, 2)), n, maxiter=120)
        except (ValueError, np.linalg.LinAlgError):
            continue
        total = float(radii.sum())
        if total > best_sum:
            best_sum, best_centers, best_radii = total, centers, radii

    if best_centers is None:
        best_centers = _interstitial(n, 5)
        best_radii = _repair(best_centers, _max_radii(best_centers))
        best_sum = float(best_radii.sum())

    return best_centers, best_radii, best_sum
```
