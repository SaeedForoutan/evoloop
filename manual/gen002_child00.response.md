Hill climbing on the centres plateaus because it only ever moves one circle and
never adjusts the radii jointly. Keep it as a cheap explorer, then polish the
configuration it finds with a joint SLSQP solve over centres and radii together.

```python
"""Hill-climbed centres, polished by a joint solve over centres and radii."""
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


def _polish(centers, n, maxiter=200):
    """Joint SLSQP over (x, y, r) with analytic constraint Jacobians."""
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


def _hill_climb(centers, deadline, rng, scale=0.08):
    n = centers.shape[0]
    centers = centers.copy()
    best = float(_max_radii(centers).sum())
    while time.monotonic() < deadline:
        index = rng.integers(n)
        saved = centers[index].copy()
        centers[index] = np.clip(saved + rng.normal(0.0, scale, size=2), 0.005, 0.995)
        total = float(_max_radii(centers).sum())
        if total > best:
            best = total
        else:
            centers[index] = saved
            scale = max(scale * 0.995, 0.004)
    return centers


def _interstitial_layout(n, coarse):
    large = [((i + 0.5) / coarse, (j + 0.5) / coarse)
             for i in range(coarse) for j in range(coarse)]
    small = [((i + 1.0) / coarse, (j + 1.0) / coarse)
             for i in range(coarse - 1) for j in range(coarse - 1)]
    centers = (large + small)[:n]
    while len(centers) < n:
        centers.append((0.5, 0.5))
    return np.clip(np.array(centers, dtype=float), 0.01, 0.99)


def construct_packing(n, random_seed: int):
    """Explore with hill climbing, then polish each result jointly."""
    rng = np.random.default_rng(random_seed)
    starts = [_interstitial_layout(n, 4), _interstitial_layout(n, 5),
              rng.uniform(0.05, 0.95, size=(n, 2))]

    slice_seconds = 16.0 / len(starts)
    best_centers = best_radii = None
    best_sum = -1.0

    for start in starts:
        explored = _hill_climb(start, time.monotonic() + slice_seconds * 0.6, rng)
        try:
            centers, radii = _polish(explored, n)
        except (ValueError, np.linalg.LinAlgError):
            centers, radii = explored, _repair(explored, _max_radii(explored))
        total = float(radii.sum())
        if total > best_sum:
            best_sum, best_centers, best_radii = total, centers, radii

    return best_centers, best_radii, best_sum
```
