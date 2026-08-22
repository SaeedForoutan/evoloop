Good packings of a square tend to be symmetric about an axis or a diagonal.
Random and row-structured starts both ignore that, so the solver spends its
iterations rediscovering symmetry. Generate starts that are symmetric by
construction — mirror a half-configuration across an axis or a diagonal — which
covers a different part of the space than the row layouts do.

```python
"""Symmetry-seeded starting configurations, each finished with a joint solve."""
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


def _solve(centers, n, maxiter=180):
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


def _mirror_vertical(half, remainder, rng):
    """Reflect a left-half configuration across x = 0.5."""
    left = half * np.array([0.5, 1.0])
    right = np.column_stack([1.0 - left[:, 0], left[:, 1]])
    spine = np.column_stack([np.full(remainder, 0.5),
                             rng.uniform(0.05, 0.95, remainder)])
    return np.vstack([left, right, spine])


def _mirror_diagonal(half, remainder, rng):
    """Reflect a below-diagonal configuration across y = x."""
    lower = half.copy()
    lower[:, 1] *= lower[:, 0]  # keep points under the diagonal
    upper = lower[:, ::-1]
    spine = np.column_stack([np.full(remainder, 0.5), np.full(remainder, 0.5)])
    spine = spine + rng.normal(0.0, 0.05, spine.shape)
    return np.vstack([lower, upper, spine])


def _symmetric_starts(n, rng, count=14):
    """Symmetric starts, half of them mirrored, half diagonal-mirrored."""
    starts = []
    half_size, remainder = divmod(n, 2)
    for index in range(count):
        half = rng.uniform(0.05, 0.95, size=(half_size, 2))
        build = _mirror_vertical if index % 2 == 0 else _mirror_diagonal
        block = build(half, remainder, rng)
        if block.shape[0] == n:
            starts.append(np.clip(block, 0.01, 0.99))
    return starts


def construct_packing(n, random_seed: int):
    rng = np.random.default_rng(random_seed)
    deadline = time.monotonic() + 19.0

    best_centers = best_radii = None
    best_sum = -1.0

    for start in _symmetric_starts(n, rng):
        if time.monotonic() > deadline - 3.0:
            break
        try:
            centers, radii = _solve(start, n)
        except (ValueError, np.linalg.LinAlgError):
            continue
        total = float(radii.sum())
        if total > best_sum:
            best_sum, best_centers, best_radii = total, centers, radii

    # Polish whatever the symmetric search found.
    while time.monotonic() < deadline and best_centers is not None:
        trial = np.clip(best_centers + rng.normal(0.0, 0.02, size=(n, 2)), 0.005, 0.995)
        try:
            centers, radii = _solve(trial, n, maxiter=120)
        except (ValueError, np.linalg.LinAlgError):
            break
        total = float(radii.sum())
        if total > best_sum:
            best_sum, best_centers, best_radii = total, centers, radii

    if best_centers is None:
        best_centers = np.clip(rng.uniform(0.05, 0.95, size=(n, 2)), 0.01, 0.99)
        best_radii = _repair(best_centers, _max_radii(best_centers))
        best_sum = float(best_radii.sum())

    return best_centers, best_radii, best_sum
```
