Enumerating layouts covers the starting points well but stops as soon as each
solve converges. After a solve there is usually one circle that ended up tiny,
squeezed into a spot with no room. Move it to wherever the packing has the most
slack and solve again — repeat while it keeps paying.

```python
"""Enumerated layouts, then repeated relocate-the-runt-and-re-solve passes."""
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


_AXIS = np.linspace(0.02, 0.98, 55)
_GX, _GY = np.meshgrid(_AXIS, _AXIS)
_GRID = np.column_stack([_GX.ravel(), _GY.ravel()])


def _roomiest_point(centers, radii, ignore):
    """Grid search for the point with the most clearance, ignoring one circle."""
    keep = np.ones(centers.shape[0], dtype=bool)
    keep[ignore] = False
    diff = _GRID[:, None, :] - centers[None, keep, :]
    room = (np.sqrt((diff ** 2).sum(-1)) - radii[keep][None, :]).min(axis=1)
    walls = np.minimum.reduce([_GRID[:, 0], _GRID[:, 1],
                               1.0 - _GRID[:, 0], 1.0 - _GRID[:, 1]])
    return _GRID[int(np.argmax(np.minimum(room, walls)))]


def _layouts(n):
    out = []
    for row_count in range(4, 8):
        base, extra = divmod(n, row_count)
        if base < 2:
            continue
        rows = [base + (1 if i < extra else 0) for i in range(row_count)]
        for stagger in (0.0, 0.25, 0.5):
            centers = []
            for row_index, count in enumerate(rows):
                y = (row_index + 0.5) / len(rows)
                shift = stagger / max(count, 1) if row_index % 2 else 0.0
                for col in range(count):
                    centers.append(((col + 0.5) / count + shift, y))
            block = np.array(centers[:n], dtype=float)
            if block.shape[0] == n:
                out.append(np.clip(block, 0.01, 0.99))
    return out


def construct_packing(n, random_seed: int):
    rng = np.random.default_rng(random_seed)
    deadline = time.monotonic() + 19.0

    best_centers = best_radii = None
    best_sum = -1.0

    for start in _layouts(n):
        if time.monotonic() > deadline - 9.0:
            break
        try:
            centers, radii = _solve(start, n)
        except (ValueError, np.linalg.LinAlgError):
            continue
        total = float(radii.sum())
        if total > best_sum:
            best_sum, best_centers, best_radii = total, centers, radii

    if best_centers is None:
        best_centers = np.clip(rng.uniform(0.05, 0.95, size=(n, 2)), 0.01, 0.99)
        best_radii = _repair(best_centers, _max_radii(best_centers))
        best_sum = float(best_radii.sum())

    # Relocate the smallest circle into the roomiest gap, re-solve, keep if better.
    while time.monotonic() < deadline:
        runt = int(np.argmin(best_radii))
        target = _roomiest_point(best_centers, best_radii, runt)

        trial = best_centers.copy()
        trial[runt] = np.clip(target + rng.normal(0.0, 0.008, size=2), 0.005, 0.995)

        try:
            centers, radii = _solve(trial, n, maxiter=140)
        except (ValueError, np.linalg.LinAlgError):
            break

        total = float(radii.sum())
        if total <= best_sum + 1e-9:
            break  # the move stopped paying
        best_sum, best_centers, best_radii = total, centers, radii

    return best_centers, best_radii, best_sum
```
