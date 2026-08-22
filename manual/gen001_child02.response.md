Choosing centres first and radii second is a two-stage approximation of one
problem. Centres and radii can be optimised jointly: maximise `sum(r)` over all
`3n` variables at once, with the containment and non-overlap conditions as
explicit nonlinear constraints. Supplying analytic Jacobians keeps it fast
enough to afford several restarts.

```python
"""Joint optimisation of centres and radii under the packing constraints."""
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
    """Best radii for fixed centres, used to warm-start the joint solve."""
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
    """Scale radii down until the packing is exactly feasible."""
    radii = np.clip(radii, 0.0, np.maximum(_wall_limits(centers), 0.0))
    dist = _pair_distances(centers)
    np.fill_diagonal(dist, np.inf)
    worst = ((radii[:, None] + radii[None, :]) / dist).max()
    if worst > 1.0:
        radii = radii / worst
    return radii * (1.0 - 1e-12)


def _solve(start_centers, n, maxiter=200):
    """Run SLSQP on the joint (x, y, r) problem from one starting point."""
    iu, ju = np.triu_indices(n, k=1)
    m = len(iu)

    def unpack(z):
        return z[:n], z[n:2 * n], z[2 * n:]

    def objective(z):
        return -z[2 * n:].sum()

    objective_grad = np.concatenate([np.zeros(2 * n), -np.ones(n)])

    def constraints(z):
        x, y, r = unpack(z)
        dx, dy = x[iu] - x[ju], y[iu] - y[ju]
        dist = np.sqrt(dx ** 2 + dy ** 2)
        return np.concatenate([
            dist - r[iu] - r[ju],       # no overlap
            x - r, y - r,               # inside the left and bottom walls
            1.0 - x - r, 1.0 - y - r,   # inside the right and top walls
        ])

    def constraints_jac(z):
        x, y, r = unpack(z)
        jac = np.zeros((m + 4 * n, 3 * n))

        dx, dy = x[iu] - x[ju], y[iu] - y[ju]
        dist = np.maximum(np.sqrt(dx ** 2 + dy ** 2), 1e-12)
        rows = np.arange(m)
        jac[rows, iu] = dx / dist
        jac[rows, ju] = -dx / dist
        jac[rows, n + iu] = dy / dist
        jac[rows, n + ju] = -dy / dist
        jac[rows, 2 * n + iu] = -1.0
        jac[rows, 2 * n + ju] = -1.0

        idx = np.arange(n)
        for k, (var_offset, sign) in enumerate(
                [(0, 1.0), (n, 1.0), (0, -1.0), (n, -1.0)]):
            block = m + k * n + idx
            jac[block, var_offset + idx] = sign
            jac[block, 2 * n + idx] = -1.0

        return jac

    radii0 = _max_radii(start_centers)
    z0 = np.concatenate([start_centers[:, 0], start_centers[:, 1], radii0])
    bounds = [(0.0, 1.0)] * (2 * n) + [(0.0, 0.5)] * n

    result = minimize(
        objective, z0, jac=lambda _: objective_grad, bounds=bounds,
        constraints=[{"type": "ineq", "fun": constraints, "jac": constraints_jac}],
        method="SLSQP", options={"maxiter": maxiter, "ftol": 1e-10},
    )

    x, y, r = unpack(result.x)
    centers = np.clip(np.column_stack([x, y]), 0.0, 1.0)
    return centers, _repair(centers, r)


def _hex_start(n, rows):
    """Row-structured start, offsetting alternate rows like a hex packing."""
    centers = []
    for row_index, count in enumerate(rows):
        y = (row_index + 0.5) / len(rows)
        shift = 0.25 / max(count, 1) if row_index % 2 else 0.0
        for col in range(count):
            centers.append(((col + 0.5) / count + shift, y))
    return np.clip(np.array(centers, dtype=float), 0.01, 0.99)[:n]


def construct_packing(n, random_seed: int):
    """Optimise from several starts within a time budget; keep the best."""
    rng = np.random.default_rng(random_seed)
    deadline = time.monotonic() + 18.0

    starts = []
    for row_count in (4, 5, 6):
        base, extra = divmod(n, row_count)
        rows = [base + (1 if i < extra else 0) for i in range(row_count)]
        if min(rows) > 0:
            starts.append(_hex_start(n, rows))

    best_centers = None
    best_radii = None
    best_sum = -1.0

    attempt = 0
    while time.monotonic() < deadline:
        if attempt < len(starts):
            start = starts[attempt]
        else:
            start = rng.uniform(0.05, 0.95, size=(n, 2))
        attempt += 1

        try:
            centers, radii = _solve(start, n)
        except (ValueError, np.linalg.LinAlgError):
            continue

        total = float(radii.sum())
        if total > best_sum:
            best_sum, best_centers, best_radii = total, centers, radii

        if attempt >= 12:
            break

    if best_centers is None:
        best_centers = _hex_start(n, [5, 5, 5, 5, 6])
        best_radii = _repair(best_centers, _max_radii(best_centers))
        best_sum = float(best_radii.sum())

    return best_centers, best_radii, best_sum
```
