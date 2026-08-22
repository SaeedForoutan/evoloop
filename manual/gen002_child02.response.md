The joint solve is finding good local optima but each restart throws away
everything learned. Basin hopping fixes that: keep an incumbent, perturb it, re-
solve from the perturbed point, and accept only improvements. Perturbing the
smallest circles is the productive move — they are the ones sitting in slack
space that a better arrangement could use.

```python
"""Joint (x, y, r) optimisation driven by basin hopping around an incumbent."""
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
    """Best radii for fixed centres, used to warm-start each joint solve."""
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


class _Problem:
    """Constraint functions and Jacobians for a fixed circle count."""

    def __init__(self, n):
        self.n = n
        self.iu, self.ju = np.triu_indices(n, k=1)
        self.m = len(self.iu)
        self.objective_grad = np.concatenate([np.zeros(2 * n), -np.ones(n)])
        self.bounds = [(0.0, 1.0)] * (2 * n) + [(0.0, 0.5)] * n
        # The wall rows are constant, so build that part of the Jacobian once.
        self._jac = np.zeros((self.m + 4 * n, 3 * n))
        idx = np.arange(n)
        for k, (offset, sign) in enumerate(
                [(0, 1.0), (n, 1.0), (0, -1.0), (n, -1.0)]):
            block = self.m + k * n + idx
            self._jac[block, offset + idx] = sign
            self._jac[block, 2 * n + idx] = -1.0

    def objective(self, z):
        return -z[2 * self.n:].sum()

    def constraints(self, z):
        n, iu, ju = self.n, self.iu, self.ju
        x, y, r = z[:n], z[n:2 * n], z[2 * n:]
        dist = np.sqrt((x[iu] - x[ju]) ** 2 + (y[iu] - y[ju]) ** 2)
        return np.concatenate([
            dist - r[iu] - r[ju],
            x - r, y - r, 1.0 - x - r, 1.0 - y - r,
        ])

    def constraints_jac(self, z):
        n, iu, ju, m = self.n, self.iu, self.ju, self.m
        x, y = z[:n], z[n:2 * n]
        dx, dy = x[iu] - x[ju], y[iu] - y[ju]
        dist = np.maximum(np.sqrt(dx ** 2 + dy ** 2), 1e-12)

        rows = np.arange(m)
        jac = self._jac
        jac[rows, :] = 0.0
        jac[rows, iu] = dx / dist
        jac[rows, ju] = -dx / dist
        jac[rows, n + iu] = dy / dist
        jac[rows, n + ju] = -dy / dist
        jac[rows, 2 * n + iu] = -1.0
        jac[rows, 2 * n + ju] = -1.0
        return jac

    def solve(self, centers, maxiter=200):
        n = self.n
        radii = _max_radii(centers)
        z0 = np.concatenate([centers[:, 0], centers[:, 1], radii])
        result = minimize(
            self.objective, z0, jac=lambda _: self.objective_grad,
            bounds=self.bounds, method="SLSQP",
            constraints=[{"type": "ineq", "fun": self.constraints,
                          "jac": self.constraints_jac}],
            options={"maxiter": maxiter, "ftol": 1e-11},
        )
        z = result.x
        out = np.clip(np.column_stack([z[:n], z[n:2 * n]]), 0.0, 1.0)
        return out, _repair(out, z[2 * n:])


def _hex_start(n, rows, rng=None):
    centers = []
    for row_index, count in enumerate(rows):
        y = (row_index + 0.5) / len(rows)
        shift = 0.25 / max(count, 1) if row_index % 2 else 0.0
        for col in range(count):
            centers.append(((col + 0.5) / count + shift, y))
    centers = np.array(centers[:n], dtype=float)
    if rng is not None:
        centers = centers + rng.normal(0.0, 0.01, centers.shape)
    return np.clip(centers, 0.01, 0.99)


def _row_structures(n):
    """Row decompositions worth trying as starting configurations."""
    structures = []
    for row_count in range(4, 8):
        base, extra = divmod(n, row_count)
        if base < 2:
            continue
        structures.append([base + (1 if i < extra else 0) for i in range(row_count)])
        alt = [base + 1 if i % 2 == 0 else base for i in range(row_count)]
        if sum(alt) == n:
            structures.append(alt)
    return structures


def construct_packing(n, random_seed: int):
    """Seed from structured starts, then basin-hop around the incumbent."""
    rng = np.random.default_rng(random_seed)
    problem = _Problem(n)
    deadline = time.monotonic() + 20.0

    best_centers = None
    best_radii = None
    best_sum = -1.0

    # Phase 1 — structured starts, to land in a good basin.
    for rows in _row_structures(n):
        if time.monotonic() > deadline - 8.0:
            break
        try:
            centers, radii = problem.solve(_hex_start(n, rows))
        except (ValueError, np.linalg.LinAlgError):
            continue
        total = float(radii.sum())
        if total > best_sum:
            best_sum, best_centers, best_radii = total, centers, radii

    if best_centers is None:
        best_centers = _hex_start(n, [5, 5, 5, 5, 6])
        best_radii = _repair(best_centers, _max_radii(best_centers))
        best_sum = float(best_radii.sum())

    # Phase 2 — perturb the incumbent and re-solve, keeping improvements.
    strength = 0.06
    while time.monotonic() < deadline:
        trial = best_centers.copy()

        # Relocate the least useful circles: the smallest ones are wasting space.
        count = max(1, int(rng.integers(1, 4)))
        smallest = np.argsort(best_radii)[:count]
        trial[smallest] = rng.uniform(0.05, 0.95, size=(count, 2))

        # Jitter everything else so the whole configuration can shift.
        mask = np.ones(n, dtype=bool)
        mask[smallest] = False
        trial[mask] += rng.normal(0.0, strength, size=(mask.sum(), 2))
        trial = np.clip(trial, 0.005, 0.995)

        try:
            centers, radii = problem.solve(trial, maxiter=150)
        except (ValueError, np.linalg.LinAlgError):
            continue

        total = float(radii.sum())
        if total > best_sum:
            best_sum, best_centers, best_radii = total, centers, radii
            strength = 0.06
        else:
            # Cool towards finer perturbations as the incumbent gets harder to beat.
            strength = max(strength * 0.97, 0.015)

    return best_centers, best_radii, best_sum
```
