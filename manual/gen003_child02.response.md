Basin hopping is working but its perturbation is blind — it drops the smallest
circles at uniformly random points, which usually lands them on top of an
existing circle and wastes the solve. Aim the move instead: find the point in
the square furthest from every existing circle and from the walls, and relocate
the smallest circle there. Adding an annealed acceptance rule lets the search
step through shallow dips rather than stalling on the first local optimum.

```python
"""Basin hopping with gap-directed relocation and annealed acceptance."""
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


class _Problem:
    """Constraints and Jacobians for the joint (x, y, r) packing problem."""

    def __init__(self, n):
        self.n = n
        self.iu, self.ju = np.triu_indices(n, k=1)
        self.m = len(self.iu)
        self.grad = np.concatenate([np.zeros(2 * n), -np.ones(n)])
        self.bounds = [(0.0, 1.0)] * (2 * n) + [(0.0, 0.5)] * n

    def _cons(self, z):
        n, iu, ju = self.n, self.iu, self.ju
        x, y, r = z[:n], z[n:2 * n], z[2 * n:]
        dist = np.sqrt((x[iu] - x[ju]) ** 2 + (y[iu] - y[ju]) ** 2)
        return np.concatenate([dist - r[iu] - r[ju], x - r, y - r,
                               1.0 - x - r, 1.0 - y - r])

    def _cons_jac(self, z):
        n, iu, ju, m = self.n, self.iu, self.ju, self.m
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

    def solve(self, centers, maxiter=180):
        n = self.n
        z0 = np.concatenate([centers[:, 0], centers[:, 1], _max_radii(centers)])
        result = minimize(lambda z: -z[2 * n:].sum(), z0, jac=lambda _: self.grad,
                          bounds=self.bounds, method="SLSQP",
                          constraints=[{"type": "ineq", "fun": self._cons,
                                        "jac": self._cons_jac}],
                          options={"maxiter": maxiter, "ftol": 1e-11})
        z = result.x
        out = np.clip(np.column_stack([z[:n], z[n:2 * n]]), 0.0, 1.0)
        return out, _repair(out, z[2 * n:])


_GRID = None


def _largest_gap(centers, radii, exclude):
    """Point in the square furthest from every kept circle and from the walls.

    Evaluated on a fixed grid — exact enough to aim a relocation, and far
    cheaper than solving the true largest-empty-circle problem.
    """
    global _GRID
    if _GRID is None:
        axis = np.linspace(0.02, 0.98, 60)
        gx, gy = np.meshgrid(axis, axis)
        _GRID = np.column_stack([gx.ravel(), gy.ravel()])

    keep = np.ones(centers.shape[0], dtype=bool)
    keep[exclude] = False
    if not keep.any():
        return np.array([0.5, 0.5])

    diff = _GRID[:, None, :] - centers[None, keep, :]
    clearance = np.sqrt((diff ** 2).sum(-1)) - radii[keep][None, :]
    room = clearance.min(axis=1)

    walls = np.minimum.reduce([
        _GRID[:, 0], _GRID[:, 1], 1.0 - _GRID[:, 0], 1.0 - _GRID[:, 1]
    ])
    return _GRID[int(np.argmax(np.minimum(room, walls)))]


def _row_starts(n):
    starts = []
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
                starts.append(np.clip(block, 0.01, 0.99))
    return starts


def construct_packing(n, random_seed: int):
    """Structured starts, then gap-directed annealed basin hopping."""
    rng = np.random.default_rng(random_seed)
    problem = _Problem(n)
    deadline = time.monotonic() + 19.0

    best_centers = best_radii = None
    best_sum = -1.0

    for start in _row_starts(n):
        if time.monotonic() > deadline - 12.0:
            break
        try:
            centers, radii = problem.solve(start)
        except (ValueError, np.linalg.LinAlgError):
            continue
        total = float(radii.sum())
        if total > best_sum:
            best_sum, best_centers, best_radii = total, centers, radii

    if best_centers is None:
        best_centers = np.clip(rng.uniform(0.05, 0.95, size=(n, 2)), 0.01, 0.99)
        best_radii = _repair(best_centers, _max_radii(best_centers))
        best_sum = float(best_radii.sum())

    # Annealed search: `current` may drift downhill, `best` never does.
    current_centers, current_radii = best_centers.copy(), best_radii.copy()
    current_sum = best_sum
    temperature = 0.004

    while time.monotonic() < deadline:
        trial = current_centers.copy()
        count = int(rng.integers(1, 3))
        smallest = np.argsort(current_radii)[:count]

        # Aim each relocation at the roomiest spot left by the others.
        for index in smallest:
            target = _largest_gap(current_centers, current_radii, smallest)
            trial[index] = np.clip(target + rng.normal(0.0, 0.01, size=2), 0.005, 0.995)

        mask = np.ones(n, dtype=bool)
        mask[smallest] = False
        trial[mask] += rng.normal(0.0, 0.02, size=(mask.sum(), 2))
        trial = np.clip(trial, 0.005, 0.995)

        try:
            centers, radii = problem.solve(trial, maxiter=140)
        except (ValueError, np.linalg.LinAlgError):
            continue

        total = float(radii.sum())
        if total > best_sum:
            best_sum, best_centers, best_radii = total, centers.copy(), radii.copy()

        # Metropolis acceptance keeps the walk moving through shallow dips.
        if total > current_sum or rng.random() < np.exp((total - current_sum) / temperature):
            current_centers, current_radii, current_sum = centers, radii, total

        temperature = max(temperature * 0.99, 5e-4)

    return best_centers, best_radii, best_sum
```
