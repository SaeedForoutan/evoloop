<!-- system -->
You are the generation half of an evolutionary coding agent working on algorithm discovery. You are shown a parent program, its measured score, and other high-scoring programs from the population. You propose one improved program.

You are optimising a real number. Small, principled changes that measurably raise the score beat sweeping rewrites that fail to run. Prior attempts and why they failed are given to you — do not repeat them.

<!-- user -->
## Task

Pack 26 non-overlapping circles inside the unit square [0,1] x [0,1] so that the **sum of their radii** is as large as possible.

Write `construct_packing(n, random_seed)` returning `(centers, radii, sum_of_radii)`:

- `centers` — numpy array, shape (26, 2), each (x, y) in the unit square
- `radii` — numpy array, shape (26,), all non-negative
- `sum_of_radii` — the sum, a float

Hard constraints, checked by the evaluator (violating any of them scores as a failure):

- every circle lies fully inside the unit square: `r_i <= x_i, y_i <= 1 - r_i`
- no two circles overlap: `r_i + r_j <= distance(center_i, center_j)` for all i != j
- all values finite, all radii non-negative
- the call should return within **20 seconds**; the evaluator kills it at 60s
  wall clock, 50s CPU, or 4GB. numpy runs single-threaded, so CPU time and wall
  clock are roughly equal — budget your search loop against `time.monotonic()`

`numpy` and `scipy` are available. The published state of the art for n=26 is about **2.635** — the seed scores far below that, so there is substantial room.

Approaches that tend to work: choose good centre positions, then solve for the largest radii those centres admit (this is a linear program in the radii); refine the centres with a local optimiser or a physics-style relaxation; restart from several initialisations and keep the best. Structured patterns (grids, hexagonal packings, corner-and-edge placements, mixed circle sizes) usually beat concentric rings.
---
Current best score in the population: **2.631730** (generation 2).

## Parent program — improve this one

### Parent (score 2.590052)
Evaluator note: valid packing; tightest pair gap 1.624e-13, tightest wall gap 6.617e-14.

```python
"""Relaxed starts handed to a joint solve over centres and radii."""
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


def _relax(centers, steps=150, step_size=0.01):
    """A short repulsion pass — enough to de-cluster a random start."""
    centers = centers.copy()
    for step in range(steps):
        diff = centers[:, None, :] - centers[None, :, :]
        dist = np.sqrt((diff ** 2).sum(-1))
        np.fill_diagonal(dist, np.inf)
        push = (diff / dist[:, :, None] / np.maximum(dist, 1e-6)[:, :, None] ** 2).sum(1)
        norm = np.linalg.norm(push, axis=1, keepdims=True)
        push = push / np.maximum(norm, 1e-12)
        centers = np.clip(centers + step_size * (1 - step / steps) * push,
                          1e-3, 1 - 1e-3)
    return centers


def construct_packing(n, random_seed: int):
    """Relax varied random starts, then solve each jointly; keep the best."""
    rng = np.random.default_rng(random_seed)
    deadline = time.monotonic() + 19.0

    best_centers = best_radii = None
    best_sum = -1.0

    while time.monotonic() < deadline:
        start = _relax(rng.uniform(0.05, 0.95, size=(n, 2)))
        try:
            centers, radii = _solve(start, n)
        except (ValueError, np.linalg.LinAlgError):
            continue
        total = float(radii.sum())
        if total > best_sum:
            best_sum, best_centers, best_radii = total, centers, radii

    if best_centers is None:
        best_centers = rng.uniform(0.1, 0.9, size=(n, 2))
        best_radii = _repair(best_centers, _max_radii(best_centers))
        best_sum = float(best_radii.sum())

    return best_centers, best_radii, best_sum
```

## Other programs in the population, for reference

### Inspiration 1 (score 2.613807)
Evaluator note: valid packing; tightest pair gap 1.509e-13, tightest wall gap 7.355e-14.

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
### Inspiration 2 (score 2.622301)
Evaluator note: valid packing; tightest pair gap 1.532e-13, tightest wall gap 6.350e-14.

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

---

## Your response

Return the complete replacement for the evolvable block: the full body of
`construct_packing` plus any helper functions it needs. Put `import` statements
for anything beyond `numpy` inside the block.

Respond with exactly one Python code block and nothing else:

```python
<your code here>
```
