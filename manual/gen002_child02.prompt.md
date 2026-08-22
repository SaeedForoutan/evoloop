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
Current best score in the population: **2.613807** (generation 1).

## Parent program — improve this one

### Parent (score 2.613807)
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

## Other programs in the population, for reference

### Inspiration 1 (score 2.541421)
Evaluator note: valid packing; tightest pair gap 1.416e-13, tightest wall gap 1.001e-13.

```python
"""Mixed-size interstitial layout, refined by per-circle hill climbing."""
import time

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
    """Largest feasible radii for fixed centres, via a linear program."""
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
    """Scale down until the packing satisfies every constraint exactly."""
    radii = np.clip(radii, 0.0, np.maximum(_wall_limits(centers), 0.0))
    dist = _pair_distances(centers)
    np.fill_diagonal(dist, np.inf)
    worst = ((radii[:, None] + radii[None, :]) / dist).max()
    if worst > 1.0:
        radii = radii / worst
    return radii * (1.0 - 1e-12)


def _interstitial_layout(n, coarse):
    """A `coarse` x `coarse` grid plus a shifted grid in the gaps between it."""
    large = [((i + 0.5) / coarse, (j + 0.5) / coarse)
             for i in range(coarse) for j in range(coarse)]
    small = [((i + 1.0) / coarse, (j + 1.0) / coarse)
             for i in range(coarse - 1) for j in range(coarse - 1)]

    centers = large + small
    # Top up along the edges if the two grids do not reach n.
    edge = 0
    while len(centers) < n:
        t = (edge + 0.5) / max(n - len(large) - len(small), 1)
        centers.append((min(max(t, 0.02), 0.98), 0.02 if edge % 2 else 0.98))
        edge += 1

    return np.clip(np.array(centers[:n], dtype=float), 0.01, 0.99)


def _hill_climb(centers, deadline, rng, scale=0.08):
    """Move one circle at a time, keeping only moves that raise the sum."""
    n = centers.shape[0]
    centers = centers.copy()
    best = float(_max_radii(centers).sum())

    while time.monotonic() < deadline:
        index = rng.integers(n)
        saved = centers[index].copy()

        proposal = saved + rng.normal(0.0, scale, size=2)
        centers[index] = np.clip(proposal, 0.005, 0.995)

        total = float(_max_radii(centers).sum())
        if total > best:
            best = total
        else:
            centers[index] = saved
            # Shrink the step as improvements get harder to find.
            scale = max(scale * 0.995, 0.004)

    return centers


def construct_packing(n, random_seed: int):
    """Try a few interstitial layouts, refine each, and keep the best."""
    rng = np.random.default_rng(random_seed)
    budget = 18.0

    candidates = [_interstitial_layout(n, coarse) for coarse in (4, 5)]
    candidates.append(rng.uniform(0.05, 0.95, size=(n, 2)))

    slice_seconds = budget / len(candidates)
    best_centers = None
    best_radii = None
    best_sum = -1.0

    for start in candidates:
        refined = _hill_climb(start, time.monotonic() + slice_seconds, rng)
        radii = _repair(refined, _max_radii(refined))
        total = float(radii.sum())
        if total > best_sum:
            best_sum, best_centers, best_radii = total, refined, radii

    return best_centers, best_radii, best_sum
```
### Inspiration 2 (score 2.500000)
Evaluator note: valid packing; tightest pair gap 1.668e-13, tightest wall gap 8.338e-14.

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

---

## Your response

Return the complete replacement for the evolvable block: the full body of
`construct_packing` plus any helper functions it needs. Put `import` statements
for anything beyond `numpy` inside the block.

Respond with exactly one Python code block and nothing else:

```python
<your code here>
```
