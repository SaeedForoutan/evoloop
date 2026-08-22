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

### Parent (score 2.322306)
Evaluator note: valid packing; tightest pair gap 1.342e-13, tightest wall gap 5.944e-14.

```python
"""Repulsion-relaxed centres, with radii solved as a linear program."""
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
    """Maximise sum(r) for fixed centres — linear in r, so use an LP."""
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
    return _repair(centers, np.maximum(result.x, 0.0))


def _repair(centers, radii):
    """Scale the radii down until every constraint holds exactly."""
    radii = np.clip(radii, 0.0, np.maximum(_wall_limits(centers), 0.0))
    dist = _pair_distances(centers)
    np.fill_diagonal(dist, np.inf)
    worst = ((radii[:, None] + radii[None, :]) / dist).max()
    if worst > 1.0:
        radii = radii / worst
    return radii * (1.0 - 1e-12)


def _relax(centers, steps=400, step_size=0.01):
    """Push centres apart, keeping them inside the square.

    Each pair repels with a force falling off as 1/d^2, and walls repel too so
    the configuration spreads into the corners instead of bunching centrally.
    """
    n = centers.shape[0]
    centers = centers.copy()

    for step in range(steps):
        diff = centers[:, None, :] - centers[None, :, :]
        dist = np.sqrt((diff ** 2).sum(-1))
        np.fill_diagonal(dist, np.inf)

        # Pairwise repulsion, normalised so no single close pair dominates.
        direction = diff / dist[:, :, None]
        magnitude = 1.0 / np.maximum(dist, 1e-6) ** 2
        force = (direction * magnitude[:, :, None]).sum(axis=1)

        # Wall repulsion, strongest as a centre approaches an edge.
        wall = np.zeros_like(centers)
        wall[:, 0] += 1.0 / np.maximum(centers[:, 0], 1e-6) ** 2
        wall[:, 0] -= 1.0 / np.maximum(1.0 - centers[:, 0], 1e-6) ** 2
        wall[:, 1] += 1.0 / np.maximum(centers[:, 1], 1e-6) ** 2
        wall[:, 1] -= 1.0 / np.maximum(1.0 - centers[:, 1], 1e-6) ** 2

        total = force + 0.35 * wall
        norm = np.linalg.norm(total, axis=1, keepdims=True)
        total = total / np.maximum(norm, 1e-12)

        # Cooling schedule: large moves early, fine adjustment late.
        rate = step_size * (1.0 - step / steps)
        centers = np.clip(centers + rate * total, 1e-4, 1.0 - 1e-4)

    return centers


def construct_packing(n, random_seed: int):
    """Relax several random starts and keep the best packing found."""
    rng = np.random.default_rng(random_seed)

    best_centers = None
    best_radii = None
    best_sum = -1.0

    for _ in range(6):
        start = rng.uniform(0.05, 0.95, size=(n, 2))
        centers = _relax(start)
        radii = _max_radii(centers)
        total = float(radii.sum())
        if total > best_sum:
            best_sum, best_centers, best_radii = total, centers, radii

    return best_centers, best_radii, best_sum
```

## Other programs in the population, for reference

### Inspiration 1 (score 0.941455)
Evaluator note: valid packing; tightest pair gap 2.453e-09, tightest wall gap 0.000e+00.

```python
"""Constructor-based circle packing for n=26 circles"""
import numpy as np


def construct_packing(n, random_seed: int):
    """Construct a specific arrangement of 26 circles in a unit square.

    The goal is to maximize the sum of their radii.

    Args:
        n: Number of circles.
        random_seed: Random seed for reproducibility.

    Returns:
        Tuple of (centers, radii, sum_of_radii)
        centers: np.array of shape (26, 2) with (x, y) coordinates
        radii: np.array of shape (26) with radius of each circle
        sum_of_radii: Sum of all radii
    """

    rng = np.random.default_rng(random_seed)
    centers = np.zeros((n, 2))

    # Place circles in a structured pattern
    # This is a simple pattern - evolution will improve this

    # First, place a large circle in the center
    centers[0] = [0.5, 0.5]

    # Place 8 circles around it in a ring
    for i in range(8):
        angle = 2 * np.pi * i / 8
        centers[i + 1] = [0.5 + 0.3 * np.cos(angle), 0.5 + 0.3 * np.sin(angle)]

    # Place 16 more circles in an outer ring
    for i in range(16):
        angle = 2 * np.pi * i / 16 * rng.uniform(0.9, 1.1)
        centers[i + 9] = [0.5 + 0.7 * np.cos(angle), 0.5 + 0.7 * np.sin(angle)]

    # Additional positioning adjustment to make sure all circles
    # are inside the square and don't overlap
    # Clip to ensure everything is inside the unit square
    centers = np.clip(centers, 0.01, 0.99)

    # Compute maximum valid radii for this configuration
    radii = compute_max_radii(centers, random_seed)

    # Calculate the sum of radii
    sum_radii = np.sum(radii)

    return centers, radii, sum_radii


def compute_max_radii(centers, random_seed: int):
    """Compute the maximum possible radii for each circle position.

    Make sure that they don't overlap and stay within the unit square.

    Args:
        centers: np.array of shape (n, 2) with (x, y) coordinates
        random_seed: Random seed for reproducibility.

    Returns:
        np.array of shape (n) with radius of each circle
    """
    del random_seed  # Unused.
    n = centers.shape[0]
    radii = np.ones(n)

    # First, limit by distance to square borders
    for i in range(n):
        x, y = centers[i]
        # Distance to borders
        radii[i] = min(x, y, 1 - x, 1 - y)

    # Then, limit by distance to other circles
    # Each pair of circles with centers at distance d can have
    # sum of radii at most d to avoid overlap
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.sqrt(np.sum((centers[i] - centers[j]) ** 2))

            # If current radii would cause overlap
            if radii[i] + radii[j] > dist:
                # Scale both radii proportionally
                scale = dist / (radii[i] + radii[j] + 1e-7)
                radii[i] *= scale
                radii[j] *= scale

    return radii
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
