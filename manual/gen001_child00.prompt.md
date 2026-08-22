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
- the whole call must return within 30 seconds of CPU time

`numpy` and `scipy` are available. The published state of the art for n=26 is about **2.635** — the seed scores far below that, so there is substantial room.

Approaches that tend to work: choose good centre positions, then solve for the largest radii those centres admit (this is a linear program in the radii); refine the centres with a local optimiser or a physics-style relaxation; restart from several initialisations and keep the best. Structured patterns (grids, hexagonal packings, corner-and-edge placements, mixed circle sizes) usually beat concentric rings.
---
Current best score in the population: **0.941455** (generation 0).

## Parent program — improve this one

### Parent (score 0.941455)
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
### Inspiration 2 (score 0.941455)
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

---

## Your response

Return the complete replacement for the evolvable block: the full body of
`construct_packing` plus any helper functions it needs. Put `import` statements
for anything beyond `numpy` inside the block.

Respond with exactly one Python code block and nothing else:

```python
<your code here>
```
