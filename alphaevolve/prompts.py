"""Prompt sampler: assembles the context shown to the generating model.

Each child gets a freshly sampled prompt — a different parent, different
inspirations, different recalled failures. That resampling is what supplies
population diversity; the current models reject a `temperature` parameter, so
diversity cannot come from sampling settings.
"""

from __future__ import annotations

from .database import Program, ProgramDatabase

SYSTEM = """\
You are the generation half of an evolutionary coding agent working on \
algorithm discovery. You are shown a parent program, its measured score, and \
other high-scoring programs from the population. You propose one improved \
program.

You are optimising a real number. Small, principled changes that measurably \
raise the score beat sweeping rewrites that fail to run. Prior attempts and \
why they failed are given to you — do not repeat them.\
"""

TASK = """\
## Task

Pack {n} non-overlapping circles inside the unit square [0,1] x [0,1] so that \
the **sum of their radii** is as large as possible.

Write `construct_packing(n, random_seed)` returning `(centers, radii, sum_of_radii)`:

- `centers` — numpy array, shape ({n}, 2), each (x, y) in the unit square
- `radii` — numpy array, shape ({n},), all non-negative
- `sum_of_radii` — the sum, a float

Hard constraints, checked by the evaluator (violating any of them scores as a failure):

- every circle lies fully inside the unit square: `r_i <= x_i, y_i <= 1 - r_i`
- no two circles overlap: `r_i + r_j <= distance(center_i, center_j)` for all i != j
- all values finite, all radii non-negative
- the call should return within **20 seconds**; the evaluator kills it at 60s
  wall clock, 50s CPU, or 4GB. numpy runs single-threaded, so CPU time and wall
  clock are roughly equal — budget your search loop against `time.monotonic()`

`numpy` and `scipy` are available. The published state of the art for n={n} is \
about **2.635** — the seed scores far below that, so there is substantial room.

Approaches that tend to work: choose good centre positions, then solve for the \
largest radii those centres admit (this is a linear program in the radii); \
refine the centres with a local optimiser or a physics-style relaxation; restart \
from several initialisations and keep the best. Structured patterns (grids, \
hexagonal packings, corner-and-edge placements, mixed circle sizes) usually beat \
concentric rings.\
"""


def _fmt(program: Program, label: str) -> str:
    status = f"score {program.score:.6f}" if program.valid else "FAILED"
    return (
        f"### {label} ({status})\n"
        f"Evaluator note: {program.insight}\n\n"
        f"```python\n{program.block}\n```\n"
    )


def build_prompt(db: ProgramDatabase, parent: Program, n: int = 26) -> str:
    """Render the full user prompt for one child."""
    parts = [TASK.format(n=n), "\n---\n"]

    best = db.best()
    if best is not None:
        parts.append(
            f"Current best score in the population: **{best.score:.6f}** "
            f"(generation {best.generation}).\n"
        )

    parts.append("\n## Parent program — improve this one\n\n")
    parts.append(_fmt(parent, "Parent"))

    inspirations = db.select_inspirations(parent)
    if inspirations:
        parts.append("\n## Other programs in the population, for reference\n\n")
        for i, insp in enumerate(inspirations, 1):
            parts.append(_fmt(insp, f"Inspiration {i}"))

    failures = db.recent_failures()
    if failures:
        parts.append("\n## Recent failed attempts — do not repeat these mistakes\n\n")
        for fail in failures:
            parts.append(f"- {fail.insight}\n")

    parts.append(f"""
---

## Your response

Return the complete replacement for the evolvable block: the full body of
`construct_packing` plus any helper functions it needs. Put `import` statements
for anything beyond `numpy` inside the block.

Respond with exactly one Python code block and nothing else:

```python
<your code here>
```
""")
    return "".join(parts)
