"""Program database: the evolving population, organised into islands.

Mirrors the AlphaEvolve loop's storage half — every candidate ever evaluated is
kept with its score, parent, and the insight the evaluator produced, so the
prompt sampler can draw both a parent to mutate and inspirations to show
alongside it.
"""

from __future__ import annotations

import json
import os
import random
from dataclasses import asdict, dataclass, field

BLOCK_START = "# EVOLVE-BLOCK-START"
BLOCK_END = "# EVOLVE-BLOCK-END"


def extract_block(source: str) -> str:
    """Return the evolvable region of a program."""
    if BLOCK_START not in source or BLOCK_END not in source:
        raise ValueError("program is missing its EVOLVE-BLOCK markers")
    return source.split(BLOCK_START, 1)[1].split(BLOCK_END, 1)[0].strip("\n")


def splice_block(source: str, block: str) -> str:
    """Return `source` with its evolvable region replaced by `block`."""
    head, rest = source.split(BLOCK_START, 1)
    _, tail = rest.split(BLOCK_END, 1)
    return f"{head}{BLOCK_START}\n{block.strip()}\n{BLOCK_END}{tail}"


@dataclass
class Program:
    id: str
    generation: int
    island: int
    parent_id: str | None
    block: str
    score: float
    valid: bool
    insight: str
    summary: str = ""
    centers: list = field(default_factory=list)
    radii: list = field(default_factory=list)


class ProgramDatabase:
    """Island-model population with periodic migration of island champions."""

    def __init__(self, path: str, islands: int = 3, seed: int = 0):
        self.path = path
        self.islands = islands
        self.programs: list[Program] = []
        self.generation = 0
        self.rng = random.Random(seed)

    # ---- persistence -----------------------------------------------------

    def save(self) -> None:
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, "w") as fh:
            json.dump({
                "islands": self.islands,
                "generation": self.generation,
                "programs": [asdict(p) for p in self.programs],
            }, fh, indent=2)

    @classmethod
    def load(cls, path: str) -> "ProgramDatabase":
        with open(path) as fh:
            blob = json.load(fh)
        db = cls(path, islands=blob["islands"])
        db.generation = blob["generation"]
        db.programs = [Program(**p) for p in blob["programs"]]
        return db

    # ---- population ------------------------------------------------------

    def add(self, program: Program) -> None:
        self.programs.append(program)

    def valid(self) -> list[Program]:
        return [p for p in self.programs if p.valid]

    def best(self, island: int | None = None) -> Program | None:
        pool = [p for p in self.valid() if island is None or p.island == island]
        return max(pool, key=lambda p: p.score) if pool else None

    def best_per_generation(self) -> list[float]:
        """Best-so-far score after each generation, for the progress plot."""
        history, running = [], None
        for gen in range(self.generation + 1):
            scored = [p.score for p in self.valid() if p.generation <= gen]
            if scored:
                running = max(scored)
            history.append(running)
        return history

    # ---- selection -------------------------------------------------------

    def select_parent(self, island: int) -> Program:
        """Tournament selection biased to the island's better programs.

        Falls back across islands (and then to any valid program) so a wiped-out
        island can still be reseeded.
        """
        pool = [p for p in self.valid() if p.island == island] or self.valid()
        if not pool:
            raise RuntimeError("no valid program to use as a parent")
        pool.sort(key=lambda p: p.score, reverse=True)
        # Favour the elite but keep a tail so the search does not collapse.
        elite = pool[: max(1, len(pool) // 2)]
        contenders = self.rng.sample(elite, min(3, len(elite)))
        return max(contenders, key=lambda p: p.score)

    def select_inspirations(self, parent: Program, count: int = 2) -> list[Program]:
        """Pick high-scoring programs other than the parent to show as context."""
        others = [p for p in self.valid() if p.id != parent.id]
        if not others:
            return []
        others.sort(key=lambda p: p.score, reverse=True)
        top = others[: max(count, min(5, len(others)))]
        return self.rng.sample(top, min(count, len(top)))

    def recent_failures(self, limit: int = 3) -> list[Program]:
        """Most recent invalid candidates, so the model can avoid repeating them."""
        failures = [p for p in self.programs if not p.valid]
        return failures[-limit:]

    def migrate(self) -> None:
        """Copy each island's champion into the next island."""
        champions = [self.best(i) for i in range(self.islands)]
        for i, champ in enumerate(champions):
            if champ is None:
                continue
            target = (i + 1) % self.islands
            self.add(Program(
                id=f"{champ.id}-mig{target}",
                generation=self.generation,
                island=target,
                parent_id=champ.id,
                block=champ.block,
                score=champ.score,
                valid=champ.valid,
                insight=champ.insight,
                summary=f"migrated from island {i}",
                centers=champ.centers,
                radii=champ.radii,
            ))
