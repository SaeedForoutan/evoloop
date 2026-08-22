#!/usr/bin/env python
"""AlphaEvolve-style evolutionary loop over the circle-packing seed program.

Subcommands:
  init      evaluate the seed and create the population
  step      run one generation (generate -> evaluate -> record)
  run       run several generations back to back
  report    print the best program and write the plots
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from alphaevolve.database import Program, ProgramDatabase, extract_block, splice_block
from alphaevolve.evaluator import evaluate_program
from alphaevolve.generators import AnthropicGenerator, ManualGenerator
from alphaevolve.prompts import SYSTEM, build_prompt

ROOT = os.path.dirname(os.path.abspath(__file__))
SEED_PATH = os.path.join(ROOT, "seed", "program.py")
DB_PATH = os.path.join(ROOT, "state", "db.json")
MANUAL_DIR = os.path.join(ROOT, "manual")
REPORT_DIR = os.path.join(ROOT, "report")

N_CIRCLES = 26


def _seed_source() -> str:
    with open(SEED_PATH) as fh:
        return fh.read()


def _make_generator(kind: str):
    return ManualGenerator(MANUAL_DIR) if kind == "manual" else AnthropicGenerator()


def cmd_init(args) -> int:
    source = _seed_source()
    block = extract_block(source)
    result = evaluate_program(source, n=N_CIRCLES)

    db = ProgramDatabase(DB_PATH, islands=args.islands)
    for island in range(args.islands):
        db.add(Program(
            id=f"seed-i{island}",
            generation=0,
            island=island,
            parent_id=None,
            block=block,
            score=result.score,
            valid=result.valid,
            insight=result.insight,
            summary="seed program (concentric rings)",
            centers=result.centers,
            radii=result.radii,
        ))
    db.save()

    print(f"seed score: {result.score:.6f}  (valid={result.valid})")
    print(f"insight   : {result.insight}")
    print(f"population: {len(db.programs)} programs across {args.islands} islands")
    return 0


def _run_generation(db: ProgramDatabase, generator, children: int) -> tuple[int, int]:
    """Generate, evaluate, and record one generation. Returns (added, improved)."""
    db.generation += 1
    gen = db.generation
    source = _seed_source()
    baseline = db.best()
    baseline_score = baseline.score if baseline else float("-inf")

    pending: list[str] = []
    added = improved = 0

    for child in range(children):
        island = child % db.islands
        parent = db.select_parent(island)
        prompt = build_prompt(db, parent, n=N_CIRCLES)
        tag = f"gen{gen:03d}_child{child:02d}"

        try:
            block = generator.generate(SYSTEM, prompt, tag)
        except ManualGenerator.PendingResponse as exc:
            pending.extend(exc.paths)
            continue
        except (RuntimeError, ValueError) as exc:
            print(f"  {tag}: generation failed — {exc}")
            continue

        result = evaluate_program(splice_block(source, block), n=N_CIRCLES)
        db.add(Program(
            id=tag,
            generation=gen,
            island=island,
            parent_id=parent.id,
            block=block,
            score=result.score,
            valid=result.valid,
            insight=result.insight,
            summary=f"child of {parent.id}",
            centers=result.centers,
            radii=result.radii,
        ))
        added += 1

        if result.valid:
            delta = result.score - parent.score
            flag = ""
            if result.score > baseline_score:
                baseline_score = result.score
                improved += 1
                flag = "  <-- new best"
            print(f"  {tag}: score {result.score:.6f} "
                  f"(parent {parent.score:.6f}, {delta:+.6f}){flag}")
        else:
            print(f"  {tag}: FAILED — {result.insight}")

    if pending:
        # Nothing was recorded for these children; roll the counter back so the
        # same generation can be retried once the responses are written.
        db.generation -= 1
        print(f"\n{len(pending)} prompt(s) written, awaiting responses:")
        for path in pending:
            print(f"  {os.path.basename(path).replace('.response.md', '.prompt.md')}"
                  f"  ->  write {os.path.basename(path)}")
        return added, improved

    if db.islands > 1 and gen % 3 == 0:
        db.migrate()
        print(f"  (migration: island champions copied onward)")

    return added, improved


def cmd_step(args) -> int:
    db = ProgramDatabase.load(DB_PATH)
    generator = _make_generator(args.generator)
    print(f"generation {db.generation + 1}:")
    _run_generation(db, generator, args.children)
    db.save()

    best = db.best()
    if best:
        print(f"\nbest so far: {best.score:.6f} ({best.id})")
    return 0


def cmd_run(args) -> int:
    db = ProgramDatabase.load(DB_PATH)
    generator = _make_generator(args.generator)

    for _ in range(args.generations):
        print(f"generation {db.generation + 1}:")
        _run_generation(db, generator, args.children)
        db.save()
        best = db.best()
        if best:
            print(f"  best so far: {best.score:.6f}\n")

    return 0


def cmd_report(args) -> int:
    db = ProgramDatabase.load(DB_PATH)
    best = db.best()
    if best is None:
        print("no valid programs in the population")
        return 1

    seed = next((p for p in db.programs if p.id.startswith("seed")), None)
    os.makedirs(REPORT_DIR, exist_ok=True)

    print(f"generations      : {db.generation}")
    print(f"programs evaluated: {len(db.programs)}  "
          f"({len(db.valid())} valid, {len(db.programs) - len(db.valid())} failed)")
    if seed:
        gain = best.score - seed.score
        print(f"seed score       : {seed.score:.6f}")
        print(f"best score       : {best.score:.6f}  ({gain:+.6f}, "
              f"{gain / seed.score * 100:+.1f}%)")
    print(f"best program     : {best.id} (generation {best.generation})")
    print(f"evaluator note   : {best.insight}")

    best_source = splice_block(_seed_source(), best.block)
    out_path = os.path.join(REPORT_DIR, "best_program.py")
    with open(out_path, "w") as fh:
        fh.write(best_source)
    print(f"\nbest program written to {out_path}")

    _write_plots(db, best, seed)
    return 0


def _write_plots(db: ProgramDatabase, best: Program, seed: Program | None) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, Rectangle

    history = db.best_per_generation()
    fig, (ax_hist, ax_pack) = plt.subplots(1, 2, figsize=(12, 5))

    ax_hist.plot(range(len(history)), history, marker="o", color="#4285F4")
    ax_hist.set_xlabel("generation")
    ax_hist.set_ylabel("best sum of radii")
    ax_hist.set_title("Best-so-far score")
    ax_hist.grid(alpha=0.3)
    scatter_gens = [p.generation for p in db.valid()]
    ax_hist.scatter(scatter_gens, [p.score for p in db.valid()],
                    s=12, alpha=0.35, color="#888", label="candidates", zorder=1)
    ax_hist.legend(loc="lower right")

    ax_pack.add_patch(Rectangle((0, 0), 1, 1, fill=False, lw=1.5))
    for (x, y), r in zip(best.centers, best.radii):
        ax_pack.add_patch(Circle((x, y), r, alpha=0.55,
                                 facecolor="#4285F4", edgecolor="#1a3d7c"))
    ax_pack.set_xlim(-0.02, 1.02)
    ax_pack.set_ylim(-0.02, 1.02)
    ax_pack.set_aspect("equal")
    ax_pack.set_title(f"Best packing — sum of radii {best.score:.4f}")

    fig.tight_layout()
    path = os.path.join(REPORT_DIR, "evolution.png")
    fig.savefig(path, dpi=140)
    print(f"plots written to {path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p_init = sub.add_parser("init", help="evaluate the seed and create the population")
    p_init.add_argument("--islands", type=int, default=3)
    p_init.set_defaults(func=cmd_init)

    for name, func, extra in (("step", cmd_step, False), ("run", cmd_run, True)):
        p = sub.add_parser(name)
        p.add_argument("--generator", choices=["anthropic", "manual"], default="anthropic")
        p.add_argument("--children", type=int, default=4)
        if extra:
            p.add_argument("--generations", type=int, default=5)
        p.set_defaults(func=func)

    p_report = sub.add_parser("report", help="summarise the run and write plots")
    p_report.set_defaults(func=cmd_report)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
