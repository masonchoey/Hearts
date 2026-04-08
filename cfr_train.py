"""
CFR Hearts — entry-point training script.

Quick start
───────────
  # Train from scratch for 10 000 iterations, checkpoint every 1 000
  python cfr_train.py

  # Longer run with more frequent evaluation
  python cfr_train.py \\
      --iterations 100000 \\
      --checkpoint-every 5000 \\
      --eval-every 10000 \\
      --eval-games 200

  # Resume from a checkpoint
  python cfr_train.py \\
      --resume cfr_checkpoints/cfr_0010000.pkl \\
      --iterations 50000

Algorithm recap
───────────────
Outcome Sampling Monte Carlo CFR (OS-MCCFR):

  Each iteration samples one complete outcome episode per player (the
  "update player").  All players' actions — including the update player's
  — are sampled, so each iteration is O(game_length) ≈ O(52) operations.
  Counterfactual regrets are computed using importance-sampling corrections.

  ES-MCCFR was evaluated but rejected: it explores all of the update
  player's actions at each of their decision nodes, giving O(13!) branching
  for a player holding a full hand of 13 cards.

  The *average* strategy (accumulated alongside regrets) converges to an
  approximate Nash equilibrium as iterations → ∞.

Checkpointing
─────────────
Checkpoints are plain pickle files containing the infostate table (regrets
+ average strategy), the iteration count, and the solver variant.  They can
be loaded with:

    from hearts_ai.cfr import HeartsCFRSolver, CFRAgent
    solver = HeartsCFRSolver.from_checkpoint("cfr_checkpoints/cfr_0010000.pkl")
    agent  = CFRAgent(solver)
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger("cfr_train")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="ES-MCCFR Hearts training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Run control
    p.add_argument("--iterations", type=int, default=10_000,
                   help="Total ES-MCCFR iterations to run")
    p.add_argument("--resume", type=str, default=None,
                   help="Path to a .pkl checkpoint to resume from")
    p.add_argument("--checkpoint-dir", type=str, default="cfr_checkpoints",
                   help="Directory for saving checkpoints and the history log")
    p.add_argument("--checkpoint-every", type=int, default=1_000,
                   help="Save a checkpoint every N iterations (0 = never)")
    p.add_argument("--log-every", type=int, default=100,
                   help="Log a progress line every N iterations")
    p.add_argument("--seed", type=int, default=None,
                   help="Global random seed (for evaluation reproducibility)")

    # (OS-MCCFR uses a fixed ε=0.6 exploration factor; no tunable averaging type)

    # Evaluation
    p.add_argument("--eval-every", type=int, default=0,
                   help="Evaluate against heuristic every N iterations (0 = never)")
    p.add_argument("--eval-games", type=int, default=100,
                   help="Number of games per evaluation run")
    p.add_argument("--eval-vs-random", action="store_true", default=False,
                   help="Also evaluate vs random opponents (in addition to heuristic)")

    return p.parse_args()


# ── History helpers ───────────────────────────────────────────────────────────

def _load_history(path: str) -> list:
    if os.path.exists(path):
        with open(path) as fh:
            return json.load(fh)
    return []


def _save_history(path: str, history: list) -> None:
    with open(path, "w") as fh:
        json.dump(history, fh, indent=2)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    if args.seed is not None:
        import random
        random.seed(args.seed)
        logger.info("Global random seed set to %d", args.seed)

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    history_path = os.path.join(args.checkpoint_dir, "history.json")
    history = _load_history(history_path)

    # ── Build or restore solver ───────────────────────────────────────────────
    from hearts_ai.cfr import CFRAgent, HeartsCFRSolver
    from hearts_ai.cfr.evaluator import evaluate_vs_heuristic, evaluate_vs_random

    if args.resume:
        logger.info("Resuming from checkpoint: %s", args.resume)
        solver = HeartsCFRSolver.from_checkpoint(args.resume)
    else:
        logger.info("Creating new HeartsCFRSolver (OS-MCCFR)")
        solver = HeartsCFRSolver()

    logger.info(
        "Solver state: %d iterations done, %d infostates in table",
        solver.iterations_done,
        solver.n_infostates,
    )
    logger.info(
        "Training plan: %d additional iterations → checkpoint every %d",
        args.iterations,
        args.checkpoint_every,
    )

    # ── Training loop ─────────────────────────────────────────────────────────
    remaining = args.iterations
    eval_countdown = args.eval_every  # iterations until next eval (0 = never)

    t_start = time.perf_counter()

    while remaining > 0:
        # How many iterations to run before the next checkpoint / eval?
        chunk = remaining
        if args.checkpoint_every > 0:
            chunk = min(chunk, args.checkpoint_every)
        if args.eval_every > 0:
            chunk = min(chunk, eval_countdown if eval_countdown > 0 else args.eval_every)

        solver.train(
            chunk,
            checkpoint_every=0,  # we handle checkpointing ourselves
            log_every=args.log_every,
        )
        remaining -= chunk

        # ── Checkpoint ───────────────────────────────────────────────────────
        if args.checkpoint_every > 0 and solver.iterations_done % args.checkpoint_every == 0:
            ckpt = os.path.join(
                args.checkpoint_dir, f"cfr_{solver.iterations_done:07d}.pkl"
            )
            solver.save(ckpt)
            logger.info("Checkpoint → %s  (%d infostates)", ckpt, solver.n_infostates)

        # ── Evaluation ───────────────────────────────────────────────────────
        if args.eval_every > 0:
            eval_countdown -= chunk
            if eval_countdown <= 0:
                eval_countdown = args.eval_every
                eval_seed = args.seed  # None is fine; gives non-deterministic evals

                agent = CFRAgent(solver)

                logger.info(
                    "=== Evaluation at iter %d (%d infostates) ===",
                    solver.iterations_done,
                    solver.n_infostates,
                )

                result_h = evaluate_vs_heuristic(
                    agent,
                    n_games=args.eval_games,
                    seed=eval_seed,
                )
                logger.info(
                    "  vs heuristic: CFR=%.2f pts, heuristic=%.2f pts",
                    result_h["cfr_avg"], result_h["opp_avg"],
                )

                entry: dict = {
                    "iteration": solver.iterations_done,
                    "n_infostates": solver.n_infostates,
                    "elapsed_s": round(time.perf_counter() - t_start, 1),
                    "vs_heuristic": {
                        "cfr_avg": result_h["cfr_avg"],
                        "opp_avg": result_h["opp_avg"],
                    },
                }

                if args.eval_vs_random:
                    result_r = evaluate_vs_random(
                        agent,
                        n_games=args.eval_games,
                        seed=eval_seed,
                    )
                    logger.info(
                        "  vs random:    CFR=%.2f pts, random=%.2f pts",
                        result_r["cfr_avg"], result_r["opp_avg"],
                    )
                    entry["vs_random"] = {
                        "cfr_avg": result_r["cfr_avg"],
                        "opp_avg": result_r["opp_avg"],
                    }

                history.append(entry)
                _save_history(history_path, history)

    # ── Final checkpoint ──────────────────────────────────────────────────────
    final_path = os.path.join(args.checkpoint_dir, "cfr_final.pkl")
    solver.save(final_path)
    elapsed = time.perf_counter() - t_start
    logger.info(
        "Training complete: %d iterations in %.1f s | %d infostates | saved → %s",
        solver.iterations_done,
        elapsed,
        solver.n_infostates,
        final_path,
    )

    # ── Final evaluation summary ──────────────────────────────────────────────
    if history:
        print("\n── Training history ──")
        print(f"{'iter':>8}  {'infostates':>12}  {'CFR vs heur':>12}  {'heur avg':>10}")
        for entry in history:
            vh = entry.get("vs_heuristic", {})
            print(
                f"{entry['iteration']:>8d}  "
                f"{entry['n_infostates']:>12d}  "
                f"{vh.get('cfr_avg', float('nan')):>12.2f}  "
                f"{vh.get('opp_avg', float('nan')):>10.2f}"
            )


if __name__ == "__main__":
    main()
