"""
AlphaZero Hearts — entry-point training script.

Quick start
───────────
  # Train from scratch (default config)
  python alphazero_train.py

  # Phase-1 supervised pre-training then Phase-2 AlphaZero
  python alphazero_train.py --pretrain-games 2000 --iterations 200

  # Customise key hyperparameters
  python alphazero_train.py \\
      --iterations 200 \\
      --games 50 \\
      --worlds 15 \\
      --depth 4 \\
      --hidden 256 \\
      --blocks 4 \\
      --checkpoint-dir my_run

  # Resume from a saved best checkpoint
  python alphazero_train.py \\
      --resume alphazero_checkpoints/best.pt \\
      --iterations 100

Architecture recap
──────────────────
  Input (5088 dims — raw OpenSpiel observation)
      │
  Linear projection → LayerNorm → GELU
      │
  N × Residual block  [pre-norm, hidden_dim × 2 inner projection]
      │
  ┌───┴───────────────────┐
  Policy head             Value head
  (52 card logits)        (sigmoid → ×26 → predicted points)

Training phases
───────────────
  Phase 1 (optional, --pretrain-games N > 0):
    Supervised learning on (obs_5088, evaluate_hand_score) pairs.
    Value-only MSE loss; policy targets are zeros.

  Phase 2 (AlphaZero loop):
    policy → MCTS visit distribution (cross-entropy)
    value  → actual game score, MSE, normalized to [0,1]

The heuristic ``evaluate_hand`` is used in two places:
  • As the ground-truth label source for Phase-1 pre-training.
  • As the fallback depth-cutoff heuristic inside WorldSolver when
    the NN is not yet available (initial random weights are poor).
"""
from __future__ import annotations

import argparse
import logging
import sys

# ── Logging setup ─────────────────────────────────────────────────────────
logging.basicConfig(
    level  = logging.INFO,
    format = "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt= "%H:%M:%S",
    stream = sys.stdout,
)
logger = logging.getLogger("alphazero_train")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="AlphaZero-style Hearts self-play training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Run control ───────────────────────────────────────────────────────
    p.add_argument("--iterations", type=int, default=100,
                   help="Number of AlphaZero pipeline iterations to run")
    p.add_argument("--resume", type=str, default=None,
                   help="Path to a .pt checkpoint to resume from")
    p.add_argument("--checkpoint-dir", type=str, default="alphazero_checkpoints",
                   help="Directory for saving checkpoints and logs")
    p.add_argument("--seed", type=int, default=None,
                   help="Global random seed (optional)")

    # ── Phase-1 pre-training ──────────────────────────────────────────────
    p.add_argument("--pretrain-games", type=int, default=0,
                   help="Games to generate for supervised pre-training on "
                        "evaluate_hand labels (0 = skip pre-training)")
    p.add_argument("--pretrain-epochs", type=int, default=5,
                   help="Epochs over the pre-training dataset")
    p.add_argument("--pretrain-lr", type=float, default=1e-3,
                   help="Learning rate for pre-training")
    p.add_argument("--pretrain-batch", type=int, default=512,
                   help="Batch size for pre-training")
    p.add_argument("--pretrain-dropout", type=float, default=0.3,
                   help="Trick-history dropout probability during pre-training "
                        "(teaches the NN to work with partial observations)")
    p.add_argument("--pretrain-players", type=str, default="heuristic",
                   choices=["heuristic", "random", "conservative"],
                   help="Player strategy for pre-training data generation. "
                        "'heuristic' runs DMCTS with evaluate_hand (best quality, "
                        "slower); 'random'/'conservative' are faster simple bots")
    p.add_argument("--pretrain-mcts-worlds", type=int, default=10,
                   help="DMCTS worlds per decision during heuristic pre-training")
    p.add_argument("--pretrain-mcts-depth", type=int, default=3,
                   help="Alpha-beta depth per world during heuristic pre-training")
    p.add_argument("--pretrain-mcts-time-ms", type=float, default=200.0,
                   help="Per-decision time cap (ms) during heuristic pre-training")

    # ── Network architecture ──────────────────────────────────────────────
    p.add_argument("--hidden", type=int, default=256,
                   help="Hidden layer width")
    p.add_argument("--blocks", type=int, default=4,
                   help="Number of residual blocks")
    p.add_argument("--dropout", type=float, default=0.1,
                   help="Dropout rate in residual blocks")

    # ── Self-play ─────────────────────────────────────────────────────────
    p.add_argument("--games", type=int, default=100,
                   help="Self-play games per iteration")
    p.add_argument("--worlds", type=int, default=20,
                   help="Determinized worlds per DMCTS decision")
    p.add_argument("--depth", type=int, default=4,
                   help="Alpha-beta search depth")
    p.add_argument("--time-ms", type=float, default=500.0,
                   help="Per-decision time budget (ms)")
    p.add_argument("--temp-tricks", type=int, default=10,
                   help="Apply temperature sampling for the first N tricks")

    # ── Training ──────────────────────────────────────────────────────────
    p.add_argument("--lr", type=float, default=1e-3,
                   help="Adam learning rate")
    p.add_argument("--wd", type=float, default=1e-4,
                   help="Adam weight decay (L2 regularisation)")
    p.add_argument("--batch-size", type=int, default=256,
                   help="Mini-batch size for training")
    p.add_argument("--n-batches", type=int, default=200,
                   help="Gradient steps per training epoch")
    p.add_argument("--buffer-size", type=int, default=100_000,
                   help="Maximum replay buffer capacity")

    # ── Evaluation ────────────────────────────────────────────────────────
    p.add_argument("--eval-games", type=int, default=100,
                   help="Head-to-head games per pit evaluation")
    p.add_argument("--pit-criterion", type=str, default="points",
                   choices=["points", "win_rate"],
                   help="How to accept the candidate after pit: "
                        "'points' = lower average points than incumbent; "
                        "'win_rate' = seat-pair win fraction ≥ --win-threshold")
    p.add_argument("--pit-points-margin", type=float, default=0.0,
                   help="When --pit-criterion=points: accept if "
                        "new_avg < old_avg − this margin (points)")
    p.add_argument("--win-threshold", type=float, default=0.55,
                   help="When --pit-criterion=win_rate: min seat win fraction (0–1)")
    p.add_argument("--baseline-every", type=int, default=5,
                   help="Evaluate vs heuristic every N iterations (0=never)")

    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Optional global seed
    if args.seed is not None:
        import random
        import numpy as np
        import torch
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        logger.info("Global seed set to %d", args.seed)

    # Build config dict from CLI args
    config = {
        "hidden_dim":              args.hidden,
        "n_blocks":                args.blocks,
        "dropout":                 args.dropout,
        "games_per_iter":          args.games,
        "n_worlds":                args.worlds,
        "max_depth":               args.depth,
        "time_limit_ms":           args.time_ms,
        "temperature_tricks":      args.temp_tricks,
        "buffer_size":             args.buffer_size,
        "batch_size":              args.batch_size,
        "n_batches":               args.n_batches,
        "lr":                      args.lr,
        "weight_decay":            args.wd,
        "eval_games":              args.eval_games,
        "pit_criterion":           args.pit_criterion,
        "pit_points_margin":       args.pit_points_margin,
        "win_threshold":           args.win_threshold,
        "baseline_every":          args.baseline_every,
        "checkpoint_dir":          args.checkpoint_dir,
        # Phase-1 pre-training
        "pretrain_games":          args.pretrain_games,
        "pretrain_epochs":         args.pretrain_epochs,
        "pretrain_lr":             args.pretrain_lr,
        "pretrain_batch_size":     args.pretrain_batch,
        "pretrain_trick_dropout":  args.pretrain_dropout,
        "pretrain_players":        args.pretrain_players,
    }
    # Only override pipeline DEFAULT_CONFIG when the user passes these flags.
    # (Previously we always passed CLI defaults here, so editing pipeline.py
    #  pretrain_mcts_* had no effect.)
    if args.pretrain_mcts_worlds is not None:
        config["pretrain_mcts_worlds"] = args.pretrain_mcts_worlds
    if args.pretrain_mcts_depth is not None:
        config["pretrain_mcts_depth"] = args.pretrain_mcts_depth
    if args.pretrain_mcts_time_ms is not None:
        config["pretrain_mcts_time_ms"] = args.pretrain_mcts_time_ms

    from hearts_ai.alphazero import AlphaZeroPipeline

    if args.resume:
        logger.info("Resuming from checkpoint: %s", args.resume)
        pipeline = AlphaZeroPipeline.from_checkpoint(args.resume, config=config)
    else:
        pipeline = AlphaZeroPipeline(config=config)

    pipeline.print_config()

    # ── Phase-1: supervised pre-training (optional) ───────────────────────
    if args.pretrain_games > 0:
        try:
            pipeline.pretrain(
                n_games  = args.pretrain_games,
                n_epochs = args.pretrain_epochs,
            )
        except KeyboardInterrupt:
            print("\nPre-training interrupted — saving current network …")
            pipeline._save_checkpoint("pretrain_interrupted")
            print(f"Saved to {args.checkpoint_dir}/pretrain_interrupted.pt")
            sys.exit(0)

    # ── Phase-2: AlphaZero self-play loop ─────────────────────────────────
    try:
        pipeline.run(n_iterations=args.iterations)
    except KeyboardInterrupt:
        print("\nInterrupted — saving current best network …")
        pipeline._save_checkpoint("interrupted")
        print(f"Saved to {args.checkpoint_dir}/interrupted.pt")

    print("\nTraining complete.")
    pipeline.print_history()


if __name__ == "__main__":
    main()
