"""
AlphaZeroPipeline: top-level orchestrator for self-play → train → pit → repeat.

One iteration of the loop:

  1. Self-play  — run ``games_per_iter`` complete games with the current best
                  network guiding DMCTS.  Collect (state, policy, value) tuples.

  2. Fill buffer — add all examples from step 1 to the rolling replay buffer.

  3. Train       — draw random mini-batches from the buffer and update the
                   candidate network's weights.

  4. Pit         — play ``eval_games`` head-to-head games between the candidate
                   (``net``) and the incumbent (``best_net``).  By default the
                   candidate is **accepted** if its average points taken is
                   lower than the old net's (see ``pit_criterion``).  Optionally
                   use seat win-rate instead (``pit_criterion="win_rate"``).

  5. Checkpoint  — save the candidate (always) and the best (when updated).

  6. Repeat.

Configuration
─────────────
Pass a dict to ``AlphaZeroPipeline.__init__`` to override any defaults.
All keys are documented in ``DEFAULT_CONFIG`` below.
"""
from __future__ import annotations

import copy
import json
import logging
import os
import time
from typing import Any, Dict, Optional

from .evaluator import evaluate_vs_heuristic, pit_networks
from .net import HeartsNet
from .replay_buffer import ReplayBuffer
from .self_play import run_self_play_game
from .trainer import Trainer

logger = logging.getLogger(__name__)

# ── Default hyperparameters ────────────────────────────────────────────────

DEFAULT_CONFIG: Dict[str, Any] = {
    # ── Network architecture ──────────────────────────────────────────────
    "hidden_dim":           256,     # Width of hidden layers
    "n_blocks":             4,       # Number of residual blocks in trunk
    "dropout":              0.1,

    # ── Self-play ─────────────────────────────────────────────────────────
    "games_per_iter":       100,     # Games generated per pipeline iteration
    "n_worlds":             20,      # Determinized worlds per DMCTS decision
    "max_depth":            4,       # Alpha-beta search depth
    "time_limit_ms":        500.0,   # Per-decision time budget (ms)
    "temperature_tricks":   10,      # Trick index after which to switch to greedy
    "dirichlet_alpha":      0.3,     # Dirichlet noise α (exploration)
    "dirichlet_eps":        0.25,    # Noise mixing weight
    "nn_blend":             0.25,    # NN policy weight in DMCTS/NN blend (0=pure DMCTS)

    # ── Replay buffer ─────────────────────────────────────────────────────
    "buffer_size":          100_000, # Max examples retained

    # ── Training ──────────────────────────────────────────────────────────
    "batch_size":           256,
    "n_batches":            200,     # Gradient steps per iteration
    "lr":                   1e-3,
    "weight_decay":         1e-4,
    "policy_weight":        1.0,
    "value_weight":         1.0,
    "grad_clip":            1.0,

    # ── Evaluation (pit) ──────────────────────────────────────────────────
    "eval_games":           100,      # Head-to-head games per pit
    "eval_worlds":          20,      # Worlds per decision during pit
    "eval_max_depth":       3,
    "eval_time_ms":         400.0,
    # Pit acceptance: "points" = lower avg points wins; "win_rate" = seat pairs
    "pit_criterion":        "points",
    "pit_points_margin":    0.0,     # Accept new if new_avg < old_avg - margin
    "win_threshold":        0.55,    # Used only when pit_criterion == "win_rate"

    # ── Heuristic baseline ────────────────────────────────────────────────
    "baseline_every":       5,       # Evaluate vs heuristic every N iterations
    "baseline_games":       100,

    # ── Phase-1 pre-training (supervised on evaluate_hand labels) ─────────
    # Set pretrain_games > 0 to run pre-training before the AlphaZero loop.
    "pretrain_games":         0,       # Games for supervised pre-training (0 = skip)
    "pretrain_epochs":        5,       # Epochs over the generated pre-training data
    "pretrain_batch_size":    512,
    "pretrain_lr":            1e-3,
    "pretrain_trick_dropout": 0.0,     # Prob of zeroing trick history per example
    "pretrain_players":       "heuristic",  # "heuristic" (MCTS), "random", "conservative"
    "pretrain_mcts_worlds":   10,      # DMCTS worlds per decision (heuristic mode)
    "pretrain_mcts_depth":    3,       # Alpha-beta depth per world (heuristic mode)
    "pretrain_mcts_time_ms":  200.0,   # Per-decision time cap ms (heuristic mode)

    # ── Checkpointing ─────────────────────────────────────────────────────
    "checkpoint_dir":       "alphazero_checkpoints",
    "save_every":           1,       # Save candidate checkpoint every N iters
    "log_every":            1,       # Print summary every N iters
}


# ── Pipeline ───────────────────────────────────────────────────────────────

class AlphaZeroPipeline:
    """
    Full AlphaZero-style training loop for the Hearts heuristic network.

    **Weights:** ``self.net`` is the candidate updated by the ``Trainer`` each
    iteration.  ``self.best_net`` is the incumbent used for self-play (data
    should track the strongest policy).  After each iteration, if pit **accepts**,
    ``best_net`` is replaced with a copy of the trained ``net`` and both point
    at the same weights; the next iteration continues SGD on that ``net`` in
    place.  If pit **rejects**, ``net`` is reset to a copy of ``best_net`` and
    the optimizer is recreated, so the next iteration trains **from the last
    accepted checkpoint**, not from the rejected candidate.

    Usage::

        pipeline = AlphaZeroPipeline({"games_per_iter": 50, "n_blocks": 2})
        pipeline.run(n_iterations=200)

    Or resume from a checkpoint::

        pipeline = AlphaZeroPipeline.from_checkpoint("alphazero_checkpoints/best.pt")
        pipeline.run(n_iterations=100)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = {**DEFAULT_CONFIG, **(config or {})}
        cfg         = self.config

        os.makedirs(cfg["checkpoint_dir"], exist_ok=True)

        # ── Networks ──────────────────────────────────────────────────────
        self.net = HeartsNet(
            hidden_dim = cfg["hidden_dim"],
            n_blocks   = cfg["n_blocks"],
            dropout    = cfg["dropout"],
        )
        # ``best_net`` is the incumbent; ``net`` is the candidate being trained.
        self.best_net = copy.deepcopy(self.net)

        # ── Infrastructure ────────────────────────────────────────────────
        self.buffer  = ReplayBuffer(max_size=cfg["buffer_size"])
        self.trainer = Trainer(
            net           = self.net,
            lr            = cfg["lr"],
            weight_decay  = cfg["weight_decay"],
            policy_weight = cfg["policy_weight"],
            value_weight  = cfg["value_weight"],
            grad_clip     = cfg["grad_clip"],
        )

        self.iteration = 0

        # ── Metrics log ───────────────────────────────────────────────────
        self.history: list = []

        self._save_config()

    # ── Config persistence ────────────────────────────────────────────────

    def _save_config(self) -> None:
        path = os.path.join(self.config["checkpoint_dir"], "config.json")
        with open(path, "w") as f:
            json.dump(self.config, f, indent=2)

    # ── Phase-1 supervised pre-training ──────────────────────────────────

    def pretrain(
        self,
        n_games: Optional[int]   = None,
        n_epochs: Optional[int]  = None,
        mcts_worlds: Optional[int] = None,
        mcts_depth: Optional[int] = None,
        mcts_time_ms: Optional[float] = None,
    ) -> None:
        """
        Phase-1 supervised pre-training on evaluate_hand() heuristic labels.

        Generates ``n_games`` Hearts games (all random players), labels each
        play-phase position with the hand-coded ``evaluate_hand`` score, and
        trains the network with value-only MSE loss.

        After pre-training, ``best_net`` is updated to match the trained
        ``net`` so the AlphaZero loop starts from the pre-trained weights.

        Args:
            n_games:  Number of games to generate (overrides
                      ``config["pretrain_games"]``).  0 is a no-op.
            n_epochs: Training epochs (overrides ``config["pretrain_epochs"]``).
            mcts_worlds, mcts_depth, mcts_time_ms: When ``pretrain_players`` is
                ``"heuristic"``, these override the corresponding
                ``pretrain_mcts_*`` entries in ``self.config`` (otherwise ignored).
        """
        from .pretrain import pretrain as _run_pretrain

        cfg     = self.config
        n_games  = n_games  if n_games  is not None else cfg["pretrain_games"]
        n_epochs = n_epochs if n_epochs is not None else cfg["pretrain_epochs"]
        pm_w = mcts_worlds if mcts_worlds is not None else cfg["pretrain_mcts_worlds"]
        pm_d = mcts_depth if mcts_depth is not None else cfg["pretrain_mcts_depth"]
        pm_t = mcts_time_ms if mcts_time_ms is not None else cfg["pretrain_mcts_time_ms"]

        if n_games <= 0:
            logger.info("pretrain: n_games=0, skipping.")
            return

        print(f"\n{'='*60}")
        print(f"[Pre-training]  Phase 1 — supervised heuristic distillation")
        players = cfg["pretrain_players"]
        print(f"  games={n_games}  epochs={n_epochs}  "
              f"batch={cfg['pretrain_batch_size']}  lr={cfg['pretrain_lr']:.1e}")
        if players == "heuristic":
            print(f"  players=heuristic MCTS  "
                  f"worlds={pm_w}  "
                  f"depth={pm_d}  "
                  f"time_ms={pm_t:.0f}")
        else:
            print(f"  players={players}  "
                  f"trick_history_dropout={cfg['pretrain_trick_dropout']}")
        print(f"{'='*60}")

        t0 = time.perf_counter()
        history = _run_pretrain(
            net                   = self.net,
            n_games               = n_games,
            n_epochs              = n_epochs,
            batch_size            = cfg["pretrain_batch_size"],
            lr                    = cfg["pretrain_lr"],
            trick_history_dropout = cfg["pretrain_trick_dropout"],
            players               = cfg["pretrain_players"],
            mcts_worlds           = pm_w,
            mcts_depth            = pm_d,
            mcts_time_ms          = pm_t,
            verbose               = True,
        )
        elapsed = time.perf_counter() - t0

        # Sync best_net to the pre-trained weights so AlphaZero starts warm
        self.best_net = copy.deepcopy(self.net)
        # Re-attach trainer to the freshly pre-trained network
        self.trainer = Trainer(
            net           = self.net,
            lr            = cfg["lr"],
            weight_decay  = cfg["weight_decay"],
            policy_weight = cfg["policy_weight"],
            value_weight  = cfg["value_weight"],
            grad_clip     = cfg["grad_clip"],
        )

        # Save pre-trained checkpoint
        self._save_checkpoint("pretrain")

        final_loss = history["value_loss"][-1] if history["value_loss"] else float("nan")
        print(f"[Pre-training]  Done — {elapsed:.1f}s  final_value_loss={final_loss:.4f}\n")

    # ── Main loop ─────────────────────────────────────────────────────────

    def run(self, n_iterations: int = 100) -> None:
        """
        Run the pipeline for ``n_iterations`` iterations starting from
        ``self.iteration``.
        """
        logger.info(
            "Starting AlphaZero pipeline: %d iterations, "
            "%d games/iter, %d worlds, depth=%d",
            n_iterations,
            self.config["games_per_iter"],
            self.config["n_worlds"],
            self.config["max_depth"],
        )
        for _ in range(n_iterations):
            self.run_iteration()

    def run_iteration(self) -> Dict[str, Any]:
        """
        Execute one full pipeline iteration and return a summary dict.
        """
        cfg   = self.config
        it    = self.iteration
        t0    = time.perf_counter()
        stats: Dict[str, Any] = {"iteration": it}

        # ── 1. Self-play ───────────────────────────────────────────────────
        print(f"\n{'='*60}")
        print(f"[Iter {it}]  Self-play  ({cfg['games_per_iter']} games) …")
        sp_start    = time.perf_counter()
        total_examples = 0

        # One OpenSpiel Environment for all self-play games this iteration
        # (avoids absl "Using game instance: …" on every Environment.__init__).
        osp_env = None
        try:
            import pyspiel
            from open_spiel.python.rl_environment import Environment as _OSP
            osp_env = _OSP(pyspiel.load_game("hearts"), players=4)
        except ImportError:
            pass

        for g in range(cfg["games_per_iter"]):
            examples = run_self_play_game(
                net               = self.best_net,
                n_worlds          = cfg["n_worlds"],
                max_depth         = cfg["max_depth"],
                time_limit_ms     = cfg["time_limit_ms"],
                temperature_tricks= cfg["temperature_tricks"],
                dirichlet_alpha   = cfg["dirichlet_alpha"],
                dirichlet_eps     = cfg["dirichlet_eps"],
                nn_blend          = cfg["nn_blend"],
                osp_env           = osp_env,
            )
            self.buffer.extend(examples)
            total_examples += len(examples)

            if (g + 1) % max(cfg["games_per_iter"] // 5, 1) == 0:
                print(
                    f"  game {g+1}/{cfg['games_per_iter']} | "
                    f"buffer={len(self.buffer):,} | "
                    f"elapsed={time.perf_counter()-sp_start:.1f}s"
                )

        stats["games"]          = cfg["games_per_iter"]
        stats["new_examples"]   = total_examples
        stats["buffer_size"]    = len(self.buffer)
        stats["self_play_time"] = time.perf_counter() - sp_start
        print(
            f"  Self-play done: {total_examples} examples | "
            f"buffer={len(self.buffer):,}"
        )

        # ── 2. Train ───────────────────────────────────────────────────────
        print(f"[Iter {it}]  Training  ({cfg['n_batches']} batches) …")
        train_start = time.perf_counter()

        self.trainer.set_cosine_schedule(total_steps=cfg["n_batches"])
        train_stats = self.trainer.train_epoch(
            buffer     = self.buffer,
            batch_size = cfg["batch_size"],
            n_batches  = cfg["n_batches"],
        )

        stats["train_loss"]        = train_stats["loss"]
        stats["train_policy_loss"] = train_stats["policy_loss"]
        stats["train_value_loss"]  = train_stats["value_loss"]
        stats["train_lr"]          = train_stats["lr"]
        stats["train_time"]        = time.perf_counter() - train_start
        print(
            f"  loss={train_stats['loss']:.4f}  "
            f"(policy={train_stats['policy_loss']:.4f}, "
            f"value={train_stats['value_loss']:.4f})  "
            f"lr={train_stats['lr']:.2e}"
        )

        # ── 3. Pit ─────────────────────────────────────────────────────────
        print(f"[Iter {it}]  Pit  ({cfg['eval_games']} games) …")
        pit_start = time.perf_counter()

        new_avg, old_avg, win_rate, accepted = pit_networks(
            net_new          = self.net,
            net_old          = self.best_net,
            n_games          = cfg["eval_games"],
            n_worlds         = cfg["eval_worlds"],
            max_depth        = cfg["eval_max_depth"],
            time_limit_ms    = cfg["eval_time_ms"],
            pit_criterion    = cfg["pit_criterion"],
            win_threshold    = cfg["win_threshold"],
            pit_points_margin= cfg["pit_points_margin"],
        )

        stats["new_avg"]   = new_avg
        stats["old_avg"]   = old_avg
        stats["win_rate"]  = win_rate
        stats["accepted"]  = accepted
        stats["pit_time"]  = time.perf_counter() - pit_start

        verdict = "ACCEPTED ✓" if accepted else "rejected"
        pc = cfg["pit_criterion"]
        if pc == "points":
            margin = cfg["pit_points_margin"]
            crit = f"points (new < old − {margin:.2f})"
        else:
            crit = f"win_rate ≥ {cfg['win_threshold']:.0%}"
        print(
            f"  new={new_avg:.2f}pts  old={old_avg:.2f}pts  "
            f"seat_win%={win_rate:.1%}  [{crit}]  → {verdict}"
        )

        if accepted:
            self.best_net = copy.deepcopy(self.net)
            self._save_checkpoint("best")
        else:
            # Reset candidate network to best to avoid drifting too far
            self.net = copy.deepcopy(self.best_net)
            # Re-attach trainer to the new (reset) candidate
            self.trainer = Trainer(
                net           = self.net,
                lr            = cfg["lr"],
                weight_decay  = cfg["weight_decay"],
                policy_weight = cfg["policy_weight"],
                value_weight  = cfg["value_weight"],
                grad_clip     = cfg["grad_clip"],
            )

        # ── 4. Heuristic baseline (periodic) ──────────────────────────────
        if cfg["baseline_every"] > 0 and (it + 1) % cfg["baseline_every"] == 0:
            print(f"[Iter {it}]  Heuristic baseline  ({cfg['baseline_games']} games) …")
            net_avg, heuristic_avg = evaluate_vs_heuristic(
                net          = self.best_net,
                n_games      = cfg["baseline_games"],
                n_worlds     = cfg["eval_worlds"],
                max_depth    = cfg["eval_max_depth"],
                time_limit_ms= cfg["eval_time_ms"],
            )
            stats["vs_heuristic_net"]       = net_avg
            stats["vs_heuristic_baseline"]  = heuristic_avg
            delta = heuristic_avg - net_avg
            print(
                f"  NN={net_avg:.2f}pts  heuristic={heuristic_avg:.2f}pts  "
                f"ΔNN={delta:+.2f} ({'better' if delta > 0 else 'worse'})"
            )

        # ── 5. Checkpoint ──────────────────────────────────────────────────
        if it % cfg["save_every"] == 0:
            self._save_checkpoint(f"iter_{it:04d}")

        stats["total_time"] = time.perf_counter() - t0
        self.history.append(stats)
        self._save_history()

        print(
            f"[Iter {it}]  Done — {stats['total_time']:.1f}s total\n"
        )

        self.iteration += 1
        return stats

    # ── Checkpointing ─────────────────────────────────────────────────────

    def _save_checkpoint(self, tag: str) -> None:
        path = os.path.join(self.config["checkpoint_dir"], f"{tag}.pt")
        self.best_net.save(path)
        logger.debug("Saved checkpoint: %s", path)

    def _save_history(self) -> None:
        path = os.path.join(self.config["checkpoint_dir"], "history.json")
        # Convert floats/bools to JSON-serializable types
        serializable = []
        for row in self.history:
            serializable.append({
                k: (bool(v) if isinstance(v, bool) else
                    float(v) if isinstance(v, float) else
                    int(v)   if isinstance(v, int)   else v)
                for k, v in row.items()
            })
        with open(path, "w") as f:
            json.dump(serializable, f, indent=2)

    # ── Resuming ──────────────────────────────────────────────────────────

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        config: Optional[Dict[str, Any]] = None,
    ) -> "AlphaZeroPipeline":
        """
        Instantiate a pipeline and load weights from a previously saved checkpoint.

        Args:
            checkpoint_path: Path to a ``.pt`` file saved by ``HeartsNet.save``.
            config:          Optional config overrides (merged over defaults).

        Returns:
            AlphaZeroPipeline with networks loaded from the checkpoint.
        """
        pipeline = cls(config=config)
        pipeline.net      = HeartsNet.load(checkpoint_path)
        pipeline.best_net = copy.deepcopy(pipeline.net)
        # Re-attach trainer to the loaded network
        pipeline.trainer = Trainer(
            net           = pipeline.net,
            lr            = pipeline.config["lr"],
            weight_decay  = pipeline.config["weight_decay"],
            policy_weight = pipeline.config["policy_weight"],
            value_weight  = pipeline.config["value_weight"],
            grad_clip     = pipeline.config["grad_clip"],
        )
        logger.info("Loaded checkpoint from %s", checkpoint_path)
        return pipeline

    # ── Introspection ─────────────────────────────────────────────────────

    def print_config(self) -> None:
        """Pretty-print the active configuration."""
        print("\nAlphaZero Configuration")
        print("─" * 40)
        for k, v in self.config.items():
            print(f"  {k:<28} {v}")
        print()

    def print_history(self, last_n: int = 10) -> None:
        """Print the last N iteration summaries."""
        rows = self.history[-last_n:]
        if not rows:
            print("No history yet.")
            return
        print(f"\n{'Iter':>4}  {'loss':>7}  {'new':>6}  {'old':>6}  {'win%':>6}  {'acc':>5}")
        print("─" * 45)
        for r in rows:
            print(
                f"{r.get('iteration',0):>4}  "
                f"{r.get('train_loss', 0):>7.4f}  "
                f"{r.get('new_avg', 0):>6.2f}  "
                f"{r.get('old_avg', 0):>6.2f}  "
                f"{r.get('win_rate', 0):>6.1%}  "
                f"{'yes' if r.get('accepted') else 'no':>5}"
            )
        print()
