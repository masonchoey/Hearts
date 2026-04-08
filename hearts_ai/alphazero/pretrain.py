"""
Phase-1 pre-training: supervised distillation of the evaluate_hand heuristic.

Pipeline
────────
  1.  Run N Hearts games using OpenSpiel.  Players can be random bots,
      conservative bots, or full MCTS players (``players="heuristic"``).
  2.  At every play-phase decision point, extract:
        obs   = ts.observations["info_state"][cp]          # 5088-dim raw obs
        world = {p: cards_in_hand_from_obs(obs_p) …}      # true hands
        play  = _state_from_obs_and_world(obs, world, cp)  # reconstruct PlayState
        label = evaluate_hand(play.hands[cp], play, cp)    # heuristic score
  3.  Train HeartsNet with **value-only MSE loss** (policy targets are zeros).

Player strategies
─────────────────
  "random"       — all four seats pick uniformly at random.
  "conservative" — avoid point cards, prefer lower cards.
  "heuristic"    — all four seats run a lightweight DMCTS loop using the
                   WorldSolver (evaluate_hand depth cutoffs, no NN).

Using ``players="heuristic"`` produces the highest-quality training data
because the resulting game states are close to the distribution seen during
AlphaZero self-play.  The trade-off is generation speed: each MCTS decision
takes O(n_worlds × minimax_depth) instead of O(1) for bot strategies.

Recommended configuration for heuristic pre-training:
    players="heuristic", mcts_worlds=10, mcts_depth=3

Training label normalization
────────────────────────────
  evaluate_hand() returns a future-point estimate (positive = bad).
  Clamped to [0, 26] so it matches the value head's output scale.
"""
from __future__ import annotations

import random
import time as _time
from typing import Any, Dict, List, Optional, Set

import numpy as np

from ..belief_state import BeliefState
from ..openspiel_utils import (
    NUM_CARDS,
    NUM_PLAYERS,
    OBS_TRICK_HISTORY,
    card_points,
    card_to_suit,
    cards_in_hand_from_obs,
)
from ..starterheartsheuristic import evaluate_hand
from ..world_solver import PlayState, WorldSolver, _state_from_obs_and_world
from .net import HeartsNet
from .replay_buffer import ReplayBuffer, TrainingExample
from .trainer import Trainer


# ── Passing-phase detection ────────────────────────────────────────────────

def _is_passing_obs(obs: np.ndarray) -> bool:
    """Return True if the observation belongs to the card-passing phase."""
    if len(obs) < 160:
        return False
    if obs[0] >= 0.99:          # pass_dir[0] == 1 → "No Pass" round
        return False
    passed   = float(np.sum(obs[56:108]))
    received = float(np.sum(obs[108:160]))
    return passed < 3 or received < 3


# ── Simple bot strategies ──────────────────────────────────────────────────

def _bot_random(legal: List[int], _rng: random.Random) -> int:
    return _rng.choice(legal)


def _bot_conservative(legal: List[int], _rng: random.Random) -> int:
    """Avoid point cards; prefer lower-ranked cards."""
    HEARTS_SUIT     = 2
    QUEEN_OF_SPADES = 43
    options = list(legal)
    non_queen = [a for a in options if a != QUEEN_OF_SPADES]
    if non_queen and QUEEN_OF_SPADES in options:
        options = non_queen
    non_hearts = [a for a in options if card_to_suit(a) != HEARTS_SUIT]
    if non_hearts:
        options = non_hearts
    options.sort()
    return _rng.choice(options[: max(1, len(options) // 2)])


_SIMPLE_BOT_FNS = {
    "random":       _bot_random,
    "conservative": _bot_conservative,
}


# ── MCTS (heuristic) player ────────────────────────────────────────────────

def _mcts_action(
    solver: WorldSolver,
    play: PlayState,
    belief: BeliefState,
    agent_id: int,
    legal: List[int],
    n_worlds: int,
    time_limit_ms: float,
) -> int:
    """
    Pick an action using a DMCTS loop with the heuristic WorldSolver.

    This is the same DMCTS logic used in the main game agent but with
    WorldSolver (evaluate_hand, no NN) so it is self-contained and fast.

    Args:
        solver:       Shared WorldSolver instance (evaluate_hand depth cutoffs).
        play:         Current true PlayState.
        belief:       BeliefState for ``agent_id``.
        agent_id:     Acting player.
        legal:        Legal actions in the real game state.
        n_worlds:     Worlds to sample per decision.
        time_limit_ms: Total time budget in ms.

    Returns:
        The action with the highest vote count.
    """
    if len(legal) == 1:
        return legal[0]

    votes: Dict[int, int] = {a: 0 for a in legal}
    real_set = set(legal)
    deadline = _time.perf_counter() + time_limit_ms / 1000.0

    for i in range(n_worlds):
        if _time.perf_counter() >= deadline:
            break
        world = belief.sample_possible_world()
        if not world:
            continue

        world_play = PlayState(
            hands          = world,
            current_player = play.current_player,
            current_trick  = list(play.current_trick),
            points         = dict(play.points),
            hearts_broken  = play.hearts_broken,
            num_played     = play.num_played,
        )
        remaining  = max(n_worlds - i, 1)
        t_left     = deadline - _time.perf_counter()
        w_deadline = _time.perf_counter() + t_left / remaining

        action, _ = solver.solve_playstate(
            world_play, agent_id,
            real_legal=legal,
            time_deadline=w_deadline,
        )
        if action in votes:
            votes[action] += 1

    if sum(votes.values()) == 0:
        votes[legal[0]] = 1

    return max(votes, key=votes.get)


# ── Data generation ────────────────────────────────────────────────────────

def generate_heuristic_data(
    n_games: int                 = 1000,
    players: str                 = "heuristic",
    mcts_worlds: int             = 10,
    mcts_depth: int              = 3,
    mcts_time_ms: float          = 200.0,
    trick_history_dropout: float = 0.0,
    seed: Optional[int]          = None,
    verbose: bool                = False,
) -> List[TrainingExample]:
    """
    Run ``n_games`` Hearts games and collect (obs_5088, evaluate_hand_score) pairs.

    Args:
        n_games:               Number of complete games to play.
        players:               Player strategy for all four seats:

                               ``"heuristic"`` (default) — full DMCTS using
                               WorldSolver (evaluate_hand depth cutoffs).
                               Generates the highest-quality, most realistic
                               positions; slower than bot strategies.

                               ``"random"``      — uniformly random.
                               ``"conservative"``— avoid point cards.

        mcts_worlds:           Worlds sampled per MCTS decision
                               (only used when ``players="heuristic"``).
        mcts_depth:            Alpha-beta search depth per world
                               (only used when ``players="heuristic"``).
        mcts_time_ms:          Per-decision time cap in ms
                               (only used when ``players="heuristic"``).
        trick_history_dropout: Probability of zeroing OBS_TRICK_HISTORY
                               [356:5088] in each example.  Default 0 since
                               the pre-trained NN is always called with a
                               real OpenSpiel observation (full history
                               available); only set > 0 if you plan to use
                               the NN in a partial-obs context.
        seed:                  Optional RNG seed.
        verbose:               Print progress every 10 % of games.

    Returns:
        List of TrainingExample objects.  ``policy`` is all-zeros
        (value-only supervision during Phase 1).
    """
    try:
        import pyspiel
        from open_spiel.python.rl_environment import Environment as _OSP
    except ImportError as exc:
        raise ImportError(
            "OpenSpiel is required for pre-training data generation.  "
            "Install it with:  pip install open_spiel"
        ) from exc

    use_mcts  = (players == "heuristic")
    simple_fn = _SIMPLE_BOT_FNS.get(players, _bot_random)
    rng       = random.Random(seed)
    game      = pyspiel.load_game("hearts")
    examples: List[TrainingExample] = []
    log_every = max(n_games // 10, 1)

    # One shared solver instance; reused across decisions for speed
    solver = WorldSolver(max_depth=mcts_depth) if use_mcts else None

    # One Environment for all games — OpenSpiel logs "Using game instance: …"
    # from Environment.__init__ on every construction; reusing avoids spam.
    env = _OSP(game, players=NUM_PLAYERS)

    for game_idx in range(n_games):
        if verbose and (game_idx % log_every == 0):
            print(f"  pretrain data: game {game_idx}/{n_games}  "
                  f"examples={len(examples)}")

        ts = env.reset()

        # Per-player BeliefStates — needed by the MCTS player; cheap to build
        # even for bot strategies (just unused).
        beliefs: Dict[int, BeliefState] = {
            p: BeliefState(my_hand=[], num_players=NUM_PLAYERS, agent_id=p)
            for p in range(NUM_PLAYERS)
        }

        while not ts.last():
            cp    = ts.observations["current_player"]
            legal = list(ts.observations["legal_actions"][cp])
            if not legal:
                break

            obs_cp = np.array(ts.observations["info_state"][cp], dtype=np.float32)

            # Passing phase: pick highest-point cards to pass; skip labelling
            if _is_passing_obs(obs_cp):
                action = max(legal, key=card_points)
                ts = env.step([action])
                continue

            # Sync belief state from full observation (needed by MCTS player)
            if use_mcts:
                beliefs[cp].update_from_openspiel_state(ts)

            # True world: each player's current hand from their own obs
            world: Dict[int, Any] = {
                p: set(cards_in_hand_from_obs(
                    np.array(ts.observations["info_state"][p], dtype=np.float32)
                ))
                for p in range(NUM_PLAYERS)
            }

            # Reconstruct PlayState for label derivation and MCTS
            play = _state_from_obs_and_world(obs_cp, world, cp)

            if play is None or not play.hands.get(cp):
                action = simple_fn(legal, rng) if not use_mcts else rng.choice(legal)
                ts = env.step([action])
                continue

            # ── evaluate_hand label ────────────────────────────────────────
            raw_score = evaluate_hand(play.hands[cp], play, cp)
            label_pts = float(max(0.0, min(26.0, raw_score)))

            # ── Optional trick-history dropout ────────────────────────────
            if trick_history_dropout > 0.0 and rng.random() < trick_history_dropout:
                obs_cp = obs_cp.copy()
                obs_cp[OBS_TRICK_HISTORY[0] : OBS_TRICK_HISTORY[1]] = 0.0

            examples.append(TrainingExample(
                features = obs_cp,
                policy   = np.zeros(NUM_CARDS, dtype=np.float32),
                value    = label_pts,
            ))

            # ── Action selection ───────────────────────────────────────────
            if use_mcts:
                action = _mcts_action(
                    solver, play, beliefs[cp], cp, legal,
                    n_worlds=mcts_worlds, time_limit_ms=mcts_time_ms,
                )
            else:
                action = simple_fn(legal, rng)

            ts = env.step([action])

    if verbose:
        print(f"  pretrain data done: {len(examples)} examples from {n_games} games")

    return examples


# ── Training ───────────────────────────────────────────────────────────────

def pretrain(
    net: HeartsNet,
    n_games: int              = 2000,
    n_epochs: int             = 5,
    batch_size: int           = 512,
    lr: float                 = 1e-3,
    weight_decay: float       = 1e-4,
    trick_history_dropout: float = 0.0,
    players: str              = "heuristic",
    mcts_worlds: int          = 10,
    mcts_depth: int           = 3,
    mcts_time_ms: float       = 200.0,
    seed: Optional[int]       = None,
    verbose: bool             = True,
) -> Dict[str, List[float]]:
    """
    Phase-1 supervised pre-training on evaluate_hand() labels.

    Generates ``n_games`` games of data then trains ``net`` for ``n_epochs``
    passes with value-only MSE loss (policy loss disabled).

    Args:
        net:                   HeartsNet to train in-place.
        n_games:               Games to generate (~52 play-phase examples each).
        n_epochs:              Full passes over the generated dataset.
        batch_size:            Mini-batch size per gradient step.
        lr:                    Learning rate for Adam.
        weight_decay:          L2 regularisation.
        trick_history_dropout: Prob of zeroing OBS_TRICK_HISTORY per example.
        players:               ``"heuristic"`` (MCTS, best quality),
                               ``"random"``, or ``"conservative"``.
        mcts_worlds:           DMCTS worlds per decision (heuristic mode only).
        mcts_depth:            Alpha-beta depth per world (heuristic mode only).
        mcts_time_ms:          Per-decision time cap ms (heuristic mode only).
        seed:                  Optional RNG seed.
        verbose:               Print per-epoch loss.

    Returns:
        Dict with key ``"value_loss"`` listing per-epoch average losses.
    """
    if verbose:
        mode_str = (
            f"heuristic MCTS (worlds={mcts_worlds}, depth={mcts_depth}, "
            f"time={mcts_time_ms:.0f}ms)"
            if players == "heuristic" else players
        )
        print(f"[pretrain] Generating data: {n_games} games, strategy='{mode_str}' …")

    examples = generate_heuristic_data(
        n_games               = n_games,
        players               = players,
        mcts_worlds           = mcts_worlds,
        mcts_depth            = mcts_depth,
        mcts_time_ms          = mcts_time_ms,
        trick_history_dropout = trick_history_dropout,
        seed                  = seed,
        verbose               = verbose,
    )

    if not examples:
        print("[pretrain] No examples generated — skipping training.")
        return {"value_loss": []}

    if verbose:
        print(f"[pretrain] Dataset: {len(examples)} examples  "
              f"(avg label = {np.mean([e.value for e in examples]):.2f} pts)")

    buffer = ReplayBuffer(max_size=len(examples) + 1)
    buffer.extend(examples)

    trainer = Trainer(
        net          = net,
        lr           = lr,
        weight_decay = weight_decay,
        policy_weight= 0.0,   # Phase 1: value supervision only
        value_weight = 1.0,
    )

    n_batches    = max(len(examples) // batch_size, 1)
    epoch_losses: List[float] = []

    for epoch in range(n_epochs):
        trainer.set_cosine_schedule(total_steps=n_batches)
        stats = trainer.train_epoch(
            buffer     = buffer,
            batch_size = batch_size,
            n_batches  = n_batches,
        )
        epoch_losses.append(stats["value_loss"])
        if verbose:
            print(f"  [pretrain epoch {epoch+1}/{n_epochs}]  "
                  f"value_loss={stats['value_loss']:.4f}  lr={stats['lr']:.2e}")

    if verbose:
        print(f"[pretrain] Done.  Final value_loss={epoch_losses[-1]:.4f}\n")

    return {"value_loss": epoch_losses}
