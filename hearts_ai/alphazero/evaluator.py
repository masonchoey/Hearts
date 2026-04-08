"""
Evaluator: pit two HeartsNet versions against each other to decide whether
to keep the newly trained network.

Design
──────
Games are driven by the OpenSpiel RL environment so that the genuine 5088-dim
observation tensor is available at every decision.  This is required to query
the NN policy head (NN input is always a real OpenSpiel obs, never a
reconstruction from PlayState).

At each ROOT decision:
  1. DMCTS vote counts are computed (evaluate_hand at depth cutoffs).
  2. The NN policy head is consulted with the real obs from OpenSpiel.
  3. The NN's top-scoring legal action is chosen (greedy, no noise) to
     reduce variance in the measurement.

Seat rotation: two networks control alternating seat pairs across games to
eliminate positional bias.

Falls back to a PlayState-based loop (no NN-guided action selection) when
OpenSpiel is not installed — keeps tests passing without the dependency.
"""
from __future__ import annotations

import random
import time as _time
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from ..belief_state import BeliefState
from ..openspiel_utils import (
    NUM_CARDS,
    NUM_PLAYERS,
    TWO_OF_CLUBS,
    card_points,
    cards_in_hand_from_obs,
)
from ..world_solver import PlayState, WorldSolver, _state_from_obs_and_world
from .net import HeartsNet
from .self_play import (
    NNWorldSolver,
    _deal_cards,
    _init_beliefs,
    _observe_played_card,
    _is_passing_obs,
    _run_dmcts,
)


# ── OpenSpiel-based evaluation game ───────────────────────────────────────

def _play_eval_game_openspiel(
    nets: Dict[int, HeartsNet],
    n_worlds: int,
    max_depth: int,
    time_limit_ms: float,
    rng: random.Random,
    env: Optional[Any] = None,
) -> Dict[int, float]:
    """
    Play one complete Hearts game driven by OpenSpiel.

    At each ROOT decision the acting player:
      1. Runs DMCTS (evaluate_hand depth cutoffs) → vote distribution.
      2. Queries their NN policy head with the real 5088-dim obs.
      3. Selects the action with the highest NN policy probability among
         the top-voted legal actions (greedy, no noise).

    This ensures that a better-trained NN (better policy) leads to better
    decisions, making pitting meaningful.
    """
    import pyspiel
    from open_spiel.python.rl_environment import Environment as _OSP

    if env is None:
        game = pyspiel.load_game("hearts")
        env  = _OSP(game, players=NUM_PLAYERS)
    ts = env.reset()

    solvers = {
        p: NNWorldSolver(net=nets[p], max_depth=max_depth)
        for p in range(NUM_PLAYERS)
    }

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

        # Passing phase: pass highest-point cards; no NN involvement
        if _is_passing_obs(obs_cp):
            action = max(legal, key=card_points)
            ts = env.step([action])
            continue

        # Sync belief from full observation
        beliefs[cp].update_from_openspiel_state(ts)

        # Reconstruct PlayState for DMCTS
        world: Dict[int, Set[int]] = {
            p: set(cards_in_hand_from_obs(
                np.array(ts.observations["info_state"][p], dtype=np.float32)
            ))
            for p in range(NUM_PLAYERS)
        }
        play = _state_from_obs_and_world(obs_cp, world, cp)

        if play is None or len(legal) == 1:
            action = legal[0]
            ts = env.step([action])
            continue

        # DMCTS votes (evaluate_hand at depth cutoffs)
        votes = _run_dmcts(
            solvers[cp], play, beliefs[cp],
            cp, n_worlds, time_limit_ms,
        )

        # Greedy action selection weighted by NN policy (real obs)
        legal_mask = np.zeros(NUM_CARDS, dtype=bool)
        for a in legal:
            legal_mask[a] = True
        nn_policy, _ = nets[cp].predict(obs_cp, legal_mask)

        # Rank by DMCTS votes first; break ties with NN policy
        action = max(legal, key=lambda a: (votes.get(a, 0), nn_policy[a]))

        ts = env.step([action])

    # Decode final scores: rewards[p] = 26 - points_taken[p]
    if ts.last() and ts.rewards is not None:
        return {p: float(26 - int(ts.rewards[p])) for p in range(NUM_PLAYERS)}
    return {p: 6.0 for p in range(NUM_PLAYERS)}


# ── PlayState-based evaluation fallback ───────────────────────────────────

def _play_eval_game_playstate(
    nets: Dict[int, HeartsNet],
    n_worlds: int,
    max_depth: int,
    time_limit_ms: float,
    rng: random.Random,
) -> Dict[int, float]:
    """
    PlayState-based evaluation used when OpenSpiel is not installed.

    The NN policy is NOT consulted (no real obs available), so pitting only
    measures evaluate_hand depth-cutoff quality — both nets play identically.
    Only used for unit-testing without the OpenSpiel dependency.
    """
    solvers = {
        p: NNWorldSolver(net=nets[p], max_depth=max_depth)
        for p in range(NUM_PLAYERS)
    }

    hands   = _deal_cards(rng)
    beliefs = _init_beliefs(hands)

    starting_player = next(
        p for p in range(NUM_PLAYERS) if TWO_OF_CLUBS in hands[p]
    )
    play = PlayState(
        hands          = hands,
        current_player = starting_player,
        current_trick  = [],
        points         = {p: 0.0 for p in range(NUM_PLAYERS)},
        hearts_broken  = False,
        num_played     = 0,
    )

    while not play.is_terminal():
        agent = play.current_player
        legal = play.legal_actions()
        if not legal:
            break

        if len(legal) == 1:
            action = legal[0]
        else:
            votes  = _run_dmcts(
                solvers[agent], play, beliefs[agent],
                agent, n_worlds, time_limit_ms,
            )
            action = max(votes, key=votes.get)

        _observe_played_card(beliefs, agent, action, list(play.current_trick))
        play.apply_action_inplace(action)

    return {p: float(play.terminal_score(p)) for p in range(NUM_PLAYERS)}


def _play_eval_game(
    nets: Dict[int, HeartsNet],
    n_worlds: int,
    max_depth: int,
    time_limit_ms: float,
    rng: random.Random,
    env: Optional[Any] = None,
) -> Dict[int, float]:
    """Dispatch to OpenSpiel or PlayState evaluator depending on availability."""
    try:
        import pyspiel  # noqa: F401
        return _play_eval_game_openspiel(
            nets, n_worlds, max_depth, time_limit_ms, rng, env=env
        )
    except ImportError:
        return _play_eval_game_playstate(nets, n_worlds, max_depth, time_limit_ms, rng)


# ── Tournament ─────────────────────────────────────────────────────────────

def pit_networks(
    net_new: HeartsNet,
    net_old: HeartsNet,
    n_games: int          = 40,
    n_worlds: int         = 20,
    max_depth: int        = 4,
    time_limit_ms: float  = 400.0,
    pit_criterion: str    = "points",
    win_threshold: float  = 0.55,
    pit_points_margin: float = 0.0,
    seed: Optional[int]   = None,
) -> Tuple[float, float, float, bool]:
    """
    Pit ``net_new`` against ``net_old`` across ``n_games`` and decide whether to
    promote the candidate.

    Seat assignment rotates each game so both networks experience every table
    position.  Metrics are **average points taken** over all seats controlled by
    each net (lower is better in Hearts).

    **Acceptance** (configurable)::

        ``pit_criterion == "points"`` (default):
            Accept if ``new_avg < old_avg - pit_points_margin``.
            Example: ``pit_points_margin=0.5`` requires the new net to average
            at least 0.5 fewer points than the old net.

        ``pit_criterion == "win_rate"``:
            Accept if the fraction of pairwise seat comparisons where the new
            net scored lower than the old net is ≥ ``win_threshold``.

    The NN's policy head guides action selection at each ROOT decision (using
    the real 5088-dim OpenSpiel obs).

    Returns:
        (new_avg_score, old_avg_score, seat_win_rate, accepted)
    """
    rng = random.Random(seed)

    new_scores: List[float] = []
    old_scores: List[float] = []

    shared_env: Optional[Any] = None
    try:
        import pyspiel
        from open_spiel.python.rl_environment import Environment as _OSP
        shared_env = _OSP(pyspiel.load_game("hearts"), players=NUM_PLAYERS)
    except ImportError:
        pass

    for game_idx in range(n_games):
        if game_idx % 2 == 0:
            new_seats: Set[int] = {0, 2}
            old_seats: Set[int] = {1, 3}
        else:
            new_seats = {1, 3}
            old_seats = {0, 2}

        nets: Dict[int, HeartsNet] = {
            p: (net_new if p in new_seats else net_old)
            for p in range(NUM_PLAYERS)
        }

        scores = _play_eval_game(
            nets, n_worlds, max_depth, time_limit_ms, rng, env=shared_env
        )

        for p in new_seats:
            new_scores.append(scores[p])
        for p in old_seats:
            old_scores.append(scores[p])

    new_avg = float(np.mean(new_scores)) if new_scores else 0.0
    old_avg = float(np.mean(old_scores)) if old_scores else 0.0

    wins      = sum(1 for n, o in zip(new_scores, old_scores) if n < o)
    total     = len(new_scores)
    win_rate  = wins / total if total > 0 else 0.5

    if pit_criterion == "points":
        accepted = new_avg < old_avg - pit_points_margin
    elif pit_criterion == "win_rate":
        accepted = win_rate >= win_threshold
    else:
        raise ValueError(
            f"pit_criterion must be 'points' or 'win_rate', got {pit_criterion!r}"
        )

    return new_avg, old_avg, win_rate, accepted


# ── Heuristic baseline comparison ─────────────────────────────────────────

def evaluate_vs_heuristic(
    net: HeartsNet,
    n_games: int         = 40,
    n_worlds: int        = 20,
    max_depth: int       = 4,
    time_limit_ms: float = 400.0,
    seed: Optional[int]  = None,
) -> Tuple[float, float]:
    """
    Compare the neural network against the hand-coded evaluate_hand heuristic.

    The heuristic uses a plain WorldSolver with evaluate_hand at depth
    cutoffs and no NN policy guidance.

    Returns:
        (net_avg_score, heuristic_avg_score)
    """
    try:
        import pyspiel  # noqa: F401
        return _evaluate_vs_heuristic_openspiel(
            net, n_games, n_worlds, max_depth, time_limit_ms, seed
        )
    except ImportError:
        return _evaluate_vs_heuristic_playstate(
            net, n_games, n_worlds, max_depth, time_limit_ms, seed
        )


def _evaluate_vs_heuristic_openspiel(
    net: HeartsNet,
    n_games: int,
    n_worlds: int,
    max_depth: int,
    time_limit_ms: float,
    seed: Optional[int],
) -> Tuple[float, float]:
    import pyspiel
    from open_spiel.python.rl_environment import Environment as _OSP

    rng  = random.Random(seed)
    game = pyspiel.load_game("hearts")

    heuristic_solver = WorldSolver(max_depth=max_depth)
    nn_solver        = NNWorldSolver(net=net, max_depth=max_depth)

    net_scores:       List[float] = []
    heuristic_scores: List[float] = []

    # Reuse one Environment across games (OpenSpiel logs on each __init__).
    env = _OSP(game, players=NUM_PLAYERS)

    for game_idx in range(n_games):
        if game_idx % 2 == 0:
            nn_seats:        Set[int] = {0, 2}
            heuristic_seats: Set[int] = {1, 3}
        else:
            nn_seats        = {1, 3}
            heuristic_seats = {0, 2}

        ts = env.reset()

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

            if _is_passing_obs(obs_cp):
                action = max(legal, key=card_points)
                ts = env.step([action])
                continue

            beliefs[cp].update_from_openspiel_state(ts)

            world: Dict[int, Set[int]] = {
                p: set(cards_in_hand_from_obs(
                    np.array(ts.observations["info_state"][p], dtype=np.float32)
                ))
                for p in range(NUM_PLAYERS)
            }
            play = _state_from_obs_and_world(obs_cp, world, cp)

            if play is None or len(legal) == 1:
                action = legal[0]
                ts = env.step([action])
                continue

            if cp in nn_seats:
                votes = _run_dmcts(
                    nn_solver, play, beliefs[cp],
                    cp, n_worlds, time_limit_ms,
                )
                legal_mask = np.zeros(NUM_CARDS, dtype=bool)
                for a in legal:
                    legal_mask[a] = True
                nn_policy, _ = net.predict(obs_cp, legal_mask)
                action = max(legal, key=lambda a: (votes.get(a, 0), nn_policy[a]))
            else:
                # Plain heuristic: DMCTS with evaluate_hand, no NN policy
                votes = _run_heuristic_dmcts(
                    heuristic_solver, play, beliefs[cp],
                    cp, n_worlds, time_limit_ms, legal,
                )
                action = max(votes, key=votes.get)

            ts = env.step([action])

        if ts.last() and ts.rewards is not None:
            scores = {p: float(26 - int(ts.rewards[p])) for p in range(NUM_PLAYERS)}
        else:
            scores = {p: 6.0 for p in range(NUM_PLAYERS)}

        for p in nn_seats:
            net_scores.append(scores[p])
        for p in heuristic_seats:
            heuristic_scores.append(scores[p])

    net_avg       = float(np.mean(net_scores))       if net_scores       else 0.0
    heuristic_avg = float(np.mean(heuristic_scores)) if heuristic_scores else 0.0
    return net_avg, heuristic_avg


def _run_heuristic_dmcts(
    solver: WorldSolver,
    play: PlayState,
    belief: BeliefState,
    agent_id: int,
    n_worlds: int,
    time_limit_ms: float,
    legal: List[int],
) -> Dict[int, int]:
    """Run DMCTS with the plain heuristic solver (no NN)."""
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

        h_legal = world_play.legal_actions()
        h_legal = [a for a in h_legal if a in real_set] or list(legal)
        if len(h_legal) == 1:
            action = h_legal[0]
        else:
            best_a = h_legal[0]
            best_v = float("inf")
            memo: dict = {}
            for a in solver._order_actions(world_play, h_legal):
                undo = world_play.apply_action_inplace(a)
                v = solver._minimax(
                    world_play, agent_id, 1, -1e9, 1e9, memo, w_deadline
                )
                world_play.undo_action(undo)
                if v < best_v:
                    best_v = v
                    best_a = a
            action = best_a

        if action in votes:
            votes[action] += 1

    if sum(votes.values()) == 0:
        votes[legal[0]] = 1
    return votes


def _evaluate_vs_heuristic_playstate(
    net: HeartsNet,
    n_games: int,
    n_worlds: int,
    max_depth: int,
    time_limit_ms: float,
    seed: Optional[int],
) -> Tuple[float, float]:
    """PlayState-based fallback for evaluate_vs_heuristic (no OpenSpiel)."""
    rng = random.Random(seed)
    nn_solver        = NNWorldSolver(net=net, max_depth=max_depth)
    heuristic_solver = WorldSolver(max_depth=max_depth)

    net_scores:       List[float] = []
    heuristic_scores: List[float] = []

    for game_idx in range(n_games):
        if game_idx % 2 == 0:
            nn_seats:        Set[int] = {0, 2}
            heuristic_seats: Set[int] = {1, 3}
        else:
            nn_seats        = {1, 3}
            heuristic_seats = {0, 2}

        hands   = _deal_cards(rng)
        beliefs = _init_beliefs(hands)

        starting = next(p for p in range(NUM_PLAYERS) if TWO_OF_CLUBS in hands[p])
        play = PlayState(
            hands=hands, current_player=starting, current_trick=[],
            points={p: 0.0 for p in range(NUM_PLAYERS)},
            hearts_broken=False, num_played=0,
        )

        while not play.is_terminal():
            agent = play.current_player
            legal = play.legal_actions()
            if not legal:
                break

            if len(legal) == 1:
                action = legal[0]
            elif agent in nn_seats:
                v = _run_dmcts(
                    nn_solver, play, beliefs[agent],
                    agent, n_worlds, time_limit_ms,
                )
                action = max(v, key=v.get)
            else:
                v = _run_heuristic_dmcts(
                    heuristic_solver, play, beliefs[agent],
                    agent, n_worlds, time_limit_ms, legal,
                )
                action = max(v, key=v.get)

            _observe_played_card(beliefs, agent, action, list(play.current_trick))
            play.apply_action_inplace(action)

        scores = {p: float(play.terminal_score(p)) for p in range(NUM_PLAYERS)}
        for p in nn_seats:
            net_scores.append(scores[p])
        for p in heuristic_seats:
            heuristic_scores.append(scores[p])

    net_avg       = float(np.mean(net_scores))       if net_scores       else 0.0
    heuristic_avg = float(np.mean(heuristic_scores)) if heuristic_scores else 0.0
    return net_avg, heuristic_avg
