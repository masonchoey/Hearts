"""
Self-play game generation for AlphaZero Hearts training.

Architecture
────────────
The 5088-dim observation comes exclusively from OpenSpiel at ROOT decision
points.  It is NEVER reconstructed from a PlayState.

At each decision the game loop does two things:

  1. ROOT DMCTS vote counting (WorldSolver with evaluate_hand depth cutoffs)
         obs_5088 = ts.observations["info_state"][cp]        ← from OpenSpiel
         play     = _state_from_obs_and_world(obs, world, cp) ← for DMCTS
         votes    = _run_dmcts(solver, play, belief, ...)

  2. NN policy consultation at ROOT
         nn_policy, _ = net.predict(obs_5088, legal_mask)     ← real obs

     The final training policy is a blend of DMCTS vote distribution and NN
     policy (controlled by ``nn_blend`` weight).  Early in training the NN
     is random, so the blend is dominated by DMCTS; as the NN improves it
     contributes more signal.

     Action selection follows the blended policy during the first
     ``temperature_tricks`` tricks (proportional sampling for exploration),
     then switches to greedy.

Depth-cutoff heuristic
──────────────────────
NNWorldSolver._estimate_score uses evaluate_hand directly — no reconstruction.
The NN is NOT called inside the minimax search; it is only called at the ROOT
where we have a genuine OpenSpiel observation.

Legacy helpers
──────────────
_deal_cards, _init_beliefs, _observe_played_card are kept for the PlayState-
based evaluator (evaluator.py) which runs without OpenSpiel.
"""
from __future__ import annotations

import random
import time
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from ..belief_state import BeliefState
from ..openspiel_utils import (
    NUM_CARDS,
    NUM_PLAYERS,
    TWO_OF_CLUBS,
    card_points,
    card_to_suit,
    cards_in_hand_from_obs,
)
from ..starterheartsheuristic import evaluate_hand
from ..world_solver import PlayState, WorldSolver, _state_from_obs_and_world
from .net import HeartsNet
from .replay_buffer import TrainingExample

HEARTS_SUIT = 2


# ── NNWorldSolver ─────────────────────────────────────────────────────────

class NNWorldSolver(WorldSolver):
    """
    WorldSolver subclass for AlphaZero self-play.

    Depth-cutoff evaluation: uses ``evaluate_hand`` directly — identical to
    the base ``WorldSolver``.  The NN is NOT called inside minimax because no
    real OpenSpiel observation is available at leaf PlayState nodes.

    ``solve_playstate`` is inherited from ``WorldSolver`` (moved there so
    the pre-training MCTS player can also use it without needing this class).

    The NN's policy head is used OUTSIDE this class, at ROOT decision points
    in the self-play and evaluation game loops where real obs is available.

    Args:
        net:       HeartsNet (used externally for ROOT policy; stored here
                   so the evaluator can pass it to pitting games).
        max_depth: Alpha-beta search depth.
    """

    def __init__(self, net: HeartsNet, max_depth: int = 4):
        super().__init__(max_depth=max_depth)
        self.net = net

    # _estimate_score is NOT overridden — inherited from WorldSolver (evaluate_hand).
    # solve_playstate is NOT overridden — inherited from WorldSolver.
    # The NN is only called at ROOT via net.predict() in the game loops.


# ── Determinized MCTS loop ─────────────────────────────────────────────────

def _run_dmcts(
    solver: NNWorldSolver,
    play: PlayState,
    belief: BeliefState,
    agent_id: int,
    n_worlds: int,
    time_limit_ms: float,
) -> Dict[int, int]:
    """
    Sample ``n_worlds`` consistent worlds from the player's belief and run
    alpha-beta minimax on each, tallying action votes.

    All depth-cutoff evaluations inside minimax use ``evaluate_hand`` (the
    solver's inherited ``_estimate_score``).

    Returns:
        Dict mapping each legal action to its vote count.
    """
    legal = play.legal_actions()
    if not legal:
        return {}
    if len(legal) == 1:
        return {legal[0]: n_worlds}

    votes: Dict[int, int] = {a: 0 for a in legal}
    deadline = time.perf_counter() + time_limit_ms / 1000.0
    n_evaluated = 0

    for i in range(n_worlds):
        if time.perf_counter() >= deadline:
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

        remaining_worlds = max(n_worlds - i, 1)
        time_left        = deadline - time.perf_counter()
        world_deadline   = time.perf_counter() + time_left / remaining_worlds

        action, _ = solver.solve_playstate(
            world_play, agent_id,
            real_legal=legal,
            time_deadline=world_deadline,
        )
        if action in votes:
            votes[action] += 1
        n_evaluated += 1

    if n_evaluated == 0 or sum(votes.values()) == 0:
        votes[legal[0]] = 1

    return votes


# ── Legacy PlayState helpers (kept for evaluator.py compatibility) ─────────

def _deal_cards(rng: random.Random) -> Dict[int, Set[int]]:
    """Uniform random deal: 52 cards → 4 players × 13 cards."""
    deck = list(range(NUM_CARDS))
    rng.shuffle(deck)
    return {p: set(deck[p * 13: (p + 1) * 13]) for p in range(NUM_PLAYERS)}


def _init_beliefs(hands: Dict[int, Set[int]]) -> Dict[int, BeliefState]:
    beliefs: Dict[int, BeliefState] = {}
    for p in range(NUM_PLAYERS):
        b = BeliefState(
            my_hand     = list(hands[p]),
            num_players = NUM_PLAYERS,
            agent_id    = p,
        )
        for q in range(NUM_PLAYERS):
            b.set_cards_remaining(q, 13)
        beliefs[p] = b
    return beliefs


def _observe_played_card(
    beliefs: Dict[int, BeliefState],
    player: int,
    card: int,
    current_trick_before_play: List[Tuple[int, int]],
) -> None:
    lead_suit = (
        card_to_suit(current_trick_before_play[0][1])
        if current_trick_before_play else None
    )
    for p in range(NUM_PLAYERS):
        if p == player:
            beliefs[p].my_hand = [c for c in beliefs[p].my_hand if c != card]
            beliefs[p].played_cards.add(card)
        else:
            beliefs[p].observe_card_played(player, card, lead_suit, current_trick_before_play)
        new_count = max(0, beliefs[p].get_cards_remaining(player) - 1)
        beliefs[p].set_cards_remaining(player, new_count)


# ── Passing-phase detection ────────────────────────────────────────────────

def _is_passing_obs(obs: np.ndarray) -> bool:
    """Return True when the observation belongs to the card-passing phase."""
    if len(obs) < 160:
        return False
    if obs[0] >= 0.99:   # OBS_PASS_DIR[0] == 1 → "No Pass" round
        return False
    passed   = float(np.sum(obs[56:108]))
    received = float(np.sum(obs[108:160]))
    return passed < 3 or received < 3


# ── Policy helper: blended DMCTS + NN at ROOT ─────────────────────────────

def _build_root_policy(
    votes: Dict[int, int],
    legal: List[int],
    obs_5088: np.ndarray,
    net: HeartsNet,
    nn_blend: float,
) -> np.ndarray:
    """
    Build the training policy as a weighted blend of DMCTS visit counts and
    the NN's policy head evaluated on the real 5088-dim ROOT observation.

    Args:
        votes:    Raw DMCTS vote counts per action.
        legal:    Legal actions in the real game state.
        obs_5088: Real 5088-dim OpenSpiel observation for the acting player.
        net:      HeartsNet to query.
        nn_blend: Weight given to the NN policy (0 = pure DMCTS, 1 = pure NN).

    Returns:
        float32 array of shape (NUM_CARDS,) — normalized over legal actions.
    """
    policy = np.zeros(NUM_CARDS, dtype=np.float32)

    # DMCTS component
    total = sum(votes.values())
    if total > 0:
        for a, v in votes.items():
            policy[a] = v / total

    # NN policy component (queried with REAL obs from OpenSpiel)
    if nn_blend > 0.0:
        legal_mask = np.zeros(NUM_CARDS, dtype=bool)
        for a in legal:
            legal_mask[a] = True
        nn_policy, _ = net.predict(obs_5088, legal_mask)
        for a in legal:
            policy[a] = (1.0 - nn_blend) * policy[a] + nn_blend * nn_policy[a]

    return policy


# ── OpenSpiel self-play game loop ──────────────────────────────────────────

def _run_openspiel_self_play_game(
    net: HeartsNet,
    n_worlds: int,
    max_depth: int,
    time_limit_ms: float,
    temperature_tricks: int,
    dirichlet_alpha: float,
    dirichlet_eps: float,
    nn_blend: float,
    rng: random.Random,
    osp_env: Optional[Any] = None,
) -> List[TrainingExample]:
    """
    Drive one Hearts hand via the OpenSpiel RL environment.

    At every ROOT decision the acting player's genuine 5088-dim observation
    is used both as the neural-network feature and to query the NN policy.
    DMCTS vote counts (using evaluate_hand at depth cutoffs) are blended with
    the NN policy to form the training target and to select the action.

    Reward decoding:  points_taken[p] = 26 - int(ts.rewards[p])
    """
    import pyspiel
    from open_spiel.python.rl_environment import Environment as _OSP

    # Reuse a caller-provided Environment to avoid OpenSpiel logging
    # "Using game instance: …" on every Environment() (see rl_environment.py).
    if osp_env is None:
        game = pyspiel.load_game("hearts")
        env  = _OSP(game, players=NUM_PLAYERS)
    else:
        env = osp_env
    ts = env.reset()
    solver = NNWorldSolver(net=net, max_depth=max_depth)

    beliefs: Dict[int, BeliefState] = {
        p: BeliefState(my_hand=[], num_players=NUM_PLAYERS, agent_id=p)
        for p in range(NUM_PLAYERS)
    }

    # (acting_player, obs_5088, blended_policy)
    pending: List[Tuple[int, np.ndarray, np.ndarray]] = []

    while not ts.last():
        cp    = ts.observations["current_player"]
        legal = list(ts.observations["legal_actions"][cp])
        if not legal:
            break

        obs_cp = np.array(ts.observations["info_state"][cp], dtype=np.float32)

        # ── Passing phase: no training data; pass highest-point cards ─────
        if _is_passing_obs(obs_cp):
            action = max(legal, key=card_points)
            ts = env.step([action])
            continue

        # ── Sync belief from full observation ─────────────────────────────
        beliefs[cp].update_from_openspiel_state(ts)

        # ── True world: each player's current hand from their own obs ─────
        world: Dict[int, Set[int]] = {
            p: set(cards_in_hand_from_obs(
                np.array(ts.observations["info_state"][p], dtype=np.float32)
            ))
            for p in range(NUM_PLAYERS)
        }
        play = _state_from_obs_and_world(obs_cp, world, cp)

        if play is None:
            action = rng.choice(legal)
            ts = env.step([action])
            continue

        # ── DMCTS vote counts (evaluate_hand at depth cutoffs) ────────────
        if len(legal) == 1:
            votes: Dict[int, int] = {legal[0]: 1}
        else:
            votes = _run_dmcts(
                solver, play, beliefs[cp],
                cp, n_worlds, time_limit_ms,
            )

        # ── Blend DMCTS votes with NN policy (real obs from OpenSpiel) ────
        policy = _build_root_policy(votes, legal, obs_cp, net, nn_blend)

        # ── Dirichlet noise for exploration ───────────────────────────────
        tricks_so_far = play.num_played // NUM_PLAYERS
        if tricks_so_far < temperature_tricks and len(legal) > 1:
            noise = np.random.dirichlet(
                [dirichlet_alpha] * len(legal)
            ).astype(np.float32)
            for i, a in enumerate(legal):
                policy[a] = (
                    (1.0 - dirichlet_eps) * policy[a]
                    + dirichlet_eps * noise[i]
                )

        # Record step: obs_5088 directly from OpenSpiel ← key requirement
        pending.append((cp, obs_cp.copy(), policy.copy()))

        # ── Action selection ───────────────────────────────────────────────
        if tricks_so_far < temperature_tricks:
            legal_probs = np.array(
                [policy[a] for a in legal], dtype=np.float64
            )
            s = legal_probs.sum()
            if s < 1e-9:
                legal_probs = np.ones(len(legal), dtype=np.float64) / len(legal)
            else:
                legal_probs /= s
            action = rng.choices(legal, weights=legal_probs.tolist())[0]
        else:
            action = max(votes, key=votes.get)

        ts = env.step([action])

    # ── Final scores (OpenSpiel rewards: rewards[p] = 26 - points_taken[p]) ─
    if ts.last() and ts.rewards is not None:
        final_scores = {p: 26 - int(ts.rewards[p]) for p in range(NUM_PLAYERS)}
    else:
        final_scores = {p: 6 for p in range(NUM_PLAYERS)}

    return [
        TrainingExample(
            features = obs,
            policy   = pol,
            value    = float(final_scores[cp]),
        )
        for cp, obs, pol in pending
    ]


# ── Legacy PlayState self-play fallback ───────────────────────────────────

def _run_playstate_self_play_game(
    net: HeartsNet,
    n_worlds: int,
    max_depth: int,
    time_limit_ms: float,
    temperature_tricks: int,
    dirichlet_alpha: float,
    dirichlet_eps: float,
    nn_blend: float,
    rng: random.Random,
) -> List[TrainingExample]:
    """
    PlayState-based fallback when OpenSpiel is not installed.

    Uses evaluate_hand for depth cutoffs and records partial obs (from
    extract_features) as features — this path is only for unit-testing
    without OpenSpiel and should NOT be used for real training.
    """
    from ..starterheartsheuristic import evaluate_hand as _eval
    from .net import OBS_DIM

    solver = NNWorldSolver(net=net, max_depth=max_depth)
    hands   = _deal_cards(rng)
    beliefs = _init_beliefs(hands)

    starting_player = next(p for p in range(NUM_PLAYERS) if TWO_OF_CLUBS in hands[p])
    play = PlayState(
        hands          = hands,
        current_player = starting_player,
        current_trick  = [],
        points         = {p: 0.0 for p in range(NUM_PLAYERS)},
        hearts_broken  = False,
        num_played     = 0,
    )

    pending: List[Tuple[int, np.ndarray, np.ndarray]] = []

    while not play.is_terminal():
        agent  = play.current_player
        legal  = play.legal_actions()
        if not legal:
            break

        # In fallback mode we have no real obs — use zeros as placeholder
        obs_placeholder = np.zeros(OBS_DIM, dtype=np.float32)

        if len(legal) == 1:
            policy           = np.zeros(NUM_CARDS, dtype=np.float32)
            policy[legal[0]] = 1.0
            action           = legal[0]
        else:
            votes = _run_dmcts(
                solver, play, beliefs[agent],
                agent, n_worlds, time_limit_ms,
            )
            policy = np.zeros(NUM_CARDS, dtype=np.float32)
            total  = sum(votes.values())
            if total > 0:
                for a, v in votes.items():
                    policy[a] = v / total

            tricks_so_far = play.num_played // NUM_PLAYERS
            if tricks_so_far < temperature_tricks:
                noise = np.random.dirichlet(
                    [dirichlet_alpha] * len(legal)
                ).astype(np.float32)
                for i, a in enumerate(legal):
                    policy[a] = (
                        (1.0 - dirichlet_eps) * policy[a]
                        + dirichlet_eps * noise[i]
                    )
                legal_probs = np.array(
                    [policy[a] for a in legal], dtype=np.float64
                )
                s = legal_probs.sum()
                if s < 1e-9:
                    legal_probs = np.ones(len(legal), dtype=np.float64) / len(legal)
                else:
                    legal_probs /= s
                action = rng.choices(legal, weights=legal_probs.tolist())[0]
            else:
                action = max(votes, key=votes.get)

        pending.append((agent, obs_placeholder, policy))
        _observe_played_card(beliefs, agent, action, list(play.current_trick))
        play.apply_action_inplace(action)

    final_scores = {p: play.terminal_score(p) for p in range(NUM_PLAYERS)}
    return [
        TrainingExample(
            features = obs,
            policy   = pol,
            value    = float(final_scores[a]),
        )
        for a, obs, pol in pending
    ]


# ── Public entry point ─────────────────────────────────────────────────────

def run_self_play_game(
    net: HeartsNet,
    n_worlds: int           = 20,
    max_depth: int          = 4,
    time_limit_ms: float    = 500.0,
    temperature_tricks: int = 10,
    dirichlet_alpha: float  = 0.3,
    dirichlet_eps: float    = 0.25,
    nn_blend: float         = 0.25,
    rng: Optional[random.Random] = None,
    osp_env: Optional[Any]  = None,
) -> List[TrainingExample]:
    """
    Play one complete Hearts hand with all four seats controlled by DMCTS+NN.

    The OpenSpiel RL environment drives the game so the raw 5088-dim
    observation tensor is available at every ROOT decision and is stored
    directly as the training feature (no reconstruction from PlayState).

    DMCTS uses ``evaluate_hand`` for depth-cutoff leaf evaluation.  The NN's
    policy head is also queried at the ROOT (with the genuine obs) and blended
    with DMCTS visit counts:

        policy[a] = (1 - nn_blend) × dmcts_visit_fraction[a]
                  + nn_blend       × nn_policy[a]

    This blend gives the NN a meaningful influence on training from the
    first iteration and improves action selection as the NN quality grows.

    Args:
        net:                HeartsNet.
        n_worlds:           DMCTS worlds per decision.
        max_depth:          Minimax search depth per world.
        time_limit_ms:      Time budget per decision (ms).
        temperature_tricks: Proportional sampling for the first N tricks,
                            greedy from trick N onward.
        dirichlet_alpha:    Dirichlet noise α (exploration).
        dirichlet_eps:      Noise mixing weight.
        nn_blend:           Weight of NN policy in the DMCTS/NN blend (0–1).
        rng:                Optional seeded RNG.
        osp_env:            Optional OpenSpiel ``Environment`` to reuse across
                            many games (avoids repeated absl "Using game …" logs).

    Returns:
        List of TrainingExample with ``features`` = raw 5088-dim OpenSpiel
        observation.  Falls back to zeros when OpenSpiel is unavailable.
    """
    rng = rng or random.Random()
    kwargs = dict(
        net               = net,
        n_worlds          = n_worlds,
        max_depth         = max_depth,
        time_limit_ms     = time_limit_ms,
        temperature_tricks= temperature_tricks,
        dirichlet_alpha   = dirichlet_alpha,
        dirichlet_eps     = dirichlet_eps,
        nn_blend          = nn_blend,
        rng               = rng,
        osp_env           = osp_env,
    )
    try:
        import pyspiel  # noqa: F401
        return _run_openspiel_self_play_game(**kwargs)
    except ImportError:
        kwargs.pop("osp_env", None)
