"""
CFR evaluation utilities.

Two evaluation functions are provided:

evaluate_vs_random(agent, n_games)
    Pit the CFR agent against three uniform-random opponents.  Useful as an
    absolute lower bound — a well-trained CFR agent should score far fewer
    points than random opponents.

evaluate_vs_heuristic(agent, n_games)
    Pit the CFR agent against three rule-based opponents.  The heuristic
    avoids winning point tricks (hearts / queen of spades) using simple
    suit-following and discard rules — a stronger but lightweight benchmark
    that does not require belief states or tree search.

Both functions rotate the CFR agent through all four seats to cancel
positional bias and return a result dict compatible with the history format
used by the AlphaZero pipeline.

Scoring convention (matching the rest of this codebase)
────────────────────────────────────────────────────────
OpenSpiel Hearts returns() gives ``26 − actual_points`` (maximising utility
= minimising points).  We convert to actual points so lower is better:
    actual_points[p] = 26 − state.returns()[p]
"""
from __future__ import annotations

import logging
import random
from typing import Any, Dict, List, Optional

import numpy as np

from ..openspiel_utils import (
    OBS_PASS_DIR,
    card_points,
    card_to_rank,
    card_to_suit,
    get_current_trick_from_obs,
    NUM_PLAYERS,
)

logger = logging.getLogger(__name__)

# ── Passing-phase detection (mirrors self_play._is_passing_obs) ───────────────

def _is_passing_phase(obs: np.ndarray) -> bool:
    """Return True if the observation indicates the card-passing phase."""
    if len(obs) < 160:
        return False
    # OBS_PASS_DIR[0] == 1 means "No Pass" round — no passing phase at all.
    if obs[OBS_PASS_DIR[0]] >= 0.99:
        return False
    passed   = float(np.sum(obs[56:108]))
    received = float(np.sum(obs[108:160]))
    return passed < 3 or received < 3


# ── Rule-based heuristic opponent ─────────────────────────────────────────────

def _heuristic_action(state, rng: random.Random) -> int:
    """
    Simple rule-based Hearts player for evaluation.

    Passing phase: pass the highest-point card (QS > hearts by rank).
    Play phase:
      - Leading:    play the lowest non-point card; fall back to lowest card.
      - Following:  play the highest card that stays under the current winner;
                    if forced to win, play the lowest card.
      - Discarding: dump the queen of spades first, then highest heart, then
                    the highest card overall.
    """
    legal = state.legal_actions()
    if len(legal) == 1:
        return legal[0]

    player = state.current_player()
    obs = np.array(state.information_state_tensor(player), dtype=np.float32)

    # ── Passing phase ────────────────────────────────────────────────────────
    if _is_passing_phase(obs):
        # Pass the card with the most points; break ties by rank (highest first).
        return max(legal, key=lambda c: (card_points(c), card_to_rank(c)))

    # ── Play phase ───────────────────────────────────────────────────────────
    current_trick = get_current_trick_from_obs(obs)

    if not current_trick:
        # Leading a new trick: play the lowest non-point card.
        safe = [c for c in legal if card_points(c) == 0]
        pool = safe if safe else legal
        return min(pool, key=card_to_rank)

    lead_suit = card_to_suit(current_trick[0][1])
    follow_suit = [c for c in legal if card_to_suit(c) == lead_suit]

    if follow_suit:
        # Must follow suit.  Find the rank of the current trick winner.
        same_suit_played = [c for _, c in current_trick if card_to_suit(c) == lead_suit]
        winning_rank = max(card_to_rank(c) for c in same_suit_played)
        # Cards that would NOT win (play under the winner).
        non_winning = [c for c in follow_suit if card_to_rank(c) < winning_rank]
        if non_winning:
            # Prefer highest non-winning card so we don't block future tricks.
            return max(non_winning, key=card_to_rank)
        # All our follow-suit cards win — play the lowest to minimise damage.
        return min(follow_suit, key=card_to_rank)

    # Discarding (can't follow suit): shed the highest-value point card first.
    QUEEN_OF_SPADES = 10 * 4 + 3  # rank 10, suit 3 (spades)
    if QUEEN_OF_SPADES in legal:
        return QUEEN_OF_SPADES
    hearts = [c for c in legal if card_to_suit(c) == 2]
    if hearts:
        return max(hearts, key=card_to_rank)
    # No point cards to dump — discard the highest card to preserve low cards.
    return max(legal, key=card_to_rank)


# ── Core game loop ─────────────────────────────────────────────────────────────

def _play_evaluation_game(
    cfr_agent,
    cfr_seats: set,
    game,
    rng: random.Random,
    use_heuristic: bool,
) -> Dict[int, float]:
    """
    Play one full Hearts game and return actual points per player.

    Parameters
    ----------
    cfr_agent:
        :class:`~hearts_ai.cfr.agent.CFRAgent` instance.
    cfr_seats:
        Set of player indices that should use the CFR policy.
    game:
        pyspiel Hearts game object.
    rng:
        Random source for chance nodes and random opponents.
    use_heuristic:
        If True, non-CFR seats use the rule-based heuristic.
        If False, they play uniformly at random.

    Returns
    -------
    dict
        ``{player_id: actual_points}`` where lower is better.
    """
    state = game.new_initial_state()

    # ── Deal (chance nodes) ──────────────────────────────────────────────────
    while state.is_chance_node():
        outcomes = state.chance_outcomes()
        actions, probs = zip(*outcomes)
        action = rng.choices(list(actions), weights=list(probs))[0]
        state.apply_action(action)

    # ── Main game loop ───────────────────────────────────────────────────────
    while not state.is_terminal():
        if state.is_chance_node():
            outcomes = state.chance_outcomes()
            actions, probs = zip(*outcomes)
            action = rng.choices(list(actions), weights=list(probs))[0]
            state.apply_action(action)
            continue

        player = state.current_player()
        legal  = state.legal_actions()

        if player in cfr_seats:
            action = cfr_agent.act(state)
            # Safety: if policy returns an illegal action, fall back.
            if action not in legal:
                logger.warning(
                    "CFR agent returned illegal action %d; falling back to random.",
                    action,
                )
                action = rng.choice(legal)
        elif use_heuristic:
            action = _heuristic_action(state, rng)
        else:
            action = rng.choice(legal)

        state.apply_action(action)

    # ── Convert returns → actual points ─────────────────────────────────────
    returns = state.returns()
    return {p: float(26 - returns[p]) for p in range(NUM_PLAYERS)}


# ── Public evaluation functions ───────────────────────────────────────────────

def evaluate_vs_random(
    agent,
    n_games: int = 100,
    *,
    cfr_seat_pairs: bool = True,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Evaluate the CFR agent against three uniform-random opponents.

    The agent occupies seats {0, 2} in even-numbered games and {1, 3} in
    odd-numbered games to cancel positional bias.

    Parameters
    ----------
    agent:
        :class:`~hearts_ai.cfr.agent.CFRAgent`.
    n_games:
        Number of evaluation games.
    cfr_seat_pairs:
        If True (default), rotate the CFR agent through seat pairs {0,2} and
        {1,3}.  If False, the agent always plays seat 0 only.
    seed:
        Optional random seed for reproducibility.

    Returns
    -------
    dict with keys:
        ``cfr_avg``      — mean points for CFR agent seats (lower is better).
        ``opp_avg``      — mean points for random-opponent seats.
        ``cfr_scores``   — per-game CFR seat scores.
        ``opp_scores``   — per-game opponent seat scores.
        ``n_games``      — number of games played.
    """
    import pyspiel

    rng  = random.Random(seed)
    game = pyspiel.load_game("hearts")

    cfr_scores: List[float] = []
    opp_scores: List[float] = []

    for g in range(n_games):
        if cfr_seat_pairs:
            cfr_seats = {0, 2} if g % 2 == 0 else {1, 3}
        else:
            cfr_seats = {0}
        opp_seats = set(range(NUM_PLAYERS)) - cfr_seats

        scores = _play_evaluation_game(agent, cfr_seats, game, rng, use_heuristic=False)

        for p in cfr_seats:
            cfr_scores.append(scores[p])
        for p in opp_seats:
            opp_scores.append(scores[p])

        if (g + 1) % 20 == 0:
            logger.info(
                "vs random [%d/%d] CFR avg=%.2f  opp avg=%.2f",
                g + 1, n_games,
                float(np.mean(cfr_scores)),
                float(np.mean(opp_scores)),
            )

    cfr_avg = float(np.mean(cfr_scores)) if cfr_scores else 0.0
    opp_avg = float(np.mean(opp_scores)) if opp_scores else 0.0

    logger.info(
        "evaluate_vs_random done: CFR %.2f pts | random %.2f pts (%d games)",
        cfr_avg, opp_avg, n_games,
    )
    return {
        "cfr_avg":    cfr_avg,
        "opp_avg":    opp_avg,
        "cfr_scores": cfr_scores,
        "opp_scores": opp_scores,
        "n_games":    n_games,
    }


def evaluate_vs_heuristic(
    agent,
    n_games: int = 100,
    *,
    cfr_seat_pairs: bool = True,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Evaluate the CFR agent against three rule-based heuristic opponents.

    The heuristic avoids winning point tricks (pass high-point cards, follow
    suit safely, discard QS/hearts when sloughing) and provides a stronger
    baseline than random play.

    Parameters
    ----------
    agent:
        :class:`~hearts_ai.cfr.agent.CFRAgent`.
    n_games:
        Number of evaluation games.
    cfr_seat_pairs:
        Rotate CFR agent through seat pairs {0,2} / {1,3} if True.
    seed:
        Optional random seed.

    Returns
    -------
    dict with keys:
        ``cfr_avg``      — mean points for CFR agent seats.
        ``opp_avg``      — mean points for heuristic-opponent seats.
        ``cfr_scores``   — per-game CFR seat scores.
        ``opp_scores``   — per-game opponent seat scores.
        ``n_games``      — number of games played.
    """
    import pyspiel

    rng  = random.Random(seed)
    game = pyspiel.load_game("hearts")

    cfr_scores: List[float] = []
    opp_scores: List[float] = []

    for g in range(n_games):
        if cfr_seat_pairs:
            cfr_seats = {0, 2} if g % 2 == 0 else {1, 3}
        else:
            cfr_seats = {0}
        opp_seats = set(range(NUM_PLAYERS)) - cfr_seats

        scores = _play_evaluation_game(agent, cfr_seats, game, rng, use_heuristic=True)

        for p in cfr_seats:
            cfr_scores.append(scores[p])
        for p in opp_seats:
            opp_scores.append(scores[p])

        if (g + 1) % 20 == 0:
            logger.info(
                "vs heuristic [%d/%d] CFR avg=%.2f  opp avg=%.2f",
                g + 1, n_games,
                float(np.mean(cfr_scores)),
                float(np.mean(opp_scores)),
            )

    cfr_avg = float(np.mean(cfr_scores)) if cfr_scores else 0.0
    opp_avg = float(np.mean(opp_scores)) if opp_scores else 0.0

    logger.info(
        "evaluate_vs_heuristic done: CFR %.2f pts | heuristic %.2f pts (%d games)",
        cfr_avg, opp_avg, n_games,
    )
    return {
        "cfr_avg":    cfr_avg,
        "opp_avg":    opp_avg,
        "cfr_scores": cfr_scores,
        "opp_scores": opp_scores,
        "n_games":    n_games,
    }
