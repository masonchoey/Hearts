"""
WorldSolver: minimax (alpha-beta) on a single fully-known world.
Uses an internal play-phase simulator so we don't require OpenSpiel state clone.
"""
from __future__ import annotations

import os
import sys
import time
from typing import Dict, Set, List, Optional, Tuple, Any

_HEARTS_DEBUG = os.environ.get("HEARTS_DEBUG", "").strip().lower() in ("1", "true", "yes")
_last_solver_log = 0.0

from .openspiel_utils import (
    card_to_suit,
    card_to_rank,
    card_points,
    NUM_PLAYERS,
    NUM_CARDS,
    NUM_TRICKS,
    TWO_OF_CLUBS,
)
from .openspiel_utils import get_trick_history_from_obs, get_current_trick_from_obs, _get_obs_from_state, cards_in_hand_from_obs
from .starterheartsheuristic import evaluate_hand

QUEEN_OF_SPADES = 10 * 4 + 3  # 43
HEARTS_SUIT = 2
_TOTAL_POINTS = 26

# Precomputed point values per card (avoids repeated function calls in _trick_points)
_CARD_POINTS: List[int] = [card_points(c) for c in range(NUM_CARDS)]

# Precomputed single-card bitmasks for O(1) hand-mask updates.
_CARD_MASK: List[int] = [1 << c for c in range(NUM_CARDS)]


def _cards_to_mask(cards) -> int:
    """Convert an iterable of card indices to an integer bitmask."""
    m = 0
    for c in cards:
        m |= _CARD_MASK[c]
    return m


def _trick_winner(lead_suit: int, cards: List[int]) -> int:
    """Index 0..3 of winner in the 4 cards (same order as players)."""
    best_i = 0
    best_rank = card_to_rank(cards[0])
    for i in range(1, 4):
        if card_to_suit(cards[i]) == lead_suit:
            r = card_to_rank(cards[i])
            if r > best_rank:
                best_rank = r
                best_i = i
    return best_i


def _trick_points(cards: List[int]) -> int:
    return sum(_CARD_POINTS[c] for c in cards)


class PlayState:
    """
    Internal state for one determinized world during play.
    hands[player] = set of card indices still in hand.

    completed_tricks is intentionally NOT stored — the minimax search only needs
    num_played (tracked incrementally) and points (updated on each trick win).
    Dropping it eliminates significant clone cost.

    hand_masks[player] is an integer bitmask (bit i = card i is held) kept in sync
    with hands[player]. This allows O(1) transposition-table key construction
    instead of O(hand_size) frozenset creation at every node.
    """
    __slots__ = ("hands", "current_player", "current_trick", "points", "hearts_broken", "num_played", "hand_masks")

    def __init__(
        self,
        hands: Dict[int, Set[int]],
        current_player: int,
        current_trick: List[Tuple[int, int]],
        points: Dict[int, float],
        hearts_broken: bool,
        num_played: Optional[int] = None,
        completed_tricks: Optional[List] = None,
    ):
        self.hands = {p: set(hands[p]) for p in range(NUM_PLAYERS)}
        self.current_player = current_player
        self.current_trick = list(current_trick)
        self.points = dict(points)
        self.hearts_broken = hearts_broken
        # num_played can be supplied directly or derived from completed_tricks for compat.
        if num_played is not None:
            self.num_played = num_played
        elif completed_tricks is not None:
            self.num_played = len(completed_tricks) * 4 + len(current_trick)
        else:
            self.num_played = len(current_trick)
        # Bitmask representation — kept in sync with hands for O(1) key hashing.
        self.hand_masks = {p: _cards_to_mask(self.hands[p]) for p in range(NUM_PLAYERS)}

    def clone(self) -> "PlayState":
        new = object.__new__(PlayState)
        new.hands = {p: set(self.hands[p]) for p in range(NUM_PLAYERS)}
        new.current_player = self.current_player
        new.current_trick = list(self.current_trick)
        new.points = dict(self.points)
        new.hearts_broken = self.hearts_broken
        new.num_played = self.num_played
        new.hand_masks = dict(self.hand_masks)
        return new

    def legal_actions(self) -> List[int]:
        """Legal card indices for current_player."""
        h = self.hands[self.current_player]
        if not h:
            return []
        # Mid-trick: must follow suit if possible
        if self.current_trick:
            lead_suit = card_to_suit(self.current_trick[0][1])
            follow = [c for c in h if card_to_suit(c) == lead_suit]
            if follow:
                return follow
            # Void: can play any
            return list(h)
        # Leading
        # First card of game: 2C
        if self.num_played == 0:
            if TWO_OF_CLUBS in h:
                return [TWO_OF_CLUBS]
            return []
        # First trick (no hearts/QS)
        if self.num_played < NUM_PLAYERS:
            return [c for c in h if c != QUEEN_OF_SPADES and card_to_suit(c) != HEARTS_SUIT]
        # Lead: no hearts until broken (but if only hearts remain, allow it)
        if not self.hearts_broken:
            non_hearts = [c for c in h if card_to_suit(c) != HEARTS_SUIT]
            return non_hearts if non_hearts else list(h)
        return list(h)

    # ------------------------------------------------------------------
    # In-place apply / undo — the primary search interface.
    # These avoid object allocation on every minimax node, which is the
    # single largest source of overhead in deep alpha-beta searches.
    # ------------------------------------------------------------------

    def apply_action_inplace(self, card: int):
        """
        Apply card in-place. Returns an opaque undo record.
        Call undo_action(record) to reverse exactly this action.
        """
        player = self.current_player
        old_hearts_broken = self.hearts_broken

        self.hands[player].discard(card)
        self.hand_masks[player] &= ~_CARD_MASK[card]
        if card_to_suit(card) == HEARTS_SUIT or card == QUEEN_OF_SPADES:
            self.hearts_broken = True
        self.num_played += 1

        if len(self.current_trick) == 3:
            # This card completes the trick — save the 3-card state for undo.
            old_trick = list(self.current_trick)
            self.current_trick.append((player, card))

            trick_cards = [self.current_trick[i][1] for i in range(4)]
            trick_players = [self.current_trick[i][0] for i in range(4)]
            lead_suit = card_to_suit(trick_cards[0])
            winner_offset = _trick_winner(lead_suit, trick_cards)
            winner = trick_players[winner_offset]
            pts = _trick_points(trick_cards)

            self.points[winner] = self.points.get(winner, 0) + pts
            self.current_trick = []
            self.current_player = winner
            return (card, player, old_hearts_broken, old_trick, True, winner, pts)
        else:
            self.current_trick.append((player, card))
            self.current_player = (player + 1) % NUM_PLAYERS
            return (card, player, old_hearts_broken, None, False, None, 0)

    def undo_action(self, undo) -> None:
        """Reverse an in-place action produced by apply_action_inplace."""
        card, player, old_hearts_broken, old_trick, completed, winner, pts = undo
        self.hands[player].add(card)
        self.hand_masks[player] |= _CARD_MASK[card]
        self.hearts_broken = old_hearts_broken
        self.current_player = player
        self.num_played -= 1
        if completed:
            self.current_trick = old_trick   # restore the 3-card pre-completion state
            self.points[winner] -= pts
        else:
            self.current_trick.pop()

    def apply_action(self, card: int) -> "PlayState":
        """Return a new state after current_player plays card (kept for external use)."""
        next_state = self.clone()
        next_state.apply_action_inplace(card)
        return next_state

    def is_terminal(self) -> bool:
        return self.num_played >= NUM_CARDS

    def _moon_check_and_adjust(self) -> None:
        """Mutating moon-shot adjustment. Kept for backward compatibility; prefer terminal_score()."""
        for p in range(NUM_PLAYERS):
            if self.points.get(p, 0) == _TOTAL_POINTS:
                self.points[p] = 0
                for q in range(NUM_PLAYERS):
                    if q != p:
                        self.points[q] = _TOTAL_POINTS
                break

    def terminal_score(self, agent_id: int) -> float:
        """
        Non-mutating terminal score with moon-shot adjustment.
        Returns the agent's final point total (lower is better).
        """
        pts = dict(self.points)
        for p in range(NUM_PLAYERS):
            if pts.get(p, 0) == _TOTAL_POINTS:
                pts[p] = 0
                for q in range(NUM_PLAYERS):
                    if q != p:
                        pts[q] = _TOTAL_POINTS
                break
        return float(pts.get(agent_id, 0))


def _state_from_obs_and_world(obs, world: Dict[int, Set[int]], agent_id: int) -> Optional[PlayState]:
    """Build PlayState from observation (trick history, current trick) and world (hands)."""
    obs_arr = _get_obs_from_state(obs, agent_id)
    if obs_arr is None:
        return None
    import numpy as np
    obs_arr = np.asarray(obs_arr)
    trick_history = get_trick_history_from_obs(obs_arr)
    current_trick = get_current_trick_from_obs(obs_arr)
    hands = {p: set(world.get(p, set())) for p in range(NUM_PLAYERS)}
    points: Dict[int, float] = {p: 0.0 for p in range(NUM_PLAYERS)}
    for trick in trick_history:
        players = [trick[i][0] for i in range(4)]
        cards = [trick[i][1] for i in range(4)]
        lead_suit = card_to_suit(cards[0])
        wi = _trick_winner(lead_suit, cards)
        winner = players[wi]
        points[winner] = points.get(winner, 0) + _trick_points(cards)
    # Current player: if current_trick empty, it's the winner of the last completed trick
    if trick_history:
        last = trick_history[-1]
        players = [last[i][0] for i in range(4)]
        cards = [last[i][1] for i in range(4)]
        wi = _trick_winner(card_to_suit(cards[0]), cards)
        current_player = players[wi]
    else:
        # First trick: holder of 2C
        for p in range(NUM_PLAYERS):
            if TWO_OF_CLUBS in hands[p]:
                current_player = p
                break
        else:
            current_player = 0
    if current_trick:
        last_played = current_trick[-1][0]
        current_player = (last_played + 1) % NUM_PLAYERS
    hearts_broken = False
    for trick in trick_history:
        for _pid, c in trick:
            if card_to_suit(c) == HEARTS_SUIT or c == QUEEN_OF_SPADES:
                hearts_broken = True
                break
    if not hearts_broken and current_trick:
        for _pid, c in current_trick:
            if card_to_suit(c) == HEARTS_SUIT or c == QUEEN_OF_SPADES:
                hearts_broken = True
                break
    return PlayState(
        hands=hands,
        current_player=current_player,
        current_trick=current_trick,
        points=points,
        hearts_broken=hearts_broken,
        num_played=len(trick_history) * 4 + len(current_trick),
    )


class WorldSolver:
    """
    Minimax with alpha-beta on a single fully-known world.
    Agent minimizes points (lower is better); moon inverts.
    """

    def __init__(self, max_depth: Optional[int]):
        # Default depth cap keeps early-game search tractable.
        # Set max_depth=None for full solve-to-terminal (can be very slow early game).
        self.max_depth = max_depth
        # Instrumentation counters — reset per best_move() call.
        self.nodes_visited: int = 0
        self.memo_hits: int = 0
        self.ab_cutoffs: int = 0
        self._node_count: int = 0  # internal ticker for time-check throttle

    def best_move(
        self,
        world: Dict[int, Set[int]],
        state: Any,
        agent_id: int = 0,
        time_deadline: Optional[float] = None,
        real_legal: Optional[List[int]] = None,
    ) -> Tuple[int, Dict[int, float]]:
        """
        Given full world and current state (observation source), return
        (best_action, action_scores) where action_scores maps each explored
        candidate action to its minimax value in this world (lower = better
        for the agent). Actions not reached before the time deadline will be
        absent from action_scores.

        real_legal: the actual legal actions for the real game state. When provided,
        the search only considers these actions (preventing the solver from picking a
        move that is legal in the simulated world but illegal in the real game).
        """
        global _last_solver_log
        self.nodes_visited = 0
        self.memo_hits = 0
        self.ab_cutoffs = 0
        self._node_count = 0
        play = _state_from_obs_and_world(state, world, agent_id)
        if play is None:
            fallback = real_legal[0] if real_legal else (_legal_actions_from_state(state, agent_id) or [0])[0]
            return fallback, {}

        # Candidates: intersection of simulated legal actions and real legal actions.
        sim_legal = play.legal_actions()
        if real_legal is not None:
            real_set = set(real_legal)
            legal = [a for a in sim_legal if a in real_set]
            if not legal:
                # Simulated world has diverged from reality — fall back to real legal,
                # ordered by the heuristic (avoid point cards where possible).
                legal = list(real_legal)
                legal.sort(key=lambda c: (card_points(c), card_to_rank(c)))
                return legal[0], {}
        else:
            legal = sim_legal
        if not legal:
            return 0, {}
        if len(legal) == 1:
            return legal[0], {legal[0]: float("inf")}

        best_action = legal[0]
        best_val = float("inf")
        action_scores: Dict[int, float] = {}
        _last_solver_log = time.perf_counter()

        # Single shared memo across all candidate actions so previously explored
        # subtrees benefit later actions.
        memo: dict = {}
        for action in self._order_actions(play, legal):
            if time_deadline is not None and time.perf_counter() >= time_deadline:
                break
            undo = play.apply_action_inplace(action)
            val = self._minimax(
                play,
                agent_id,
                depth=1,
                alpha=-1e9,
                beta=1e9,
                memo=memo,
                time_deadline=time_deadline,
            )
            play.undo_action(undo)
            action_scores[action] = val
            if val < best_val or (val == best_val and card_to_rank(action) > card_to_rank(best_action)):
                best_val = val
                best_action = action
        return best_action, action_scores

    def _order_actions(self, play: PlayState, legal: List[int]) -> List[int]:
        """
        Move ordering + equivalence pruning to improve alpha-beta.

        Following suit:
          - Prefer low cards first (avoid taking the trick).
          - Equivalence pruning: when all legal cards are losing (no card beats
            the current trick high card), consecutive rank blocks within the suit
            are strategically equivalent — only the lowest of each block is kept.
            E.g. holding [4♥, 5♥, 7♥] when 9♥ already leads: 4♥ and 5♥ are in
            the same block (5 is adjacent to 4, no gap), so we only try 4♥.
            7♥ is a separate block (gap at 6♥) so we try it too.
          - When void: dump point cards first (prune the most for the minimizer).
        Leading: low point-risk cards first so safer lines are explored early.
        """
        if play.current_trick:
            lead_suit = card_to_suit(play.current_trick[0][1])
            following = [c for c in legal if card_to_suit(c) == lead_suit]
            if following:
                following_sorted = sorted(following, key=card_to_rank)
                # Find current trick high card in lead suit.
                high_rank = max(
                    card_to_rank(c) for _, c in play.current_trick
                    if card_to_suit(c) == lead_suit
                )
                losers = [c for c in following_sorted if card_to_rank(c) < high_rank]
                winners = [c for c in following_sorted if card_to_rank(c) > high_rank]
                # Apply equivalence pruning only to the losing group.
                # Two adjacent losers are equivalent when the rank directly below
                # the higher card is either already played or in some player's hand
                # (which it is, since all cards exist somewhere). Simplification:
                # prune consecutive ranks — keep only the lowest of each run.
                if len(losers) > 1:
                    pruned = [losers[0]]
                    for i in range(1, len(losers)):
                        prev_rank = card_to_rank(losers[i - 1])
                        curr_rank = card_to_rank(losers[i])
                        if curr_rank > prev_rank + 1:
                            # Gap: different equivalence block, keep it.
                            pruned.append(losers[i])
                        # Adjacent: same block, skip the higher one.
                    losers = pruned
                return losers + winners
            # Void: dump point cards first (QoS, hearts), then low cards
            return sorted(legal, key=lambda c: (-_CARD_POINTS[c], card_to_rank(c)))
        # Leading: low point-risk cards first so we explore safer lines sooner
        return sorted(legal, key=lambda c: (_CARD_POINTS[c], -card_to_rank(c)))

    def _minimax(
        self,
        play: PlayState,
        agent_id: int,
        depth: int,
        alpha: float,
        beta: float,
        memo: dict,
        time_deadline: Optional[float],
    ) -> float:
        global _last_solver_log

        # Time-check throttle: perf_counter() is a syscall (~100ns on macOS).
        # Checking every 256 nodes instead of every node reduces this overhead
        # by ~99% with at most ~256 node overshoot — negligible for correctness.
        self._node_count += 1
        self.nodes_visited += 1
        # if time_deadline is not None and (self._node_count & 255) == 0:
        #     if time.perf_counter() >= time_deadline:
        #         return self._estimate_score(play, agent_id)

        # if _HEARTS_DEBUG:
        #     # now = time.perf_counter()
        #     #print every 100000 minimax searches:
        #     if self.nodes_visited % 100000 == 0:
        #         print(f"      solver: minimax depth={depth} nodes={self.nodes_visited} memo={len(memo)}", file=sys.stderr, flush=True)

        if play.is_terminal():
            return play.terminal_score(agent_id)
        if self.max_depth is not None and depth >= self.max_depth:
            return self._estimate_score(play, agent_id)

        # Transposition key: bitmask per hand (O(1) read) instead of frozenset
        # construction (O(hand_size)) — single biggest per-node cost reduction.
        # remaining_depth is included so states at different depths don't share
        # entries when max_depth is set (full solves use -1, which is fine since
        # all nodes in a full solve share the same remaining horizon).
        remaining = (self.max_depth - depth) if self.max_depth is not None else -1
        key = (
            play.hand_masks[0], play.hand_masks[1],
            play.hand_masks[2], play.hand_masks[3],
            play.current_player,
            tuple(play.current_trick),
            remaining,
        )
        if key in memo:
            self.memo_hits += 1
            return memo[key]

        legal = play.legal_actions()
        if not legal:
            return play.terminal_score(agent_id)

        if play.current_player == agent_id:
            # Minimize our points
            val = float("inf")
            for action in self._order_actions(play, legal):
                undo = play.apply_action_inplace(action)
                v = self._minimax(play, agent_id, depth + 1, alpha, beta, memo, time_deadline)
                play.undo_action(undo)
                if v < val:
                    val = v
                beta = min(beta, val)
                if beta <= alpha:
                    self.ab_cutoffs += 1
                    break
        else:
            # Maximize agent's points (opponent perspective)
            val = float("-inf")
            for action in self._order_actions(play, legal):
                undo = play.apply_action_inplace(action)
                v = self._minimax(play, agent_id, depth + 1, alpha, beta, memo, time_deadline)
                play.undo_action(undo)
                if v > val:
                    val = v
                alpha = max(alpha, val)
                if beta <= alpha:
                    self.ab_cutoffs += 1
                    break

        memo[key] = val
        return val

    def _estimate_score(self, play: PlayState, agent_id: int) -> float:
        """
        Heuristic score at depth cutoff. Returns accumulated points plus
        evaluate_hand(), which estimates future point exposure using the
        structured hand-evaluation heuristic from starterheartsheuristic.py.
        """
        pts = float(play.points.get(agent_id, 0))
        hand = play.hands[agent_id]
        if not hand:
            return pts
        return pts + evaluate_hand(hand, play, agent_id)

    def solve_playstate(
        self,
        play: PlayState,
        agent_id: int,
        real_legal: Optional[List[int]] = None,
        time_deadline: Optional[float] = None,
    ) -> Tuple[int, Dict[int, float]]:
        """
        Run alpha-beta minimax directly on a PlayState (no OpenSpiel state needed).

        Unlike ``best_move``, which reconstructs a PlayState from an OpenSpiel
        observation + world dict, this method accepts a fully-known PlayState
        directly.  Used by the DMCTS loop and the pre-training MCTS player.

        Args:
            play:          Fully-known PlayState (determinized world).
            agent_id:      The player whose points we are minimising.
            real_legal:    Legal actions in the *real* game state.  When
                           provided, only actions in this set are explored so
                           we never return a card that is illegal in reality.
            time_deadline: Optional ``time.perf_counter()`` deadline; search
                           is cut short if exceeded.

        Returns:
            (best_action, action_scores) where action_scores maps each explored
            action to its minimax value (lower = better for agent_id).
        """
        self.nodes_visited = 0
        self.memo_hits     = 0
        self.ab_cutoffs    = 0
        self._node_count   = 0

        legal = play.legal_actions()
        if real_legal is not None:
            real_set = set(real_legal)
            legal    = [a for a in legal if a in real_set]
            if not legal:
                legal = list(real_legal)
                return legal[0], {}
        if not legal:
            return 0, {}
        if len(legal) == 1:
            return legal[0], {legal[0]: float("inf")}

        best_action   = legal[0]
        best_val      = float("inf")
        action_scores: Dict[int, float] = {}
        memo: dict    = {}

        for action in self._order_actions(play, legal):
            if time_deadline is not None and time.perf_counter() >= time_deadline:
                break
            undo = play.apply_action_inplace(action)
            val  = self._minimax(
                play, agent_id, depth=1,
                alpha=-1e9, beta=1e9,
                memo=memo, time_deadline=time_deadline,
            )
            play.undo_action(undo)
            action_scores[action] = val
            if val < best_val:
                best_val    = val
                best_action = action

        return best_action, action_scores


def _legal_actions_from_state(state: Any, agent_id: int) -> List[int]:
    """Extract legal actions from state (timestep or game)."""
    if hasattr(state, "observations") and isinstance(state.observations, dict):
        la = state.observations.get("legal_actions")
        cp = state.observations.get("current_player", agent_id)
        if la is not None:
            return list(la[cp]) if hasattr(la[cp], "__iter__") else list(la)
    if hasattr(state, "get_legal_actions"):
        return list(state.get_legal_actions())
    return []
