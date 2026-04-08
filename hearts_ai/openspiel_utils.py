"""
OpenSpiel Hearts bridge: card encoding, observation parsing, state helpers.

OpenSpiel encoding: card = rank * 4 + suit
- Suit: 0=Clubs, 1=Diamonds, 2=Hearts, 3=Spades
- Rank: 0=2, 1=3, ..., 12=Ace
"""
from __future__ import annotations

import numpy as np
from typing import List, Tuple, Optional, Any

# Observation layout (from hearts.h / hearts.cc)
OBS_PASS_DIR = (0, 4)
OBS_DEALT_HAND = (4, 56)
OBS_PASSED_CARDS = (56, 108)
OBS_RECEIVED_CARDS = (108, 160)
OBS_CURRENT_HAND = (160, 212)
OBS_POINTS = (212, 356)
OBS_TRICK_HISTORY = (356, 5088)

NUM_PLAYERS = 4
NUM_CARDS = 52
NUM_SUITS = 4
CARDS_PER_SUIT = 13
TRICK_TENSOR_SIZE = 364  # 7 * 52 per trick
NUM_TRICKS = 13

# 2 of Clubs = 0 (rank 0, suit 0)
TWO_OF_CLUBS = 0


def card_to_suit(card: int) -> int:
    """Suit of card: 0=Clubs, 1=Diamonds, 2=Hearts, 3=Spades."""
    return card % NUM_SUITS


def card_to_rank(card: int) -> int:
    """Rank of card: 0=2, 1=3, ..., 12=Ace."""
    return card // NUM_SUITS


def card_points(card: int, hearts_broken: bool = True) -> int:
    """Points for a card. QS=13, each heart=1, JD optional -10 (not used by default)."""
    s = card_to_suit(card)
    r = card_to_rank(card)
    if s == 2:  # Hearts
        return 1
    if s == 3 and r == 10:  # Queen of Spades
        return 13
    return 0


def get_trick_history_from_obs(obs: np.ndarray) -> List[List[Tuple[int, int]]]:
    """
    Parse observation tensor into list of completed tricks.
    Each trick is list of (player_id, card) in play order (leader first).

    OpenSpiel C++ layout per trick (364 bytes = 7 × 52):
      - Segments 0 .. leader-1:   zero-padded
      - Segment  leader:           leader's card (one-hot, 52 bits)
      - Segment  leader+1:         next player's card
      - Segment  leader+2:         ...
      - Segment  leader+3:         last player's card
      - Segments leader+4 .. 6:   zero-padded

    Because of this layout, segment_index % NUM_PLAYERS == player_id directly,
    regardless of who the trick leader is.
    """
    if obs is None or len(obs) < OBS_TRICK_HISTORY[1]:
        return []
    arr = obs[OBS_TRICK_HISTORY[0] : OBS_TRICK_HISTORY[1]]
    tricks: List[List[Tuple[int, int]]] = []
    for t in range(NUM_TRICKS):
        base = t * TRICK_TENSOR_SIZE
        trick_cards: List[Tuple[int, int]] = []
        for seg in range(7):
            block = arr[base + seg * NUM_CARDS : base + (seg + 1) * NUM_CARDS]
            if np.any(block > 0):
                card = int(np.argmax(block))
                player = seg % NUM_PLAYERS
                trick_cards.append((player, card))
        if len(trick_cards) < 4:
            break
        tricks.append(trick_cards)
    return tricks


def get_current_trick_from_obs(obs: np.ndarray) -> List[Tuple[int, int]]:
    """
    Current (incomplete) trick as (player_id, card) list, leader first.
    Uses the same segment-index % NUM_PLAYERS == player_id encoding.
    """
    if obs is None or len(obs) < OBS_TRICK_HISTORY[1]:
        return []
    arr = obs[OBS_TRICK_HISTORY[0] : OBS_TRICK_HISTORY[1]]
    completed = get_trick_history_from_obs(obs)
    num_completed = len(completed)
    if num_completed >= NUM_TRICKS:
        return []
    base = num_completed * TRICK_TENSOR_SIZE
    current: List[Tuple[int, int]] = []
    for seg in range(7):
        block = arr[base + seg * NUM_CARDS : base + (seg + 1) * NUM_CARDS]
        if np.any(block > 0):
            card = int(np.argmax(block))
            player = seg % NUM_PLAYERS
            current.append((player, card))
    return current


def get_lead_suit_from_trick(trick_cards: List[Tuple[int, int]]) -> Optional[int]:
    """Lead suit of the trick (suit of first card), or None if empty."""
    if not trick_cards:
        return None
    return card_to_suit(trick_cards[0][1])


def cards_in_hand_from_obs(obs: np.ndarray) -> List[int]:
    """Agent's current hand from observation (160:212) as list of card indices."""
    if obs is None or len(obs) < OBS_CURRENT_HAND[1]:
        return []
    hand_slice = obs[OBS_CURRENT_HAND[0] : OBS_CURRENT_HAND[1]]
    return [c for c in range(NUM_CARDS) if hand_slice[c] > 0]


def get_lead_suit(state: Any) -> Optional[int]:
    """
    Lead suit of current trick from state.
    state can be OpenSpiel-like (has observation) or provide obs + current_trick.
    """
    if hasattr(state, "observations"):
        obs = state.observations.get("info_state") if isinstance(state.observations, dict) else None
        if obs is not None:
            cp = state.observations.get("current_player", 0)
            cur = get_current_trick_from_obs(obs[cp] if hasattr(obs, "__getitem__") else obs)
            return get_lead_suit_from_trick(cur)
    if hasattr(state, "get_observation") and hasattr(state, "current_player"):
        obs = state.get_observation(state.current_player())
        cur = get_current_trick_from_obs(np.asarray(obs))
        return get_lead_suit_from_trick(cur)
    return None


def get_trick_history(state: Any, player_id: int = 0) -> List[List[Tuple[int, int]]]:
    """Full trick history from state (observation for player_id)."""
    obs = _get_obs_from_state(state, player_id)
    return get_trick_history_from_obs(obs) if obs is not None else []


def get_current_trick(state: Any, player_id: int = 0) -> List[Tuple[int, int]]:
    """Current incomplete trick from state."""
    obs = _get_obs_from_state(state, player_id)
    return get_current_trick_from_obs(obs) if obs is not None else []


def _get_obs_from_state(state: Any, player_id: int) -> Optional[np.ndarray]:
    if state is None:
        return None
    # Already the per-player observation tensor (e.g. from ts.observations["info_state"][cp]).
    # Raw arrays are not wrapped in a timestep, so the branches below would return None.
    if isinstance(state, np.ndarray):
        return np.asarray(state, dtype=np.float32)
    if hasattr(state, "observations") and isinstance(state.observations, dict):
        info = state.observations.get("info_state")
        if info is not None:
            o = info[player_id] if hasattr(info, "__getitem__") else info
            return np.asarray(o, dtype=np.float32)
    if hasattr(state, "get_observation"):
        o = state.get_observation(player_id)
        return np.asarray(o, dtype=np.float32) if o is not None else None
    return None


def cards_in_hand(state: Any, player_id: int) -> List[int]:
    """Cards in hand for player_id from state (only current player's hand is visible in obs)."""
    obs = _get_obs_from_state(state, player_id)
    return cards_in_hand_from_obs(obs) if obs is not None else []


def all_played_cards_from_history(trick_history: List[List[Tuple[int, int]]]) -> set:
    """Set of all cards that have been played (from full trick history)."""
    out = set()
    for trick in trick_history:
        for _pid, card in trick:
            out.add(card)
    return out


def count_cards_played(trick_history: List[List[Tuple[int, int]]], current_trick: List[Tuple[int, int]]) -> int:
    """Total number of cards played so far."""
    return len(trick_history) * 4 + len(current_trick)


def clone_state_with_world(state: Any, world: dict, agent_id: int) -> Any:
    """
    Create a copy of state with opponents' hands set to the sampled world.
    Used by WorldSolver for minimax on a determinized world.
    If state is from RL env (no raw OpenSpiel state), returns (state, world) for internal use.
    """
    # Backend uses RL env; we don't have raw OpenSpiel state. Return (state, world) so
    # WorldSolver uses internal simulator with world.
    return (state, world)
