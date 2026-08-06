"""
Per-deal scoring for the native Hearts engine.

Pure functions over "who took which cards", so they can be unit-tested without
running a whole game. See ``customscoring.md`` for the source rules; the moon
algorithm here reproduces every worked example in that doc.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

from .rules import RuleConfig

Card = Tuple[str, str]  # (suit, rank)

QUEEN_OF_SPADES: Card = ("S", "Q")
JACK_OF_DIAMONDS: Card = ("D", "J")
TEN_OF_CLUBS: Card = ("C", "T")

# Base penalty points available in a standard deal: 13 hearts + Q♠ = 26.
MOON_TOTAL = 26


def _base_penalty(cards: List[Card]) -> int:
    """Standard hearts (+1 each) and Q♠ (+13), before any optional rule."""
    points = 0
    for suit, rank in cards:
        if suit == "H":
            points += 1
        elif (suit, rank) == QUEEN_OF_SPADES:
            points += 13
    return points


def _holder(taken: Dict[int, List[Card]], card: Card) -> int | None:
    for seat, cards in taken.items():
        if card in cards:
            return seat
    return None


def _moon_shooter(taken: Dict[int, List[Card]]) -> int | None:
    """
    Return the seat that took ALL 13 hearts and the Q♠ (a shot moon), else None.
    JD / 10♣ are irrelevant to whether a moon occurred.
    """
    for seat, cards in taken.items():
        hearts = sum(1 for suit, _ in cards if suit == "H")
        has_qs = QUEEN_OF_SPADES in cards
        if hearts == 13 and has_qs:
            return seat
    return None


def score_deal(
    taken_cards_by_seat: Dict[int, List[Card]],
    rules: RuleConfig,
) -> Dict[int, int]:
    """
    Compute each seat's penalty points for one completed deal.

    Args:
        taken_cards_by_seat: seat -> list of every card that seat won in tricks.
        rules: active rule config (player count + optional toggles).

    Returns:
        seat -> integer score for the deal (can be negative).
    """
    seats = list(range(rules.player_count))
    taken = {s: taken_cards_by_seat.get(s, []) for s in seats}

    jd_seat = _holder(taken, JACK_OF_DIAMONDS) if rules.jd_bonus else None
    ten_seat = _holder(taken, TEN_OF_CLUBS) if rules.ten_club_doubler else None

    shooter = _moon_shooter(taken)

    if shooter is not None:
        # 1. base moon: shooter 0, everyone else 26
        scores = {s: (0 if s == shooter else MOON_TOTAL) for s in seats}
        # 2. Jack of Diamonds (scored before the 10)
        if jd_seat is not None:
            scores[jd_seat] += -10
        # 3. Ten of Clubs doubler
        if ten_seat is not None:
            if ten_seat == shooter:
                # shooter took the 10 -> everyone doubles (incl. negatives)
                scores = {s: v * 2 for s, v in scores.items()}
            else:
                scores[ten_seat] *= 2
        return scores

    # Non-moon deal.
    scores = {s: _base_penalty(taken[s]) for s in seats}
    # Jack of Diamonds before the 10 of Clubs.
    if jd_seat is not None:
        scores[jd_seat] += -10
    if ten_seat is not None:
        scores[ten_seat] *= 2
    return scores
