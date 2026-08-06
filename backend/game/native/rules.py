"""
Rule configuration for the native Hearts engine.

Owns everything that varies by player count or host toggle:
- which cards are removed from the deck (and therefore hand size),
- how seats are arranged and what each pass direction means,
- which optional scoring rules are active.

Seat geometry convention
-------------------------
Seats are numbered ``0 .. player_count-1`` arranged clockwise around the table.
"Right" (the neighbour you pass to first) is the next seat clockwise, i.e.
``(seat + 1) % player_count``. All pass directions are expressed as a positive
clockwise offset from the passer to the receiver.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple

# Card identity helpers. A card is a (suit, rank) tuple of single-char codes,
# matching backend.schemas.types.Card.suit / .rank.
SUITS = ("C", "D", "H", "S")
RANKS = ("2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K", "A")
RANK_ORDER: Dict[str, int] = {r: i for i, r in enumerate(RANKS)}

SUPPORTED_PLAYER_COUNTS = (3, 4, 5)

# Cards removed from the standard 52-card deck for each player count so the deck
# divides evenly. 3p: 51/3 = 17 each. 4p: 52/4 = 13 each. 5p: 50/5 = 10 each.
_REMOVED_CARDS: Dict[int, Tuple[Tuple[str, str], ...]] = {
    3: (("C", "2"),),
    4: (),
    5: (("C", "2"), ("D", "2")),
}

# Clockwise pass-direction sequence per player count, as offsets from passer to
# receiver. Deal 0 always passes "right" (offset 1). Only the first entry matters
# for the current single-deal scope; the full sequence is here for the separate
# multi-deal branch to consume.
#   4p: right(1) -> across(2) -> left(3)
#   5p: right(1) -> two-right(2) -> two-left(3) -> left(4)
#   3p: right(1) -> left(2)   (doc unspecified beyond first deal; sensible default)
_PASS_SEQUENCE: Dict[int, Tuple[int, ...]] = {
    3: (1, 2),
    4: (1, 2, 3),
    5: (1, 2, 3, 4),
}


@dataclass(frozen=True)
class RuleConfig:
    """Immutable per-game rule set."""

    player_count: int = 4
    jd_bonus: bool = False          # Jack of Diamonds is worth -10
    ten_club_doubler: bool = False  # taker of 10 of Clubs doubles their points

    def __post_init__(self) -> None:
        if self.player_count not in SUPPORTED_PLAYER_COUNTS:
            raise ValueError(
                f"player_count must be one of {SUPPORTED_PLAYER_COUNTS}, "
                f"got {self.player_count}"
            )

    # ── Deck ────────────────────────────────────────────────────────────────

    @property
    def removed_cards(self) -> Set[Tuple[str, str]]:
        return set(_REMOVED_CARDS[self.player_count])

    def build_deck(self) -> List[Tuple[str, str]]:
        """Full ordered deck for this player count (removed cards excluded)."""
        removed = self.removed_cards
        return [
            (suit, rank)
            for suit in SUITS
            for rank in RANKS
            if (suit, rank) not in removed
        ]

    @property
    def hand_size(self) -> int:
        return len(self.build_deck()) // self.player_count

    # ── Seat geometry / passing ──────────────────────────────────────────────

    def pass_offset_for_deal(self, deal_index: int) -> int:
        """Clockwise seat offset (passer -> receiver) for the given deal."""
        seq = _PASS_SEQUENCE[self.player_count]
        return seq[deal_index % len(seq)]

    def pass_receiver(self, passer_seat: int, deal_index: int = 0) -> int:
        return (passer_seat + self.pass_offset_for_deal(deal_index)) % self.player_count

    def pass_direction_label(self, deal_index: int = 0) -> str:
        """Human-readable label for the deal's pass direction."""
        offset = self.pass_offset_for_deal(deal_index)
        # Distance measured the short way around the table.
        n = self.player_count
        if offset == 1:
            return "Right"
        if offset == n - 1:
            return "Left"
        if offset == 2:
            return "Two Right"
        if offset == n - 2:
            return "Two Left"
        if offset * 2 == n:
            return "Across"
        return f"+{offset}"

    def to_dict(self) -> Dict:
        return {
            "player_count": self.player_count,
            "jd_bonus": self.jd_bonus,
            "ten_club_doubler": self.ten_club_doubler,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "RuleConfig":
        data = data or {}
        return cls(
            player_count=int(data.get("player_count", 4)),
            jd_bonus=bool(data.get("jd_bonus", False)),
            ten_club_doubler=bool(data.get("ten_club_doubler", False)),
        )
