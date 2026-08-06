"""
Native Hearts game engine (no OpenSpiel).

Plays a single deal for 3/4/5 players with configurable rules:
- deck construction / hand size per player count,
- simultaneous passing in a configurable direction (no "no-pass" round),
- standard trick-taking (follow suit, hearts-broken, no-blood-on-first-trick),
- lowest-club leads the first trick,
- custom per-deal scoring via ``scoring.score_deal``.

Internally cards are ``(suit, rank)`` tuples for simplicity; the public methods
accept and return ``backend.schemas.types.Card`` objects.
"""
from __future__ import annotations

import random
from typing import Dict, List, Optional, Tuple

from ...schemas.types import Card
from .rules import RuleConfig, RANK_ORDER, SUITS
from .scoring import score_deal, QUEEN_OF_SPADES

CardT = Tuple[str, str]  # (suit, rank)

_SUIT_ORDER = {s: i for i, s in enumerate(SUITS)}


def _sort_key(card: CardT) -> Tuple[int, int]:
    suit, rank = card
    return (_SUIT_ORDER[suit], RANK_ORDER[rank])


def _to_tuple(card: Card) -> CardT:
    return (card.suit, card.rank)


def _to_card(card: CardT) -> Card:
    return Card(suit=card[0], rank=card[1])


class NativeHeartsGame:
    """One deal of Hearts. Not thread-safe; guarded per-room by the caller."""

    def __init__(
        self,
        rules: RuleConfig,
        deal_index: int = 0,
        seed: Optional[int] = None,
    ):
        self.rules = rules
        self.n = rules.player_count
        self.deal_index = deal_index
        self._rng = random.Random(seed)
        self.reset()

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def reset(self) -> None:
        deck = self.rules.build_deck()
        self._rng.shuffle(deck)
        hs = self.rules.hand_size
        self.hands: Dict[int, List[CardT]] = {
            s: sorted(deck[s * hs : (s + 1) * hs], key=_sort_key) for s in range(self.n)
        }
        self.taken: Dict[int, List[CardT]] = {s: [] for s in range(self.n)}
        self.current_trick: List[Tuple[int, CardT]] = []
        # The most recently completed trick, for animation: {cards, winner, points}.
        self.last_trick: Optional[dict] = None
        self.hearts_broken = False
        self.tricks_played = 0
        self.first_trick = True
        self._pending_passes: Dict[int, List[CardT]] = {}
        self._final_scores: Optional[Dict[int, int]] = None
        # Every deal passes (no no-pass round). Passing is simultaneous.
        self.phase = "passing"  # "passing" | "playing" | "done"
        self._current_player: Optional[int] = None

    # ── Queries ───────────────────────────────────────────────────────────────

    def current_player(self) -> int:
        return -1 if self._current_player is None else self._current_player

    def is_passing_phase(self) -> bool:
        return self.phase == "passing"

    def is_terminal(self) -> bool:
        return self.phase == "done"

    def pass_direction(self) -> str:
        return self.rules.pass_direction_label(self.deal_index)

    def pass_receiver(self, seat: int) -> int:
        return self.rules.pass_receiver(seat, self.deal_index)

    def get_player_hand(self, seat: int) -> List[Card]:
        return [_to_card(c) for c in self.hands.get(seat, [])]

    def hand_count(self, seat: int) -> int:
        return len(self.hands.get(seat, []))

    def current_trick_cards(self) -> List[Tuple[int, Card]]:
        return [(s, _to_card(c)) for (s, c) in self.current_trick]

    def has_submitted_pass(self, seat: int) -> bool:
        return seat in self._pending_passes

    def submitted_pass_seats(self) -> List[int]:
        return sorted(self._pending_passes.keys())

    # ── Passing ───────────────────────────────────────────────────────────────

    def submit_pass(self, seat: int, cards: List[Card]) -> bool:
        """
        Record a seat's 3-card pass. When every seat has submitted, all passes
        are applied simultaneously and play begins. Returns True if this call
        triggered the simultaneous apply.
        """
        if self.phase != "passing":
            raise ValueError("Not in passing phase")
        if seat in self._pending_passes:
            raise ValueError("You have already submitted your pass")
        if len(cards) != 3:
            raise ValueError("Must pass exactly 3 cards")

        tuples = [_to_tuple(c) for c in cards]
        if len(set(tuples)) != 3:
            raise ValueError("Pass cards must be distinct")
        hand = set(self.hands[seat])
        for t in tuples:
            if t not in hand:
                raise ValueError(f"Card {t[1]}{t[0]} is not in your hand")

        self._pending_passes[seat] = tuples
        if len(self._pending_passes) == self.n:
            self._apply_all_passes()
            return True
        return False

    def _apply_all_passes(self) -> None:
        outgoing = self._pending_passes
        for seat, cards in outgoing.items():
            for c in cards:
                self.hands[seat].remove(c)
        for seat, cards in outgoing.items():
            receiver = self.pass_receiver(seat)
            self.hands[receiver].extend(cards)
        for seat in range(self.n):
            self.hands[seat].sort(key=_sort_key)
        self._pending_passes = {}
        self.phase = "playing"
        self._current_player = self._first_leader()

    # ── Playing ───────────────────────────────────────────────────────────────

    def _lowest_club_card(self) -> Optional[CardT]:
        best: Optional[CardT] = None
        for hand in self.hands.values():
            for card in hand:
                if card[0] == "C" and (best is None or RANK_ORDER[card[1]] < RANK_ORDER[best[1]]):
                    best = card
        return best

    def _first_leader(self) -> int:
        """Holder of the lowest club leads the first trick (2♣ for 4p; 3♣ when 2♣ removed)."""
        lead = self._lowest_club_card()
        if lead is not None:
            for seat, hand in self.hands.items():
                if lead in hand:
                    return seat
        # No clubs anywhere (astronomically unlikely) — fall back to seat 0.
        return 0

    def legal_moves(self, seat: int) -> List[Card]:
        return [_to_card(c) for c in self._legal_tuples(seat)]

    def _legal_tuples(self, seat: int) -> List[CardT]:
        if self.phase != "playing" or seat != self._current_player:
            return []
        hand = self.hands[seat]

        # Leading a trick.
        if not self.current_trick:
            if self.first_trick:
                lead = self._lowest_club_card()
                if lead in hand:
                    return [lead]
            non_hearts = [c for c in hand if c[0] != "H"]
            if not self.hearts_broken and non_hearts:
                return non_hearts
            return list(hand)

        # Following a trick.
        led_suit = self.current_trick[0][1][0]
        same_suit = [c for c in hand if c[0] == led_suit]
        if same_suit:
            return same_suit
        # Void in led suit: may play anything, but no hearts / Q♠ on the first
        # trick unless that is all that remains.
        if self.first_trick:
            safe = [c for c in hand if c[0] != "H" and c != QUEEN_OF_SPADES]
            if safe:
                return safe
        return list(hand)

    def validate_move(self, seat: int, card: Card) -> bool:
        return _to_tuple(card) in self._legal_tuples(seat)

    def apply_move(self, seat: int, card: Card) -> None:
        if self.phase != "playing":
            raise ValueError("Not in playing phase")
        if seat != self._current_player:
            raise ValueError(f"It is not seat {seat}'s turn")
        t = _to_tuple(card)
        if t not in self._legal_tuples(seat):
            raise ValueError(f"Card {t[1]}{t[0]} is not a legal move for seat {seat}")

        self.hands[seat].remove(t)
        self.current_trick.append((seat, t))
        if t[0] == "H":
            self.hearts_broken = True

        if len(self.current_trick) == self.n:
            self._resolve_trick()
        else:
            self._current_player = (seat + 1) % self.n

    def _resolve_trick(self) -> None:
        led_suit = self.current_trick[0][1][0]
        winner_seat, _ = max(
            ((s, c) for (s, c) in self.current_trick if c[0] == led_suit),
            key=lambda sc: RANK_ORDER[sc[1][1]],
        )
        trick_cards = [(s, c) for (s, c) in self.current_trick]
        self.taken[winner_seat].extend(c for (_, c) in trick_cards)
        trick_points = sum(
            (1 if c[0] == "H" else 13 if (c[0], c[1]) == QUEEN_OF_SPADES else 0)
            for (_, c) in trick_cards
        )
        self.last_trick = {
            "cards": [(s, _to_card(c)) for (s, c) in trick_cards],
            "winner": winner_seat,
            "points": trick_points,
        }
        self.current_trick = []
        self.tricks_played += 1
        self.first_trick = False

        if all(len(h) == 0 for h in self.hands.values()):
            self.phase = "done"
            self._current_player = None
            self._final_scores = score_deal(self.taken, self.rules)
        else:
            self._current_player = winner_seat

    # ── Scores ────────────────────────────────────────────────────────────────

    def final_scores(self) -> Optional[Dict[int, int]]:
        """Custom per-deal scores once the deal is complete, else None."""
        return dict(self._final_scores) if self._final_scores is not None else None

    def running_points(self) -> Dict[int, int]:
        """
        Base penalty points (hearts +1, Q♠ +13) taken so far per seat — a live
        tally for the scoreboard mid-deal. Optional rules are only applied in
        ``final_scores`` at deal end.
        """
        points: Dict[int, int] = {}
        for seat in range(self.n):
            total = 0
            for suit, rank in self.taken[seat]:
                if suit == "H":
                    total += 1
                elif (suit, rank) == QUEEN_OF_SPADES:
                    total += 13
            points[seat] = total
        return points

    def winner(self) -> Optional[int]:
        """Seat with the lowest final score (deal winner)."""
        if self._final_scores is None:
            return None
        return min(self._final_scores, key=lambda s: self._final_scores[s])
