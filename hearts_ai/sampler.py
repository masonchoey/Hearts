"""
Constrained deal sampler: uniform samples over card assignments that respect
voids and card counts. Used by BeliefState.sample_possible_world().
"""
from __future__ import annotations

import random
from typing import Dict, Set, List, Optional

from .openspiel_utils import card_to_suit, NUM_PLAYERS, NUM_CARDS


class ConstrainedDealSampler:
    """
    Samples a valid assignment of unknown_cards to players such that:
    - Each player p gets exactly cards_needed[p] cards.
    - For each card c, if c's suit is in voids[p], c cannot be assigned to p.
    - known_holdings[p] are already fixed for player p (and count toward cards_needed[p]).
    """

    def __init__(
        self,
        unknown_cards: Set[int],
        voids: Dict[int, Set[int]],
        cards_needed: Dict[int, int],
        known_holdings: Dict[int, Set[int]],
    ):
        self.unknown_cards = set(unknown_cards)
        self.voids = {p: set(voids.get(p, set())) for p in range(NUM_PLAYERS)}
        self.cards_needed = dict(cards_needed)
        self.known_holdings = {p: set(known_holdings.get(p, set())) for p in range(NUM_PLAYERS)}
        # Cards we must assign (exclude known)
        self.to_assign = list(self.unknown_cards - set().union(*(self.known_holdings.values())))
        # Remaining slots per player (after known)
        self.slots: Dict[int, int] = {}
        for p in range(NUM_PLAYERS):
            self.slots[p] = self.cards_needed.get(p, 0) - len(self.known_holdings.get(p, set()))

    def _can_receive(self, card: int, player: int) -> bool:
        if self.slots.get(player, 0) <= 0:
            return False
        suit = card_to_suit(card)
        return suit not in self.voids.get(player, set())

    def sample(self, rng: Optional[random.Random] = None) -> Optional[Dict[int, Set[int]]]:
        """
        Returns one valid assignment {player_id: set of cards}, or None if impossible.
        Fast path when no voids; otherwise randomized assignment with limited backtracking.
        """
        rng = rng or random.Random()
        if not self.voids or all(not self.voids[p] for p in range(NUM_PLAYERS)):
            return self._sample_no_constraints(rng)
        for _restart in range(10):
            result = self._try_sample(rng)
            if result is not None:
                return result
        return None

    def _sample_no_constraints(self, rng: random.Random) -> Dict[int, Set[int]]:
        """No voids: shuffle and assign first n0 to p0, next n1 to p1, etc."""
        cards = list(self.to_assign)
        rng.shuffle(cards)
        out: Dict[int, Set[int]] = {p: set(self.known_holdings.get(p, set())) for p in range(NUM_PLAYERS)}
        k = 0
        for p in range(NUM_PLAYERS):
            n = self.slots[p]
            for _ in range(n):
                if k < len(cards):
                    out[p].add(cards[k])
                    k += 1
        return out

    def _try_sample(self, rng: random.Random) -> Optional[Dict[int, Set[int]]]:
        slots = {p: self.slots[p] for p in range(NUM_PLAYERS)}
        cards = list(self.to_assign)
        rng.shuffle(cards)
        assignment: Dict[int, List[int]] = {p: [] for p in range(NUM_PLAYERS)}
        for p in range(NUM_PLAYERS):
            assignment[p] = list(self.known_holdings.get(p, set()))

        backtrack_limit = 50
        backtrack_count = 0
        i = 0
        while i < len(cards):
            card = cards[i]
            valid_players = [p for p in range(NUM_PLAYERS) if slots[p] > 0 and self._can_receive(card, p)]
            if not valid_players:
                if backtrack_count >= backtrack_limit:
                    return None
                backtrack_count += 1
                if i == 0:
                    return None
                i -= 1
                card = cards[i]
                for p in range(NUM_PLAYERS):
                    if card in assignment[p]:
                        assignment[p].remove(card)
                        slots[p] += 1
                        break
                continue
            weights = [slots[p] for p in valid_players]
            total = sum(weights)
            if total <= 0:
                if backtrack_count >= backtrack_limit:
                    return None
                backtrack_count += 1
                if i == 0:
                    return None
                i -= 1
                card = cards[i]
                for p in range(NUM_PLAYERS):
                    if card in assignment[p]:
                        assignment[p].remove(card)
                        slots[p] += 1
                        break
                continue
            r = rng.random() * total
            idx = 0
            for k, p in enumerate(valid_players):
                r -= weights[k]
                if r <= 0:
                    idx = k
                    break
            chosen = valid_players[idx]
            assignment[chosen].append(card)
            slots[chosen] -= 1
            i += 1

        out = {p: set(assignment[p]) for p in range(NUM_PLAYERS)}
        return out
