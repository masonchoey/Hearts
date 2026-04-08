"""
BeliefState: probability distribution over opponent holdings, updated from play history.
Tracks voids, played cards, and samples constraint-consistent possible worlds.
"""
from __future__ import annotations

import random
from typing import Dict, Set, List, Optional

import numpy as np

from .openspiel_utils import (
    card_to_suit,
    NUM_PLAYERS,
    NUM_CARDS,
    get_trick_history_from_obs,
    get_current_trick_from_obs,
    all_played_cards_from_history,
    count_cards_played,
)
from .sampler import ConstrainedDealSampler


class BeliefState:
    """
    Tracks constraints and probability over unseen cards.
    - my_hand: agent's current hand (known)
    - played_cards: all cards played so far
    - voids[p]: suits player p is void in
    - known_holdings[p]: cards we know p holds (e.g. from pass to us)
    - card_probs: card -> {player_id: unnormalized prob}; we renormalize after updates
    """

    def __init__(self, my_hand: List[int], num_players: int = 4, agent_id: int = 0):
        self.agent_id = agent_id
        self.num_players = num_players
        self.my_hand = list(my_hand)
        self.played_cards: Set[int] = set()
        self.voids: Dict[int, Set[int]] = {p: set() for p in range(num_players)}
        self.known_holdings: Dict[int, Set[int]] = {p: set() for p in range(num_players)}
        # card_probs[card][player_id] = unnormalized probability; missing = 0
        self.card_probs: Dict[int, Dict[int, float]] = {}
        # Cards remaining per player (updated from state)
        self._cards_remaining: Dict[int, int] = {p: 13 for p in range(num_players)}

    def observe_card_played(
        self,
        player_id: int,
        card: int,
        lead_suit: Optional[int],
        trick_cards_so_far: List[tuple],
    ) -> None:
        """Update belief: card was played by player_id; detect void if they didn't follow suit."""
        self.played_cards.add(card)
        # Remove card from all card_probs
        if card in self.card_probs:
            del self.card_probs[card]
        for c in self.card_probs:
            self.card_probs[c].pop(player_id, None)
        # Void: first time this player didn't follow lead suit
        if lead_suit is not None and card_to_suit(card) != lead_suit:
            self.voids[player_id].add(lead_suit)
        self._renormalize_card_probs()

    def _renormalize_card_probs(self) -> None:
        """Zero out impossible assignments (voids, already played, in my_hand), redistribute mass."""
        my_hand_set = set(self.my_hand)
        for card in list(self.card_probs.keys()):
            if card in self.played_cards or card in my_hand_set:
                del self.card_probs[card]
                continue
            probs = self.card_probs[card]
            for p in list(probs.keys()):
                if card_to_suit(card) in self.voids.get(p, set()):
                    del probs[p]
            if not probs:
                del self.card_probs[card]
        # Optional: normalize so each card sums to 1 over players (we use uniform prior in sampling anyway)
        for card in self.card_probs:
            probs = self.card_probs[card]
            total = sum(probs.values())
            if total > 0:
                for p in probs:
                    probs[p] /= total

    def observe_pass(self, receiver_id: int, cards_received: List[int]) -> None:
        """We only know cards passed to us (receiver_id == agent_id). Lock those to us."""
        if receiver_id != self.agent_id:
            return
        for card in cards_received:
            self.known_holdings[self.agent_id].add(card)
            if card in self.card_probs:
                del self.card_probs[card]
            for c in self.card_probs:
                self.card_probs[c].pop(receiver_id, None)
        self._renormalize_card_probs()

    def sample_possible_world(self, rng: Optional[random.Random] = None) -> Dict[int, Set[int]]:
        """
        One consistent assignment of all unplayed, unseen cards to players.
        Returns {player_id: set of cards} for all players (including agent).
        """
        rng = rng or random.Random()
        unknown_cards = set(range(NUM_CARDS)) - self.played_cards - set(self.my_hand)
        # Known holdings: agent's hand + any locked cards
        known = {p: set(self.known_holdings.get(p, set())) for p in range(self.num_players)}
        known[self.agent_id].update(self.my_hand)
        # Cards to assign: unknown minus any already in known
        to_assign = unknown_cards - set().union(*known.values())
        cards_needed = {}
        for p in range(self.num_players):
            cards_needed[p] = self.get_cards_remaining(p)
        # Slots for non-agent: cards_needed - len(known)
        for p in range(self.num_players):
            need = cards_needed[p]
            have = len(known.get(p, set()))
            if have > need:
                # Inconsistent state; still try to assign
                cards_needed[p] = have
            else:
                cards_needed[p] = need
        sampler = ConstrainedDealSampler(
            unknown_cards=unknown_cards,
            voids=self.voids,
            cards_needed=cards_needed,
            known_holdings=known,
        )
        result = sampler.sample(rng)
        if result is None:
            # Fallback: assign remaining cards arbitrarily (shouldn't happen if consistent)
            result = {p: set() for p in range(self.num_players)}
            for p in range(self.num_players):
                result[p].update(known.get(p, set()))
            remaining = list(to_assign)
            rng.shuffle(remaining)
            idx = 0
            for p in range(self.num_players):
                need = cards_needed[p] - len(result[p])
                for _ in range(need):
                    if idx < len(remaining):
                        result[p].add(remaining[idx])
                        idx += 1
        else:
            # Ensure agent's hand is in the result
            result[self.agent_id] = set(self.my_hand) | result.get(self.agent_id, set())
        return result

    def get_cards_remaining(self, player_id: int) -> int:
        """Number of cards that player still holds (use _cards_remaining from state sync)."""
        return self._cards_remaining.get(player_id, 13)

    def set_cards_remaining(self, player_id: int, n: int) -> None:
        self._cards_remaining[player_id] = n

    def update_from_openspiel_state(self, state) -> None:
        """
        Sync belief from state: replay history to update played_cards and voids.
        state: object with get_observation(player_id) and current_player(), or observations["info_state"].
        """
        from .openspiel_utils import _get_obs_from_state, get_lead_suit_from_trick, cards_in_hand_from_obs

        obs = _get_obs_from_state(state, self.agent_id)
        if obs is None:
            return
        obs = np.asarray(obs) if not isinstance(obs, np.ndarray) else obs
        trick_history = get_trick_history_from_obs(obs)
        current_trick = get_current_trick_from_obs(obs)
        # Cards remaining per player
        played_by_player: Dict[int, Set[int]] = {p: set() for p in range(self.num_players)}
        for trick in trick_history:
            for pid, card in trick:
                played_by_player[pid].add(card)
        for pid, card in current_trick:
            played_by_player[pid].add(card)
        for p in range(self.num_players):
            self._cards_remaining[p] = 13 - len(played_by_player[p])
        # Build played_cards and voids from full history (don't call observe_card_played to avoid double-apply)
        self.played_cards = set().union(*played_by_player.values())
        for trick in trick_history:
            if not trick:
                continue
            lead_suit = card_to_suit(trick[0][1])
            for i, (pid, card) in enumerate(trick):
                if i > 0 and card_to_suit(card) != lead_suit:
                    self.voids[pid].add(lead_suit)
        if current_trick:
            lead_suit = card_to_suit(current_trick[0][1])
            for i, (pid, card) in enumerate(current_trick):
                if i > 0 and card_to_suit(card) != lead_suit:
                    self.voids[pid].add(lead_suit)
        self.my_hand = cards_in_hand_from_obs(obs)
        # Clear card_probs for played/held cards and renormalize
        self._renormalize_card_probs()
