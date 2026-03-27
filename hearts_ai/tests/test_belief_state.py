"""Tests for BeliefState."""
import pytest
from hearts_ai.belief_state import BeliefState
from hearts_ai.openspiel_utils import card_to_suit, NUM_CARDS

# OpenSpiel: suit = card % 4 (C=0, D=1, H=2, S=3), rank = card // 4


def test_void_after_play():
    """After player voids spades, card_probs for spades should be 0 for that player."""
    # Spades = suit 3. Cards 39-51 are spades (rank 9..12 * 4 + 3 = 39,43,47,51; actually 3,7,...,51)
    belief = BeliefState(my_hand=[0, 1, 2], num_players=4, agent_id=0)
    belief.played_cards = set()
    belief.voids = {0: set(), 1: set(), 2: set(), 3: set()}
    # Player 1 plays a diamond (suit 1) when lead was spades (suit 3) -> void spades for player 1
    belief.observe_card_played(1, 5, lead_suit=3, trick_cards_so_far=[])
    assert 3 in belief.voids[1]
    # Any spade card should not be assignable to player 1 in sampling (handled by sampler)


def test_played_cards_removed():
    """After all cards are played, sample_possible_world still returns valid assignment."""
    belief = BeliefState(my_hand=[], num_players=4, agent_id=0)
    belief.played_cards = set(range(52))
    belief.my_hand = []
    belief._cards_remaining = {0: 0, 1: 0, 2: 0, 3: 0}
    # No unknown cards; sampler should return known only
    world = belief.sample_possible_world()
    assert world is not None
    for p in range(4):
        assert len(world[p]) == 0


def test_sample_possible_world_valid():
    """sample_possible_world returns valid assignment: no void violations, correct counts."""
    my_hand = [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48]  # 13 cards
    belief = BeliefState(my_hand=my_hand, num_players=4, agent_id=0)
    belief.played_cards = set()
    belief._cards_remaining = {0: 13, 1: 13, 2: 13, 3: 13}
    world = belief.sample_possible_world()
    assert world is not None
    assert len(world[0]) == 13
    assert len(world[1]) == 13
    assert len(world[2]) == 13
    assert len(world[3]) == 13
    all_cards = set()
    for p in range(4):
        all_cards |= world[p]
    assert len(all_cards) == 52
    assert all_cards == set(range(52))
    # Agent's hand should be in world[0]
    assert set(my_hand) <= world[0]


def test_void_respected_in_sample():
    """With one player void in spades, sampled world gives them no spades."""
    my_hand = list(range(13))  # 0-12 (all clubs and some diamonds)
    belief = BeliefState(my_hand=my_hand, num_players=4, agent_id=0)
    belief.played_cards = set()
    belief.voids[1] = {3}  # Player 1 void in spades (suit 3)
    belief._cards_remaining = {0: 13, 1: 13, 2: 13, 3: 13}
    for _ in range(20):
        world = belief.sample_possible_world()
        assert world is not None
        for c in world[1]:
            assert card_to_suit(c) != 3
