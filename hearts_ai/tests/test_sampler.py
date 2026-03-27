"""Tests for ConstrainedDealSampler."""
import random
from collections import Counter
import pytest
from hearts_ai.sampler import ConstrainedDealSampler
from hearts_ai.openspiel_utils import card_to_suit, NUM_PLAYERS, NUM_CARDS


def test_no_constraints_uniform():
    """With no constraints, distribution of card-to-player should be approximately uniform (chi-squared)."""
    unknown = set(range(NUM_CARDS))
    voids = {p: set() for p in range(NUM_PLAYERS)}
    cards_needed = {p: 13 for p in range(NUM_PLAYERS)}
    known_holdings = {p: set() for p in range(NUM_PLAYERS)}
    sampler = ConstrainedDealSampler(unknown, voids, cards_needed, known_holdings)
    rng = random.Random(42)
    # Count how often card 0 goes to player 0 over many samples
    counts = Counter()
    for _ in range(80):
        w = sampler.sample(rng)
        assert w is not None
        for p in range(NUM_PLAYERS):
            assert len(w[p]) == 13
        for card in range(NUM_CARDS):
            for p in range(NUM_PLAYERS):
                if card in w[p]:
                    counts[(card, p)] += 1
                    break
    # Each (card, player) should appear ~20/80 = 0.25
    for card in range(min(5, NUM_CARDS)):
        for p in range(NUM_PLAYERS):
            c = counts.get((card, p), 0)
            assert 5 <= c <= 45  # rough uniformity


def test_tight_constraints():
    """With tight constraints (e.g. one player void in two suits), still find valid assignment."""
    # Player 0 gets only clubs (0) and diamonds (1); player 1 only hearts (2) and spades (3)
    unknown = set(range(NUM_CARDS))
    voids = {
        0: {2, 3},
        1: {0, 1},
        2: set(),
        3: set(),
    }
    cards_needed = {0: 13, 1: 13, 2: 13, 3: 13}
    known_holdings = {p: set() for p in range(NUM_PLAYERS)}
    sampler = ConstrainedDealSampler(unknown, voids, cards_needed, known_holdings)
    rng = random.Random(123)
    for _ in range(10):
        w = sampler.sample(rng)
        assert w is not None
        for c in w[0]:
            assert card_to_suit(c) in (0, 1)
        for c in w[1]:
            assert card_to_suit(c) in (2, 3)


def test_impossible_constraints():
    """With impossible constraints, sampler may return None (or fail after restarts)."""
    unknown = set(range(20))  # only 20 cards
    voids = {0: {0, 1, 2, 3}, 1: set(), 2: set(), 3: set()}  # player 0 void in all
    cards_needed = {0: 10, 1: 4, 2: 3, 3: 3}
    known_holdings = {p: set() for p in range(NUM_PLAYERS)}
    sampler = ConstrainedDealSampler(unknown, voids, cards_needed, known_holdings)
    rng = random.Random(456)
    result = sampler.sample(rng)
    # Player 0 needs 10 cards but is void in all suits -> impossible
    assert result is None or len(result[0]) < 10
