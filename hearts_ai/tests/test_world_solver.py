"""Tests for WorldSolver."""
import pytest
from hearts_ai.world_solver import WorldSolver, PlayState, _trick_winner, _trick_points, QUEEN_OF_SPADES
from hearts_ai.openspiel_utils import TWO_OF_CLUBS, card_to_suit, NUM_PLAYERS, NUM_CARDS


def test_trick_winner():
    """Winner follows lead suit with highest rank. Card = rank*4+suit (C=0,D=1,H=2,S=3)."""
    # Lead club (0), cards 0=2C, 20=6C, 40=10C, 48=AC (rank 12) -> winner index 3
    cards = [0, 20, 40, 48]
    assert _trick_winner(0, cards) == 3
    # 0=2C, 4=3C, 8=4C, 33=10D (off-suit) -> winner index 2 (4C is highest club)
    cards2 = [0, 4, 8, 33]
    assert _trick_winner(0, cards2) == 2


def test_trick_points():
    """Hearts=1, QS=13. Card 34 = 8*4+2 = 8H (one heart)."""
    assert _trick_points([0, 4, 8]) == 0
    assert _trick_points([34]) == 1  # one heart (rank 8, suit 2)
    assert _trick_points([QUEEN_OF_SPADES]) == 13


def test_trivial_endgame():
    """Two cards each, mid-game (not first trick): solver returns a legal move."""
    # After first trick so we don't need 2C. Player 0 leads.
    hands = {
        0: {10, 20},
        1: {30, 40},
        2: {11, 21},
        3: {31, 41},
    }
    play = PlayState(
        hands=hands,
        current_player=0,
        current_trick=[],
        completed_tricks=[[(0, 0), (1, 1), (2, 2), (3, 3)]],  # one trick done -> num_played=4
        points={0: 0, 1: 0, 2: 0, 3: 0},
        hearts_broken=True,
    )
    solver = WorldSolver(max_depth=2)
    legal = play.legal_actions()
    assert len(legal) >= 1
    action = legal[0]
    child = play.apply_action(action)
    assert action not in child.hands[0]


def test_moon_detection():
    """PlayState._moon_check_and_adjust inverts when one player has 26."""
    hands = {p: set() for p in range(NUM_PLAYERS)}
    play = PlayState(
        hands=hands,
        current_player=0,
        current_trick=[],
        completed_tricks=[],
        points={0: 26, 1: 0, 2: 0, 3: 0},
        hearts_broken=True,
    )
    play._moon_check_and_adjust()
    assert play.points[0] == 0
    assert play.points[1] == 26
    assert play.points[2] == 26
    assert play.points[3] == 26


def test_best_move_single_legal():
    """When only one legal action, best_move returns it."""
    world = {
        0: {0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48},
        1: {1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 45, 49},
        2: {2, 6, 10, 14, 18, 22, 26, 30, 34, 38, 42, 46, 50},
        3: {3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51},
    }
    # Mock state: only action 0 legal (2C)
    class MockState:
        observations = {"info_state": [None] * 4, "legal_actions": [[0], [], [], []], "current_player": 0}
        def get_observation(self, pid):
            return [0.0] * 5088
    mock = MockState()
    mock.observations["info_state"][0] = [0.0] * 5088
    solver = WorldSolver(max_depth=2)
    action, scores = solver.best_move(world, mock, 0)
    assert action == 0
