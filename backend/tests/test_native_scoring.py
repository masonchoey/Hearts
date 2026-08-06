"""
Unit tests for the native engine's scoring and core rules.

Runnable two ways:
  - pytest backend/tests/test_native_scoring.py
  - python -m backend.tests.test_native_scoring   (from repo root)
"""
from backend.game.native.rules import RuleConfig, RANKS
from backend.game.native.scoring import score_deal
from backend.game.native.engine import NativeHeartsGame

HEARTS = [("H", r) for r in RANKS]        # all 13 hearts
QS = ("S", "Q")
JD = ("D", "J")
TC = ("C", "T")
ALL_MOON = HEARTS + [QS]                   # 26-point moon set

BOTH = RuleConfig(player_count=4, jd_bonus=True, ten_club_doubler=True)


def _multiset(d):
    return sorted(d.values())


# ── Doc's five worked shoot-the-moon examples (4 players) ────────────────────

def test_moon_shooter_takes_jd_and_tc():
    # shooter(0) takes everything incl. JD + 10♣  ->  -20 52 52 52
    taken = {0: ALL_MOON + [JD, TC], 1: [], 2: [], 3: []}
    assert _multiset(score_deal(taken, BOTH)) == [-20, 52, 52, 52]


def test_moon_shooter_jd_nonshooter_tc():
    # shooter(0) has JD; seat1 has 10♣  ->  -10 26 26 52
    taken = {0: ALL_MOON + [JD], 1: [TC], 2: [], 3: []}
    assert _multiset(score_deal(taken, BOTH)) == [-10, 26, 26, 52]


def test_moon_shooter_tc_nonshooter_jd():
    # shooter(0) has 10♣ (=> everyone doubles); seat1 has JD  ->  0 52 52 32
    taken = {0: ALL_MOON + [TC], 1: [JD], 2: [], 3: []}
    assert _multiset(score_deal(taken, BOTH)) == [0, 32, 52, 52]


def test_moon_jd_and_tc_same_nonshooter():
    # shooter(0) plain; seat1 has both JD and 10♣  ->  0 26 26 32
    taken = {0: ALL_MOON, 1: [JD, TC], 2: [], 3: []}
    assert _multiset(score_deal(taken, BOTH)) == [0, 26, 26, 32]


def test_moon_jd_and_tc_different_nonshooters():
    # shooter(0) plain; seat1 JD; seat2 10♣  ->  0 16 26 52
    taken = {0: ALL_MOON, 1: [JD], 2: [TC], 3: []}
    assert _multiset(score_deal(taken, BOTH)) == [0, 16, 26, 52]


# ── Non-moon scoring ─────────────────────────────────────────────────────────

def test_plain_no_rules():
    rules = RuleConfig(player_count=4)
    taken = {0: [QS] + HEARTS[:3], 1: HEARTS[3:], 2: [], 3: []}
    scores = score_deal(taken, rules)
    assert scores == {0: 16, 1: 10, 2: 0, 3: 0}
    assert sum(scores.values()) == 26


def test_jd_bonus_only():
    rules = RuleConfig(player_count=4, jd_bonus=True)
    taken = {0: HEARTS[:5] + [JD], 1: HEARTS[5:] + [QS], 2: [], 3: []}
    # seat0: 5 - 10 = -5 ; seat1: 8 + 13 = 21
    assert score_deal(taken, rules) == {0: -5, 1: 21, 2: 0, 3: 0}


def test_ten_club_doubler_only():
    rules = RuleConfig(player_count=4, ten_club_doubler=True)
    taken = {0: [QS, TC], 1: HEARTS, 2: [], 3: []}
    # seat0: 13 * 2 = 26 ; seat1: 13 hearts
    assert score_deal(taken, rules) == {0: 26, 1: 13, 2: 0, 3: 0}


def test_jd_then_tc_ordering():
    # JD applied before doubling: (2 - 10) * 2 = -16
    rules = RuleConfig(player_count=4, jd_bonus=True, ten_club_doubler=True)
    taken = {0: HEARTS[:2] + [JD, TC], 1: HEARTS[2:] + [QS], 2: [], 3: []}
    assert score_deal(taken, rules)[0] == -16


def test_all_hearts_without_qs_is_not_a_moon():
    # Taking all 13 hearts but NOT the Q♠ is a plain 13, not a shot moon.
    rules = RuleConfig(player_count=4)
    taken = {0: HEARTS, 1: [QS], 2: [], 3: []}
    assert score_deal(taken, rules) == {0: 13, 1: 13, 2: 0, 3: 0}


# ── Engine: deck / dealing / rules integration ───────────────────────────────

def test_deck_and_hand_sizes():
    assert RuleConfig(player_count=3).hand_size == 17
    assert RuleConfig(player_count=4).hand_size == 13
    assert RuleConfig(player_count=5).hand_size == 10
    assert ("C", "2") not in RuleConfig(player_count=3).build_deck()
    deck5 = RuleConfig(player_count=5).build_deck()
    assert ("C", "2") not in deck5 and ("D", "2") not in deck5


def _play_full_deal(rules, seed):
    g = NativeHeartsGame(rules, seed=seed)
    # Everyone passes their first three cards.
    for seat in range(rules.player_count):
        g.submit_pass(seat, g.get_player_hand(seat)[:3])
    assert g.is_passing_phase() is False
    # First trick must be led with the lowest club.
    leader = g.current_player()
    lead_legal = g.legal_moves(leader)
    assert len(lead_legal) == 1 and lead_legal[0].suit == "C"
    # Greedily play the first legal move until the deal ends.
    guard = 0
    while not g.is_terminal():
        seat = g.current_player()
        moves = g.legal_moves(seat)
        assert moves, f"no legal moves for seat {seat}"
        g.apply_move(seat, moves[0])
        guard += 1
        assert guard < 500
    return g


def test_full_deal_reaches_terminal_and_conserves_points():
    for n in (3, 4, 5):
        rules = RuleConfig(player_count=n, jd_bonus=True, ten_club_doubler=True)
        g = _play_full_deal(rules, seed=n * 7)
        assert all(g.hand_count(s) == 0 for s in range(n))
        total_cards = sum(len(g.taken[s]) for s in range(n))
        assert total_cards == len(rules.build_deck())
        # Base penalty (hearts + Q♠) always totals 26 regardless of who took what.
        assert sum(g.running_points().values()) == 26
        assert g.final_scores() is not None


def test_passing_moves_cards_to_the_right():
    rules = RuleConfig(player_count=4)
    g = NativeHeartsGame(rules, seed=123)
    before = {s: set(g.hands[s]) for s in range(4)}
    passes = {s: g.get_player_hand(s)[:3] for s in range(4)}
    for s in range(4):
        g.submit_pass(s, passes[s])
    # seat s passed to seat (s+1)%4 (right). Those exact cards should now be
    # in the receiver's hand and gone from the passer's.
    for s in range(4):
        receiver = (s + 1) % 4
        passed = {(c.suit, c.rank) for c in passes[s]}
        assert passed <= set(g.hands[receiver])
        assert not (passed & set(g.hands[s]) & before[s] - passed)


if __name__ == "__main__":
    import sys, traceback

    funcs = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in funcs:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except Exception:
            failed += 1
            print(f"FAIL {fn.__name__}")
            traceback.print_exc()
    print(f"\n{len(funcs) - failed}/{len(funcs)} passed")
    sys.exit(1 if failed else 0)
