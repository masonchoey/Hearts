"""
Hearts Hand Evaluation Heuristic — Scoring Constants
=====================================================
All weights are in "centipunishment" units: positive = bad for you (expected
points you'll likely take), negative = good for you (expected point avoidance).
 
Tune these via self-play: pit version A vs version B, log point totals across
~10k games, and adjust weights in the direction that reduces your average score.
 
Usage:
    score = evaluate_hand(hand, game_state)
    # Lower score = better hand position.
"""
 
from __future__ import annotations
from dataclasses import dataclass, field
 
 
# ---------------------------------------------------------------------------
# 1. SPADE DANGER — Queen / King / Ace of Spades
# ---------------------------------------------------------------------------
# The danger of holding QKA of spades depends heavily on how many low spades
# you hold as "cover" (cards that force opponents to lead into you first).
 
# Base danger for holding each high spade with NO cover at all
SPADE_QUEEN_NO_COVER    = 12.0   # Queen of spades, 0 low spades held
SPADE_KING_NO_COVER     =  7.0   # King (can be forced to cover Q)
SPADE_ACE_NO_COVER      =  9.0   # Ace (always wins a spade trick)
 
# Danger reduction per low spade held (stacks per card, diminishing above 4)
SPADE_COVER_PER_CARD    = -2.5   # Each low spade below Q reduces danger
SPADE_COVER_DIMINISH_AT =  4     # Above this many low spades, half value
SPADE_COVER_DIMINISH_MULT = 0.5  # Multiplier for cover cards beyond the threshold
 
# If you hold QKA together, the compounding danger (they protect each other less)
SPADE_QK_TOGETHER_BONUS  = -1.5  # Slight reduction: K absorbs tricks before Q
SPADE_QKA_TOGETHER_BONUS = -2.0  # A further reduces — you control spade tempo
 
# Danger if spades have NOT been led yet (QoS still safely hidden)
SPADE_UNLED_DISCOUNT     = -2.0  # Leading hasn't happened; you may discard first
 
 
# ---------------------------------------------------------------------------
# 2. HEART DANGER — Middling hearts without low escape cards
# ---------------------------------------------------------------------------
# Mid hearts (6–9) can't win tricks to dump elsewhere, and can't underplay.
# Having low hearts (2–5) gives you an escape route.
 
HEART_MID_NO_LOW_PER_CARD   =  2.5   # Per mid heart (6–9) when holding no low hearts
HEART_MID_WITH_LOW_DISCOUNT = -1.0   # Per mid heart if you DO hold at least one low heart
HEART_LOW_ESCAPE_THRESHOLD  =  2     # Need at least this many low hearts to feel safe
HEART_HIGH_NO_MID_PER_CARD  =  1.5   # A/K/Q/J of hearts without low hearts — still risky
 
# Each raw heart point card (any heart) in your hand at game start
# (tempered heavily by the shoot-the-moon check below)
HEART_RAW_POINT_PER_CARD    =  0.8
 
 
# ---------------------------------------------------------------------------
# 3. SUIT LENGTH DANGER — Long suit with no low cards
# ---------------------------------------------------------------------------
# Holding 5+ cards in a suit but missing the lowest 2–3 means opponents
# can lead that suit and you'll be forced to win tricks with your high cards.
 
LONG_SUIT_LENGTH_THRESHOLD  =  4     # Suits longer than this trigger danger
LONG_SUIT_NO_LOW_PER_CARD   =  1.8   # Per card above threshold if missing lows
LONG_SUIT_HAS_LOW_DISCOUNT  = -0.9   # Per card above threshold if you DO have lows
 
# Penalty when an opponent is known void in a suit you lead
# (they can dump high cards or the QoS on your tricks)
OPPONENT_VOID_IN_YOUR_SUIT  =  3.5   # Per known void opponent
 
 
# ---------------------------------------------------------------------------
# 4. BEING VOID IN A SUIT — Generally good
# ---------------------------------------------------------------------------
# Void = you can dump your worst cards whenever that suit is led.
# Value depends on what dangerous cards you hold.
 
VOID_BASE_VALUE              = -4.0  # Flat value for being void in any suit
VOID_WITH_HIGH_HEARTS        = -5.5  # Extra value if you have high hearts to shed
VOID_WITH_QOS                = -8.0  # Very strong: can always dump QoS on that lead
VOID_IN_SPADES_WITH_QOS      = -6.0  # Void in spades + holding QoS = mixed (risky)
 
# Penalty offset: being void when you're ALSO trying to shoot the moon
# reduces your ability to stay in the trick-winning flow
VOID_WHILE_SHOOTING_PENALTY  =  3.0
 
 
# ---------------------------------------------------------------------------
# 5. QUEEN OF SPADES LOCATION (positional danger)
# ---------------------------------------------------------------------------
# If you don't hold the QoS, its estimated location relative to your seat
# matters a lot. Seat positions are 0=you, 1=left, 2=across, 3=right.
# "Left" = plays immediately after you; "right" = plays immediately before you.
 
QOS_HELD_BY_YOU              =  0.0  # Handled by section 1
QOS_ESTIMATED_LEFT           =  5.0  # Worst: they play after you, hard to avoid
QOS_ESTIMATED_ACROSS         =  2.5  # Medium: some information before they play
QOS_ESTIMATED_RIGHT          = -1.0  # Best: they play before you, you can react
 
# Confidence scaling: if QoS location is uncertain, scale by probability
# penalty = QOS_ESTIMATED_LEFT * p_left + QOS_ESTIMATED_ACROSS * p_across + ...
# (apply this in your evaluate_hand function, not as a constant)
 
# Late-game certainty bonus: if QoS location is narrowed to one player
QOS_LOCATION_KNOWN_BONUS     = -1.5  # Knowing exactly where it is reduces surprise
 
 
# ---------------------------------------------------------------------------
# 6. SHOOT THE MOON DETECTION
# ---------------------------------------------------------------------------
# If YOU might be shooting, your evaluation should invert: points = good.
# If an OPPONENT might be shooting, you need to interfere.
 
# Thresholds to flag potential shooting
SHOOT_MIN_HEARTS_FOR_ALERT   =  6    # Holding this many hearts triggers moon check
SHOOT_MIN_HIGH_CARDS         =  3    # High cards (A/K/Q across suits) needed alongside
SHOOT_SELF_INVERT_MULTIPLIER = -0.9  # Multiply your heart danger by this if shooting
 
# Opponent shoot risk: how many hearts + points they've taken so far
OPPONENT_SHOOT_HEARTS_TAKEN_ALERT  =  5   # If opponent has taken this many hearts...
OPPONENT_SHOOT_NO_OTHER_POINTS     =  1   # ...and this few non-heart points, alert
OPPONENT_SHOOT_INTERFERENCE_VALUE  =  8.0 # How important it is to give them a trick
 
 
# ---------------------------------------------------------------------------
# 7. CARD COUNTING / VOID INFERENCE
# ---------------------------------------------------------------------------
# Derived from observed plays; these weight how much inferred voids matter.
 
INFERRED_VOID_CONFIDENCE_THRESHOLD = 0.85  # Min probability to treat as "known void"
INFERRED_VOID_WEIGHT                = 0.7  # Scale danger by this if void is inferred
# (vs 1.0 if void is certain from an observed discard)
 
 
# ---------------------------------------------------------------------------
# 8. TURN ORDER CONTEXT
# ---------------------------------------------------------------------------
# Leading a trick vs. following changes how dangerous certain holdings are.
 
LEADING_WITH_NO_SAFE_CARD    =  3.0  # You must lead but have no card below 7
LEADING_HEARTS_WHEN_UNBROKEN =  4.0  # Illegal + indicates poor hand structure model
HOLDING_2_OF_CLUBS           =  1.5  # Forced to lead trick 1; slight constraint penalty
 
# Following: danger of being "sandwiched" (dangerous card on your right)
SANDWICH_DANGER_RIGHT_HIGH   =  2.0  # High card to your right + you must play mid
 
 
# ---------------------------------------------------------------------------
# 9. GAME PHASE SCALING
# ---------------------------------------------------------------------------
# Heuristic values should scale differently early vs. late game.
# Multiply each category's score by the phase weight.
 
PHASE_EARLY_TRICKS  = (1, 4)    # Tricks 1–4
PHASE_MID_TRICKS    = (5, 9)    # Tricks 5–9
PHASE_LATE_TRICKS   = (10, 13)  # Tricks 10–13
 
# Multipliers per phase (early, mid, late)
PHASE_MULTIPLIER_SPADE_DANGER  = (1.0, 1.2, 1.5)   # Gets worse late
PHASE_MULTIPLIER_HEART_DANGER  = (0.8, 1.0, 1.3)
PHASE_MULTIPLIER_LONG_SUIT     = (1.2, 1.0, 0.6)   # Less relevant late
PHASE_MULTIPLIER_VOID_VALUE    = (0.6, 1.0, 1.4)   # More valuable late
PHASE_MULTIPLIER_MOON_SHOOT    = (0.5, 1.0, 1.5)   # Shoot attempts clarify late
 
 
# ---------------------------------------------------------------------------
# 10. HAND SCORE NORMALIZATION
# ---------------------------------------------------------------------------
# Raw scores can vary wildly; clamp before passing to MCTS rollout policy.
 
SCORE_CLAMP_MIN  = -30.0
SCORE_CLAMP_MAX  =  50.0
 
# If using a softmax policy in MCTS, temperature for converting scores to probs.
# Lower = more greedy, higher = more exploratory.
MCTS_EVAL_TEMPERATURE = 1.2
 
 
# ---------------------------------------------------------------------------
# FEATURE VECTOR SPEC (for future NN input encoding)
# ---------------------------------------------------------------------------
# If you move to an MLP, this documents the input feature order.
# Each entry: (feature_name, size, notes)
 
NN_FEATURE_SPEC = [
    ("card_presence",           52,  "Binary: 1 if card in hand"),
    ("suit_lengths",             4,  "0–13 normalized to 0–1"),
    ("spade_cover_count",        1,  "Low spades below Q, normalized /5"),
    ("hearts_broken",            1,  "Binary flag"),
    ("qos_played",               1,  "Binary flag"),
    ("qos_in_hand",              1,  "Binary flag"),
    ("opponent_void_flags",     12,  "4 suits × 3 opponents, inferred"),
    ("current_scores",           4,  "Each player's score, normalized /26"),
    ("tricks_remaining",         1,  "Normalized 0–1"),
    ("moon_shoot_risk",          3,  "Per opponent, probability 0–1"),
    ("phase",                    1,  "0=early, 0.5=mid, 1=late"),
    ("seat_relative_to_qos",     4,  "One-hot: you/left/across/right holds QoS"),
]
 
TOTAL_NN_INPUT_DIM = sum(size for _, size, _ in NN_FEATURE_SPEC)  # = 85


# ---------------------------------------------------------------------------
# HAND EVALUATOR
# ---------------------------------------------------------------------------

def evaluate_hand(hand: set, play, agent_id: int) -> float:
    """
    Estimate expected future point exposure for agent_id given the current hand
    and PlayState. Returns a value in ~points units: positive = more expected
    points (worse), negative = fewer expected points (better).

    Intended as the depth-cutoff heuristic for WorldSolver minimax, replacing
    the previous proportional-share estimate.
    """
    from .openspiel_utils import card_to_suit, card_to_rank, NUM_PLAYERS

    CLUBS_SUIT    = 0
    DIAMONDS_SUIT = 1
    HEARTS_SUIT   = 2
    SPADES_SUIT   = 3

    QUEEN_OF_SPADES = 43  # rank 10 * 4 + suit 3
    KING_OF_SPADES  = 47  # rank 11 * 4 + suit 3
    ACE_OF_SPADES   = 51  # rank 12 * 4 + suit 3
    QUEEN_RANK      = 10

    if not hand:
        return 0.0

    tricks_played = play.num_played // NUM_PLAYERS
    if 13 - tricks_played <= 0:
        return 0.0

    # Game phase index: 0=early, 1=mid, 2=late
    if tricks_played < PHASE_EARLY_TRICKS[1]:
        phase = 0
    elif tricks_played < PHASE_MID_TRICKS[1]:
        phase = 1
    else:
        phase = 2

    # Suit breakdown; all_suits indexed by suit constant (0–3)
    spades   = [c for c in hand if card_to_suit(c) == SPADES_SUIT]
    hearts   = [c for c in hand if card_to_suit(c) == HEARTS_SUIT]
    clubs    = [c for c in hand if card_to_suit(c) == CLUBS_SUIT]
    diamonds = [c for c in hand if card_to_suit(c) == DIAMONDS_SUIT]
    all_suits = [clubs, diamonds, hearts, spades]

    qs_in_hand = QUEEN_OF_SPADES in hand
    ks_in_hand = KING_OF_SPADES  in hand
    as_in_hand = ACE_OF_SPADES   in hand

    # Low spades = cards below the Queen that provide cover
    n_low_spades = sum(1 for c in spades if card_to_rank(c) < QUEEN_RANK)

    score = 0.0

    # ── 1. Spade danger ───────────────────────────────────────────────────
    def _cover_reduction(n: int) -> float:
        full  = min(n, SPADE_COVER_DIMINISH_AT)
        extra = max(0, n - SPADE_COVER_DIMINISH_AT)
        return (full * SPADE_COVER_PER_CARD
                + extra * SPADE_COVER_PER_CARD * SPADE_COVER_DIMINISH_MULT)

    spade_danger = 0.0
    if qs_in_hand:
        spade_danger += SPADE_QUEEN_NO_COVER + _cover_reduction(n_low_spades)
        if ks_in_hand:
            spade_danger += SPADE_QK_TOGETHER_BONUS
        if ks_in_hand and as_in_hand:
            spade_danger += SPADE_QKA_TOGETHER_BONUS
        if not play.hearts_broken:
            spade_danger += SPADE_UNLED_DISCOUNT
    if ks_in_hand and not qs_in_hand:
        spade_danger += SPADE_KING_NO_COVER + _cover_reduction(n_low_spades)
    if as_in_hand and not qs_in_hand:
        spade_danger += SPADE_ACE_NO_COVER + _cover_reduction(n_low_spades)

    score += spade_danger * PHASE_MULTIPLIER_SPADE_DANGER[phase]

    # ── 2. Heart danger ───────────────────────────────────────────────────
    # rank 0–3 = 2–5 (low), 4–7 = 6–9 (mid), 9–12 = J–A (high); rank 8 = T
    low_hearts  = [c for c in hearts if card_to_rank(c) <= 3]
    mid_hearts  = [c for c in hearts if 4 <= card_to_rank(c) <= 7]
    high_hearts = [c for c in hearts if card_to_rank(c) >= 9]

    has_low_escape = len(low_hearts) >= HEART_LOW_ESCAPE_THRESHOLD

    heart_danger = len(hearts) * HEART_RAW_POINT_PER_CARD
    for _ in mid_hearts:
        heart_danger += HEART_MID_WITH_LOW_DISCOUNT if has_low_escape else HEART_MID_NO_LOW_PER_CARD
    for _ in high_hearts:
        if not has_low_escape:
            heart_danger += HEART_HIGH_NO_MID_PER_CARD

    score += heart_danger * PHASE_MULTIPLIER_HEART_DANGER[phase]

    # ── 3. Long-suit danger ───────────────────────────────────────────────
    long_mult = PHASE_MULTIPLIER_LONG_SUIT[phase]
    for s_cards in all_suits:
        if len(s_cards) > LONG_SUIT_LENGTH_THRESHOLD:
            excess = len(s_cards) - LONG_SUIT_LENGTH_THRESHOLD
            has_low = any(card_to_rank(c) <= 2 for c in s_cards)
            per_card = LONG_SUIT_HAS_LOW_DISCOUNT if has_low else LONG_SUIT_NO_LOW_PER_CARD
            score += excess * per_card * long_mult

    # ── 4. Void value ─────────────────────────────────────────────────────
    # Only meaningful once tricks have started (voids at trick 0 aren't informative)
    if tricks_played > 0:
        void_mult = PHASE_MULTIPLIER_VOID_VALUE[phase]
        for s_idx, s_cards in enumerate(all_suits):
            if len(s_cards) == 0:
                if qs_in_hand and s_idx == SPADES_SUIT:
                    score += VOID_IN_SPADES_WITH_QOS * void_mult
                elif qs_in_hand:
                    score += VOID_WITH_QOS * void_mult
                elif high_hearts and s_idx != HEARTS_SUIT:
                    score += VOID_WITH_HIGH_HEARTS * void_mult
                else:
                    score += VOID_BASE_VALUE * void_mult

    # ── 5. Shoot-the-moon check ───────────────────────────────────────────
    # If we look like a moon-shot candidate, invert most of the heart danger:
    # holding lots of hearts is a sign of strength, not exposure.
    high_card_count = sum(1 for c in hand if card_to_rank(c) >= 10)  # Q, K, A across suits
    if len(hearts) >= SHOOT_MIN_HEARTS_FOR_ALERT and high_card_count >= SHOOT_MIN_HIGH_CARDS:
        score += heart_danger * SHOOT_SELF_INVERT_MULTIPLIER * PHASE_MULTIPLIER_MOON_SHOOT[phase]

    return max(SCORE_CLAMP_MIN, min(SCORE_CLAMP_MAX, score))