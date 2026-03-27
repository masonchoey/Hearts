"""
DMCTS vs Bots Hearts Simulation

Runs Hearts games with the DMCTS HeartsAgent (player 0) against bot opponents
and reports performance statistics. Mirrors the structure of rl_vs_bots.py.

USAGE:
1. Basic run (100 games vs random bots):
   python dmcts_vs_bots.py

2. Choose bot type:
   BOT_TYPE=conservative python dmcts_vs_bots.py

3. Control number of games / DMCTS settings:
   NUM_GAMES=50 N_WORLDS=100 TIME_LIMIT_MS=1000 python dmcts_vs_bots.py

4. Full in-game debug output (trick by trick, DMCTS votes):
   HEARTS_DEBUG=1 NUM_GAMES=3 VERBOSE_FREQ=1 python dmcts_vs_bots.py

5. Suppress DMCTS internal logs but keep game display:
   VERBOSE_FREQ=1 NUM_GAMES=5 python dmcts_vs_bots.py

ENVIRONMENT VARIABLES: (no defaults, all must be set explicitly through .env file)
- BOT_TYPE:       "random" or "conservative"
- NUM_GAMES:      number of games to play
- N_WORLDS:       DMCTS determinizations
- TIME_LIMIT_MS:  DMCTS time budget per move
- HEARTS_DEBUG:   "1" adds remaining-cards detail inside each trick
- MAX_DEPTH:      DMCTS search depth
- VERBOSE_FREQ:   show full game output every N games
"""

from dotenv import load_dotenv
load_dotenv()

import json
import os
import random
import sys
import time
from datetime import datetime

import numpy as np

try:
    import pyspiel
    from open_spiel.python.rl_environment import Environment as OSPSingle
except ImportError:
    print("ERROR: OpenSpiel is not installed. Install it before running this script.")
    sys.exit(1)

from hearts_ai.agent import HeartsAgent
from hearts_ai.openspiel_utils import (
    card_points,
    card_to_rank,
    card_to_suit,
    cards_in_hand_from_obs,
)

# HEARTS_DEBUG=1 enables remaining-cards detail inside tricks
DEBUG = os.environ.get("HEARTS_DEBUG", "").strip().lower() in ("1", "true", "yes")

# Suits per openspiel_utils encoding: 0=Clubs 1=Diamonds 2=Hearts 3=Spades
_SUITS = ["♣", "♦", "♥", "♠"]
_SUIT_NAMES = ["clubs", "diamonds", "hearts", "spades"]
HEARTS_SUIT = 2
QUEEN_OF_SPADES = 43  # rank 10 * 4 + suit 3
HEARTS_CARDS = [c for c in range(52) if card_to_suit(c) == HEARTS_SUIT]


def _is_in_passing_phase(ts, cp: int) -> bool:
    """
    Determine if player `cp` is currently in the passing phase using their own
    observation slice — the same logic agent._is_passing_phase uses for P0.
    Works for any player without needing a step counter or in_play_phase flag.
    """
    obs_all = ts.observations.get("info_state")
    if obs_all is None:
        return False
    o = np.asarray(obs_all[cp], dtype=np.float32)
    if len(o) < 160:
        return False
    # pass_dir[0] == 1 means "No Pass" round — never passing
    if o[0] >= 0.99:
        return False
    passed = float(np.sum(o[56:108]))
    received = float(np.sum(o[108:160]))
    return passed < 3 or received < 3


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def show_card(card: int) -> str:
    """Return a human-readable card string, e.g. 'Q♠', '2♣'."""
    rank = "23456789TJQKA"[card_to_rank(card)]
    suit = _SUITS[card_to_suit(card)]
    return f"{rank}{suit}"


def fmt_hand(cards) -> str:
    by_suit = [sorted([c for c in cards if card_to_suit(c) == s]) for s in range(4)]
    parts = []
    for suit_cards in by_suit:
        if suit_cards:
            parts.append("  ".join(show_card(c) for c in suit_cards))
    return "   ".join(parts)


def fmt_score_line(points: dict) -> str:
    return "  ".join(f"P{p}={points[p]}" for p in range(4))


def _trick_winner_idx(trick: list) -> int:
    """Return index (0-3) in `trick` of the winning play. trick = [(player, card), ...]"""
    lead_suit = card_to_suit(trick[0][1])
    best_rank = card_to_rank(trick[0][1])
    wi = 0
    for i in range(1, 4):
        if card_to_suit(trick[i][1]) == lead_suit:
            r = card_to_rank(trick[i][1])
            if r > best_rank:
                best_rank = r
                wi = i
    return wi


def _trick_pts(trick: list) -> int:
    return sum(card_points(c) for _, c in trick)


# ---------------------------------------------------------------------------
# Bot strategies
# ---------------------------------------------------------------------------

def bot_random(legal_actions):
    return random.choice(legal_actions)


def bot_conservative(legal_actions):
    """Avoid point cards; prefer lower cards."""
    options = list(legal_actions)
    non_queen = [a for a in options if a != QUEEN_OF_SPADES]
    if non_queen and QUEEN_OF_SPADES in options:
        options = non_queen
    non_hearts = [a for a in options if card_to_suit(a) != HEARTS_SUIT]
    if non_hearts:
        options = non_hearts
    options.sort()
    return random.choice(options[: max(1, len(options) // 2)])


BOT_STRATEGIES = {
    "random": bot_random,
    "conservative": bot_conservative,
}


# ---------------------------------------------------------------------------
# Per-move display helpers
# ---------------------------------------------------------------------------

def _print_votes(votes: dict, n_worlds: int, indent: str = "       ", suffix: str = "") -> None:
    """Print DMCTS vote breakdown sorted by vote count descending."""
    if not votes or n_worlds == 0:
        return
    total = max(n_worlds, 1)
    sorted_votes = sorted(votes.items(), key=lambda kv: -kv[1])
    parts = []
    for card, v in sorted_votes:
        if v > 0:
            pct = v / total * 100
            parts.append(f"{show_card(card)} {v}/{total} ({pct:.0f}%)")
    if parts:
        print(f"{indent}Votes{suffix}:  " + "   ".join(parts))


def _print_avg_scores(avg_scores: dict, chosen: int, indent: str = "       ") -> None:
    """
    Print per-action average minimax scores across all sampled worlds.
    Sorted best (lowest) to worst. The chosen action is flagged with *.
    """
    if not avg_scores:
        return
    sorted_scores = sorted(avg_scores.items(), key=lambda kv: kv[1])
    parts = []
    for card, score in sorted_scores:
        marker = "*" if card == chosen else " "
        parts.append(f"{marker}{show_card(card)} {score:.1f}pts")
    print(f"{indent}Avg scores:  " + "   ".join(parts))


def _print_remaining_point_cards(played: set, indent: str = "       ") -> None:
    """Show which hearts and the QS are still unplayed."""
    rem_hearts = [c for c in HEARTS_CARDS if c not in played]
    qs_played = QUEEN_OF_SPADES in played
    parts = []
    if rem_hearts:
        parts.append("♥ remaining: " + "  ".join(show_card(c) for c in rem_hearts))
    else:
        parts.append("♥ remaining: none")
    parts.append(f"Q♠: {'played' if qs_played else 'still out'}")
    print(f"{indent}" + "   |   ".join(parts))


def _print_remaining_unseen_cards(my_hand: list, played: set, indent: str = "       ") -> None:
    """Show cards still in opponents' hands — the unknown pool DMCTS samples over."""
    unseen = set(range(52)) - played - set(my_hand)
    suit_parts = []
    for s in range(4):
        suit_cards = sorted([c for c in unseen if card_to_suit(c) == s])
        cards_str = "  ".join(show_card(c) for c in suit_cards) if suit_cards else "—"
        suit_parts.append(f"{_SUITS[s]} {cards_str}")
    print(f"{indent}Opp cards ({len(unseen):2d}): " + "   ".join(suit_parts))


# ---------------------------------------------------------------------------
# Simulator
# ---------------------------------------------------------------------------

class DMCTSSimulator:
    """Run Hearts games: DMCTS agent (player 0) vs bots (players 1-3)."""

    def __init__(self, bot_type: str, n_worlds: int, time_limit_ms: int, max_depth: int):
        if bot_type not in BOT_STRATEGIES:
            raise ValueError(
                f"Unknown bot type '{bot_type}'. Choose from: {list(BOT_STRATEGIES)}"
            )
        self.bot_type = bot_type
        self.bot_fn = BOT_STRATEGIES[bot_type]
        self.n_worlds = n_worlds
        self.time_limit_ms = time_limit_ms
        self.max_depth = max_depth
        self.game = pyspiel.load_game("hearts")
        self.game_results = []
        self.total_agent_moves = 0
        self.total_agent_time = 0.0

    # ------------------------------------------------------------------
    # Single game
    # ------------------------------------------------------------------

    def play_single_game(self, verbose: bool = False) -> dict:
        env = OSPSingle(self.game, players=4)
        agent = HeartsAgent(
            player_id=0,
            n_worlds=self.n_worlds,
            time_limit_ms=self.time_limit_ms,
            max_depth=self.max_depth,
        )

        ts = env.reset()
        agent.reset(initial_hand=None)

        # --- game tracking ---
        current_trick: list = []   # [(player_id, card), ...]
        trick_num: int = 0
        points: dict = {0: 0, 1: 0, 2: 0, 3: 0}
        hearts_broken: bool = False
        all_played: set = set()

        agent_time = 0.0
        agent_turns = 0
        step = 0

        if verbose:
            print(f"\n{'═'*60}")
            print(f"  New game  |  DMCTS (P0) vs {self.bot_type} bots (P1–P3)")
            print(f"  n_worlds={self.n_worlds}  time_limit={self.time_limit_ms}ms  max_depth={self.max_depth}")
            print(f"{'═'*60}")

        while not ts.last() and step < 250:
            cp = ts.observations["current_player"]
            legal = list(ts.observations["legal_actions"][cp])
            if not legal:
                break

            # ── Phase detection ────────────────────────────────────────
            # Use each player's own observation slice (same logic as
            # agent._is_passing_phase) so the detection is correct even
            # when non-P0 players lead the first trick.
            is_passing = _is_in_passing_phase(ts, cp)

            # ── Trick header (printed BEFORE any card display) ─────────
            if verbose and not is_passing and len(current_trick) == 0:
                print(f"\n  {'─'*10} Trick {trick_num + 1} {'─'*10}")

            # ── Get action and display ─────────────────────────────────
            if cp == 0:
                t0 = time.perf_counter()
                action = agent.step(ts)
                dt = time.perf_counter() - t0
                agent_time += dt
                if not is_passing:
                    agent_turns += 1
                    self.total_agent_moves += 1
                    self.total_agent_time += dt

                if verbose and is_passing:
                    print(f"  [pass]  P0 passes: {show_card(action)}")

                elif verbose:
                    leading = len(current_trick) == 0
                    if leading:
                        ctx = "leading"
                    else:
                        lead_suit = card_to_suit(current_trick[0][1])
                        ctx = f"following {_SUITS[lead_suit]}"

                    obs_arr = np.asarray(ts.observations["info_state"][0], dtype=np.float32)
                    hand = cards_in_hand_from_obs(obs_arr)
                    hand_str = fmt_hand(hand) if hand else "(unknown)"

                    print(f"\n  ── P0 [DMCTS] {ctx} ──")
                    print(f"       Hand  ({len(hand):2d}): {hand_str}")
                    legal_str = "  ".join(show_card(c) for c in sorted(legal))
                    print(f"       Legal ({len(legal):2d}): {legal_str}")

                    if len(legal) == 1:
                        # agent.step() returned early — dmcts was never called,
                        # last_votes would be stale from a previous decision.
                        print(f"       Votes:  (only legal move)")
                    else:
                        votes = agent.dmcts.last_votes
                        avg_scores = agent.dmcts.last_avg_scores
                        n_w = agent.dmcts.last_n_worlds
                        _print_votes(votes, n_w, suffix=f" [{n_w} worlds]")
                        _print_avg_scores(avg_scores, chosen=action)
                        # Solver instrumentation
                        nodes = agent.dmcts.last_total_nodes
                        hits = agent.dmcts.last_memo_hits
                        cuts = agent.dmcts.last_ab_cutoffs
                        hit_pct = hits / max(nodes, 1) * 100
                        cut_pct = cuts / max(nodes, 1) * 100
                        print(f"       Solver:  {nodes:,} nodes  memo_hit={hit_pct:.0f}%  α/β_cut={cut_pct:.0f}%")

                    if DEBUG:
                        _print_remaining_point_cards(all_played)
                        _print_remaining_unseen_cards(hand, all_played)

                    print(f"       → Plays: {show_card(action)}   ({dt*1000:.0f}ms)")

            else:
                action = self.bot_fn(legal)

                if verbose and is_passing:
                    print(f"  [pass]  P{cp} passes: {show_card(action)}")
                elif verbose:
                    print(f"  P{cp} [{self.bot_type[:6]:6s}] plays: {show_card(action)}")

            # ── Update trick state ─────────────────────────────────────
            if not is_passing:
                current_trick.append((cp, action))
                all_played.add(action)

                if not hearts_broken and (card_to_suit(action) == HEARTS_SUIT or action == QUEEN_OF_SPADES):
                    hearts_broken = True
                    if verbose:
                        card_name = "Q♠" if action == QUEEN_OF_SPADES else show_card(action)
                        print(f"  ♥  Hearts broken! ({card_name} played by P{cp})")

                if len(current_trick) == 4:
                    wi = _trick_winner_idx(current_trick)
                    winner = current_trick[wi][0]
                    pts = _trick_pts(current_trick)
                    points[winner] += pts
                    trick_num += 1

                    if verbose:
                        cards_str = "   ".join(
                            f"P{p}:{show_card(c)}" for p, c in current_trick
                        )
                        pts_str = f"{pts} pt{'s' if pts != 1 else ''}"
                        scores_str = fmt_score_line(points)
                        winner_label = "DMCTS" if winner == 0 else self.bot_type
                        print(
                            f"  ► P{winner} ({winner_label}) wins  |  {pts_str}  |  {cards_str}"
                        )
                        print(f"    Running scores: {scores_str}")

                    current_trick = []

            ts = env.step([action])
            step += 1

        # --- final scores from OpenSpiel rewards ---
        if ts.last() and ts.rewards is not None:
            raw = list(ts.rewards)
            scores = [int(26 - r) for r in raw]
        else:
            scores = list(points.values())  # fallback to tracked points

        min_score = min(scores)
        winners = [i for i, s in enumerate(scores) if s == min_score]
        winner = winners[0]

        if verbose:
            print(f"\n  {'═'*56}")
            print(f"  Final scores")
            print(f"  {'─'*56}")
            for i in range(4):
                label = "DMCTS" if i == 0 else self.bot_type
                rank = sorted(scores).index(scores[i]) + 1
                tag = "  ← winner!" if i in winners else ""
                print(f"    P{i} [{label:12s}]: {scores[i]:2d} pts  (rank {rank}){tag}")
            print(f"  {'─'*56}")
            print(f"  DMCTS: {agent_turns} decisions  |  avg {agent_time/max(agent_turns,1)*1000:.0f}ms/move  |  total {agent_time:.1f}s")
            print(f"  {'═'*56}")

        EXPECTED = 6.5
        return {
            "timestamp": datetime.now().isoformat(),
            "player_scores": scores,
            "agent_score": scores[0],
            "bot_scores": scores[1:],
            "winner": winner,
            "winners": winners,
            "agent_rank": sorted(scores).index(scores[0]) + 1,
            "agent_percentage": (scores[0] / EXPECTED) * 100,
            "game_steps": step,
            "agent_turns": agent_turns,
            "agent_time_s": round(agent_time, 3),
        }

    # ------------------------------------------------------------------
    # Multi-game run
    # ------------------------------------------------------------------

    def run_simulation(self, num_games: int, verbose_freq: int = 10) -> dict:
        print(f"\n{'='*60}")
        print(f"  DMCTS vs {self.bot_type} bots  |  {num_games} games")
        print(f"  n_worlds={self.n_worlds}  time_limit={self.time_limit_ms}ms  max_depth={self.max_depth}")
        print(f"  HEARTS_DEBUG={'on — remaining cards shown per trick' if DEBUG else 'off'}")
        print(f"{'='*60}")

        self.game_results = []
        self.total_agent_moves = 0
        self.total_agent_time = 0.0

        for i in range(num_games):
            verbose = ((i + 1) % verbose_freq == 0) or i == 0
            if verbose:
                print(f"\nGame {i+1}/{num_games}...")

            result = self.play_single_game(verbose=verbose)
            self.game_results.append(result)

            if verbose:
                recent = self.game_results[-verbose_freq:]
                avg = np.mean([r["agent_score"] for r in recent])
                cumulative_pct = self._cumulative_pct()
                print(
                    f"\n  Progress: avg score (last {len(recent)}): {avg:.1f}  |"
                    f"  cumulative%: {cumulative_pct:.1f}%"
                )

        print(f"\nSimulation complete: {len(self.game_results)} games")
        return self.analyze_results()

    # ------------------------------------------------------------------
    # Analysis
    # ------------------------------------------------------------------

    def _cumulative_pct(self) -> float:
        if not self.game_results:
            return 0.0
        total_pts = sum(r["agent_score"] for r in self.game_results)
        expected = len(self.game_results) * 6.5
        return (total_pts / expected) * 100

    def analyze_results(self) -> dict:
        results = self.game_results
        if not results:
            print("No results to analyze.")
            return {}

        EXPECTED = 6.5
        scores = [r["agent_score"] for r in results]
        ranks = [r["agent_rank"] for r in results]
        wins = sum(1 for r in results if 0 in r["winners"])
        pcts = [r["agent_percentage"] for r in results]

        total_pts = sum(scores)
        n = len(results)
        cumulative_pct = (total_pts / (n * EXPECTED)) * 100

        bot_scores_by_slot = [[r["bot_scores"][i] for r in results] for i in range(3)]
        bot_avgs = [np.mean(s) for s in bot_scores_by_slot]

        analysis = {
            "total_games": n,
            "bot_type": self.bot_type,
            "n_worlds": self.n_worlds,
            "time_limit_ms": self.time_limit_ms,
            "agent": {
                "avg_score": float(np.mean(scores)),
                "std_score": float(np.std(scores)),
                "best_score": int(min(scores)),
                "worst_score": int(max(scores)),
                "avg_rank": float(np.mean(ranks)),
                "wins": wins,
                "win_rate": wins / n,
                "cumulative_percentage": cumulative_pct,
                "avg_percentage": float(np.mean(pcts)),
                "total_agent_moves": self.total_agent_moves,
                "avg_time_per_move_ms": (
                    (self.total_agent_time / self.total_agent_moves) * 1000
                    if self.total_agent_moves else 0.0
                ),
                "rank_distribution": {
                    f"rank_{r}": sum(1 for x in ranks if x == r) for r in range(1, 5)
                },
            },
            "bots": {
                "avg_score_per_slot": [float(a) for a in bot_avgs],
                "avg_score_overall": float(np.mean(bot_avgs)),
            },
        }

        print("\n" + "=" * 60)
        print(f"  RESULTS: DMCTS vs {self.bot_type} bots  ({n} games)")
        print("=" * 60)

        a = analysis["agent"]
        print(f"\n  DMCTS Agent (Player 0):")
        print(f"    Avg score:          {a['avg_score']:.2f} ± {a['std_score']:.2f}")
        print(f"    Best / Worst:       {a['best_score']} / {a['worst_score']}")
        print(f"    Avg rank:           {a['avg_rank']:.2f}/4")
        print(f"    Win rate:           {a['wins']}/{n}  ({a['win_rate']*100:.1f}%)")
        print(f"    Cumulative %:       {a['cumulative_percentage']:.1f}%  (100% = avg)")
        print(f"    Avg time/move:      {a['avg_time_per_move_ms']:.0f}ms")

        print(f"\n  Rank distribution:")
        for r in range(1, 5):
            count = a["rank_distribution"][f"rank_{r}"]
            print(f"    Rank {r}: {count:4d} games  ({count/n*100:.1f}%)")

        print(f"\n  {self.bot_type.title()} bots (players 1-3):")
        for i, avg in enumerate(analysis["bots"]["avg_score_per_slot"]):
            print(f"    P{i+1} avg score: {avg:.2f}")
        print(f"    Overall bot avg:    {analysis['bots']['avg_score_overall']:.2f}")

        return analysis

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    def save_results(self, analysis: dict) -> str:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"dmcts_vs_{self.bot_type}_results_{timestamp}.json"

        def _to_python(obj):
            if isinstance(obj, np.number):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, dict):
                return {k: _to_python(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [_to_python(x) for x in obj]
            return obj

        save_data = {
            "simulation_info": {
                "timestamp": timestamp,
                "total_games": len(self.game_results),
                "bot_type": self.bot_type,
                "n_worlds": self.n_worlds,
                "time_limit_ms": self.time_limit_ms,
                "max_depth": self.max_depth,
            },
            "analysis": _to_python(analysis),
            "sample_games": self.game_results[:10],
        }

        with open(filename, "w") as f:
            json.dump(save_data, f, indent=2)

        print(f"\n  Results saved to: {filename}")
        return filename


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    bot_type = os.environ.get("BOT_TYPE").lower()
    num_games = int(os.environ.get("NUM_GAMES"))
    n_worlds = int(os.environ.get("N_WORLDS"))
    time_limit_ms = int(os.environ.get("TIME_LIMIT_MS"))
    verbose_freq = int(os.environ.get("VERBOSE_FREQ"))
    max_depth = int(os.environ.get("MAX_DEPTH"))

    print("DMCTS vs Bots  |  Hearts Simulation")
    print(f"  BOT_TYPE={bot_type}  NUM_GAMES={num_games}")
    print(f"  N_WORLDS={n_worlds}  TIME_LIMIT_MS={time_limit_ms}ms")
    print(f"  VERBOSE_FREQ={verbose_freq}  HEARTS_DEBUG={'on' if DEBUG else 'off'}")

    try:
        sim = DMCTSSimulator(
            bot_type=bot_type,
            n_worlds=n_worlds,
            time_limit_ms=time_limit_ms,
            max_depth=max_depth,
        )
    except ValueError as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    analysis = sim.run_simulation(num_games=num_games, verbose_freq=verbose_freq)

    if analysis:
        sim.save_results(analysis)
        print("\nDone.")


if __name__ == "__main__":
    main()
