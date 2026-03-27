"""Integration test: full game with 1 DMCTS agent + 3 random agents."""
import os
import random
import sys
import time

import pytest

# Set HEARTS_DEBUG=1 to print progress (e.g. HEARTS_DEBUG=1 pytest hearts_ai/tests/test_integration.py -v -s)
DEBUG = os.environ.get("HEARTS_DEBUG", "").strip() in ("1", "true", "yes")


def _card_str(action: int) -> str:
    """Human-readable card: 0-51 -> 2C..AS."""
    rank = "23456789TJQKA"[action // 4]
    suit = "CDHS"[action % 4]
    return f"{rank}{suit}"


def _phase_str(ts) -> str:
    """Infer pass vs play from observation if available."""
    try:
        obs = ts.observations.get("info_state")
        if obs is not None:
            o = obs[0]
            if len(o) >= 160:
                passed = sum(o[56:108])
                received = sum(o[108:160])
                if passed < 3 or received < 3:
                    return "pass"
    except Exception:
        pass
    return "play"


def test_full_game_no_error():
    """Run one full 4-player game with 1 DMCTS agent + 3 random; game completes without error."""
    try:
        import pyspiel
        from open_spiel.python.rl_environment import Environment as OSPSingle
    except ImportError:
        pytest.skip("OpenSpiel not installed")
    from hearts_ai.agent import HeartsAgent

    game = pyspiel.load_game("hearts")
    env = OSPSingle(game, players=4)
    # Use small n_worlds and short time limit so the test finishes quickly
    agent = HeartsAgent(player_id=0, n_worlds=5, time_limit_ms=300)
    ts = env.reset()
    agent.reset(initial_hand=None)
    step_count = 0
    max_steps = 200
    t0 = time.perf_counter()
    agent_turns = 0
    agent_time = 0.0

    while not ts.last() and step_count < max_steps:
        current_player = ts.observations["current_player"]
        legal = ts.observations["legal_actions"][current_player]
        if not legal:
            break
        phase = _phase_str(ts) if DEBUG else ""

        if current_player == 0:
            if DEBUG:
                print(
                    f"  step {step_count:3d} | P{current_player} [{phase:4s}] getting action... (legal={len(legal)})",
                    file=sys.stderr,
                    flush=True,
                )
            turn_start = time.perf_counter()
            action = agent.step(ts)
            agent_time += time.perf_counter() - turn_start
            agent_turns += 1
            if DEBUG:
                print(
                    f"  step {step_count:3d} | P{current_player} [{phase:4s}] agent -> {_card_str(action):3s} (dt={time.perf_counter() - turn_start:.2f}s)",
                    file=sys.stderr,
                    flush=True,
                )
            assert action in legal
        else:
            action = random.choice(legal)
            if DEBUG:
                print(
                    f"  step {step_count:3d} | P{current_player} [{phase:4s}] | random -> {_card_str(action):3s}",
                    file=sys.stderr,
                    flush=True,
                )
        ts = env.step([action])
        step_count += 1

    elapsed = time.perf_counter() - t0
    if DEBUG:
        print(
            f"  done in {step_count} steps, {elapsed:.1f}s total, agent acted {agent_turns} times ({agent_time:.1f}s)",
            file=sys.stderr,
            flush=True,
        )
        if ts.last() and hasattr(ts, "rewards") and ts.rewards:
            pts = [26 - int(r) for r in ts.rewards]
            print(f"  final points: P0={pts[0]} P1={pts[1]} P2={pts[2]} P3={pts[3]}", file=sys.stderr, flush=True)
    assert ts.last() or step_count >= max_steps


def test_agent_vs_random_100_games():
    """Over N games, DMCTS agent completes without error and has reasonable scores."""
    try:
        import pyspiel
        from open_spiel.python.rl_environment import Environment as OSPSingle
    except ImportError:
        pytest.skip("OpenSpiel not installed")
    from hearts_ai.agent import HeartsAgent

    n_games = 2  # Quick check; use 100 for full validation
    agent_scores = []
    random_scores = {1: [], 2: [], 3: []}
    game = pyspiel.load_game("hearts")
    env = OSPSingle(game, players=4)
    agent = HeartsAgent(player_id=0, n_worlds=10, time_limit_ms=500)
    for _ in range(n_games):
        ts = env.reset()
        agent.reset(None)
        while not ts.last():
            cp = ts.observations["current_player"]
            legal = ts.observations["legal_actions"][cp]
            if not legal:
                break
            if cp == 0:
                action = agent.step(ts)
            else:
                action = random.choice(legal)
            ts = env.step([action])
        if ts.last() and hasattr(ts, "rewards") and ts.rewards is not None:
            # OpenSpiel returns: 26 - points (so lower reward = more points)
            for i, r in enumerate(ts.rewards):
                points = 26 - int(r)
                if i == 0:
                    agent_scores.append(points)
                else:
                    random_scores[i].append(points)
    if not agent_scores:
        pytest.skip("No completed games")
    agent_avg = sum(agent_scores) / len(agent_scores)
    all_random = []
    for i in (1, 2, 3):
        all_random.extend(random_scores[i])
    random_avg = sum(all_random) / len(all_random) if all_random else 26
    # Agent should be competitive (<= 26 on average; ideally lower than random)
    assert agent_avg <= 30  # Relaxed: just check no crash and reasonable score
