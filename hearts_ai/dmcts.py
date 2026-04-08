"""
DMCTSSearch: sample N possible worlds, run WorldSolver on each, aggregate votes for best action.
"""
from __future__ import annotations

import os
import sys
import time
from typing import Dict, List, Optional

from .belief_state import BeliefState
from .world_solver import WorldSolver

_HEARTS_DEBUG = os.environ.get("HEARTS_DEBUG", "").strip().lower() in ("1", "true", "yes")


class DMCTSSearch:
    """
    Determinized MCTS: sample possible worlds from belief, solve each with minimax, vote.
    """

    def __init__(
        self,
        n_worlds,
        solver,
        time_limit_ms,
        agent_id,
        max_depth,
    ):
        self.n_worlds = n_worlds
        self.solver = solver or WorldSolver(max_depth=max_depth)
        self.time_limit_ms = time_limit_ms
        self.max_depth = max_depth
        self.agent_id = agent_id
        # Populated after each select_action call; readable by callers for display/debug.
        self.last_votes: Dict[int, int] = {}
        self.last_avg_scores: Dict[int, float] = {}   # action -> mean minimax score across worlds
        self.last_n_worlds: int = 0
        # Per-move solver instrumentation (aggregated across worlds).
        self.last_total_nodes: int = 0
        self.last_memo_hits: int = 0
        self.last_ab_cutoffs: int = 0

    def select_action(self, state, belief: BeliefState) -> int:
        """
        Main method: return best legal action for current state.
        If only one legal action, return it. Else sample worlds, solve each, return majority vote.
        After each call, last_votes and last_avg_scores are populated with per-action statistics.
        """
        legal = self._legal_actions(state)
        if not legal:
            self.last_votes = {}
            self.last_avg_scores = {}
            self.last_n_worlds = 0
            return 0
        if len(legal) == 1:
            self.last_votes = {legal[0]: 1}
            self.last_avg_scores = {}
            self.last_n_worlds = 1
            return legal[0]
        legal_set = set(legal)
        deadline = time.perf_counter() + (self.time_limit_ms / 1000.0)
        action_votes: Dict[int, int] = {a: 0 for a in legal}
        # Accumulate scores across worlds to compute per-action averages.
        score_sums: Dict[int, float] = {a: 0.0 for a in legal}
        score_counts: Dict[int, int] = {a: 0 for a in legal}
        total_nodes = 0
        total_memo_hits = 0
        total_ab_cutoffs = 0
        n = 0
        if _HEARTS_DEBUG:
            print(f"    dmcts: start (n_worlds={self.n_worlds}, time_limit_ms={self.time_limit_ms})", file=sys.stderr, flush=True)
        for i in range(self.n_worlds):
            if time.perf_counter() >= deadline:
                if _HEARTS_DEBUG:
                    print(f"    dmcts: time limit after {n} worlds", file=sys.stderr, flush=True)
                break
            world = belief.sample_possible_world()
            if not world:
                if _HEARTS_DEBUG:
                    print(f"    dmcts: world {i+1}/{self.n_worlds} sample failed, retry", file=sys.stderr, flush=True)
                continue
            # Give each remaining world an equal share of the remaining time budget
            # so that all n_worlds get evaluated rather than one world hogging the budget.
            remaining_worlds = self.n_worlds - i
            time_remaining = deadline - time.perf_counter()
            world_deadline = time.perf_counter() + time_remaining / remaining_worlds
            action, scores = self.solver.best_move(
                world, state, self.agent_id, time_deadline=world_deadline, real_legal=legal
            )
            if action in action_votes:
                action_votes[action] += 1
            for a, s in scores.items():
                if a in score_sums:
                    score_sums[a] += s
                    score_counts[a] += 1
            total_nodes += self.solver.nodes_visited
            total_memo_hits += self.solver.memo_hits
            total_ab_cutoffs += self.solver.ab_cutoffs
            n += 1
            # if _HEARTS_DEBUG:
            #     print(f"    dmcts: world {n}/{self.n_worlds}", file=sys.stderr, flush=True)

        self.last_votes = dict(action_votes)
        self.last_avg_scores = {
            a: score_sums[a] / score_counts[a]
            for a in legal
            if score_counts.get(a, 0) > 0
        }
        self.last_n_worlds = n
        self.last_total_nodes = total_nodes
        self.last_memo_hits = total_memo_hits
        self.last_ab_cutoffs = total_ab_cutoffs
        if n == 0:
            return legal[0]
        best = max(action_votes, key=action_votes.get)
        # Safety: guarantee the returned action is in the real legal set
        return best if best in legal_set else legal[0]

    def _legal_actions(self, state) -> List[int]:
        if hasattr(state, "observations") and isinstance(state.observations, dict):
            la = state.observations.get("legal_actions")
            cp = state.observations.get("current_player", self.agent_id)
            if la is not None:
                return list(la[cp]) if hasattr(la[cp], "__iter__") else list(la)
        if hasattr(state, "get_legal_actions"):
            return list(state.get_legal_actions())
        return []

