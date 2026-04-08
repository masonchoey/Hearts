"""
HeartsAgent: top-level DMCTS agent that interfaces with OpenSpiel / backend state.
"""
from __future__ import annotations

import os
import sys
from typing import List, Optional

from .belief_state import BeliefState
from .dmcts import DMCTSSearch
from .world_solver import WorldSolver
from .openspiel_utils import cards_in_hand

_HEARTS_DEBUG = os.environ.get("HEARTS_DEBUG", "").strip().lower() in ("1", "true", "yes")


class HeartsAgent:
    """
    DMCTS Hearts agent. Use reset() at start of each hand, then step(state) to get actions.
    """

    def __init__(
        self,
        player_id,
        n_worlds,
        time_limit_ms,
        max_depth,
    ):
        self.player_id = player_id
        self.belief: Optional[BeliefState] = None
        self.dmcts = DMCTSSearch(
            n_worlds=n_worlds,
            solver=WorldSolver(max_depth=max_depth),
            time_limit_ms=time_limit_ms,
            agent_id=player_id,
            max_depth=max_depth,
        )

    def reset(self, initial_hand: Optional[List[int]] = None) -> None:
        """Called at start of each hand. Initializes BeliefState."""
        if initial_hand is not None:
            self.belief = BeliefState(my_hand=list(initial_hand), agent_id=self.player_id)
        else:
            self.belief = BeliefState(my_hand=[], agent_id=self.player_id)

    def step(self, state) -> int:
        """
        Main entry: given state (OpenSpiel timestep or backend game), return action.
        During passing phase: pass highest point cards. During play: DMCTS.
        """
        legal = self._legal_actions(state)
        if not legal:
            return 0
        if len(legal) == 1:
            return legal[0]
        passing = self._is_passing_phase(state)
        if _HEARTS_DEBUG:
            print(f"    agent.step: legal={len(legal)} passing_phase={passing}", file=sys.stderr, flush=True)
        # Passing phase: simple heuristic — pass highest point cards first
        if passing:
            from .openspiel_utils import card_points
            return max(legal, key=card_points)
        if self.belief is None:
            hand = cards_in_hand(state, self.player_id)
            self.belief = BeliefState(my_hand=hand, agent_id=self.player_id)
        self.belief.update_from_openspiel_state(state)
        return self.dmcts.select_action(state, self.belief)

    def _legal_actions(self, state) -> List[int]:
        if hasattr(state, "observations") and isinstance(state.observations, dict):
            la = state.observations.get("legal_actions")
            cp = state.observations.get("current_player", self.player_id)
            if la is not None:
                return list(la[cp]) if hasattr(la[cp], "__iter__") else list(la)
        if hasattr(state, "get_legal_actions"):
            return list(state.get_legal_actions())
        return []

    def _is_passing_phase(self, state) -> bool:
        if hasattr(state, "is_passing_phase"):
            return state.is_passing_phase(self.player_id)
        if hasattr(state, "observations") and isinstance(state.observations, dict):
            obs = state.observations.get("info_state")
            if obs is not None:
                o = obs[self.player_id] if hasattr(obs, "__getitem__") else obs
                import numpy as np
                o = np.asarray(o)
                if len(o) < 160:
                    return False
                pass_dir = o[0:4]
                if pass_dir[0] >= 0.99:  # No Pass
                    return False
                passed = np.sum(o[56:108])
                received = np.sum(o[108:160])
                return passed < 3 or received < 3
        return False

    def observe_game_action(self, player_id: int, action: int, state) -> None:
        """
        Call after every player's action (if you want incremental belief updates).
        Updates belief with new observation. step() already syncs from state; this is optional.
        """
        if self.belief is None:
            return
        from .openspiel_utils import card_to_suit, get_current_trick, get_lead_suit_from_trick
        trick = get_current_trick(state, self.player_id)
        lead_suit = get_lead_suit_from_trick(trick) if trick else None
        self.belief.observe_card_played(player_id, action, lead_suit, trick)
