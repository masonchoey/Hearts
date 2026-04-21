"""
Shared helpers that combine DMCTS with an AlphaZero ``HeartsNet`` checkpoint.

These utilities replace the heuristic leaf evaluator (``evaluate_hand``) with
the network's value head inside the minimax depth cutoff, and expose the
network's policy head for passing-phase decisions.

Used both by ``dmcts_vs_bots.py`` and by the backend's DMCTS opponent
controller so there is a single source of truth for the NN-augmented behaviour.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np

from .alphazero.net import HeartsNet, OBS_DIM
from .openspiel_utils import OBS_CURRENT_HAND
from .world_solver import WorldSolver


__all__ = [
    "NNValueWorldSolver",
    "is_passing_phase_for_player",
    "nn_pass_action",
]


class NNValueWorldSolver(WorldSolver):
    """
    ``WorldSolver`` variant whose depth-cutoff evaluator is the ``HeartsNet``
    value head.

    At each leaf the agent's remaining hand is encoded into the current-hand
    slice of the 5088-dim OpenSpiel observation (everything else zeroed) and
    passed to ``net.predict_value``, which returns the agent's expected future
    point contribution.  The net was pre-trained with trick-history dropout so
    it is robust to this sparse input.

    A per-instance cache keyed on the agent's hand bitmask avoids redundant
    forward passes across minimax paths.  The cache is safe to share across
    games because the network is deterministic in eval mode: same hand mask →
    same features → same output.
    """

    def __init__(self, max_depth: int, net: HeartsNet):
        super().__init__(max_depth=max_depth)
        self.net = net
        self._value_cache: dict = {}

    def _estimate_score(self, play, agent_id: int) -> float:
        pts = float(play.points.get(agent_id, 0))
        hand = play.hands[agent_id]
        if not hand:
            return pts

        key = play.hand_masks[agent_id]
        if key in self._value_cache:
            return pts + self._value_cache[key]

        features = np.zeros(OBS_DIM, dtype=np.float32)
        for card in hand:
            features[OBS_CURRENT_HAND[0] + card] = 1.0

        future_pts = self.net.predict_value(features)
        self._value_cache[key] = future_pts
        return pts + future_pts


def is_passing_phase_for_player(ts, player_id: int) -> bool:
    """
    Detect passing phase from ``player_id``'s own observation slice.

    Mirrors ``HeartsAgent._is_passing_phase`` but takes an arbitrary player id
    instead of assuming the agent's seat — useful when the backend drives
    multiple DMCTS agents through a shared timestep.
    """
    obs_all = ts.observations.get("info_state") if hasattr(ts, "observations") else None
    if obs_all is None:
        return False
    o = np.asarray(obs_all[player_id], dtype=np.float32)
    if len(o) < 160:
        return False
    # pass_dir[0] == 1 means "No Pass" round.
    if o[0] >= 0.99:
        return False
    passed = float(np.sum(o[56:108]))
    received = float(np.sum(o[108:160]))
    return passed < 3 or received < 3


def nn_pass_action(
    net: HeartsNet,
    ts,
    legal: List[int],
    player_id: int,
) -> int:
    """
    Pick a pass card using ``HeartsNet``'s policy head constrained to legal
    actions.  Falls back to ``legal[0]`` if the legal list is empty (caller
    should ensure that cannot happen for a real timestep).
    """
    if not legal:
        raise ValueError("nn_pass_action called with no legal actions")
    obs = np.asarray(ts.observations["info_state"][player_id], dtype=np.float32)
    legal_mask = np.zeros(52, dtype=bool)
    for c in legal:
        legal_mask[c] = True
    probs, _ = net.predict(obs, legal_mask=legal_mask)
    return max(legal, key=lambda c: probs[c])
