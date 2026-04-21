"""
DMCTS + AlphaZero opponent controller for the backend.

Drives the three AI seats (players 1-3) in the Human-vs-AI environment with
independent ``HeartsAgent`` instances that share a single ``HeartsNet``
checkpoint and a single NN value-head ``WorldSolver``.  Mirrors the behaviour
of ``dmcts_vs_bots.py`` so opponent play in the backend matches what we
benchmark against bots.

Configuration is read from environment variables (loaded via ``python-dotenv``
by the caller, typically ``backend.main``):

    ALPHAZERO_CHECKPOINT  Path to a HeartsNet ``.pt`` checkpoint.  If empty or
                          missing, the controller falls back to the heuristic
                          ``evaluate_hand`` leaf evaluator and the point-max
                          passing heuristic (same fallback as dmcts_vs_bots).
    N_WORLDS              Number of DMCTS determinizations per decision.
    TIME_LIMIT_MS         DMCTS time budget per decision (milliseconds).
    MAX_DEPTH             Alpha-beta search depth.
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional

import numpy as np

from hearts_ai.agent import HeartsAgent
from hearts_ai.alphazero.net import HeartsNet
from hearts_ai.dmcts_alphazero_bridge import (
    NNValueWorldSolver,
    is_passing_phase_for_player,
    nn_pass_action,
)


def _require_int_env(name: str) -> int:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        raise RuntimeError(
            f"Environment variable {name} must be set (check .env) for the DMCTS opponent controller."
        )
    try:
        return int(raw)
    except ValueError as exc:
        raise RuntimeError(f"Environment variable {name}={raw!r} is not an integer") from exc


class DMCTSOpponentController:
    """
    Manages three ``HeartsAgent`` DMCTS players (seats 1-3) that share one
    ``HeartsNet`` checkpoint.

    The controller exposes a single ``choose_action(timestep, player_id)``
    entry point that the gym environment calls in place of the old
    ``HeartsAIModel.get_action``.
    """

    def __init__(self, human_player_id: int = 0) -> None:
        if human_player_id != 0:
            # HeartsAgent seats are hard-coded as 1/2/3 by convention in the
            # backend.  If this ever needs to change, generalize the seat list.
            raise NotImplementedError(
                f"DMCTSOpponentController currently assumes human_player_id=0 (got {human_player_id})."
            )

        self.human_player_id = human_player_id
        self.ai_seats: List[int] = [1, 2, 3]

        self.n_worlds = _require_int_env("N_WORLDS")
        self.time_limit_ms = _require_int_env("TIME_LIMIT_MS")
        self.max_depth = _require_int_env("MAX_DEPTH")

        checkpoint_path = os.environ.get("ALPHAZERO_CHECKPOINT", "").strip()
        self.checkpoint_path: Optional[str] = checkpoint_path or None
        self.net: Optional[HeartsNet] = None
        self._nn_solver: Optional[NNValueWorldSolver] = None

        print("=" * 60)
        print("🤖 DMCTSOpponentController")
        print(f"   seats                : {self.ai_seats} (human is P{self.human_player_id})")
        print(f"   N_WORLDS             : {self.n_worlds}")
        print(f"   TIME_LIMIT_MS        : {self.time_limit_ms}")
        print(f"   MAX_DEPTH            : {self.max_depth}")
        print(f"   ALPHAZERO_CHECKPOINT : {self.checkpoint_path or '(none — heuristic fallback)'}")

        if self.checkpoint_path:
            if os.path.isfile(self.checkpoint_path):
                try:
                    self.net = HeartsNet.load(self.checkpoint_path)
                    self._nn_solver = NNValueWorldSolver(max_depth=self.max_depth, net=self.net)
                    print("   ✓ HeartsNet loaded")
                    print("     depth-cutoff evaluator → value head")
                    print("     passing-phase policy   → policy head")
                except Exception as exc:  # noqa: BLE001
                    print(f"   ✗ Failed to load checkpoint: {exc}")
                    print("     Falling back to evaluate_hand / point-max passing.")
                    self.net = None
                    self._nn_solver = None
            else:
                print(f"   ✗ Checkpoint file not found: {self.checkpoint_path}")
                print("     Falling back to evaluate_hand / point-max passing.")

        self._agents: Dict[int, HeartsAgent] = {
            seat: HeartsAgent(
                player_id=seat,
                n_worlds=self.n_worlds,
                time_limit_ms=self.time_limit_ms,
                max_depth=self.max_depth,
            )
            for seat in self.ai_seats
        }
        if self._nn_solver is not None:
            # All three agents share the same solver instance.  The solver's
            # value cache is safe to share: the network is deterministic in
            # eval mode, so identical hand masks always map to the same value
            # regardless of which agent produced them.  Sharing also means the
            # three seats collectively warm the cache over the course of a
            # game and across successive games.
            for agent in self._agents.values():
                agent.dmcts.solver = self._nn_solver

        print("=" * 60)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def reset_episode(self) -> None:
        """Reset per-hand state on all three DMCTS agents.

        ``HeartsAgent.step`` will lazily (re)initialize each agent's belief
        from the first observed timestep, so we pass ``initial_hand=None``
        here — the hand is extracted from the OpenSpiel observation on the
        first ``step`` call.
        """
        for agent in self._agents.values():
            agent.reset(initial_hand=None)

    def shutdown(self) -> None:
        """Release NN references (no Ray / process teardown needed)."""
        self._agents.clear()
        self._nn_solver = None
        self.net = None

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def choose_action(self, timestep, player_id: int) -> int:
        """Return the action the DMCTS agent for ``player_id`` wants to play.

        Args:
            timestep: OpenSpiel ``TimeStep`` for the current decision point.
            player_id: Seat of the AI player to act (must be in ``ai_seats``).
        """
        if player_id not in self._agents:
            raise ValueError(
                f"player_id {player_id} is not controlled by DMCTSOpponentController "
                f"(controlled seats: {sorted(self._agents)})"
            )

        legal = list(timestep.observations["legal_actions"][player_id])
        if not legal:
            return 0
        if len(legal) == 1:
            return int(legal[0])

        if self.net is not None and is_passing_phase_for_player(timestep, player_id):
            return int(nn_pass_action(self.net, timestep, legal, player_id))

        return int(self._agents[player_id].step(timestep))

    # ------------------------------------------------------------------
    # Backwards-compatibility shim
    # ------------------------------------------------------------------

    def get_action(self, observation: np.ndarray, legal_actions: List[int]) -> int:
        """
        Legacy signature kept so any caller that still passes ``(obs, legal)``
        without a timestep raises a clear error instead of silently producing
        bad DMCTS behaviour.
        """
        raise RuntimeError(
            "DMCTSOpponentController.get_action(obs, legal) is not supported. "
            "Callers must invoke choose_action(timestep, player_id) instead."
        )
