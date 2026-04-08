"""
CFRAgent — play-time agent backed by a trained HeartsCFRSolver.

Usage
─────
    from hearts_ai.cfr import HeartsCFRSolver, CFRAgent

    # After training (or loading from checkpoint):
    solver = HeartsCFRSolver.from_checkpoint("cfr_checkpoints/cfr_0010000.pkl")
    agent  = CFRAgent(solver)

    # During a pyspiel game loop:
    action = agent.act(state)        # sampled from average policy
    action = agent.act_greedy(state) # argmax of average policy

The agent caches the ``AveragePolicy`` object on first use so repeated calls
within the same game (or across games) do not re-create it.  If the solver
continues training after the agent is constructed, call
``agent.refresh_policy()`` to pick up the updated policy.
"""
from __future__ import annotations

import random
from typing import Union

from .solver import HeartsCFRSolver


class CFRAgent:
    """
    Hearts agent that acts according to an ES-MCCFR average policy.

    Parameters
    ----------
    solver_or_path:
        Either a trained :class:`HeartsCFRSolver` instance or a ``str`` path
        to a checkpoint file.  When a path is given the solver is loaded via
        :meth:`HeartsCFRSolver.from_checkpoint`.
    rng:
        Optional :class:`random.Random` instance for reproducible sampling.
        Defaults to the module-level ``random`` functions.
    """

    def __init__(
        self,
        solver_or_path: Union[HeartsCFRSolver, str],
        rng: random.Random | None = None,
    ) -> None:
        if isinstance(solver_or_path, str):
            self._solver = HeartsCFRSolver.from_checkpoint(solver_or_path)
        else:
            self._solver = solver_or_path

        self._rng = rng
        self._policy = None  # lazily cached

    # ── Policy access ─────────────────────────────────────────────────────────

    def _get_policy(self):
        """Return the cached average policy, building it on first call."""
        if self._policy is None:
            self._policy = self._solver.average_policy()
        return self._policy

    def refresh_policy(self) -> None:
        """Invalidate the cached policy so the next call rebuilds it.

        Call this after the solver has run additional training iterations.
        """
        self._policy = None

    # ── Action selection ──────────────────────────────────────────────────────

    def act(self, state) -> int:
        """
        Sample an action from the average policy for the current state.

        For information states not seen during training the policy falls back
        to a uniform distribution over legal actions (provided by OpenSpiel's
        ``AveragePolicy``).

        Parameters
        ----------
        state:
            A ``pyspiel.State`` object at a decision node (not terminal, not
            chance).

        Returns
        -------
        int
            A legal action index.
        """
        policy = self._get_policy()
        probs: dict = policy.action_probabilities(state)
        if not probs:
            return state.legal_actions()[0]

        actions = list(probs.keys())
        weights = list(probs.values())

        if self._rng is not None:
            return self._rng.choices(actions, weights=weights)[0]
        return random.choices(actions, weights=weights)[0]

    def act_greedy(self, state) -> int:
        """
        Return the action with the highest probability under the average policy.

        Greedy selection is useful for deterministic evaluation.

        Parameters
        ----------
        state:
            A ``pyspiel.State`` object at a decision node.

        Returns
        -------
        int
            The highest-probability legal action.
        """
        policy = self._get_policy()
        probs: dict = policy.action_probabilities(state)
        if not probs:
            return state.legal_actions()[0]
        return max(probs, key=probs.get)

    # ── Introspection ─────────────────────────────────────────────────────────

    @property
    def solver(self) -> HeartsCFRSolver:
        return self._solver

    def action_probabilities(self, state) -> dict:
        """Return the full action-probability dict for *state* (for debugging)."""
        return self._get_policy().action_probabilities(state)

    def __repr__(self) -> str:
        return f"CFRAgent(solver={self._solver!r})"
