"""
Model-agnostic AI seat controller.

This is the seam for substituting a bot into a seat (e.g. when a human
disconnects mid-game). The real trained model — DMCTS / AlphaZero — is still in
progress, so the default here is a lightweight legal-move bot. Swap the
implementation returned by ``get_ai_controller`` (or register a different
default) once the model is ready; nothing else in the game engine needs to know
which brain is behind a seat.
"""
from __future__ import annotations

import random
from abc import ABC, abstractmethod
from typing import List, Optional

from ..schemas.types import Card


class AIController(ABC):
    """Decides moves for one seat. Implementations must return *legal* choices."""

    @abstractmethod
    def choose_pass(self, hand: List[Card], pass_direction: str) -> List[Card]:
        """Pick exactly 3 cards from ``hand`` to pass."""

    @abstractmethod
    def choose_play(self, legal_cards: List[Card], game: "MultiplayerGameInstance", seat: int) -> Card:
        """Pick one card from ``legal_cards`` to play."""


class RandomLegalAI(AIController):
    """Placeholder bot: makes only legal moves, no strategy.

    Deliberately simple — it exists so AI takeover works end-to-end before the
    trained model is wired in. Replace via ``get_ai_controller``.
    """

    def choose_pass(self, hand: List[Card], pass_direction: str) -> List[Card]:
        # Hands arrive sorted low→high; passing the three highest is a reasonable
        # trivial heuristic and always legal.
        return list(hand[-3:]) if len(hand) >= 3 else list(hand)

    def choose_play(self, legal_cards: List[Card], game, seat: int) -> Card:
        return random.choice(legal_cards)


# ── Factory ──────────────────────────────────────────────────────────────────
# Central place to choose which brain controls an AI seat. When the trained
# model is ready, return it here (optionally keyed on difficulty/seat).
_DEFAULT_FACTORY = RandomLegalAI


def get_ai_controller(seat: Optional[int] = None) -> AIController:
    return _DEFAULT_FACTORY()
