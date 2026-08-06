"""
Native Python Hearts engine for multiplayer (human-only) games.

This package intentionally has NO dependency on OpenSpiel. It supports custom
player counts (3/4/5) and host-selectable scoring rules that OpenSpiel's fixed
4-player Hearts implementation cannot express. The single-player-vs-AI path
continues to use OpenSpiel (see ``backend/game/hearts_logic.py``).
"""
from .rules import RuleConfig
from .engine import NativeHeartsGame

__all__ = ["RuleConfig", "NativeHeartsGame"]
