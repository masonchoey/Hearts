"""
Multiplayer Hearts game manager.
Wraps the native Hearts engine for a room with N human players (no AI).
Builds per-seat game-state views (each player sees only their own hand).

Supports custom player counts (3/4/5) and host-selectable scoring rules via
``RuleConfig``. This path is fully independent of OpenSpiel — the single-player
vs-AI path (``state_manager.py`` / ``hearts_logic.py``) still uses OpenSpiel.
"""
from typing import Dict, List, Optional, Tuple

from .native import NativeHeartsGame, RuleConfig
from ..schemas.types import Card, Player


class MultiplayerGameInstance:
    """
    Holds one Hearts game for a room.
    seat_to_user: maps seat (0..N-1) → user_id
    seat_to_name: maps seat (0..N-1) → display name
    rules:        player count + scoring toggles for this game
    """

    def __init__(
        self,
        room_id: str,
        seat_to_user: Dict[int, str],
        seat_to_name: Dict[int, str],
        rules: Optional[RuleConfig] = None,
    ):
        self.room_id = room_id
        self.rules = rules or RuleConfig(player_count=len(seat_to_user))
        self.n = self.rules.player_count
        self.seat_to_user = seat_to_user  # {0: user_id, ...}
        self.seat_to_name = seat_to_name  # {0: "Alice", ...}
        self.user_to_seat: Dict[str, int] = {v: k for k, v in seat_to_user.items()}
        self.game = NativeHeartsGame(self.rules)

    # ------------------------------------------------------------------
    # State helpers
    # ------------------------------------------------------------------

    def current_player(self) -> int:
        return self.game.current_player()

    def is_terminal(self) -> bool:
        return self.game.is_terminal()

    def is_passing_phase(self) -> bool:
        return self.game.is_passing_phase()

    def pass_direction(self) -> str:
        return self.game.pass_direction()

    # ------------------------------------------------------------------
    # Move processing
    # ------------------------------------------------------------------

    def apply_move(self, seat: int, card: Card) -> None:
        """Apply a card play from the given seat. Raises ValueError on illegal move."""
        self.game.apply_move(seat, card)

    def queue_pass(self, seat: int, cards: List[Card]) -> None:
        """
        Submit a seat's 3-card pass. Passing is simultaneous: once every seat has
        submitted, the native engine applies all passes at once and play begins.
        """
        if seat not in range(self.n):
            raise ValueError(f"Invalid seat {seat}")
        self.game.submit_pass(seat, cards)

    def process_pending_passes(self) -> List[Tuple[int, Card]]:
        """
        Retained for route compatibility. The native engine applies passes
        atomically inside ``queue_pass``, so there are no incremental pass moves
        to return here.
        """
        return []

    # ------------------------------------------------------------------
    # State serialisation
    # ------------------------------------------------------------------

    def _build_player_list(self, viewer_seat: int) -> List[Player]:
        """Build the players list as seen by viewer_seat — only viewer's hand is full."""
        scores = self._compute_scores()
        players = []
        for seat in range(self.n):
            hand = self.game.get_player_hand(seat) if seat == viewer_seat else []
            players.append(
                Player(
                    id=seat,
                    name=self.seat_to_name.get(seat, f"Player {seat}"),
                    hand=hand,
                    score=scores.get(seat, 0),
                    round_score=scores.get(seat, 0),
                    is_ai=False,
                )
            )
        return players

    def build_state_for_seat(self, viewer_seat: int) -> dict:
        """
        Return a serialisable game state dict personalised for viewer_seat.
        Includes full hand for viewer_seat, card counts for others.
        """
        players = self._build_player_list(viewer_seat)
        hand_counts = {seat: self.game.hand_count(seat) for seat in range(self.n)}
        is_passing = self.is_passing_phase()

        current_trick = [[s, c.dict()] for (s, c) in self.game.current_trick_cards()]
        last_trick = self._serialise_last_trick()

        return {
            "my_seat": viewer_seat,
            "player_count": self.n,
            "rules": self.rules.to_dict(),
            "players": [p.dict() for p in players],
            "hand_counts": hand_counts,
            "current_player": self.current_player(),
            "current_trick": current_trick,
            "last_trick": last_trick,
            "move_sequence": [],
            "is_passing_phase": is_passing,
            "pass_direction": self.pass_direction() if is_passing else None,
            "game_over": self.is_terminal(),
            "hearts_broken": self.game.hearts_broken,
            "round_number": 1,
            "tricks_played": self.game.tricks_played,
            "scores": self._compute_scores(),
            "winner": self._compute_winner(),
            "passes_submitted": self.game.submitted_pass_seats(),
            "my_pass_submitted": self.game.has_submitted_pass(viewer_seat),
        }

    def build_state_with_move(self, viewer_seat: int, move_sequence: list) -> dict:
        """Same as build_state_for_seat but includes the move_sequence for animation."""
        state = self.build_state_for_seat(viewer_seat)
        state["move_sequence"] = move_sequence
        return state

    def _serialise_last_trick(self) -> Optional[dict]:
        lt = self.game.last_trick
        if not lt:
            return None
        return {
            "cards": [[s, c.dict()] for (s, c) in lt["cards"]],
            "winner": lt["winner"],
            "points": lt["points"],
        }

    def _compute_scores(self) -> Dict[int, int]:
        """
        Final custom scores once the deal is over, otherwise a live running tally
        of base penalty points (hearts + Q♠) taken so far.
        """
        final = self.game.final_scores()
        if final is not None:
            return final
        return self.game.running_points()

    def _compute_winner(self) -> Optional[int]:
        return self.game.winner()


class MultiplayerGameManager:
    """
    Global registry of in-progress multiplayer games, keyed by room_id.
    """

    def __init__(self):
        self._games: Dict[str, MultiplayerGameInstance] = {}

    def create_game(
        self,
        room_id: str,
        seat_to_user: Dict[int, str],
        seat_to_name: Dict[int, str],
        rules: Optional[RuleConfig] = None,
    ) -> MultiplayerGameInstance:
        instance = MultiplayerGameInstance(room_id, seat_to_user, seat_to_name, rules)
        self._games[room_id] = instance
        return instance

    def get_game(self, room_id: str) -> Optional[MultiplayerGameInstance]:
        return self._games.get(room_id)

    def delete_game(self, room_id: str):
        self._games.pop(room_id, None)

    def has_game(self, room_id: str) -> bool:
        return room_id in self._games


# Singleton
multiplayer_manager = MultiplayerGameManager()
