"""
Multiplayer Hearts game manager.
Wraps HeartsGame for a room with 4 human players — no AI.
Builds per-seat game-state views (each player sees only their own hand).
"""
from typing import Dict, List, Optional, Tuple
from .hearts_logic import HeartsGame
from ..schemas.types import Card, Player, GameState


class MultiplayerGameInstance:
    """
    Holds one Hearts game for a room.
    seat_to_user: maps seat (0-3) → user_id
    seat_to_name: maps seat (0-3) → display name
    """

    def __init__(
        self,
        room_id: str,
        seat_to_user: Dict[int, str],
        seat_to_name: Dict[int, str],
    ):
        self.room_id = room_id
        self.seat_to_user = seat_to_user  # {0: user_id, ...}
        self.seat_to_name = seat_to_name  # {0: "Alice", ...}
        self.user_to_seat: Dict[str, int] = {v: k for k, v in seat_to_user.items()}
        self.game = HeartsGame()
        self.game.reset()

    # ------------------------------------------------------------------
    # State helpers
    # ------------------------------------------------------------------

    def current_player(self) -> int:
        return self.game.current_player()

    def is_terminal(self) -> bool:
        return self.game.is_terminal()

    def is_passing_phase(self) -> bool:
        return self.game.is_passing_phase(0)

    def pass_direction(self) -> str:
        return self.game.get_pass_direction(0)

    # ------------------------------------------------------------------
    # Move processing
    # ------------------------------------------------------------------

    def apply_move(self, seat: int, card: Card) -> None:
        """Apply a card play or pass from the given seat. Raises ValueError on illegal move."""
        if self.game.current_player() != seat:
            raise ValueError(f"It is not seat {seat}'s turn")
        if not self.game.validate_move(seat, card):
            raise ValueError(f"Card {card} is not a legal move for seat {seat}")
        action = self.game.card_to_action(card)
        self.game.apply_action(action)

    # ------------------------------------------------------------------
    # State serialisation
    # ------------------------------------------------------------------

    def _build_player_list(self, viewer_seat: int) -> List[Player]:
        """Build the players list as seen by viewer_seat — only viewer's hand is full."""
        players = []
        for seat in range(4):
            hand = self.game.get_player_hand(seat) if seat == viewer_seat else []
            players.append(
                Player(
                    id=seat,
                    name=self.seat_to_name.get(seat, f"Player {seat}"),
                    hand=hand,
                    score=0,
                    round_score=0,
                    is_ai=False,
                )
            )
        return players

    def _current_trick(self) -> List[Tuple[int, Card]]:
        # OpenSpiel doesn't expose trick cards directly through the RL env,
        # so we track it via the observation's played-cards section.
        # For now return empty; it's populated from move_sequence in practice.
        return []

    def build_state_for_seat(self, viewer_seat: int) -> dict:
        """
        Return a serialisable game state dict personalised for viewer_seat.
        Includes full hand for viewer_seat, card counts for others.
        """
        players = self._build_player_list(viewer_seat)

        # Compute hand counts for all players from their observations
        hand_counts = {}
        for seat in range(4):
            hand_counts[seat] = len(self.game.get_player_hand(seat))

        is_passing = self.is_passing_phase()
        return {
            "my_seat": viewer_seat,
            "players": [p.dict() for p in players],
            "hand_counts": hand_counts,
            "current_player": self.current_player(),
            "current_trick": [],
            "move_sequence": [],
            "is_passing_phase": is_passing,
            "pass_direction": self.pass_direction() if is_passing else None,
            "game_over": self.is_terminal(),
            "hearts_broken": self.game.hearts_broken,
            "round_number": 1,
            "tricks_played": 0,
            "scores": self._compute_scores(),
            "winner": self._compute_winner(),
        }

    def build_state_with_move(self, viewer_seat: int, move_sequence: list) -> dict:
        """Same as build_state_for_seat but includes the move_sequence for animation."""
        state = self.build_state_for_seat(viewer_seat)
        state["move_sequence"] = move_sequence
        return state

    def _compute_scores(self) -> Dict[int, int]:
        """Return cumulative scores for all seats."""
        if self.is_terminal():
            returns = self.game.get_returns()
            return {i: max(0, 26 - int(r)) for i, r in enumerate(returns)}
        # Scores mid-game are tracked per-trick; approximate from observations
        return {i: 0 for i in range(4)}

    def _compute_winner(self) -> Optional[int]:
        if not self.is_terminal():
            return None
        scores = self._compute_scores()
        return min(scores, key=lambda k: scores[k])


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
    ) -> MultiplayerGameInstance:
        instance = MultiplayerGameInstance(room_id, seat_to_user, seat_to_name)
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
