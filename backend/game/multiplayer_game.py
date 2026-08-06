"""
Multiplayer Hearts game manager.

Runs a room of N human players (3/4/5) on the native Hearts engine — no
OpenSpiel, no AI. Supports:
  • custom rules — player count + host-selectable scoring (Jack of Diamonds −10,
    10 of Clubs doubler) via ``RuleConfig``.
  • multi-round play — deals repeat, penalty points accumulate across rounds
    until a seat reaches the host's target score (or forever, if infinite),
    then the lowest total wins (P1-1).

Builds per-seat game-state views (each player sees only their own hand).

NOTE: AI takeover of disconnected seats (previously P1-2) has been removed for
now; a disconnected seat simply pauses the game until the player returns or the
match is ended. The AI seam can be reintroduced later.
"""
from typing import Dict, List, Optional, Tuple

from .native import NativeHeartsGame, RuleConfig
from ..schemas.types import Card, Player


class MultiplayerGameInstance:
    """
    Holds one Hearts *match* (one or more rounds) for a room.
    seat_to_user: seat (0..N-1) → user_id ; seat_to_name: seat (0..N-1) → display name.
    rules: player count + scoring toggles. target_score None ⇒ play until host ends.
    """

    def __init__(
        self,
        room_id: str,
        seat_to_user: Dict[int, str],
        seat_to_name: Dict[int, str],
        rules: Optional[RuleConfig] = None,
        target_score: Optional[int] = 100,
    ):
        self.room_id = room_id
        self.rules = rules or RuleConfig(player_count=len(seat_to_user))
        self.n = self.rules.player_count
        self.seat_to_user = seat_to_user
        self.seat_to_name = seat_to_name
        self.user_to_seat: Dict[str, int] = {v: k for k, v in seat_to_user.items()}
        self.target_score = target_score

        # Match-level state
        self.round_number = 1
        self.cumulative_scores: Dict[int, int] = {i: 0 for i in range(self.n)}  # completed rounds only
        self._match_over = False

        self._new_deal()

    # ------------------------------------------------------------------
    # Round lifecycle
    # ------------------------------------------------------------------

    def _new_deal(self) -> None:
        """Start a fresh deal (with this round's pass direction) and reset round-local state."""
        self.game = NativeHeartsGame(self.rules, deal_index=self.round_number - 1)
        self._round_finalized = False
        self._last_round_scores: Optional[Dict[int, int]] = None

    def finalize_round_if_over(self) -> None:
        """If the current deal has ended, snapshot the round and decide whether
        the whole match is complete. Idempotent."""
        if not self.game.is_terminal() or self._round_finalized:
            return
        self._round_finalized = True
        pts = self.game.final_scores() or {i: 0 for i in range(self.n)}
        self._last_round_scores = dict(pts)
        totals = {i: self.cumulative_scores[i] + pts.get(i, 0) for i in range(self.n)}
        if self.target_score is not None and max(totals.values()) >= self.target_score:
            self._match_over = True

    def start_next_round(self) -> bool:
        """Fold the finished deal into cumulative totals and deal again.
        No-op (returns False) if the round isn't finished or the match is over."""
        self.finalize_round_if_over()
        if not self._round_finalized or self._match_over:
            return False
        pts = self.game.final_scores() or {i: 0 for i in range(self.n)}
        for i in range(self.n):
            self.cumulative_scores[i] += pts.get(i, 0)
        self.round_number += 1
        self._new_deal()
        return True

    def end_match(self) -> None:
        """End the whole match now (host abandon / infinite-mode stop)."""
        if not self._round_finalized and not self.game.is_terminal():
            self._last_round_scores = dict(self.game.running_points())
        self._match_over = True

    # ------------------------------------------------------------------
    # State helpers
    # ------------------------------------------------------------------

    def current_player(self) -> int:
        return self.game.current_player()

    def is_terminal(self) -> bool:
        """True when the current *deal* has ended (not necessarily the match)."""
        return self.game.is_terminal()

    def is_match_over(self) -> bool:
        return self._match_over

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
    # Scores
    # ------------------------------------------------------------------

    def _round_scores(self) -> Dict[int, int]:
        """This deal's points per seat: final custom scores once the deal ends,
        otherwise a running base-penalty tally (hearts + Q♠) mid-deal."""
        if self.game.is_terminal():
            final = self.game.final_scores()
            if final is not None:
                return dict(final)
        return dict(self.game.running_points())

    def _display_scores(self) -> Dict[int, int]:
        """Cumulative match totals = completed rounds + this deal's points."""
        rnd = self._round_scores()
        return {i: self.cumulative_scores[i] + rnd.get(i, 0) for i in range(self.n)}

    # ------------------------------------------------------------------
    # State serialisation
    # ------------------------------------------------------------------

    def _build_player_list(self, viewer_seat: int) -> List[Player]:
        """Build the players list as seen by viewer_seat — only viewer's hand is full."""
        display = self._display_scores()
        rounds = self._round_scores()
        players = []
        for seat in range(self.n):
            hand = self.game.get_player_hand(seat) if seat == viewer_seat else []
            players.append(
                Player(
                    id=seat,
                    name=self.seat_to_name.get(seat, f"Player {seat}"),
                    hand=hand,
                    score=display.get(seat, 0),       # cumulative match total
                    round_score=rounds.get(seat, 0),  # this round's points
                    is_ai=False,
                )
            )
        return players

    def build_state_for_seat(self, viewer_seat: int) -> dict:
        """Return a serialisable game state dict personalised for viewer_seat."""
        # Keep match/round status consistent regardless of caller ordering.
        self.finalize_round_if_over()

        players = self._build_player_list(viewer_seat)
        hand_counts = {seat: self.game.hand_count(seat) for seat in range(self.n)}

        is_passing = self.is_passing_phase()
        deal_over = self.game.is_terminal()
        display = self._display_scores()

        overall_winner = min(display, key=lambda k: display[k]) if self._match_over else None
        round_winner = (
            min(self._last_round_scores, key=lambda k: self._last_round_scores[k])
            if (deal_over and self._last_round_scores) else None
        )

        return {
            "my_seat": viewer_seat,
            "player_count": self.n,
            "rules": self.rules.to_dict(),
            "players": [p.dict() for p in players],
            "hand_counts": hand_counts,
            "current_player": self.current_player(),
            # Absolute seats (0..N-1); the client rotates to its own viewpoint.
            "current_trick": [[s, c.dict()] for (s, c) in self.game.current_trick_cards()],
            "last_trick": self._serialise_last_trick(),
            "move_sequence": [],
            "is_passing_phase": is_passing,
            "pass_direction": self.pass_direction() if is_passing else None,
            # Match vs. round distinction:
            "game_over": self._match_over,                     # whole match complete
            "round_over": deal_over and not self._match_over,  # deal done, more to come
            "hearts_broken": self.game.hearts_broken,
            "round_number": self.round_number,
            "target_score": self.target_score,
            "tricks_played": self.game.tricks_played,
            "scores": display,                             # cumulative totals ("Score")
            "round_scores": self._round_scores(),          # this round's points
            "last_round_scores": self._last_round_scores,  # snapshot for the summary
            "winner": overall_winner,                      # only when game_over
            "round_winner": round_winner,                  # only when round_over
            "passes_submitted": self.game.submitted_pass_seats(),
            "my_pass_submitted": self.game.has_submitted_pass(viewer_seat),
        }

    def build_state_with_move(self, viewer_seat: int, move_sequence: list) -> dict:
        """Kept for API compatibility; move_sequence is no longer used for MP."""
        return self.build_state_for_seat(viewer_seat)

    def _serialise_last_trick(self) -> Optional[dict]:
        lt = self.game.last_trick
        if not lt:
            return None
        return {
            "cards": [[s, c.dict()] for (s, c) in lt["cards"]],
            "winner": lt["winner"],
            "points": lt["points"],
        }


class MultiplayerGameManager:
    """Global registry of in-progress multiplayer games, keyed by room_id."""

    def __init__(self):
        self._games: Dict[str, MultiplayerGameInstance] = {}

    def create_game(
        self,
        room_id: str,
        seat_to_user: Dict[int, str],
        seat_to_name: Dict[int, str],
        rules: Optional[RuleConfig] = None,
        target_score: Optional[int] = 100,
    ) -> MultiplayerGameInstance:
        instance = MultiplayerGameInstance(room_id, seat_to_user, seat_to_name, rules, target_score)
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
