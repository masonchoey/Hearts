"""
Multiplayer Hearts game manager.

Wraps HeartsGame for a room of up to 4 players. Supports:
  • multi-round play — deals repeat, penalty points accumulate across rounds
    until a seat reaches the host's target score (or forever, if infinite),
    then the lowest total wins (P1-1).
  • AI takeover — a disconnected seat can be handed to a model-agnostic bot so
    the game doesn't stall (P1-2).

Builds per-seat game-state views (each player sees only their own hand).
"""
from typing import Dict, List, Optional, Tuple

from .hearts_logic import HeartsGame
from .ai_controller import AIController, get_ai_controller
from ..schemas.types import Card, Player


class MultiplayerGameInstance:
    """
    Holds one Hearts *match* (one or more rounds) for a room.
    seat_to_user: seat (0-3) → user_id ; seat_to_name: seat (0-3) → display name.
    target_score: None means play forever until the host ends the match.
    """

    def __init__(
        self,
        room_id: str,
        seat_to_user: Dict[int, str],
        seat_to_name: Dict[int, str],
        target_score: Optional[int] = 100,
    ):
        self.room_id = room_id
        self.seat_to_user = seat_to_user
        self.seat_to_name = seat_to_name
        self.user_to_seat: Dict[str, int] = {v: k for k, v in seat_to_user.items()}
        self.target_score = target_score

        # Match-level state
        self.round_number = 1
        self.cumulative_scores: Dict[int, int] = {i: 0 for i in range(4)}  # completed rounds only
        self._match_over = False

        # AI-controlled seats (model-agnostic controllers)
        self.ai_seats: set[int] = set()
        self._ai: Dict[int, AIController] = {}
        # seat → wall-clock time it disconnected (for the "sub in AI" grace window)
        self.disconnect_times: Dict[int, float] = {}

        self._new_deal()

    # ------------------------------------------------------------------
    # Round lifecycle
    # ------------------------------------------------------------------

    def _new_deal(self) -> None:
        """Start a fresh deal and reset all round-local state."""
        self.game = HeartsGame()
        self.game.reset()
        self._pending_passes: Dict[int, List[Card]] = {}
        self._passes_submitted: set[int] = set()
        self._current_trick: List[Tuple[int, Card]] = []
        self._tricks_completed = 0
        self._round_finalized = False
        self._last_round_scores: Optional[Dict[int, int]] = None

    def finalize_round_if_over(self) -> None:
        """If the current deal has ended, snapshot the round and decide whether
        the whole match is complete. Idempotent."""
        if not self.game.is_terminal() or self._round_finalized:
            return
        self._round_finalized = True
        pts = self.game.get_points()
        self._last_round_scores = {i: pts[i] for i in range(4)}
        totals = {i: self.cumulative_scores[i] + pts[i] for i in range(4)}
        if self.target_score is not None and max(totals.values()) >= self.target_score:
            self._match_over = True

    def start_next_round(self) -> bool:
        """Fold the finished deal into cumulative totals and deal again.
        No-op (returns False) if the round isn't finished or the match is over."""
        self.finalize_round_if_over()
        if not self._round_finalized or self._match_over:
            return False
        pts = self.game.get_points()
        for i in range(4):
            self.cumulative_scores[i] += pts[i]
        self.round_number += 1
        self._new_deal()
        return True

    def end_match(self) -> None:
        """End the whole match now (host abandon / infinite-mode stop)."""
        if not self._round_finalized and not self.game.is_terminal():
            pts = self.game.get_points()
            self._last_round_scores = {i: pts[i] for i in range(4)}
        self._match_over = True

    # ------------------------------------------------------------------
    # AI takeover
    # ------------------------------------------------------------------

    def sub_in_ai(self, seat: int) -> None:
        self.ai_seats.add(seat)
        self._ai[seat] = get_ai_controller(seat)

    def remove_ai(self, seat: int) -> None:
        self.ai_seats.discard(seat)
        self._ai.pop(seat, None)

    def mark_disconnected(self, seat: int, now: float) -> None:
        self.disconnect_times.setdefault(seat, now)

    def mark_reconnected(self, seat: int) -> None:
        """A human reclaimed their seat — drop the disconnect timer and any AI."""
        self.disconnect_times.pop(seat, None)
        self.remove_ai(seat)

    def seat_disconnect_elapsed(self, seat: int, now: float) -> Optional[float]:
        t = self.disconnect_times.get(seat)
        return None if t is None else now - t

    def advance_ai(self) -> bool:
        """Auto-play/pass for AI-controlled seats until it's a human's turn (or the
        deal ends). Returns True if any move was made."""
        moved = False
        guard = 0
        while not self.game.is_terminal() and guard < 120:
            guard += 1
            if self.is_passing_phase():
                submitted = False
                for seat in sorted(self.ai_seats):
                    if seat not in self._passes_submitted:
                        hand = self.game.get_player_hand(seat)
                        cards = self._ai[seat].choose_pass(hand, self.pass_direction())
                        try:
                            self.queue_pass(seat, cards)
                            submitted = True
                        except ValueError:
                            pass
                self.process_pending_passes()
                moved = moved or submitted
                if self.is_passing_phase():
                    break  # still waiting on human passes
                continue
            cp = self.current_player()
            if cp in self.ai_seats and cp >= 0:
                legal = [self.game.action_to_card(a) for a in self.game.get_legal_actions()]
                if not legal:
                    break
                card = self._ai[cp].choose_play(legal, self, cp)
                self.apply_move(cp, card)
                moved = True
            else:
                break
        return moved

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
        was_passing = self.is_passing_phase()
        action = self.game.card_to_action(card)
        self.game.apply_action(action)

        # Maintain the visible trick during the playing phase (pass cards excluded).
        if not was_passing:
            if len(self._current_trick) >= 4:
                # Previous trick is complete — clear it now that a new card lands.
                self._current_trick = []
            self._current_trick.append((seat, card))
            if len(self._current_trick) == 4:
                self._tricks_completed += 1

    def queue_pass(self, seat: int, cards: List[Card]) -> None:
        """Queue a 3-card pass for a seat. Applied when OpenSpiel reaches that seat's turn."""
        if not self.is_passing_phase():
            raise ValueError("Not in passing phase")
        if seat in self._passes_submitted:
            raise ValueError("You have already submitted your pass")
        if len(cards) != 3:
            raise ValueError("Must pass exactly 3 cards")

        hand = self.game.get_player_hand(seat)
        hand_set = {(c.suit, c.rank) for c in hand}
        card_keys = [(c.suit, c.rank) for c in cards]
        if len(set(card_keys)) != 3:
            raise ValueError("Pass cards must be distinct")
        for key in card_keys:
            if key not in hand_set:
                raise ValueError(f"Card {key} is not in your hand")

        self._pending_passes[seat] = list(cards)
        self._passes_submitted.add(seat)

    def process_pending_passes(self) -> List[Tuple[int, Card]]:
        """Apply queued pass cards in OpenSpiel turn order until no more can be applied."""
        moves: List[Tuple[int, Card]] = []
        for _ in range(12):  # safety cap — at most 12 single-card pass steps per round
            if not self.is_passing_phase():
                break
            seat = self.current_player()
            queue = self._pending_passes.get(seat)
            if not queue:
                break
            card = queue.pop(0)
            if not queue:
                del self._pending_passes[seat]
            self.apply_move(seat, card)
            moves.append((seat, card))
        return moves

    # ------------------------------------------------------------------
    # Scores
    # ------------------------------------------------------------------

    def _round_scores(self) -> Dict[int, int]:
        """This deal's running penalty points per seat (from OpenSpiel)."""
        pts = self.game.get_points()
        return {i: pts[i] for i in range(4)}

    def _display_scores(self) -> Dict[int, int]:
        """Cumulative match totals = completed rounds + this deal's running points."""
        pts = self.game.get_points()
        return {i: self.cumulative_scores[i] + pts[i] for i in range(4)}

    # ------------------------------------------------------------------
    # State serialisation
    # ------------------------------------------------------------------

    def _build_player_list(self, viewer_seat: int) -> List[Player]:
        """Build the players list as seen by viewer_seat — only viewer's hand is full."""
        display = self._display_scores()
        rounds = self._round_scores()
        players = []
        for seat in range(4):
            hand = self.game.get_player_hand(seat) if seat == viewer_seat else []
            players.append(
                Player(
                    id=seat,
                    name=self.seat_to_name.get(seat, f"Player {seat}"),
                    hand=hand,
                    score=display[seat],        # cumulative match total
                    round_score=rounds[seat],   # this round's points
                    is_ai=seat in self.ai_seats,
                )
            )
        return players

    def build_state_for_seat(self, viewer_seat: int) -> dict:
        """Return a serialisable game state dict personalised for viewer_seat."""
        # Keep match/round status consistent regardless of caller ordering.
        self.finalize_round_if_over()

        players = self._build_player_list(viewer_seat)
        hand_counts = {seat: len(self.game.get_player_hand(seat)) for seat in range(4)}

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
            "players": [p.dict() for p in players],
            "hand_counts": hand_counts,
            "current_player": self.current_player(),
            # Absolute seats (0-3); the client rotates to its own viewpoint.
            "current_trick": [[seat, card.dict()] for seat, card in self._current_trick],
            "move_sequence": [],
            "is_passing_phase": is_passing,
            "pass_direction": self.pass_direction() if is_passing else None,
            # Match vs. round distinction:
            "game_over": self._match_over,                 # whole match complete
            "round_over": deal_over and not self._match_over,  # deal done, more to come
            "hearts_broken": self.game.get_hearts_broken(),
            "round_number": self.round_number,
            "target_score": self.target_score,
            "tricks_played": self._tricks_completed,
            "scores": display,                             # cumulative totals ("Score")
            "round_scores": self._round_scores(),          # this round's points
            "last_round_scores": self._last_round_scores,  # snapshot for the summary
            "winner": overall_winner,                      # only when game_over
            "round_winner": round_winner,                  # only when round_over
            "ai_seats": sorted(self.ai_seats),
            "passes_submitted": sorted(self._passes_submitted),
            "my_pass_submitted": viewer_seat in self._passes_submitted,
        }

    def build_state_with_move(self, viewer_seat: int, move_sequence: list) -> dict:
        """Kept for API compatibility; move_sequence is no longer used for MP."""
        return self.build_state_for_seat(viewer_seat)


class MultiplayerGameManager:
    """Global registry of in-progress multiplayer games, keyed by room_id."""

    def __init__(self):
        self._games: Dict[str, MultiplayerGameInstance] = {}

    def create_game(
        self,
        room_id: str,
        seat_to_user: Dict[int, str],
        seat_to_name: Dict[int, str],
        target_score: Optional[int] = 100,
    ) -> MultiplayerGameInstance:
        instance = MultiplayerGameInstance(room_id, seat_to_user, seat_to_name, target_score)
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
