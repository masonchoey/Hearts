"""
Pydantic models for request/response schemas
"""
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any


class Card(BaseModel):
    """Represents a single playing card"""
    suit: str  # "C", "D", "H", "S"
    rank: str  # "2", "3", ..., "T", "J", "Q", "K", "A"
    
    def __str__(self):
        return f"{self.rank}{self.suit}"
    
    def to_string(self):
        """Convert to OpenSpiel format string"""
        return f"{self.rank}{self.suit}"


class Player(BaseModel):
    """Represents a player in the game"""
    id: int = Field(..., ge=0, le=4)  # up to 5 seats (0-4) for custom multiplayer
    name: str
    hand: list[Card] = []
    score: int = 0
    round_score: int = 0
    is_ai: bool = True


class Trick(BaseModel):
    """Represents a trick (4 cards played)"""
    cards: List[tuple[int, Card]] = []  # [(player_id, card), ...]
    winner: Optional[int] = None
    points: int = 0


class GameState(BaseModel):
    """Complete game state"""
    players: List[Player]
    current_trick: List[tuple[int, Card]] = []  # Cards played in current trick
    move_sequence: List[tuple[int, Card]] = []  # Sequence of moves that happened in last step (for animation)
    current_player: int = 0  # Player whose turn it is
    round_number: int = 1
    tricks_played: int = 0
    game_over: bool = False
    hearts_broken: bool = False
    winner: Optional[int] = None
    observation: Optional[List[float]] = None  # 5088-length OpenSpiel observation
    legal_actions: Optional[List[int]] = None  # Legal action indices
    is_passing_phase: bool = False  # Whether we're in the passing phase
    pass_direction: Optional[str] = None  # Pass direction: "No Pass", "Left", "Across", "Right"


class PlayMoveRequest(BaseModel):
    """Request to play a card"""
    player_id: int = Field(..., ge=0, le=4)
    card: Card


class PassCardsRequest(BaseModel):
    """Request to pass multiple cards during passing phase"""
    player_id: int = Field(..., ge=0, le=4)
    cards: list[Card] = Field(..., min_items=3, max_items=3)


class PlayMoveResponse(BaseModel):
    """Response after playing a card"""
    state: GameState
    valid_move: bool = True
    message: Optional[str] = None


