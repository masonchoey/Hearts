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
    id: int = Field(..., ge=0, le=3)
    name: str
    hand: List[Card] = []
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
    current_player: int = 0  # Player whose turn it is
    round_number: int = 1
    tricks_played: int = 0
    game_over: bool = False
    hearts_broken: bool = False
    winner: Optional[int] = None
    observation: Optional[List[float]] = None  # 5088-length OpenSpiel observation
    legal_actions: Optional[List[int]] = None  # Legal action indices


class PlayMoveRequest(BaseModel):
    """Request to play a card"""
    player_id: int = Field(..., ge=0, le=3)
    card: Card


class PlayMoveResponse(BaseModel):
    """Response after playing a card"""
    state: GameState
    valid_move: bool = True
    message: Optional[str] = None


