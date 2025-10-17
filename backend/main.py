"""
FastAPI Backend for Hearts Game
Integrates with RLlib-trained models and OpenSpiel Hearts environment
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import uuid

from .game.state_manager import GameStateManager
from .schemas.types import GameState, PlayMoveRequest, PlayMoveResponse, PassCardsRequest

app = FastAPI(title="Hearts Game API", version="1.0.0")

# CORS middleware for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],  # React dev servers
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Game state manager (in-memory for now)
game_manager = GameStateManager()


@app.get("/")
async def root():
    """Health check endpoint"""
    return {"status": "ok", "message": "Hearts Game API is running"}


@app.post("/start")
async def start_game():
    """
    Initialize a new Hearts game
    Returns the initial game state with dealt cards
    """
    try:
        game_id = str(uuid.uuid4())
        game_state = game_manager.create_game(game_id)
        return {
            "game_id": game_id,
            "state": game_state.dict()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start game: {str(e)}")


@app.get("/state/{game_id}")
async def get_state(game_id: str):
    """
    Get current game state for a given game ID
    """
    game_state = game_manager.get_game(game_id)
    if not game_state:
        raise HTTPException(status_code=404, detail="Game not found")
    return {"state": game_state.dict()}


@app.post("/play/{game_id}")
async def play_move(game_id: str, request: PlayMoveRequest):
    """
    Process a player's move WITHOUT auto-processing AI turns
    Returns updated game state immediately after the player's move
    Frontend should call /ai-move endpoint for each AI turn
    """
    game_state = game_manager.get_game(game_id)
    if not game_state:
        raise HTTPException(status_code=404, detail="Game not found")
    
    try:
        # Process the player's move only
        updated_state = game_manager.play_card(game_id, request.player_id, request.card)
        
        return {
            "state": updated_state.dict(),
            "valid_move": True
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process move: {str(e)}")


@app.post("/pass/{game_id}")
async def pass_cards(game_id: str, request: PassCardsRequest):
    """Pass 3 cards during the passing phase"""
    try:
        game_state = game_manager.pass_cards(game_id, request.player_id, request.cards)
        return {"state": game_state.dict()}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to pass cards: {str(e)}")


@app.post("/reset/{game_id}")
async def reset_game(game_id: str):
    """
    Reset an existing game or create a new one
    """
    try:
        game_state = game_manager.reset_game(game_id)
        return {"state": game_state.dict()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to reset game: {str(e)}")


@app.post("/ai-move/{game_id}")
async def process_single_ai_move(game_id: str):
    """
    Process a single AI move (one card play)
    Returns updated game state after one AI card is played
    """
    game_state = game_manager.get_game(game_id)
    if not game_state:
        raise HTTPException(status_code=404, detail="Game not found")
    
    try:
        updated_state = game_manager.process_single_ai_move(game_id)
        return {
            "state": updated_state.dict(),
            "ai_move_processed": True
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process AI move: {str(e)}")


@app.post("/ai-turns/{game_id}")
async def process_ai_turns(game_id: str):
    """
    Process AI turns until it's the human player's turn or trick is complete
    Useful for debugging or manual AI turn triggering
    """
    game_state = game_manager.get_game(game_id)
    if not game_state:
        raise HTTPException(status_code=404, detail="Game not found")
    
    try:
        updated_state = game_manager.process_ai_turns(game_id)
        return {
            "state": updated_state.dict(),
            "ai_turns_processed": True
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process AI turns: {str(e)}")


@app.delete("/game/{game_id}")
async def delete_game(game_id: str):
    """
    Delete a game session
    """
    success = game_manager.delete_game(game_id)
    if not success:
        raise HTTPException(status_code=404, detail="Game not found")
    return {"message": "Game deleted successfully"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


