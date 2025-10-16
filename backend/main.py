"""
FastAPI Backend for Hearts Game
Integrates with RLlib-trained models and OpenSpiel Hearts environment
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import uuid

from game.state_manager import GameStateManager
from schemas.types import GameState, PlayMoveRequest, PlayMoveResponse

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
    Process a player's move and trigger AI responses
    Returns updated game state after all players have played
    """
    game_state = game_manager.get_game(game_id)
    if not game_state:
        raise HTTPException(status_code=404, detail="Game not found")
    
    try:
        # Process the player's move
        updated_state = game_manager.play_card(game_id, request.player_id, request.card)
        
        # If it's not game over, process AI moves
        if not updated_state.game_over:
            updated_state = game_manager.process_ai_turns(game_id)
        
        return {
            "state": updated_state.dict(),
            "valid_move": True
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process move: {str(e)}")


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


