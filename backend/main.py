"""
FastAPI Backend for Hearts Game
Integrates with RLlib-trained models and OpenSpiel Hearts environment
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import uuid
from contextlib import asynccontextmanager
import ray

from .game.state_manager import GameStateManager
from .schemas.types import GameState, PlayMoveRequest, PlayMoveResponse, PassCardsRequest

# Global game manager instance
game_manager: Optional[GameStateManager] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for FastAPI application
    Handles startup and shutdown events for resource management
    """
    # Startup: Initialize Ray and load AI model
    global game_manager
    print("=" * 60)
    print("🎮 Hearts Game Backend Starting Up...")
    print("=" * 60)
    
    try:
        # Initialize game manager with eager loading
        # This will initialize Ray and load the model immediately
        game_manager = GameStateManager(eager_load=True)
        print("✓ Game manager initialized with AI model ready")
        print("=" * 60)
        print("🚀 Backend ready to accept requests")
        print("=" * 60)
    except Exception as e:
        print(f"✗ Error during startup: {e}")
        print("Backend will continue but AI may not be available")
    
    yield
    
    # Shutdown: Clean up resources
    print("=" * 60)
    print("🛑 Hearts Game Backend Shutting Down...")
    print("=" * 60)
    
    if game_manager:
        # Shutdown all active game wrappers
        for game_id, wrapper in list(game_manager.gym_wrappers.items()):
            wrapper.shutdown()
        print("✓ All game wrappers shut down")
    
    if ray.is_initialized():
        ray.shutdown()
        print("✓ Ray shut down")
    
    print("=" * 60)
    print("👋 Backend shutdown complete")
    print("=" * 60)


app = FastAPI(
    title="Hearts Game API", 
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],  # React dev servers
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "ok",
        "message": "Hearts Game API is running (Gymnasium-based)",
        "ray_initialized": ray.is_initialized() if ray else False,
        "active_games": len(game_manager.gym_wrappers) if game_manager else 0
    }


@app.get("/health")
async def health_check():
    """
    Detailed health check endpoint
    Returns status of Ray and active games
    """
    try:
        ray_status = "initialized" if ray.is_initialized() else "not initialized"
        active_games = len(game_manager.gym_wrappers) if game_manager else 0
        checkpoint_configured = game_manager.checkpoint_path is not None if game_manager else False
        
        return {
            "status": "healthy",
            "ray": ray_status,
            "checkpoint_configured": checkpoint_configured,
            "active_games": active_games,
            "ready_for_inference": game_manager is not None
        }
    except Exception as e:
        return {
            "status": "degraded",
            "error": str(e)
        }


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
    Ensures is_passing_phase is up-to-date
    """
    game_state = game_manager.get_game(game_id)
    if not game_state:
        raise HTTPException(status_code=404, detail="Game not found")
    
    # Refresh is_passing_phase from the actual game state
    wrapper = game_manager.gym_wrappers.get(game_id)
    if wrapper and wrapper.is_game_active():
        from .game.hearts_logic import HeartsGame
        internal_game = HeartsGame()
        internal_game._env = wrapper.env._base_env
        internal_game._timestep = wrapper.env._last_timestep
        game_state.is_passing_phase = internal_game.is_passing_phase(0)
        game_state.pass_direction = internal_game.get_pass_direction(0)
    
    return {"state": game_state.dict()}


@app.post("/play/{game_id}")
async def play_move(game_id: str, request: PlayMoveRequest):
    """
    Process a player's move and automatically process all AI turns.
    The Gymnasium environment automatically handles all AI moves until it's the human's turn again.
    Returns updated game state with AI moves already completed.
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


