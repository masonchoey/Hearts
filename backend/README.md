# Hearts Game Backend

FastAPI backend for Hearts game with RLlib AI integration.

## Setup

1. Make sure you have the dependencies from the main requirements.txt installed.

2. Create `.env` file in the project root:
```bash
cp .env.example .env
```

3. Edit `.env` and set your checkpoint path:
```
CHECKPOINT_PATH=./PPO_2025-10-07_04-21-40/PPO_hearts_env_self_play_1f830_00000_0_2025-10-07_04-21-40/checkpoint_000013
```

4. Run the server:
```bash
cd backend
python main.py
```

Or with uvicorn:
```bash
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`

## API Endpoints

- `POST /start` - Start a new game
- `GET /state/{game_id}` - Get current game state
- `POST /play/{game_id}` - Play a card
- `POST /reset/{game_id}` - Reset game
- `DELETE /game/{game_id}` - Delete game session

## Project Structure

```
backend/
├── main.py              # FastAPI app
├── game/
│   ├── hearts_logic.py  # OpenSpiel wrapper
│   └── state_manager.py # Game state management
├── models/
│   └── hearts_model.py  # RLlib model wrapper
└── schemas/
    └── types.py         # Pydantic models
```

## OpenSpiel Integration

The backend uses OpenSpiel's Hearts environment which provides:
- 5088-length observation vectors
- Game rules and validation
- Legal action computation
- Score tracking

## RLlib Model

The AI players use your pre-trained RLlib PPO model. The model is loaded from the checkpoint path specified in `.env`.


