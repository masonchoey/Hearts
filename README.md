# OpenSpiel Hearts - RL Training & Interactive Game

Developing an RL training pipeline with Ray RLlib + PyTorch to train PPO agents for the Hearts card game via self-play. Currently using action-masked attention-based neural networks and an evaluation framework benchmarking performance with a **35%** win rate against diverse bot strategies in a 4-player setting.

## Features

- **RL Training Pipeline**: Self-play training with PPO agents
- **Attention-Based Neural Networks**: Action-masked models for strategic play
- **Evaluation Framework**: Benchmark against diverse bot strategies
- **Interactive Web Interface**: Play against your trained AI models!
  - React frontend with elegant card table UI
  - FastAPI backend with OpenSpiel integration
  - Real-time gameplay with 3 AI opponents

## Documentation

- **[DOCS_INDEX.md](DOCS_INDEX.md)** - Complete documentation index
- **[QUICKSTART.md](QUICKSTART.md)** - Get started in 5 minutes
- **[GAME_SETUP.md](GAME_SETUP.md)** - Detailed setup guide
- **[ARCHITECTURE.md](ARCHITECTURE.md)** - System architecture
- **[DEPLOYMENT.md](DEPLOYMENT.md)** - Production deployment

## Quick Start - Play the Game!

Want to play against your trained models? See **[QUICKSTART.md](QUICKSTART.md)** for a 5-minute guide.

### TL;DR

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure your checkpoint
cp .env.example .env
# Edit .env and set CHECKPOINT_PATH

# 3. Run backend
./run_backend.sh

# 4. Run frontend (in new terminal)
./run_frontend.sh

# 5. Play at http://localhost:3000
```

## Project Structure

```
OpenSpiel-Hearts/
├── backend/              # FastAPI game server
├── frontend/             # React web interface
├── hearts_env_*.py       # OpenSpiel environment wrappers
├── main_self_play*.py    # Training scripts
├── rl_vs_bots.py        # Evaluation against bots
├── attention_model.py    # Neural network architectures
└── PPO_*/               # Trained checkpoints
```

## Training

For training your own models, see the training scripts:
- `main_self_play.py` - Basic self-play training
- `main_self_play_optimized.py` - Optimized training pipeline

## Web Interface

The web interface provides:
- 4-player Hearts game board
- Interactive card playing
- Real-time AI opponent moves
- Score tracking and leaderboard
- OpenSpiel integration (5088-length observations)
- RLlib model inference

See [GAME_SETUP.md](GAME_SETUP.md) for complete setup and usage instructions. 
