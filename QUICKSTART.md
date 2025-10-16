# Quick Start Guide

Get up and running with the Hearts game in 5 minutes!

## Prerequisites Check

Before starting, make sure you have:
- ✅ Python 3.8+ installed
- ✅ Node.js 18+ installed  
- ✅ A trained RLlib checkpoint

## 1. Verify Your Setup

Run the verification script:

```bash
python verify_setup.py
```

This will check if everything is installed correctly.

## 2. Install Dependencies

```bash
# Python dependencies
pip install -r requirements.txt

# Frontend dependencies (in a new terminal)
cd frontend
npm install
cd ..
```

## 3. Configure Environment

```bash
# Copy the example env file
cp .env.example .env

# Edit .env and set your checkpoint path
# Example:
# CHECKPOINT_PATH=./PPO_2025-10-07_04-21-40/PPO_hearts_env_self_play_1f830_00000_0_2025-10-07_04-21-40/checkpoint_000013
nano .env  # or use your preferred editor
```

**Finding your checkpoint:**
```bash
# List available checkpoints
find . -name "checkpoint_*" -type d | head -5
```

## 4. Start the Backend

Open a terminal and run:

```bash
chmod +x run_backend.sh
./run_backend.sh
```

You should see:
```
INFO:     Started server process [12345]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000
```

## 5. Start the Frontend

Open a **new terminal** and run:

```bash
chmod +x run_frontend.sh
./run_frontend.sh
```

You should see:
```
  VITE v5.0.0  ready in 1234 ms

  ➜  Local:   http://localhost:3000/
  ➜  Network: use --host to expose
```

## 6. Play!

1. Open your browser to **http://localhost:3000**
2. Click **"Start Game"**
3. Select a card from your hand (bottom of screen)
4. Click **"Play Card"**
5. Watch the AI opponents play automatically!

## Game Controls

- **Select Card**: Click on a card in your hand
- **Play Card**: Click the green "Play Card" button
- **View Scores**: Click "📊 Scores" button (bottom-right)
- **New Round**: Click "🔄 New Round" button

## Understanding the UI

```
        [AI Player 2]
             ↑
[AI 3] ←  [Table]  → [AI 1]
             ↓
          [You]
```

- **Your Hand**: Bottom - click to select/play cards
- **AI Hands**: Other positions - shown as card backs
- **Table Center**: Middle - shows current trick
- **Scores**: Display next to each player name

## Troubleshooting

### Backend won't start

**Problem**: Port 8000 already in use
```bash
# Find and kill the process
lsof -ti:8000 | xargs kill -9
```

**Problem**: Checkpoint not found
```bash
# Verify your checkpoint path
ls -la $CHECKPOINT_PATH  # Replace with your path

# If path doesn't exist, update .env
nano .env
```

**Problem**: Ray errors
```bash
# Clean Ray temp files
ray stop
rm -rf /tmp/ray

# Try starting backend again
./run_backend.sh
```

### Frontend won't start

**Problem**: Port 3000 already in use
```bash
# Kill the process
lsof -ti:3000 | xargs kill -9
```

**Problem**: npm install fails
```bash
cd frontend
rm -rf node_modules package-lock.json
npm install
```

**Problem**: Can't connect to backend
- Make sure backend is running on port 8000
- Check browser console for errors (F12)
- Try accessing http://localhost:8000 directly

### Game issues

**Problem**: Cards won't play
- Only your turn! Wait for AI players to finish
- Make sure you've selected a card first
- Check that the move is legal (follow suit)

**Problem**: AI taking too long
- First move might be slow (model loading)
- Check backend terminal for errors
- Model inference typically takes <100ms

**Problem**: Page won't load
- Clear browser cache (Ctrl+Shift+R)
- Check browser console (F12)
- Make sure both backend and frontend are running

## Next Steps

Once you're up and running:

1. **Read the full docs**:
   - [GAME_SETUP.md](GAME_SETUP.md) - Detailed setup
   - [ARCHITECTURE.md](ARCHITECTURE.md) - How it works
   - [DEPLOYMENT.md](DEPLOYMENT.md) - Production deployment

2. **Customize**:
   - Try different checkpoints
   - Modify the UI colors/layout
   - Adjust AI difficulty

3. **Develop**:
   - Add new features
   - Improve the AI
   - Deploy to production

## Testing the API

Want to test the backend directly?

```bash
# Start a game
curl -X POST http://localhost:8000/start

# The response will include a game_id, use it for other commands:
GAME_ID="your-game-id-here"

# Get game state
curl http://localhost:8000/state/$GAME_ID

# Play a card (2 of clubs)
curl -X POST http://localhost:8000/play/$GAME_ID \
  -H "Content-Type: application/json" \
  -d '{"player_id": 0, "card": {"suit": "C", "rank": "2"}}'
```

## Docker Alternative

Prefer Docker? Run everything with one command:

```bash
# Make sure .env is configured
cp .env.example .env
nano .env  # Set CHECKPOINT_PATH

# Start with Docker Compose
docker-compose up --build

# Access at http://localhost:3000
```

## Getting Help

- Check the troubleshooting section above
- Review error messages in terminal/browser console
- Read the detailed docs in [GAME_SETUP.md](GAME_SETUP.md)
- Check the [ARCHITECTURE.md](ARCHITECTURE.md) for how things work

## Common Questions

**Q: Do I need a trained model?**  
A: Yes, but the backend will use random moves if checkpoint is not found. You won't get smart AI, but the game will still work.

**Q: Can I play against humans?**  
A: Not yet - currently only single player vs 3 AI. Multiplayer is a planned feature.

**Q: How do I train my own model?**  
A: Use the existing training scripts (`main_self_play.py`). The web interface uses those checkpoints.

**Q: Can I deploy this online?**  
A: Yes! See [DEPLOYMENT.md](DEPLOYMENT.md) for various deployment options.

**Q: Is there a mobile version?**  
A: Not yet, but the web UI works on mobile browsers. A native app is a future enhancement.

## Success Checklist

- [ ] Backend running on port 8000
- [ ] Frontend running on port 3000  
- [ ] Browser shows the welcome screen
- [ ] Can start a new game
- [ ] Can see cards in hand
- [ ] Can select and play cards
- [ ] AI players respond automatically
- [ ] Scores update correctly

If all checked, you're all set! Enjoy playing Hearts! 🎴♥️

---

**Need more help?** Check out:
- [GAME_SETUP.md](GAME_SETUP.md) for detailed instructions
- [ARCHITECTURE.md](ARCHITECTURE.md) to understand the system
- [DEPLOYMENT.md](DEPLOYMENT.md) for production deployment


