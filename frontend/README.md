# Hearts Game Frontend

React-based frontend for the Hearts card game with AI opponents.

## Setup

1. Install dependencies:
```bash
npm install
```

2. Create `.env` file (optional):
```bash
cp .env.example .env
```

3. Run development server:
```bash
npm run dev
```

The app will be available at `http://localhost:3000`

## Project Structure

```
src/
├── components/     # React components
│   ├── GameBoard.jsx
│   ├── PlayerHand.jsx
│   ├── TableCenter.jsx
│   ├── Card.jsx
│   ├── Scoreboard.jsx
│   └── Controls.jsx
├── hooks/         # Custom hooks
│   └── useGameState.js
├── api/           # API communication
│   └── backend.js
└── App.jsx        # Main app component
```

## Features

- 4-player Hearts game interface
- Real-time game state updates
- Interactive card playing
- Score tracking
- Game controls (reset, scoreboard)
- Responsive design with animations


