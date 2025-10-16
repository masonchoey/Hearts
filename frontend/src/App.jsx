import { useState, useEffect } from 'react'
import GameBoard from './components/GameBoard'
import Controls from './components/Controls'
import Scoreboard from './components/Scoreboard'
import { useGameStore } from './hooks/useGameState'
import './App.css'

function App() {
  const { gameId, gameState, startNewGame, isLoading } = useGameStore()
  const [showScoreboard, setShowScoreboard] = useState(false)

  return (
    <div className="app">
      <main className="app-main">
        {!gameId ? (
          <div className="welcome-screen">
            <h2>Welcome to Hearts</h2>
            <p>Challenge three AI opponents in this classic card game</p>
            <button 
              className="start-button"
              onClick={startNewGame}
              disabled={isLoading}
            >
              {isLoading ? 'Starting...' : 'Start Game'}
            </button>
          </div>
        ) : (
          <>
            <GameBoard />
            <Controls onShowScoreboard={() => setShowScoreboard(true)} />
          </>
        )}
      </main>

      {showScoreboard && (
        <Scoreboard onClose={() => setShowScoreboard(false)} />
      )}
    </div>
  )
}

export default App


