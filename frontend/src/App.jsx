import { useState, useEffect } from 'react'
import { AuthProvider, useAuth } from './auth/AuthContext'
import LandingPage from './components/LandingPage'
import MultiplayerLobby from './components/MultiplayerLobby'
import MultiplayerGame from './components/MultiplayerGame'
import GameBoard from './components/GameBoard'
import Controls from './components/Controls'
import Scoreboard from './components/Scoreboard'
import { useGameStore } from './hooks/useGameState'
import './App.css'

// ── View constants ──────────────────────────────────────────────────────────
const VIEW = {
  LANDING: 'landing',
  AI_GAME: 'ai_game',
  MP_LOBBY: 'mp_lobby',
  MP_GAME: 'mp_game',
}

// ── Detect invite-link in URL (/join/CODE or ?invite=CODE) ─────────────────
function getInviteCodeFromUrl() {
  const path = window.location.pathname
  const match = path.match(/\/join\/([A-Z0-9]+)/i)
  if (match) return match[1].toUpperCase()
  const params = new URLSearchParams(window.location.search)
  return params.get('invite')?.toUpperCase() || null
}

// ── Inner app (has access to Auth context) ──────────────────────────────────
function AppInner() {
  const { user } = useAuth()
  const { gameId, gameState, startNewGame, isLoading } = useGameStore()
  const [showScoreboard, setShowScoreboard] = useState(false)
  const [view, setView] = useState(VIEW.LANDING)
  const [activeRoom, setActiveRoom] = useState(null)

  // If URL contains an invite code, jump straight into the multiplayer lobby
  const inviteCode = getInviteCodeFromUrl()
  useEffect(() => {
    if (inviteCode) {
      if (user) {
        setView(VIEW.MP_LOBBY)
      }
      // If not logged in, the user will see the landing page with sign-in first;
      // after login the invite code is still in the URL.
    }
  }, [user, inviteCode])

  const handlePlayAI = async () => {
    setView(VIEW.AI_GAME)
    await startNewGame()
  }

  const handlePlayMultiplayer = () => {
    setView(VIEW.MP_LOBBY)
  }

  const handleGameStart = (room) => {
    setActiveRoom(room)
    setView(VIEW.MP_GAME)
  }

  const handleLeaveGame = () => {
    setActiveRoom(null)
    setView(VIEW.LANDING)
    // Clear game state
    useGameStore.setState({ gameId: null, gameState: null })
  }

  // ── Render ────────────────────────────────────────────────────────────────

  if (view === VIEW.LANDING) {
    return (
      <LandingPage
        onPlayAI={handlePlayAI}
        onPlayMultiplayer={handlePlayMultiplayer}
      />
    )
  }

  if (view === VIEW.AI_GAME) {
    return (
      <div className="app">
        <main className="app-main">
          {!gameId ? (
            <div className="welcome-screen">
              <h2>Welcome to Hearts</h2>
              <p>Challenge three AI opponents in this classic card game</p>
              <div className="welcome-actions">
                <button
                  className="start-button"
                  onClick={startNewGame}
                  disabled={isLoading}
                >
                  {isLoading ? 'Starting...' : 'Start Game'}
                </button>
                <button className="back-button" onClick={() => setView(VIEW.LANDING)}>
                  ← Back
                </button>
              </div>
            </div>
          ) : (
            <>
              <GameBoard />
              <Controls onShowScoreboard={() => setShowScoreboard(true)} />
            </>
          )}
        </main>
        {showScoreboard && <Scoreboard onClose={() => setShowScoreboard(false)} />}
      </div>
    )
  }

  if (view === VIEW.MP_LOBBY) {
    return (
      <MultiplayerLobby
        inviteCode={inviteCode}
        onGameStart={handleGameStart}
        onBack={() => setView(VIEW.LANDING)}
      />
    )
  }

  if (view === VIEW.MP_GAME && activeRoom) {
    return (
      <MultiplayerGame
        room={activeRoom}
        onLeave={handleLeaveGame}
      />
    )
  }

  return null
}

// ── Root app with providers ──────────────────────────────────────────────────
export default function App() {
  return (
    <AuthProvider>
      <AppInner />
    </AuthProvider>
  )
}
