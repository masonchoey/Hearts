/**
 * Multiplayer game wrapper.
 * Bridges the WebSocket state from useMultiplayerGame into the existing
 * Zustand store so all existing GameBoard / PlayerHand components work unchanged.
 * Players array is rotated so the current user always appears at seat 0 (bottom).
 */
import { useEffect, useRef, useState } from 'react'
import { useAuth } from '../auth/AuthContext'
import { useMultiplayerGame } from '../hooks/useMultiplayerGame'
import { useGameStore } from '../hooks/useGameState'
import { startRoom } from '../api/multiplayer'
import GameBoard from './GameBoard'
import Controls from './Controls'
import Scoreboard from './Scoreboard'
import './MultiplayerGame.css'

function rotateArray(arr, by) {
  if (!arr || arr.length === 0) return arr
  const n = arr.length
  const offset = ((by % n) + n) % n
  return [...arr.slice(offset), ...arr.slice(0, offset)]
}

export default function MultiplayerGame({ room, onLeave }) {
  const { user, token } = useAuth()
  const {
    gameState,
    mySeat,
    connectionStatus,
    connectedSeats,
    error: wsError,
    playCard: wsPlayCard,
    passCards: wsPassCards,
  } = useMultiplayerGame(room.room_id, token)
  const [showScoreboard, setShowScoreboard] = useState(false)
  const [starting, setStarting] = useState(false)
  const [startError, setStartError] = useState(null)

  // Track original playCard so we can restore on unmount
  const origPlayCardRef = useRef(null)
  const origPassCardsRef = useRef(null)

  useEffect(() => {
    if (gameState) setStarting(false)
  }, [gameState])

  // ── Sync WebSocket state → Zustand store ──────────────────────────────────
  useEffect(() => {
    if (!gameState || mySeat === null) return

    // Rotate so mySeat appears at rotated index 0 (bottom / "human" slot)
    const rotatedHandCounts = rotateArray(
      [0, 1, 2, 3].map(seat => gameState.hand_counts?.[seat] ?? 0),
      mySeat,
    )
    const rotatedPlayers = rotateArray(gameState.players, mySeat).map((p, i) => ({
      ...p,
      id: i,
      is_ai: false,
      // Backend only sends full hand for the viewer; use hand_counts for card backs
      hand: p.hand?.length ? p.hand : Array(rotatedHandCounts[i]).fill(null),
    }))

    const rotatedCurrentPlayer = ((gameState.current_player ?? 0) - mySeat + 4) % 4
    const rotatedPassesSubmitted = (gameState.passes_submitted ?? []).map(
      s => (s - mySeat + 4) % 4,
    )

    useGameStore.setState({
      gameId: `mp_${room.room_id}`,
      gameState: {
        ...gameState,
        players: rotatedPlayers,
        current_player: rotatedCurrentPlayer,
        passes_submitted: rotatedPassesSubmitted,
        my_pass_submitted: gameState.my_pass_submitted ?? false,
      },
      isLoading: false,
      error: null,
    })
  }, [gameState, mySeat, room.room_id])

  // ── Override playCard and passCards to use WebSocket ─────────────────────
  useEffect(() => {
    const store = useGameStore.getState()
    origPlayCardRef.current = store.playCard
    origPassCardsRef.current = store.passCards

    // Override playCard: ignore rotated playerId, always play for the real me (mySeat)
    useGameStore.setState({
      playCard: async (_rotatedPlayerId, card) => {
        wsPlayCard(card)
        return null
      },
      // Queue full 3-card pass — backend applies when OpenSpiel reaches this seat
      passCards: async (_rotatedPlayerId, cards) => {
        const store = useGameStore.getState()
        const currentGameState = store.gameState
        if (!currentGameState) return

        // Optimistic: remove cards from hand immediately
        const updatedPlayers = currentGameState.players.map(player => {
          if (player.id !== 0) return player
          return {
            ...player,
            hand: player.hand.filter(
              c => c && !cards.some(p => p.suit === c.suit && p.rank === c.rank),
            ),
          }
        })

        useGameStore.setState({
          gameState: {
            ...currentGameState,
            players: updatedPlayers,
            my_pass_submitted: true,
          },
          selectedCards: [],
        })

        wsPassCards(cards)
      },
    })

    return () => {
      useGameStore.setState({
        playCard: origPlayCardRef.current,
        passCards: origPassCardsRef.current,
        gameId: null,
        gameState: null,
      })
    }
  }, [wsPlayCard, wsPassCards])

  // ── Connection status banner ──────────────────────────────────────────────
  const statusLabel = {
    connecting: '🔄 Connecting…',
    open: null,
    closed: '🔴 Disconnected — reconnecting…',
    error: '⚠️ Connection error — reconnecting…',
  }[connectionStatus]

  const myUserId = user?.id
  const myRoomSeat = room.players.find(p => p.user_id === myUserId)?.seat
  const connectedSet = new Set(connectedSeats)
  if (connectionStatus === 'open' && myRoomSeat !== undefined) {
    connectedSet.add(myRoomSeat)
  }
  const sortedPlayers = [...room.players].sort((a, b) => a.seat - b.seat)
  const connectedCount = sortedPlayers.filter(p => connectedSet.has(p.seat)).length
  const isHost = room.host_id === myUserId
  const allConnected = sortedPlayers.length === 4 && connectedCount === 4
  const canStart = isHost && allConnected && !starting

  const handleStartGame = async () => {
    if (!canStart) return
    setStarting(true)
    setStartError(null)
    try {
      await startRoom(token, room.room_id)
    } catch (e) {
      setStartError(e.response?.data?.detail || 'Failed to start game')
      setStarting(false)
    }
  }

  return (
    <div className="mp-game-root">
      {/* Top bar */}
      <div className="mp-topbar">
        <button className="mp-leave-btn" onClick={onLeave}>← Leave Game</button>
        <div className="mp-room-info">
          Room <strong>{room.invite_code}</strong>
        </div>
        {statusLabel && <div className="mp-status-banner">{statusLabel}</div>}
      </div>

      {/* Waiting for game to start */}
      {!gameState ? (
        <div className="mp-waiting">
          <div className="spinner" />
          <p>Waiting for all players to connect…</p>
          <p className="mp-waiting-count">{connectedCount} / {sortedPlayers.length} connected</p>

          <ul className="mp-connection-list">
            {sortedPlayers.map(player => {
              const isMe = player.user_id === myUserId
              const isConnected = connectedSet.has(player.seat)
              return (
                <li
                  key={player.seat}
                  className={`mp-connection-item ${isConnected ? 'mp-connection-item--connected' : 'mp-connection-item--waiting'}`}
                >
                  <span className="mp-connection-avatar">
                    {player.picture ? (
                      <img src={player.picture} alt="" referrerPolicy="no-referrer" />
                    ) : (
                      <span className="mp-connection-avatar-fallback">👤</span>
                    )}
                  </span>
                  <span className="mp-connection-name">
                    {player.name}{isMe ? ' (You)' : ''}
                  </span>
                  <span className="mp-connection-status">
                    {isConnected ? '✓ Connected' : '⏳ Waiting'}
                  </span>
                </li>
              )
            })}
          </ul>

          {isHost ? (
            <button
              className="mp-start-btn"
              onClick={handleStartGame}
              disabled={!canStart}
            >
              {starting ? 'Starting…' : 'Start Game'}
            </button>
          ) : (
            <p className="mp-waiting-host">Waiting for host to start the game…</p>
          )}

          {isHost && !allConnected && (
            <p className="mp-start-hint">All 4 players must be connected before you can start.</p>
          )}

          {startError && <p className="mp-start-error">{startError}</p>}

          <p className="mp-room-code">Room code: <strong>{room.invite_code}</strong></p>
        </div>
      ) : (
        <div className="mp-game-area">
          <GameBoard />
          <Controls onShowScoreboard={() => setShowScoreboard(true)} />
        </div>
      )}

      {showScoreboard && <Scoreboard onClose={() => setShowScoreboard(false)} />}

      {wsError && (
        <div className="mp-error-toast">{wsError}</div>
      )}
    </div>
  )
}
