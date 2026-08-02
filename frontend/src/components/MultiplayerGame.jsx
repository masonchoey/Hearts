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
    nextRound: wsNextRound,
    subInAI: wsSubInAI,
    endMatch: wsEndMatch,
  } = useMultiplayerGame(room.room_id, token)
  const [showScoreboard, setShowScoreboard] = useState(false)
  const [starting, setStarting] = useState(false)
  const [startError, setStartError] = useState(null)
  const [, setTick] = useState(0) // 1s ticker to animate the sub-AI countdown
  const pauseBaseRef = useRef(null)

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

    // Re-derive rotatedPlayers name to flag AI seats for the board.
    const labeledPlayers = rotatedPlayers.map((p, i) => ({
      ...p,
      name: gameState.players?.[(i + mySeat) % 4]?.is_ai ? `${p.name} (AI)` : p.name,
    }))

    const rotSeat = s => (s - mySeat + 4) % 4
    const cp = gameState.current_player ?? -1
    const rotatedCurrentPlayer = cp >= 0 ? rotSeat(cp) : -1 // -1 during round-over/terminal
    const rotatedPassesSubmitted = (gameState.passes_submitted ?? []).map(rotSeat)

    // Trick cards carry absolute seats; rotate them to this viewer's layout and
    // hand them to TableCenter via `animatedTrick` (its render source).
    const rotatedTrick = (gameState.current_trick ?? []).map(
      ([s, card]) => [rotSeat(s), card],
    )

    useGameStore.setState({
      gameId: `mp_${room.room_id}`,
      gameState: {
        ...gameState,
        players: labeledPlayers,
        current_player: rotatedCurrentPlayer,
        passes_submitted: rotatedPassesSubmitted,
        my_pass_submitted: gameState.my_pass_submitted ?? false,
        // MultiplayerGame renders its own round/match modals; suppress the
        // single-player GameBoard modal and its initial-trick animation.
        game_over: false,
        winner: null,
        current_trick: [],
      },
      animatedTrick: rotatedTrick,
      isLoading: false,
      error: null,
    })
  }, [gameState, mySeat, room.room_id])

  // ── Sub-AI grace countdown: tick every second while paused ─────────────────
  useEffect(() => {
    if (!gameState?.paused) {
      pauseBaseRef.current = null
      return
    }
    const seat = gameState.current_player
    if (!pauseBaseRef.current || pauseBaseRef.current.seat !== seat) {
      pauseBaseRef.current = {
        seat,
        baseElapsed: gameState.disconnect_elapsed?.[seat] ?? 0,
        baseAt: Date.now(),
      }
    }
    const id = setInterval(() => setTick(t => t + 1), 1000)
    return () => clearInterval(id)
  }, [gameState?.paused, gameState?.current_player, gameState?.disconnect_elapsed])

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

  // ── Derived round / match / pause state (absolute seats from the raw state) ──
  const nameOf = seat => gameState?.players?.[seat]?.name ?? `Player ${seat}`
  const roundOver = !!gameState?.round_over
  const matchOver = !!gameState?.game_over
  const paused = !!gameState?.paused && !roundOver && !matchOver
  const targetLabel = gameState?.target_score == null ? '∞' : gameState.target_score

  // Sub-AI countdown for the stalled seat
  const pausedSeat = paused ? gameState.current_player : null
  const grace = gameState?.ai_sub_grace_seconds ?? 60
  let subRemaining = null
  if (pausedSeat !== null && pauseBaseRef.current?.seat === pausedSeat) {
    const { baseElapsed, baseAt } = pauseBaseRef.current
    const elapsed = baseElapsed + (Date.now() - baseAt) / 1000
    subRemaining = Math.max(0, Math.ceil(grace - elapsed))
  }
  const canSubAI = pausedSeat !== null && subRemaining === 0

  const seatRows = [0, 1, 2, 3] // absolute seats for the summary tables

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

      {/* ── Pause overlay: waiting for a disconnected player ─────────────── */}
      {gameState && paused && (
        <div className="mp-overlay">
          <div className="mp-overlay-card">
            <div className="spinner" />
            <h3>Waiting for {nameOf(pausedSeat)} to reconnect…</h3>
            <p className="mp-overlay-sub">The game is paused until they return or an AI takes over.</p>
            <div className="mp-overlay-actions">
              <button
                className="mp-overlay-btn mp-overlay-btn--primary"
                onClick={() => wsSubInAI(pausedSeat)}
                disabled={!canSubAI}
              >
                {canSubAI
                  ? `Play ${nameOf(pausedSeat)}'s seat as AI`
                  : `Sub in AI available in ${subRemaining}s`}
              </button>
              <button className="mp-overlay-btn mp-overlay-btn--danger" onClick={wsEndMatch}>
                End match for everyone
              </button>
            </div>
          </div>
        </div>
      )}

      {/* ── Round-over modal ─────────────────────────────────────────────── */}
      {gameState && roundOver && (
        <div className="mp-overlay">
          <div className="mp-overlay-card">
            <h2>Round {gameState.round_number} complete</h2>
            <p className="mp-overlay-sub">Playing to {targetLabel} — lowest total wins</p>
            <table className="mp-score-table">
              <thead>
                <tr><th>Player</th><th>This round</th><th>Total</th></tr>
              </thead>
              <tbody>
                {seatRows.map(seat => (
                  <tr key={seat} className={seat === gameState.round_winner ? 'mp-row-best' : ''}>
                    <td>{nameOf(seat)}{gameState.ai_seats?.includes(seat) ? ' (AI)' : ''}</td>
                    <td>{gameState.round_scores?.[seat] ?? 0}</td>
                    <td>{gameState.scores?.[seat] ?? 0}</td>
                  </tr>
                ))}
              </tbody>
            </table>
            <div className="mp-overlay-actions">
              <button className="mp-overlay-btn mp-overlay-btn--primary" onClick={wsNextRound}>
                Next Round →
              </button>
              <button className="mp-overlay-btn mp-overlay-btn--danger" onClick={wsEndMatch}>
                End Match
              </button>
            </div>
          </div>
        </div>
      )}

      {/* ── Match-over modal ─────────────────────────────────────────────── */}
      {gameState && matchOver && (
        <div className="mp-overlay">
          <div className="mp-overlay-card">
            <h2>🏆 Game Over</h2>
            <p className="mp-overlay-sub">
              {nameOf(gameState.winner)} wins with the lowest total!
            </p>
            <table className="mp-score-table">
              <thead>
                <tr><th>Player</th><th>Final total</th></tr>
              </thead>
              <tbody>
                {[...seatRows]
                  .sort((a, b) => (gameState.scores?.[a] ?? 0) - (gameState.scores?.[b] ?? 0))
                  .map(seat => (
                    <tr key={seat} className={seat === gameState.winner ? 'mp-row-best' : ''}>
                      <td>{seat === gameState.winner ? '👑 ' : ''}{nameOf(seat)}</td>
                      <td>{gameState.scores?.[seat] ?? 0}</td>
                    </tr>
                  ))}
              </tbody>
            </table>
            <div className="mp-overlay-actions">
              <button className="mp-overlay-btn mp-overlay-btn--primary" onClick={onLeave}>
                Leave Game
              </button>
            </div>
          </div>
        </div>
      )}

      {wsError && (
        <div className="mp-error-toast">{wsError}</div>
      )}
    </div>
  )
}
