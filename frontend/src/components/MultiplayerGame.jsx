/**
 * Multiplayer game wrapper.
 * Bridges the WebSocket state from useMultiplayerGame into the existing
 * Zustand store so all existing GameBoard / PlayerHand components work unchanged.
 * Players array is rotated so the current user always appears at seat 0 (bottom).
 * Supports 3/4/5 players and multi-round matches.
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
    endMatch: wsEndMatch,
  } = useMultiplayerGame(room.room_id, token)
  const animationDelay = useGameStore(s => s.animationDelay)
  const [showScoreboard, setShowScoreboard] = useState(false)
  const [starting, setStarting] = useState(false)
  const [startError, setStartError] = useState(null)
  const trickClearTimer = useRef(null)     // pending "clear completed trick" timeout
  const clearedTrickSig = useRef(null)     // signature of the trick we've already cleared

  // Track original playCard so we can restore on unmount
  const origPlayCardRef = useRef(null)
  const origPassCardsRef = useRef(null)

  useEffect(() => {
    if (gameState) setStarting(false)
  }, [gameState])

  // ── Sync WebSocket state → Zustand store ──────────────────────────────────
  useEffect(() => {
    if (!gameState || mySeat === null) return

    const n = gameState.player_count || gameState.players?.length || 4
    const rotSeat = s => (((s - mySeat) % n) + n) % n

    // Rotate so mySeat appears at rotated index 0 (bottom / "human" slot)
    const rotatedHandCounts = rotateArray(
      Array.from({ length: n }, (_, seat) => gameState.hand_counts?.[seat] ?? 0),
      mySeat,
    )
    const disconnectedSeats = gameState.disconnected_seats ?? []
    const rotatedPlayers = rotateArray(gameState.players, mySeat).map((p, i) => {
      const absSeat = (i + mySeat) % n
      return {
        ...p,
        id: i,
        is_ai: false,
        disconnected: disconnectedSeats.includes(absSeat), // flag dropped players
        // Backend only sends full hand for the viewer; use hand_counts for card backs
        hand: p.hand?.length ? p.hand : Array(rotatedHandCounts[i]).fill(null),
      }
    })

    const cp = gameState.current_player ?? -1
    const rotatedCurrentPlayer = cp >= 0 ? rotSeat(cp) : -1 // -1 during passing/round-over/terminal
    const rotatedPassesSubmitted = (gameState.passes_submitted ?? []).map(rotSeat)

    useGameStore.setState({
      gameId: `mp_${room.room_id}`,
      gameState: {
        ...gameState,
        players: rotatedPlayers,
        current_player: rotatedCurrentPlayer,
        passes_submitted: rotatedPassesSubmitted,
        my_pass_submitted: gameState.my_pass_submitted ?? false,
        // MultiplayerGame renders its own round/match modals; suppress the
        // single-player GameBoard modal and its initial-trick animation.
        game_over: false,
        winner: null,
        current_trick: [],
      },
      // animatedTrick is owned by the trick-display effect below.
      isLoading: false,
      error: null,
    })
  }, [gameState, mySeat, room.room_id])

  // ── Trick display: rotate to this viewer, and auto-clear a COMPLETED trick ──
  // The server keeps the N-card trick until the next card is played; we instead
  // show it briefly then clear it, so the table doesn't stay full waiting on the
  // next play. Linger scales with the animation-delay slider.
  useEffect(() => {
    if (!gameState || mySeat === null) return
    const n = gameState.player_count || gameState.players?.length || 4
    const rotSeat = s => (((s - mySeat) % n) + n) % n
    const rawTrick = gameState.current_trick ?? []
    const rotated = rawTrick.map(([s, card]) => [rotSeat(s), card])
    const sig = JSON.stringify(rawTrick)

    clearTimeout(trickClearTimer.current)

    if (rawTrick.length < n) {
      // Growing (or empty) trick — show it live and reset the cleared marker.
      clearedTrickSig.current = null
      useGameStore.setState({ animatedTrick: rotated })
      return
    }

    // Completed trick (all N seats played).
    if (clearedTrickSig.current === sig) {
      // Already cleared this one (re-broadcast from an unrelated update) — keep empty.
      useGameStore.setState({ animatedTrick: [] })
      return
    }
    // New completed trick: show it, then clear after a proportional linger.
    useGameStore.setState({ animatedTrick: rotated })
    const linger = Math.max(1000, Math.min(animationDelay * 4, 3500))
    trickClearTimer.current = setTimeout(() => {
      clearedTrickSig.current = sig
      useGameStore.setState({ animatedTrick: [] })
    }, linger)
  }, [gameState, mySeat, animationDelay])

  useEffect(() => () => clearTimeout(trickClearTimer.current), [])

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
      // Queue full 3-card pass — backend applies once every seat has submitted
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

  const requiredPlayers = room.player_count || 4
  const myUserId = user?.id
  const myRoomSeat = room.players.find(p => p.user_id === myUserId)?.seat
  const connectedSet = new Set(connectedSeats)
  if (connectionStatus === 'open' && myRoomSeat !== undefined) {
    connectedSet.add(myRoomSeat)
  }
  const sortedPlayers = [...room.players].sort((a, b) => a.seat - b.seat)
  const connectedCount = sortedPlayers.filter(p => connectedSet.has(p.seat)).length
  const isHost = room.host_id === myUserId
  const allConnected = sortedPlayers.length === requiredPlayers && connectedCount === requiredPlayers
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
  const n = gameState?.player_count || gameState?.players?.length || 4
  const nameOf = seat => gameState?.players?.[seat]?.name ?? `Player ${seat}`
  const roundOver = !!gameState?.round_over
  const matchOver = !!gameState?.game_over
  const paused = !!gameState?.paused && !roundOver && !matchOver
  const targetLabel = gameState?.target_score == null ? '∞' : gameState.target_score
  const pausedSeat = paused ? gameState.current_player : null

  const seatRows = Array.from({ length: n }, (_, i) => i) // absolute seats for the summary tables

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
            <p className="mp-start-hint">All {requiredPlayers} players must be connected before you can start.</p>
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
            <p className="mp-overlay-sub">The game is paused until they return.</p>
            <div className="mp-overlay-actions">
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
                    <td>{nameOf(seat)}</td>
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
