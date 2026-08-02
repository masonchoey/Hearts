import { useState, useEffect, useRef } from 'react'
import { useAuth } from '../auth/AuthContext'
import { createRoom, joinRoom, getRoom, listFriends } from '../api/multiplayer'
import { useMultiplayerGame } from '../hooks/useMultiplayerGame'
import FriendsPanel from './FriendsPanel'
import './MultiplayerLobby.css'

const SEATS = ['South (You)', 'West', 'North', 'East']

export default function MultiplayerLobby({ onGameStart, onBack, inviteCode: initialInviteCode, roomId: initialRoomId }) {
  const { user, token } = useAuth()
  const [view, setView] = useState(initialRoomId || initialInviteCode ? 'joining' : 'menu') // menu | create | join | waiting | joining
  const [room, setRoom] = useState(null)
  const [inviteInput, setInviteInput] = useState(initialInviteCode || '')
  const [targetScore, setTargetScore] = useState(100) // null = infinite
  const [friends, setFriends] = useState([])
  const [error, setError] = useState(null)
  const [loading, setLoading] = useState(false)
  const [copied, setCopied] = useState(false)
  const [showFriends, setShowFriends] = useState(false)
  const gameStartedRef = useRef(false)

  // Connect WebSocket while waiting so we receive game_started in real time
  const { gameState: wsGameState, connectionStatus } = useMultiplayerGame(
    view === 'waiting' && room ? room.room_id : null,
    token,
  )

  // Load friends for the sidebar
  useEffect(() => {
    if (!token) return
    listFriends(token).then(setFriends).catch(() => {})
  }, [token])

  // Direct room link (?room=ID): load the existing room and go straight to the
  // waiting screen — no join attempt, so it works for the host and joiners alike.
  useEffect(() => {
    if (!initialRoomId || !token) return
    let cancelled = false
    ;(async () => {
      try {
        const r = await getRoom(token, initialRoomId)
        if (cancelled) return
        setRoom(r)
        if (r.players.length === 4) startGame(r)
        else setView('waiting')
      } catch (e) {
        if (cancelled) return
        setError('Could not load room — it may have ended.')
        setView('menu')
      }
    })()
    return () => { cancelled = true }
  }, [initialRoomId, token])

  // Auto-join if an invite code was passed in (e.g. from URL)
  useEffect(() => {
    if (initialInviteCode && !initialRoomId) handleJoin(initialInviteCode)
  }, [])

  const startGame = (roomData) => {
    if (gameStartedRef.current) return
    gameStartedRef.current = true
    onGameStart(roomData)
  }

  // Transition when WebSocket delivers game_started (preferred path)
  useEffect(() => {
    if (view !== 'waiting' || !room || !wsGameState) return
    startGame(room)
  }, [view, room, wsGameState])

  // Poll room state as a fallback when WebSocket is unavailable
  useEffect(() => {
    if (view !== 'waiting' || !room) return
    const interval = setInterval(async () => {
      try {
        const updated = await getRoom(token, room.room_id)
        setRoom(updated)
        if (updated.players.length === 4) {
          clearInterval(interval)
          startGame(updated)
        }
      } catch {}
    }, 2000)
    return () => clearInterval(interval)
  }, [view, room, token])

  const handleCreate = async () => {
    setLoading(true)
    setError(null)
    try {
      const newRoom = await createRoom(token, targetScore)
      setRoom(newRoom)
      setView('waiting')
    } catch (e) {
      setError(e.response?.data?.detail || 'Failed to create room')
    } finally {
      setLoading(false)
    }
  }

  const handleJoin = async (code) => {
    const useCode = (code || inviteInput).trim().toUpperCase()
    if (!useCode) return
    setLoading(true)
    setError(null)
    try {
      const joinedRoom = await joinRoom(token, useCode)
      setRoom(joinedRoom)
      setView('waiting')
      if (joinedRoom.players.length === 4) startGame(joinedRoom)
    } catch (e) {
      setError(e.response?.data?.detail || 'Could not join room — check the invite code')
      setView('join')
    } finally {
      setLoading(false)
    }
  }

  const copyInviteLink = () => {
    if (!room) return
    const url = `${window.location.origin}/join/${room.invite_code}`
    navigator.clipboard.writeText(url).then(() => {
      setCopied(true)
      setTimeout(() => setCopied(false), 2500)
    })
  }

  const copyInviteCode = () => {
    if (!room) return
    navigator.clipboard.writeText(room.invite_code).then(() => {
      setCopied(true)
      setTimeout(() => setCopied(false), 2500)
    })
  }

  // ── Render waiting room ──────────────────────────────────────────────────

  if (view === 'waiting' && room) {
    const myUserId = user?.id
    const mySeat = room.players.find(p => p.user_id === myUserId)?.seat ?? 0
    const playerBySeat = Object.fromEntries(room.players.map(p => [p.seat, p]))

    return (
      <div className="lobby-screen">
        <div className="lobby-card lobby-card--wide">
          <button className="lobby-back" onClick={onBack}>← Back</button>
          <h2 className="lobby-heading">Waiting for Players</h2>
          <p className="lobby-subtext">{room.players.length} / 4 players joined</p>

          {/* Seat grid */}
          <div className="lobby-seats">
            {[0, 1, 2, 3].map(seat => {
              const p = playerBySeat[seat]
              const isMe = p?.user_id === myUserId
              return (
                <div key={seat} className={`lobby-seat ${p ? 'lobby-seat--filled' : 'lobby-seat--empty'} ${isMe ? 'lobby-seat--me' : ''}`}>
                  {p ? (
                    <>
                      {p.picture && <img src={p.picture} alt={p.name} className="lobby-seat-avatar" referrerPolicy="no-referrer" />}
                      <span className="lobby-seat-name">{p.name}{isMe ? ' (You)' : ''}</span>
                    </>
                  ) : (
                    <>
                      <span className="lobby-seat-icon">👤</span>
                      <span className="lobby-seat-name">Waiting...</span>
                    </>
                  )}
                </div>
              )
            })}
          </div>

          {/* Invite section */}
          <div className="lobby-invite-section">
            <p className="lobby-invite-label">Invite your friends</p>
            <div className="lobby-invite-code-box">
              <span className="lobby-invite-code">{room.invite_code}</span>
              <button className="lobby-btn lobby-btn-secondary" onClick={copyInviteCode}>
                {copied ? '✓ Copied' : 'Copy Code'}
              </button>
            </div>
            <button className="lobby-btn lobby-btn-link-btn" onClick={copyInviteLink}>
              📋 Copy Invite Link
            </button>
          </div>

          {/* Friends quick-invite */}
          {friends.length > 0 && (
            <div className="lobby-friends-quick">
              <p className="lobby-invite-label">Invite a friend</p>
              <div className="lobby-friends-list">
                {friends.map(f => (
                  <div key={f.id} className="lobby-friend-row">
                    {f.picture && <img src={f.picture} alt={f.name} className="lobby-friend-avatar" referrerPolicy="no-referrer" />}
                    <span className="lobby-friend-name">{f.name}</span>
                    <button
                      className="lobby-btn lobby-btn-xs"
                      onClick={() => {
                        navigator.clipboard.writeText(`${window.location.origin}/join/${room.invite_code}`)
                        setCopied(true)
                        setTimeout(() => setCopied(false), 2500)
                      }}
                    >
                      Copy Link
                    </button>
                  </div>
                ))}
              </div>
            </div>
          )}

          <div className="lobby-spinner">
            <div className="spinner" />
            {connectionStatus === 'error' || connectionStatus === 'closed' ? (
              <p>Reconnecting to game server…</p>
            ) : (
              <p>Waiting for {4 - room.players.length} more player{4 - room.players.length !== 1 ? 's' : ''}…</p>
            )}
          </div>
        </div>
      </div>
    )
  }

  // ── Main menu / join / create ────────────────────────────────────────────

  return (
    <div className="lobby-screen">
      <div className="lobby-card">
        <button className="lobby-back" onClick={onBack}>← Back</button>
        <h2 className="lobby-heading">Multiplayer</h2>
        <p className="lobby-subtext">Play Hearts with up to 3 friends in real time</p>

        {error && <div className="lobby-error">{error}</div>}

        {view === 'menu' && (
          <div className="lobby-menu">
            <button
              className="lobby-mode-btn"
              onClick={() => setView('create')}
            >
              <span className="lobby-mode-icon">🎮</span>
              <div>
                <strong>Create a Room</strong>
                <p>Host a game and share an invite link</p>
              </div>
            </button>

            <button
              className="lobby-mode-btn"
              onClick={() => setView('join')}
            >
              <span className="lobby-mode-icon">🔗</span>
              <div>
                <strong>Join a Room</strong>
                <p>Enter an invite code to join a friend's game</p>
              </div>
            </button>

            <button
              className="lobby-mode-btn lobby-mode-btn--secondary"
              onClick={() => setShowFriends(true)}
            >
              <span className="lobby-mode-icon">👥</span>
              <div>
                <strong>Friends</strong>
                <p>Manage your friends list and requests</p>
              </div>
            </button>
          </div>
        )}

        {view === 'create' && (
          <div className="lobby-action">
            <p>A new room will be created and you'll receive an invite link to share with friends.</p>

            <label className="lobby-label">Play until a player reaches</label>
            <div className="lobby-target-options">
              {[50, 100, 150].map(v => (
                <button
                  key={v}
                  type="button"
                  className={`lobby-target-btn ${targetScore === v ? 'lobby-target-btn--active' : ''}`}
                  onClick={() => setTargetScore(v)}
                >
                  {v}
                </button>
              ))}
              <button
                type="button"
                className={`lobby-target-btn ${targetScore === null ? 'lobby-target-btn--active' : ''}`}
                onClick={() => setTargetScore(null)}
                title="Play forever until the host ends the match"
              >
                ∞ Infinite
              </button>
            </div>
            <p className="lobby-target-hint">
              {targetScore === null
                ? 'Rounds continue until the host ends the match — lowest total wins.'
                : `First to ${targetScore} ends the match — lowest total wins.`}
            </p>

            <button className="lobby-btn lobby-btn-primary" onClick={handleCreate} disabled={loading}>
              {loading ? 'Creating…' : 'Create Room'}
            </button>
            <button className="lobby-btn-link" onClick={() => setView('menu')}>Cancel</button>
          </div>
        )}

        {view === 'join' && (
          <div className="lobby-action">
            <label className="lobby-label">Invite Code</label>
            <input
              className="lobby-input"
              value={inviteInput}
              onChange={e => setInviteInput(e.target.value.toUpperCase())}
              placeholder="e.g. AB1C2D"
              maxLength={8}
              autoFocus
              onKeyDown={e => e.key === 'Enter' && handleJoin()}
            />
            <button className="lobby-btn lobby-btn-primary" onClick={() => handleJoin()} disabled={loading || !inviteInput}>
              {loading ? 'Joining…' : 'Join Room'}
            </button>
            <button className="lobby-btn-link" onClick={() => setView('menu')}>Cancel</button>
          </div>
        )}

        {view === 'joining' && (
          <div className="lobby-action">
            <div className="lobby-spinner">
              <div className="spinner" />
              <p>Joining room {initialInviteCode}…</p>
            </div>
          </div>
        )}
      </div>

      {showFriends && (
        <FriendsPanel onClose={() => setShowFriends(false)} />
      )}
    </div>
  )
}
