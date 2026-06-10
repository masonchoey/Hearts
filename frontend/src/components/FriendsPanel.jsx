import { useState, useEffect } from 'react'
import { useAuth } from '../auth/AuthContext'
import {
  listFriends,
  listFriendRequests,
  searchUsers,
  sendFriendRequest,
  acceptFriendRequest,
  rejectFriendRequest,
} from '../api/multiplayer'
import './FriendsPanel.css'

export default function FriendsPanel({ onClose }) {
  const { token } = useAuth()
  const [tab, setTab] = useState('friends') // friends | requests | search
  const [friends, setFriends] = useState([])
  const [requests, setRequests] = useState([])
  const [searchQuery, setSearchQuery] = useState('')
  const [searchResults, setSearchResults] = useState([])
  const [pendingSent, setPendingSent] = useState(new Set())
  const [loading, setLoading] = useState(false)

  const reload = async () => {
    try {
      const [f, r] = await Promise.all([listFriends(token), listFriendRequests(token)])
      setFriends(f)
      setRequests(r)
    } catch {}
  }

  useEffect(() => {
    reload()
  }, [token])

  const doSearch = async (q) => {
    setSearchQuery(q)
    if (q.length < 2) { setSearchResults([]); return }
    setLoading(true)
    try {
      const results = await searchUsers(token, q)
      setSearchResults(results)
    } catch {
      setSearchResults([])
    } finally {
      setLoading(false)
    }
  }

  const sendRequest = async (userId) => {
    try {
      await sendFriendRequest(token, userId)
      setPendingSent(s => new Set([...s, userId]))
    } catch {}
  }

  const acceptReq = async (reqId) => {
    try {
      await acceptFriendRequest(token, reqId)
      await reload()
    } catch {}
  }

  const rejectReq = async (reqId) => {
    try {
      await rejectFriendRequest(token, reqId)
      setRequests(r => r.filter(x => x.id !== reqId))
    } catch {}
  }

  return (
    <div className="friends-overlay" onClick={onClose}>
      <div className="friends-panel" onClick={e => e.stopPropagation()}>
        <div className="friends-header">
          <h3>Friends</h3>
          <button className="friends-close" onClick={onClose}>✕</button>
        </div>

        {/* Tabs */}
        <div className="friends-tabs">
          <button className={`ftab ${tab === 'friends' ? 'ftab--active' : ''}`} onClick={() => setTab('friends')}>
            Friends {friends.length > 0 && <span className="ftab-badge">{friends.length}</span>}
          </button>
          <button className={`ftab ${tab === 'requests' ? 'ftab--active' : ''}`} onClick={() => setTab('requests')}>
            Requests {requests.length > 0 && <span className="ftab-badge ftab-badge--red">{requests.length}</span>}
          </button>
          <button className={`ftab ${tab === 'search' ? 'ftab--active' : ''}`} onClick={() => setTab('search')}>
            Add
          </button>
        </div>

        <div className="friends-body">
          {/* ── Friends list ──────────────────────────────────────── */}
          {tab === 'friends' && (
            <div className="friends-list">
              {friends.length === 0 ? (
                <p className="friends-empty">No friends yet. Search to add some!</p>
              ) : (
                friends.map(f => (
                  <div key={f.id} className="friend-row">
                    {f.picture
                      ? <img src={f.picture} alt={f.name} className="friend-avatar" referrerPolicy="no-referrer" />
                      : <div className="friend-avatar-placeholder">{f.name[0]}</div>
                    }
                    <div className="friend-info">
                      <span className="friend-name">{f.name}</span>
                      <span className="friend-email">{f.email}</span>
                    </div>
                  </div>
                ))
              )}
            </div>
          )}

          {/* ── Incoming requests ─────────────────────────────────── */}
          {tab === 'requests' && (
            <div className="friends-list">
              {requests.length === 0 ? (
                <p className="friends-empty">No pending requests.</p>
              ) : (
                requests.map(req => (
                  <div key={req.id} className="friend-row">
                    {req.from_user.picture
                      ? <img src={req.from_user.picture} alt={req.from_user.name} className="friend-avatar" referrerPolicy="no-referrer" />
                      : <div className="friend-avatar-placeholder">{req.from_user.name[0]}</div>
                    }
                    <div className="friend-info">
                      <span className="friend-name">{req.from_user.name}</span>
                      <span className="friend-email">{req.from_user.email}</span>
                    </div>
                    <div className="friend-actions">
                      <button className="friend-btn friend-btn--accept" onClick={() => acceptReq(req.id)}>✓</button>
                      <button className="friend-btn friend-btn--reject" onClick={() => rejectReq(req.id)}>✕</button>
                    </div>
                  </div>
                ))
              )}
            </div>
          )}

          {/* ── Search ───────────────────────────────────────────── */}
          {tab === 'search' && (
            <div className="friends-search">
              <input
                className="friends-search-input"
                placeholder="Search by name or email…"
                value={searchQuery}
                onChange={e => doSearch(e.target.value)}
                autoFocus
              />
              {loading && <p className="friends-empty">Searching…</p>}
              <div className="friends-list">
                {searchResults.map(u => {
                  const sent = pendingSent.has(u.id)
                  return (
                    <div key={u.id} className="friend-row">
                      {u.picture
                        ? <img src={u.picture} alt={u.name} className="friend-avatar" referrerPolicy="no-referrer" />
                        : <div className="friend-avatar-placeholder">{u.name[0]}</div>
                      }
                      <div className="friend-info">
                        <span className="friend-name">{u.name}</span>
                        <span className="friend-email">{u.email}</span>
                      </div>
                      <button
                        className={`friend-btn ${sent ? 'friend-btn--sent' : 'friend-btn--add'}`}
                        onClick={() => sendRequest(u.id)}
                        disabled={sent}
                      >
                        {sent ? 'Sent ✓' : '+ Add'}
                      </button>
                    </div>
                  )
                })}
                {!loading && searchQuery.length >= 2 && searchResults.length === 0 && (
                  <p className="friends-empty">No users found.</p>
                )}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
