import axios from 'axios'

const API_BASE = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000'
const WS_BASE = import.meta.env.VITE_WS_BASE_URL || 'ws://localhost:8000'

const makeApi = (token) =>
  axios.create({
    baseURL: API_BASE,
    timeout: 15000,
    headers: { Authorization: `Bearer ${token}` },
  })

// ── Rooms ─────────────────────────────────────────────────────────────────

export const createRoom = async (token, targetScore = 100) => {
  // targetScore null => infinite play
  const { data } = await makeApi(token).post('/mp/rooms', { target_score: targetScore })
  return data // { room_id, invite_code, host_id, status, target_score, players }
}

export const joinRoom = async (token, inviteCode) => {
  const { data } = await makeApi(token).post(`/mp/rooms/join/${inviteCode}`)
  return data
}

export const getRoom = async (token, roomId) => {
  const { data } = await makeApi(token).get(`/mp/rooms/${roomId}`)
  return data
}

export const startRoom = async (token, roomId) => {
  const { data } = await makeApi(token).post(`/mp/rooms/${roomId}/start`)
  return data
}

// ── Friends ───────────────────────────────────────────────────────────────

export const searchUsers = async (token, q) => {
  const { data } = await makeApi(token).get('/friends/search', { params: { q } })
  return data
}

export const sendFriendRequest = async (token, targetUserId) => {
  const { data } = await makeApi(token).post(`/friends/request/${targetUserId}`)
  return data
}

export const acceptFriendRequest = async (token, requestId) => {
  const { data } = await makeApi(token).post(`/friends/accept/${requestId}`)
  return data
}

export const rejectFriendRequest = async (token, requestId) => {
  const { data } = await makeApi(token).post(`/friends/reject/${requestId}`)
  return data
}

export const listFriends = async (token) => {
  const { data } = await makeApi(token).get('/friends/')
  return data
}

export const listFriendRequests = async (token) => {
  const { data } = await makeApi(token).get('/friends/requests')
  return data
}

// ── WebSocket factory ─────────────────────────────────────────────────────

export const createGameWebSocket = (roomId, token) => {
  const url = `${WS_BASE}/mp/ws/${roomId}?token=${encodeURIComponent(token)}`
  return new WebSocket(url)
}
