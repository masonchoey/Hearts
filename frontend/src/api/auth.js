import axios from 'axios'

const API_BASE = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000'

const authApi = axios.create({ baseURL: API_BASE, timeout: 15000 })

export const getMe = async (token) => {
  const { data } = await authApi.get('/auth/me', {
    headers: { Authorization: `Bearer ${token}` },
  })
  return data
}

/** Sync Neon Auth JWT into the app users table. */
export const syncSession = async (neonToken) => {
  const { data } = await authApi.post('/auth/session', null, {
    headers: { Authorization: `Bearer ${neonToken}` },
  })
  return data // { token, user }
}
