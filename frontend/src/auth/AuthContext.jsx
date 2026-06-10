import { createContext, useContext, useState, useEffect, useCallback, useRef } from 'react'
import { getMe, syncSession } from '../api/auth'
import {
  isNeonAuthConfigured,
  getNeonSession,
  getNeonToken,
  neonSignIn,
  neonSignUp,
  neonSignInWithGoogle,
  neonSignOut,
} from '../api/neonAuth'

const AuthContext = createContext(null)

const TOKEN_KEY = 'hearts_auth_token'

// Neon Auth JWTs expire after ~15 min — refresh before that.
const TOKEN_REFRESH_MS = 10 * 60 * 1000

export function AuthProvider({ children }) {
  const [user, setUser] = useState(null)
  const [token, setToken] = useState(() => localStorage.getItem(TOKEN_KEY))
  const [loading, setLoading] = useState(true)

  const refreshTimer = useRef(null)

  const clearSession = useCallback(() => {
    localStorage.removeItem(TOKEN_KEY)
    setToken(null)
    setUser(null)
    if (refreshTimer.current) {
      clearInterval(refreshTimer.current)
      refreshTimer.current = null
    }
  }, [])

  const refreshSession = useCallback(async () => {
    const neonToken = await getNeonToken()
    const { user: appUser } = await syncSession(neonToken)
    localStorage.setItem(TOKEN_KEY, neonToken)
    setToken(neonToken)
    setUser(appUser)
    return appUser
  }, [])

  const startRefreshLoop = useCallback(() => {
    if (refreshTimer.current) clearInterval(refreshTimer.current)
    refreshTimer.current = setInterval(() => {
      refreshSession().catch(() => clearSession())
    }, TOKEN_REFRESH_MS)
  }, [refreshSession, clearSession])

  useEffect(() => {
    let cancelled = false

    async function restore() {
      if (!isNeonAuthConfigured) {
        if (!cancelled) setLoading(false)
        return
      }

      // Prefer Neon cookie session (survives JWT expiry).
      try {
        const session = await getNeonSession()
        if (session) {
          await refreshSession()
          if (!cancelled) startRefreshLoop()
          if (!cancelled) setLoading(false)
          return
        }
      } catch {
        /* no cookie session */
      }

      // Fall back to stored JWT if still valid.
      const stored = localStorage.getItem(TOKEN_KEY)
      if (stored) {
        try {
          const u = await getMe(stored)
          if (!cancelled) {
            setToken(stored)
            setUser(u)
            startRefreshLoop()
          }
          if (!cancelled) setLoading(false)
          return
        } catch {
          clearSession()
        }
      }

      if (!cancelled) setLoading(false)
    }

    restore()
    return () => {
      cancelled = true
    }
  }, [refreshSession, startRefreshLoop, clearSession])

  useEffect(() => {
    return () => {
      if (refreshTimer.current) clearInterval(refreshTimer.current)
    }
  }, [])

  const loginWithEmail = useCallback(async (email, password) => {
    await neonSignIn(email, password)
    const appUser = await refreshSession()
    startRefreshLoop()
    return appUser
  }, [refreshSession, startRefreshLoop])

  const signupWithEmail = useCallback(async (email, password, name) => {
    await neonSignUp(email, password, name)
    const appUser = await refreshSession()
    startRefreshLoop()
    return appUser
  }, [refreshSession, startRefreshLoop])

  const loginWithGoogle = useCallback(async () => {
    await neonSignInWithGoogle(window.location.href)
  }, [])

  const logout = useCallback(async () => {
    await neonSignOut()
    clearSession()
  }, [clearSession])

  return (
    <AuthContext.Provider
      value={{
        user,
        token,
        loading,
        neonAuthEnabled: isNeonAuthConfigured,
        loginWithEmail,
        signupWithEmail,
        loginWithGoogle,
        logout,
      }}
    >
      {children}
    </AuthContext.Provider>
  )
}

export const useAuth = () => {
  const ctx = useContext(AuthContext)
  if (!ctx) throw new Error('useAuth must be used inside AuthProvider')
  return ctx
}
