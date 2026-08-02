import { useState } from 'react'
import { useAuth } from '../auth/AuthContext'
import './LandingPage.css'

export default function LandingPage({ onPlayAI, onPlayMultiplayer }) {
  const {
    user,
    logout,
    loading,
    neonAuthEnabled,
    loginWithEmail,
    signupWithEmail,
    loginWithGoogle,
  } = useAuth()
  const [loginError, setLoginError] = useState(null)
  const [authMode, setAuthMode] = useState('signin')
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [name, setName] = useState('')
  const [busy, setBusy] = useState(false)

  const handleSubmit = async (e) => {
    e.preventDefault()
    setLoginError(null)
    setBusy(true)
    try {
      if (authMode === 'signup') {
        await signupWithEmail(email, password, name)
      } else {
        await loginWithEmail(email, password)
      }
    } catch (err) {
      const msg =
        err?.response?.data?.detail ||
        err?.message ||
        'Sign-in failed. Please try again.'
      setLoginError(typeof msg === 'string' ? msg : JSON.stringify(msg))
    } finally {
      setBusy(false)
    }
  }

  const handleGoogle = async () => {
    setLoginError(null)
    setBusy(true)
    try {
      await loginWithGoogle()
    } catch (err) {
      setLoginError(err?.message || 'Google sign-in failed.')
      setBusy(false)
    }
  }

  if (loading) {
    return (
      <div className="landing-screen">
        <div className="landing-card">
          <div className="landing-logo">♥</div>
          <p className="landing-loading">Loading...</p>
        </div>
      </div>
    )
  }

  return (
    <div className="landing-screen">
      <div className="landing-card">
        <div className="landing-logo">♥</div>
        <h1 className="landing-title">Hearts</h1>
        <p className="landing-subtitle">The classic trick-taking card game</p>

        {!user ? (
          <div className="landing-auth">
            <div className="landing-play-guest">
              <h3>Play as Guest</h3>
              <p>Challenge the AI without an account</p>
              <button className="landing-btn landing-btn-primary" onClick={onPlayAI}>
                Play vs AI
              </button>
            </div>

            <div className="landing-divider">
              <span>or sign in for multiplayer</span>
            </div>

            {neonAuthEnabled ? (
              <>
                <button
                  type="button"
                  className="landing-btn landing-neon-google"
                  onClick={handleGoogle}
                  disabled={busy}
                >
                  Continue with Google
                </button>

                <form className="landing-neon-form" onSubmit={handleSubmit}>
                  {authMode === 'signup' && (
                    <input
                      className="landing-input"
                      type="text"
                      placeholder="Display name"
                      value={name}
                      onChange={(e) => setName(e.target.value)}
                    />
                  )}
                  <input
                    className="landing-input"
                    type="email"
                    placeholder="Email"
                    value={email}
                    onChange={(e) => setEmail(e.target.value)}
                    required
                  />
                  <input
                    className="landing-input"
                    type="password"
                    placeholder="Password"
                    value={password}
                    onChange={(e) => setPassword(e.target.value)}
                    required
                  />
                  <button
                    type="submit"
                    className="landing-btn landing-btn-primary"
                    disabled={busy}
                  >
                    {busy
                      ? 'Please wait…'
                      : authMode === 'signup'
                        ? 'Create account'
                        : 'Sign in with email'}
                  </button>
                </form>

                <button
                  type="button"
                  className="landing-btn-link"
                  onClick={() => {
                    setLoginError(null)
                    setAuthMode((m) => (m === 'signup' ? 'signin' : 'signup'))
                  }}
                >
                  {authMode === 'signup'
                    ? 'Already have an account? Sign in'
                    : "Don't have an account? Sign up"}
                </button>
              </>
            ) : (
              <p className="landing-error">
                Multiplayer sign-in is not configured. Set VITE_NEON_AUTH_URL in your frontend .env.
              </p>
            )}

            {loginError && <p className="landing-error">{loginError}</p>}
          </div>
        ) : (
          <div className="landing-signed-in">
            <div className="landing-user-row">
              {user.picture && (
                <img
                  className="landing-avatar"
                  src={user.picture}
                  alt={user.name}
                  referrerPolicy="no-referrer"
                />
              )}
              <div>
                <p className="landing-user-name">{user.name}</p>
                <button className="landing-btn-link" onClick={logout}>
                  Sign out
                </button>
              </div>
            </div>

            <div className="landing-modes">
              <button className="landing-mode-card" onClick={onPlayAI}>
                <span className="mode-icon">🤖</span>
                <h3>Play vs AI</h3>
                <p>Challenge three AI opponents powered by DMCTS + AlphaZero</p>
              </button>

              <button
                className="landing-mode-card landing-mode-card--multi"
                onClick={onPlayMultiplayer}
              >
                <span className="mode-icon">👥</span>
                <h3>Multiplayer</h3>
                <p>Play with friends in real-time — invite up to 3 others via link</p>
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
