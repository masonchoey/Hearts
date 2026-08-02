/**
 * Neon Auth (Better Auth) client.
 *
 * Uses createInternalNeonAuth so we can call getJWTToken() — the supported way to
 * obtain a JWT for our backend. createAuthClient() only returns the Better Auth
 * adapter and does not expose getJWTToken.
 */
import { createInternalNeonAuth } from '@neondatabase/neon-js/auth'

const NEON_AUTH_URL = import.meta.env.VITE_NEON_AUTH_URL || ''

export const isNeonAuthConfigured = Boolean(NEON_AUTH_URL)

const neonAuth = NEON_AUTH_URL ? createInternalNeonAuth(NEON_AUTH_URL) : null

/** Better Auth adapter (signIn, signUp, getSession, signOut, …). */
export const authClient = neonAuth?.adapter ?? null

export async function getNeonSession() {
  if (!authClient) return null
  const result = await authClient.getSession()
  if (result?.data?.session && result?.data?.user) {
    return { session: result.data.session, user: result.data.user }
  }
  return null
}

/** JWT for backend Bearer auth (~15 min lifetime). */
export async function getNeonToken() {
  if (!neonAuth) throw new Error('Neon Auth is not configured')
  const token = await neonAuth.getJWTToken()
  if (!token) throw new Error('Not signed in — no JWT available')
  return token
}

export async function neonSignIn(email, password) {
  if (!authClient) throw new Error('Neon Auth is not configured')
  const { error } = await authClient.signIn.email({ email, password })
  if (error) throw new Error(error.message || 'Sign-in failed')
}

export async function neonSignUp(email, password, name) {
  if (!authClient) throw new Error('Neon Auth is not configured')
  const { error } = await authClient.signUp.email({
    name: name || email.split('@')[0] || 'Player',
    email,
    password,
  })
  if (error) throw new Error(error.message || 'Sign-up failed')
}

export async function neonSignInWithGoogle(callbackURL = window.location.href) {
  if (!authClient) throw new Error('Neon Auth is not configured')
  await authClient.signIn.social({ provider: 'google', callbackURL })
}

export async function neonSignOut() {
  if (!authClient) return
  try {
    await authClient.signOut()
  } catch {
    /* ignore */
  }
}
