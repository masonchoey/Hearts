"""
FastAPI auth dependencies — inject the current authenticated user.

All authentication flows through Neon Auth (Better Auth), including Google
via Neon's OAuth provider. Requests carry a Neon Auth JWT (EdDSA) verified
against the branch JWKS.
"""
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from ..db.database import get_db
from ..db.models import User
from .neon_auth import verify_neon_auth_token, NeonAuthError, is_configured as neon_auth_configured

bearer = HTTPBearer(auto_error=False)


async def upsert_neon_user(db: AsyncSession, claims: dict) -> User:
    """Find-or-create the app user for a verified Neon Auth token."""
    neon_id = claims["neon_auth_id"]

    result = await db.execute(select(User).where(User.neon_auth_id == neon_id))
    user = result.scalar_one_or_none()

    if user is None and claims.get("email"):
        result = await db.execute(select(User).where(User.email == claims["email"]))
        user = result.scalar_one_or_none()
        if user is not None:
            user.neon_auth_id = neon_id

    if user is None:
        user = User(
            neon_auth_id=neon_id,
            email=claims.get("email") or f"{neon_id}@neon.local",
            name=claims.get("name") or claims.get("email") or "Player",
            picture=claims.get("picture") or None,
            auth_provider="neon",
        )
        db.add(user)
    else:
        if claims.get("name"):
            user.name = claims["name"]
        if claims.get("picture"):
            user.picture = claims["picture"]
        user.auth_provider = "neon"

    await db.commit()
    await db.refresh(user)
    return user


async def resolve_user_from_token(token: str, db: AsyncSession) -> User | None:
    """Resolve a Neon Auth bearer token to a User."""
    if not token or not neon_auth_configured():
        return None
    try:
        claims = verify_neon_auth_token(token)
    except NeonAuthError:
        return None
    return await upsert_neon_user(db, claims)


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(bearer),
    db: AsyncSession = Depends(get_db),
) -> User:
    if credentials is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Not authenticated")
    if not neon_auth_configured():
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Neon Auth is not configured")

    user = await resolve_user_from_token(credentials.credentials, db)
    if user is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or expired token")
    return user


async def get_optional_user(
    credentials: HTTPAuthorizationCredentials = Depends(bearer),
    db: AsyncSession = Depends(get_db),
) -> User | None:
    if credentials is None:
        return None
    return await resolve_user_from_token(credentials.credentials, db)
