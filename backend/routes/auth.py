"""
Auth routes — Neon Auth (Better Auth) only.
"""
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from ..db.database import get_db
from ..db.models import User
from ..auth.dependencies import get_current_user, upsert_neon_user
from ..auth.neon_auth import verify_neon_auth_token, NeonAuthError, is_configured as neon_auth_configured

router = APIRouter(prefix="/auth", tags=["auth"])

_bearer = HTTPBearer(auto_error=False)


class UserResponse(BaseModel):
    id: str
    email: str
    name: str
    picture: str | None
    auth_provider: str | None = None


class LoginResponse(BaseModel):
    token: str
    user: UserResponse


@router.post("/session", response_model=LoginResponse)
async def sync_session(
    credentials: HTTPAuthorizationCredentials = Depends(_bearer),
    db: AsyncSession = Depends(get_db),
):
    """Sync a Neon Auth session into the app ``users`` table.

    The frontend signs in via Neon Auth (email/password or Google OAuth through
    Neon), obtains a JWT via ``getJWTToken()``, and sends it as a Bearer token.
    """
    if not neon_auth_configured():
        raise HTTPException(status_code=503, detail="Neon Auth is not configured on the server")
    if credentials is None:
        raise HTTPException(status_code=401, detail="Missing auth token")

    try:
        claims = verify_neon_auth_token(credentials.credentials)
    except NeonAuthError as e:
        raise HTTPException(status_code=401, detail=f"Invalid auth token: {e}")

    user = await upsert_neon_user(db, claims)
    return LoginResponse(token=credentials.credentials, user=_to_user_response(user))


# Backwards-compatible alias while clients migrate
@router.post("/neon", response_model=LoginResponse, include_in_schema=False)
async def sync_session_legacy(
    credentials: HTTPAuthorizationCredentials = Depends(_bearer),
    db: AsyncSession = Depends(get_db),
):
    return await sync_session(credentials, db)


@router.get("/me", response_model=UserResponse)
async def get_me(current_user: User = Depends(get_current_user)):
    return _to_user_response(current_user)


def _to_user_response(user: User) -> UserResponse:
    return UserResponse(
        id=user.id,
        email=user.email,
        name=user.name,
        picture=user.picture,
        auth_provider=user.auth_provider,
    )
