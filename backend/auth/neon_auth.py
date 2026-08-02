"""
Neon Auth (Better Auth) JWT verification.

Neon Auth issues EdDSA (Ed25519) signed JWTs. We verify them against the
branch's JWKS endpoint, derived from ``NEON_AUTH_BASE_URL``.
"""
import os
from urllib.parse import urlparse

import jwt
from jwt import PyJWKClient

NEON_AUTH_BASE_URL = os.getenv("NEON_AUTH_BASE_URL", "").strip().rstrip("/")
_JWKS_URL = f"{NEON_AUTH_BASE_URL}/.well-known/jwks.json" if NEON_AUTH_BASE_URL else None

_parsed = urlparse(NEON_AUTH_BASE_URL) if NEON_AUTH_BASE_URL else None
_ORIGIN = f"{_parsed.scheme}://{_parsed.netloc}" if _parsed and _parsed.netloc else None

_jwk_client = PyJWKClient(_JWKS_URL) if _JWKS_URL else None


class NeonAuthError(Exception):
    """Raised when a Neon Auth token cannot be verified."""


def is_configured() -> bool:
    return _jwk_client is not None


def _decode_payload(token: str) -> dict:
    """Verify signature + expiry, trying common Neon/Better Auth iss/aud patterns."""
    signing_key = _jwk_client.get_signing_key_from_jwt(token)

    attempts: list[dict] = []
    if _ORIGIN:
        attempts.append({"issuer": _ORIGIN, "audience": _ORIGIN})
    if NEON_AUTH_BASE_URL:
        attempts.append({"issuer": NEON_AUTH_BASE_URL, "audience": NEON_AUTH_BASE_URL})
    if _ORIGIN:
        attempts.append({"issuer": _ORIGIN, "options": {"verify_aud": False}})
    attempts.append({"options": {"verify_aud": False, "verify_iss": False}})

    last_err: Exception | None = None
    for kwargs in attempts:
        try:
            opts = kwargs.pop("options", {})
            return jwt.decode(
                token,
                signing_key.key,
                algorithms=["EdDSA"],
                options=opts,
                **kwargs,
            )
        except Exception as e:
            last_err = e

    raise NeonAuthError(str(last_err) if last_err else "JWT verification failed")


def verify_neon_auth_token(token: str) -> dict:
    """Verify a Neon Auth JWT and return normalised user claims.

    Returns a dict with: neon_auth_id, email, name, picture.
    Raises NeonAuthError on any verification failure.
    """
    if _jwk_client is None:
        raise NeonAuthError("NEON_AUTH_BASE_URL is not configured")

    if not token or token.count(".") != 2:
        raise NeonAuthError("Value is not a JWT")

    try:
        payload = _decode_payload(token)
    except NeonAuthError:
        raise
    except Exception as e:
        raise NeonAuthError(str(e)) from e

    sub = payload.get("sub") or payload.get("id")
    if not sub:
        raise NeonAuthError("Token missing subject (sub) claim")

    nested = payload.get("user") or {}

    def _pick(*keys):
        for src in (payload, nested):
            for k in keys:
                v = src.get(k)
                if v:
                    return v
        return None

    return {
        "neon_auth_id": str(sub),
        "email": _pick("email") or "",
        "name": _pick("name", "fullName") or "",
        "picture": _pick("picture", "image", "avatar") or "",
    }
