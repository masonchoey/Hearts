"""
Dev-only auth bypass for local testing.

When the ``DEV_AUTH`` env var is truthy, the app accepts bearer tokens of the
form ``dev:<name>`` (e.g. ``dev:alice``) and resolves them to a stable fake
user. This lets a full 4-player game be played locally without setting up Neon
Auth / Google OAuth.

SECURITY: this bypasses all real authentication. It is gated behind DEV_AUTH,
which must be blank/absent in any non-local environment.
"""
import os

_TRUTHY = {"1", "true", "yes", "on"}


def dev_auth_enabled() -> bool:
    return os.getenv("DEV_AUTH", "").strip().lower() in _TRUTHY


def parse_dev_token(token: str) -> dict | None:
    """Return normalised claims for a ``dev:<name>`` token, or None if not one."""
    if not token or not token.startswith("dev:"):
        return None
    name = token[len("dev:"):].strip().lower()
    if not name:
        return None
    return {
        "neon_auth_id": f"dev:{name}",
        "email": f"{name}@dev.local",
        "name": name.capitalize(),
        "picture": "",
    }
