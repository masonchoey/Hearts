"""
Async database setup using SQLAlchemy.

Uses the Neon Postgres instance pointed to by ``DATABASE_URL`` (see backend/.env).
Falls back to a local SQLite file when ``DATABASE_URL`` is not configured so the
app still runs in a bare dev environment.
"""
import os
import ssl
from urllib.parse import urlsplit, urlunsplit, parse_qsl

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase

# Application tables live in this Postgres schema. The Neon Auth (Better Auth)
# tables live in a separate ``neon_auth`` schema that Neon manages for us.
APP_SCHEMA = "public"

_SQLITE_FALLBACK = "sqlite+aiosqlite:///./hearts_multiplayer.db"


def _build_engine_config() -> tuple[str, dict]:
    """Return (sqlalchemy_url, connect_args) derived from DATABASE_URL.

    Neon connection strings look like::

        postgresql://user:pass@host/db?sslmode=require&channel_binding=require

    The asyncpg driver does not understand the libpq ``sslmode`` /
    ``channel_binding`` query parameters, so we strip them out of the URL and
    translate them into an explicit SSL context passed via ``connect_args``.
    """
    raw_url = os.getenv("DATABASE_URL", "").strip()
    if not raw_url:
        return _SQLITE_FALLBACK, {}

    parts = urlsplit(raw_url)

    # Normalise the scheme to the asyncpg dialect.
    scheme = parts.scheme
    if scheme in ("postgres", "postgresql"):
        scheme = "postgresql+asyncpg"
    elif scheme == "postgresql+psycopg2":
        scheme = "postgresql+asyncpg"

    # Pull libpq-only params out of the query string.
    query_params = dict(parse_qsl(parts.query, keep_blank_values=True))
    sslmode = query_params.pop("sslmode", None)
    query_params.pop("channel_binding", None)  # negotiated automatically by asyncpg

    cleaned = urlunsplit(
        (scheme, parts.netloc, parts.path, "", parts.fragment)
    )

    connect_args: dict = {}
    if sslmode and sslmode != "disable":
        # Neon requires TLS. Verify the server certificate against system roots.
        connect_args["ssl"] = ssl.create_default_context()

    return cleaned, connect_args


DATABASE_URL, _CONNECT_ARGS = _build_engine_config()
IS_POSTGRES = DATABASE_URL.startswith("postgresql")

engine = create_async_engine(
    DATABASE_URL,
    echo=False,
    pool_pre_ping=True,
    connect_args=_CONNECT_ARGS,
)

AsyncSessionLocal = async_sessionmaker(
    engine, class_=AsyncSession, expire_on_commit=False
)


class Base(DeclarativeBase):
    pass


async def init_db():
    """Create application tables on startup (idempotent).

    Neon-Auth-managed tables (``neon_auth`` schema) are NOT touched here — Neon
    owns them. We only create the app tables defined in ``models``.
    """
    from . import models  # noqa: F401  ensure models are registered on Base.metadata

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def get_db():
    """FastAPI dependency that yields a DB session."""
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()
