from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase

from app.core.config import settings

# ── Engine ────────────────────────────────────────────────────────────────────
engine = create_async_engine(
    settings.async_database_url,
    echo=settings.DEBUG,  # SQL logging; disable in production
    pool_size=10,
    max_overflow=20,
    pool_pre_ping=True,  # Recycles stale connections automatically
)

# ── Session factory ───────────────────────────────────────────────────────────
AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,  # Keep attribute access after commit
    autoflush=False,
    autocommit=False,
)


# ── Declarative base ──────────────────────────────────────────────────────────
class Base(DeclarativeBase):
    """Shared declarative base for all ORM models."""

    pass


# ── FastAPI dependency ────────────────────────────────────────────────────────
async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    Yields a transactional AsyncSession per request.
    Rolls back automatically on any unhandled exception.
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
