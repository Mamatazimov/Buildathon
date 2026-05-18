"""
Application factory
===================
Assembles the FastAPI instance, registers routers, mounts global exception
handlers, and manages the DB lifespan (database creation, extension, and table creation on startup).
"""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError, OperationalError
from sqlalchemy.ext.asyncio import create_async_engine

from app.core.config import settings
from app.core.database import Base, engine
from app.routers import recognition, users


# ── Database Initialization Logic ─────────────────────────────────────────────
async def ensure_database_exists() -> None:
    """
    Default 'postgres' ma'lumotlar bazasiga ulanib, loyiha uchun kerakli
    baza mavjudligini tekshiradi va yo'q bo'lsa uni avtomat yaratadi.
    """
    # CREATE DATABASE tranzaksiya ichida ishlamasligi uchun AUTOCOMMIT ishlatamiz
    default_engine = create_async_engine(
        settings.default_database_url, isolation_level="AUTOCOMMIT"
    )

    async with default_engine.connect() as conn:
        # Baza mavjudligini tekshirish query'si
        result = await conn.execute(
            text(
                f"SELECT 1 FROM pg_database WHERE datname = '{settings.DATABASE_NAME}'"
            )
        )
        db_exists = result.scalar()

        if not db_exists:
            print(
                f"⚠️ '{settings.DATABASE_NAME}' ma'lumotlar bazasi topilmadi. Yaratilmoqda..."
            )
            await conn.execute(text(f"CREATE DATABASE {settings.DATABASE_NAME}"))
            print(
                f"✅ '{settings.DATABASE_NAME}' ma'lumotlar bazasi muvaffaqiyatli yaratildi."
            )

    await default_engine.dispose()


# ── Lifespan ──────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Startup: ensure DB exists, `pgvector` extension and all tables exist.
    Shutdown: dispose connection pool gracefully.
    """
    # 1. Lokal muhitda baza borligini tekshirib, yaratib olamiz
    try:
        await ensure_database_exists()
    except Exception as e:
        print(f"❌ Ma'lumotlar bazasini tekshirish/yaratishda xatolik: {e}")
        raise e

    # 2. 'pgvector' extensionni alohida va xavfsiz yoqamiz (Engine.connect orqali)
    # connect() asynctron muhitda ko'proq izolyatsiya beradi
    try:
        async with engine.connect() as conn:
            await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            await conn.commit()  # O'zgarishni darhol saqlaymiz
            print("✅ 'pgvector' muvaffaqiyatli tekshirildi.")
    except (IntegrityError, OperationalError) as e:
        # Agar baza allaqachon bor deb unikal indeks xatosi bersa, buni o'tkazib yuboramiz
        if "already exists" in str(e):
            print("ℹ️ 'pgvector' allaqachon bazada mavjud, davom etamiz.")
        else:
            raise e

    # 3. Jadvallarni yaratish (alohida tranzaksiyada)
    async with engine.begin() as conn:
        print("🛠️ Ma'lumotlar bazasi jadvallari tekshirilmoqda...")
        await conn.run_sync(Base.metadata.create_all)
        print("✅ Barcha jadvallar tayyor.")

    yield  # ← loyiha (FastAPI) shu yerda ishga tushadi

    # Shutdown: ulanishlar pulini tozalaymiz
    await engine.dispose()


# ── App factory ───────────────────────────────────────────────────────────────
def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.APP_TITLE,
        version=settings.APP_VERSION,
        description=(
            "A production-ready, fully asynchronous API for real-time face "
            "recognition, attendance tracking, and AI-powered greeting generation."
        ),
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    # ── Routers ───────────────────────────────────────────────────────────────
    app.include_router(users.router)
    app.include_router(recognition.router)

    # ── Global exception handler ──────────────────────────────────────────────
    @app.exception_handler(Exception)
    async def unhandled_exception_handler(
        request: Request, exc: Exception
    ) -> JSONResponse:
        """
        Catch-all for any unexpected server errors.
        Avoids leaking stack traces to the client.
        """
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "detail": "An internal server error occurred. Please try again later."
            },
        )

    # ── Health check ──────────────────────────────────────────────────────────
    @app.get("/health", tags=["Health"], summary="Liveness probe")
    async def health_check() -> dict[str, str]:
        return {"status": "ok", "version": settings.APP_VERSION}

    return app


app = create_app()
