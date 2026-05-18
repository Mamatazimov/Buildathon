from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # ── Database ──────────────────────────────────────────────────────────────
    DATABASE_HOST: str = "localhost"
    DATABASE_PORT: int = 5432
    DATABASE_NAME: str = "smart_greeting_db"
    DATABASE_USER: str = "postgres"
    DATABASE_PASSWORD: str = "postgres"

    # ── Recognition ───────────────────────────────────────────────────────────
    RECOGNITION_THRESHOLD: float = 1  # L2 distance; lower = stricter match
    EMBEDDING_DIMENSION: int = 128

    # ── App ───────────────────────────────────────────────────────────────────
    APP_TITLE: str = "Smart AI-Powered Greeting System"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False

    @property
    def async_database_url(self) -> str:
        """Asosiy loyiha bazasiga ulanish URL-manzili"""
        return (
            f"postgresql+psycopg://{self.DATABASE_USER}:{self.DATABASE_PASSWORD}"
            f"@{self.DATABASE_HOST}:{self.DATABASE_PORT}/{self.DATABASE_NAME}"
        )

    @property
    def default_database_url(self) -> str:
        """Default 'postgres' bazasiga ulanish URL-manzili (yangi baza yaratish uchun)"""
        return (
            f"postgresql+psycopg://{self.DATABASE_USER}:{self.DATABASE_PASSWORD}"
            f"@{self.DATABASE_HOST}:{self.DATABASE_PORT}/postgres"
        )


settings = Settings()
