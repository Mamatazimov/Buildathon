from datetime import datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import DateTime, ForeignKey, Integer, String, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base


class User(Base):
    """Registered personnel with a name and organizational role."""

    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    full_name: Mapped[str] = mapped_column(String(255), nullable=False)
    role: Mapped[str] = mapped_column(String(128), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    # ── Relationships ─────────────────────────────────────────────────────────
    face_vector: Mapped["FaceVector"] = relationship(
        back_populates="user",
        cascade="all, delete-orphan",
        uselist=False,
    )
    attendance_logs: Mapped[list["AttendanceLog"]] = relationship(
        back_populates="user",
        cascade="all, delete-orphan",
        order_by="AttendanceLog.timestamp.desc()",
    )

    def __repr__(self) -> str:  # pragma: no cover
        return f"<User id={self.id} full_name={self.full_name!r}>"


class FaceVector(Base):
    """
    Stores one 128-dimensional face embedding per user.
    The pgvector `Vector` column type enables cosine / L2 similarity search.
    """

    __tablename__ = "face_vectors"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("users.id", ondelete="CASCADE"),
        unique=True,  # Enforces one-to-one at the DB level
        nullable=False,
        index=True,
    )
    embedding: Mapped[list[float]] = mapped_column(
        Vector(512),  # pgvector type; dimension must match model output
        nullable=False,
    )

    # ── Relationships ─────────────────────────────────────────────────────────
    user: Mapped["User"] = relationship(back_populates="face_vector")

    def __repr__(self) -> str:  # pragma: no cover
        return f"<FaceVector id={self.id} user_id={self.user_id}>"


class AttendanceLog(Base):
    """
    Immutable event log written each time a face is successfully recognized.
    Used to compute time-delta for the greeting rule engine.
    """

    __tablename__ = "attendance_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    # ── Relationships ─────────────────────────────────────────────────────────
    user: Mapped["User"] = relationship(back_populates="attendance_logs")

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"<AttendanceLog id={self.id} user_id={self.user_id} ts={self.timestamp}>"
        )
