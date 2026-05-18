"""
Recognition Router
==================
POST /recognition/evaluate

Accepts a raw 128-D face embedding from an edge device, performs a nearest-
neighbour search against stored embeddings using pgvector's L2 (<->) operator,
logs the attendance event, and returns a dynamically generated greeting message
produced by the rule engine.
"""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.database import get_db
from app.models.models import AttendanceLog, FaceVector, User
from app.schemas.schemas import EvaluateRequest, EvaluateResponse
from app.services.greeting import build_greeting

router = APIRouter(prefix="/recognition", tags=["Recognition"])

DB = Annotated[AsyncSession, Depends(get_db)]


@router.post(
    "/evaluate",
    response_model=EvaluateResponse,
    summary="Real-time face recognition + dynamic greeting evaluation",
)
async def evaluate(payload: EvaluateRequest, db: DB) -> EvaluateResponse:
    """
    Full recognition pipeline executed per camera frame / edge event:

    1. **Vector search** – find the nearest stored embedding via L2 distance.
    2. **Threshold guard** – reject if distance > configured threshold.
    3. **Attendance log** – persist the current visit immediately.
    4. **Context fetch** – retrieve the *previous* log entry to compute Δt.
    5. **Rule engine** – produce a tailored greeting message.
    """

    # ── Step 1: Nearest-neighbour search via pgvector ─────────────────────────
    #
    # `FaceVector.embedding.l2_distance(vector)` generates the SQL expression:
    #   embedding <-> '[0.1,0.2,...]'::vector
    # SQLAlchemy 2.0 requires .label() or direct use inside order_by/where.
    #
    print(1)
    distance_expr = FaceVector.embedding.l2_distance(payload.detected_vector)
    print(distance_expr)

    stmt = (
        select(FaceVector, distance_expr.label("distance"))
        .order_by(distance_expr)
        .limit(1)
    )
    result = await db.execute(stmt)
    row = result.first()

    if row is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No face embeddings found in the database.",
        )
    print(2)

    face_vector: FaceVector = row[0]
    distance: float = float(row[1])

    # ── Step 2: Threshold guard ───────────────────────────────────────────────
    print(row)
    if distance > settings.RECOGNITION_THRESHOLD:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=(
                f"Face not recognised. "
                f"Closest match distance={distance:.4f} exceeds threshold={settings.RECOGNITION_THRESHOLD}."
            ),
        )
    print(3)

    user_id: int = face_vector.user_id

    # ── Step 3: Fetch user profile ────────────────────────────────────────────
    user_result = await db.execute(select(User).where(User.id == user_id))
    user: User | None = user_result.scalar_one_or_none()
    if user is None:
        # Defensive: should not happen due to FK constraint, but guard anyway
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Data integrity error: user_id={user_id} missing.",
        )
    print(4)

    # ── Step 4: Write attendance log ──────────────────────────────────────────
    new_log = AttendanceLog(user_id=user_id)
    db.add(new_log)
    await db.flush()  # Persist & obtain generated PK + server timestamp
    await db.refresh(new_log)

    # ── Step 5: Fetch previous attendance log (second-most-recent entry) ──────
    #
    # After the flush above, the DB now holds the new log as the most recent.
    # We fetch the *two* most recent rows and pick the second one as "previous".
    #
    logs_stmt = (
        select(AttendanceLog)
        .where(AttendanceLog.user_id == user_id)
        .order_by(AttendanceLog.timestamp.desc())  # type: ignore[arg-type]
        .limit(2)
    )
    logs_result = await db.execute(logs_stmt)
    logs = logs_result.scalars().all()

    previous_timestamp = logs[1].timestamp if len(logs) >= 2 else None

    # ── Step 6: Rule engine → greeting message ────────────────────────────────
    message = build_greeting(
        full_name=user.full_name,
        weather_condition=payload.weather_condition,
        current_emotion=payload.current_emotion,
        previous_log_timestamp=previous_timestamp,
    )

    return EvaluateResponse(
        user_id=user.id,
        full_name=user.full_name,
        role=user.role,
        distance=round(distance, 6),
        log_id=new_log.id,
        message=message,
    )
