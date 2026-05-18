from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.core.database import get_db
from app.models.models import FaceVector, User
from app.schemas.schemas import (
    FaceVectorCreate,
    FaceVectorResponse,
    UserCreate,
    UserResponse,
)

router = APIRouter(prefix="/users", tags=["Users"])

# Type alias for the injected session dependency
DB = Annotated[AsyncSession, Depends(get_db)]


# ─────────────────────────────────────────────────────────────────────────────
# POST /users
# ─────────────────────────────────────────────────────────────────────────────
@router.post(
    "",
    response_model=UserResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new user profile",
)
async def create_user(payload: UserCreate, db: DB) -> User:
    """
    Register a new user with a name and role.
    Returns the created user record including the server-generated primary key
    and creation timestamp.
    """
    user = User(full_name=payload.full_name, role=payload.role)
    db.add(user)
    await db.flush()  # Populate `user.id` without committing (commit in middleware)
    await db.refresh(user)
    return user


# ─────────────────────────────────────────────────────────────────────────────
# GET /users/{user_id}
# ─────────────────────────────────────────────────────────────────────────────
@router.get(
    "/{user_id}",
    response_model=UserResponse,
    summary="Retrieve a user profile by ID",
)
async def get_user(user_id: int, db: DB) -> User:
    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"User with id={user_id} not found.",
        )
    return user


# ─────────────────────────────────────────────────────────────────────────────
# DELETE /users/{user_id}
# ─────────────────────────────────────────────────────────────────────────────
@router.delete(
    "/{user_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a user and all associated data",
)
async def delete_user(user_id: int, db: DB) -> None:
    """
    Hard-delete the user.  Because both `FaceVector` and `AttendanceLog` FK
    columns reference `users.id` with `ondelete=CASCADE`, the database
    automatically removes all child rows.
    """
    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"User with id={user_id} not found.",
        )
    await db.delete(user)


# ─────────────────────────────────────────────────────────────────────────────
# POST /users/{user_id}/vectors
# ─────────────────────────────────────────────────────────────────────────────
@router.post(
    "/{user_id}/vectors",
    response_model=FaceVectorResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Save or replace a user's 512-D face embedding",
)
async def save_face_vector(
    user_id: int, payload: FaceVectorCreate, db: DB
) -> FaceVector:
    """
    Upsert the face embedding for the given user.

    * If no embedding exists yet → create a new `FaceVector` row.
    * If an embedding already exists → overwrite `embedding` in-place
      (the `UNIQUE` constraint on `user_id` allows only one row per user).
    """
    # Verify user exists first
    user_result = await db.execute(select(User).where(User.id == user_id))
    if user_result.scalar_one_or_none() is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"User with id={user_id} not found.",
        )

    # Load existing vector (if any) with the user eager-loaded
    vec_result = await db.execute(
        select(FaceVector)
        .options(selectinload(FaceVector.user))
        .where(FaceVector.user_id == user_id)
    )
    face_vector = vec_result.scalar_one_or_none()

    if face_vector is None:
        face_vector = FaceVector(user_id=user_id, embedding=payload.embedding)
        db.add(face_vector)
    else:
        face_vector.embedding = payload.embedding  # In-place update (upsert)

    await db.flush()
    await db.refresh(face_vector)
    return face_vector
