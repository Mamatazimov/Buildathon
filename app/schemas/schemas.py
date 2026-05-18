from datetime import datetime
from typing import Annotated

from pydantic import BaseModel, Field, field_validator


# ── Shared config ─────────────────────────────────────────────────────────────
class _OrmBase(BaseModel):
    """All response schemas inherit from here to enable ORM mode."""

    model_config = {"from_attributes": True}


# ─────────────────────────────────────────────────────────────────────────────
# User
# ─────────────────────────────────────────────────────────────────────────────
class UserCreate(BaseModel):
    full_name: Annotated[
        str, Field(min_length=1, max_length=255, examples=["Asilbek Yusupov"])
    ]
    role: Annotated[str, Field(min_length=1, max_length=512, examples=["engineer"])]


class UserResponse(_OrmBase):
    id: int
    full_name: str
    role: str
    created_at: datetime


# ─────────────────────────────────────────────────────────────────────────────
# FaceVector
# ─────────────────────────────────────────────────────────────────────────────
class FaceVectorCreate(BaseModel):
    embedding: Annotated[
        list[float],
        Field(
            description="512-dimensional face embedding produced by the recognition model.",
            examples=[[0.1] * 512],
        ),
    ]

    @field_validator("embedding")
    @classmethod
    def validate_dimension(cls, v: list[float]) -> list[float]:
        if len(v) != 512:
            raise ValueError(
                f"Embedding must have exactly 512 dimensions, got {len(v)}."
            )
        return v


class FaceVectorResponse(_OrmBase):
    id: int
    user_id: int


# ─────────────────────────────────────────────────────────────────────────────
# Recognition / Evaluation
# ─────────────────────────────────────────────────────────────────────────────
VALID_WEATHER = {"rainy", "hot", "snowy", "sunny", "cloudy"}
VALID_EMOTIONS = {"sad", "tired", "happy", "neutral", "angry", "surprised"}


class EvaluateRequest(BaseModel):
    detected_vector: Annotated[
        list[float],
        Field(description="512-dimensional face embedding from the camera feed."),
    ]
    weather_condition: Annotated[
        str,
        Field(
            description=f"Current weather. Accepted values: {VALID_WEATHER}.",
            examples=["rainy"],
        ),
    ]
    current_emotion: Annotated[
        str,
        Field(
            description=f"Detected facial emotion. Accepted values: {VALID_EMOTIONS}.",
            examples=["sad"],
        ),
    ]

    @field_validator("detected_vector")
    @classmethod
    def validate_dimension(cls, v: list[float]) -> list[float]:
        if len(v) != 512:
            raise ValueError(f"detected_vector must have 512 dimensions, got {len(v)}.")
        return v

    @field_validator("weather_condition")
    @classmethod
    def normalise_weather(cls, v: str) -> str:
        return v.strip().lower()

    @field_validator("current_emotion")
    @classmethod
    def normalise_emotion(cls, v: str) -> str:
        return v.strip().lower()


class EvaluateResponse(BaseModel):
    user_id: int
    full_name: str
    role: str
    distance: float
    log_id: int
    message: str
