"""
Greeting Rule Engine
====================
Stateless service that applies a strict priority-ordered ruleset
to produce a single localised greeting message.

Priority order (highest → lowest):
  1. Absence rule  – last visit was > 7 days ago
  2. Weather rule  – actionable weather notice
  3. Emotion rule  – empathetic mood response
  4. Default rule  – standard welcome message
"""

from datetime import datetime, timezone

ABSENCE_THRESHOLD_DAYS: int = 7


def _days_since(previous_timestamp: datetime) -> float:
    """Return the number of days elapsed since *previous_timestamp*."""
    now = datetime.now(tz=timezone.utc)
    # Ensure both datetimes are timezone-aware for safe subtraction
    if previous_timestamp.tzinfo is None:
        previous_timestamp = previous_timestamp.replace(tzinfo=timezone.utc)
    return (now - previous_timestamp).total_seconds() / 86_400


def build_greeting(
    full_name: str,
    weather_condition: str,
    current_emotion: str,
    previous_log_timestamp: datetime | None,
) -> str:
    """
    Evaluate all rules in priority order and return the appropriate message.

    Parameters
    ----------
    full_name:
        Display name of the recognised user.
    weather_condition:
        Normalised weather string (already lowercased by Pydantic validator).
    current_emotion:
        Normalised emotion string (already lowercased by Pydantic validator).
    previous_log_timestamp:
        UTC datetime of the user's second-most-recent attendance log entry,
        or ``None`` if this is their first recorded visit.
    """

    # ── Rule 1: Absence ───────────────────────────────────────────────────────
    if previous_log_timestamp is not None:
        days_away = _days_since(previous_log_timestamp)
        if days_away > ABSENCE_THRESHOLD_DAYS:
            return (
                "Ko'rishmaganimizga ancha bo'libdi, qayerlarda edingiz? Sog'indik! 🤗"
            )

    # ── Rule 2: Weather ───────────────────────────────────────────────────────
    weather_messages: dict[str, str] = {
        "rainy": "Bugun yomg'ir yog'ayapti, soyabonni unutmang! 🌧️",
        "hot": "Kun juda issiq, ko'proq suv ichishni unutmang! ☀️",
    }
    if weather_condition in weather_messages:
        return weather_messages[weather_condition]

    # ── Rule 3: Emotion ───────────────────────────────────────────────────────
    emotion_messages: dict[str, str] = {
        "sad": "Kayfiyatingizni tushirmang, bugungi kuningiz ajoyib o'tadi! ✨",
        "tired": "Charchadingizmi? Bir finjon qahva ichib dam oling! ☕",
    }
    if current_emotion in emotion_messages:
        return emotion_messages[current_emotion]

    # ── Rule 4: Default ───────────────────────────────────────────────────────
    return f"Xush kelibsiz, {full_name}! Kuningiz unumli o'tsin! 🚀"
