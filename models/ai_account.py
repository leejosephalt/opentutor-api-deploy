"""Per-user AI plan, locked model preference, and usage counters.

These are created automatically by ``Base.metadata.create_all`` at startup, so
no Alembic migration is required for SQLite local/public deployments.
"""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import Boolean, DateTime, ForeignKey, Integer, String, func
from sqlalchemy.orm import Mapped, mapped_column

from database import Base
from models.compat import CompatUUID


class UserAISettings(Base):
    __tablename__ = "user_ai_settings"

    user_id: Mapped[uuid.UUID] = mapped_column(
        CompatUUID,
        ForeignKey("users.id", ondelete="CASCADE"),
        primary_key=True,
    )
    plan: Mapped[str] = mapped_column(String(20), default="free", nullable=False)
    model_tier: Mapped[str] = mapped_column(String(16), default="fast", nullable=False)
    is_admin: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    last_request_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )


class AIUsageCounter(Base):
    __tablename__ = "ai_usage_counters"

    user_id: Mapped[uuid.UUID] = mapped_column(
        CompatUUID,
        ForeignKey("users.id", ondelete="CASCADE"),
        primary_key=True,
    )
    period_date: Mapped[str] = mapped_column(String(10), primary_key=True)
    requests: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    tokens: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    smart_calls: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )
