"""AI plan, locked-model preference, and per-user quota enforcement.

Free tier:
* conversational AI chat is disabled (handled by the chat router),
* one-shot AI actions are rate limited by a cooldown and daily caps.

Pro/admin tier: higher caps.

All counters are stored in the database so they survive restarts and cannot be
reset by clearing local storage.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone
from typing import Any

from sqlalchemy import select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from config import settings
from libs.exceptions import AIQuotaError
from models.ai_account import AIUsageCounter, UserAISettings

PLAN_FREE = "free"
PLAN_PRO = "pro"
MODEL_TIERS = ("fast", "smart")
DEFAULT_TIER = "fast"


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _as_utc(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _today() -> str:
    return _utc_now().strftime("%Y-%m-%d")


def _seconds_until_midnight() -> int:
    now = _utc_now()
    tomorrow = (now + timedelta(days=1)).replace(
        hour=0, minute=0, second=0, microsecond=0
    )
    return max(1, int((tomorrow - now).total_seconds()))


def limits_for(plan: str) -> dict[str, int]:
    if plan == PLAN_PRO:
        return {
            "daily_requests": max(1, settings.ai_pro_daily_limit),
            "daily_tokens": max(1, settings.ai_pro_daily_token_limit),
            "smart_calls": max(0, settings.ai_pro_smart_daily_limit),
        }
    return {
        "daily_requests": max(1, settings.ai_daily_limit),
        "daily_tokens": max(1, settings.ai_daily_token_limit),
        "smart_calls": max(0, settings.ai_smart_daily_limit),
    }


def _default_plan() -> str:
    """Local single-user installs are the owner: full access, no quotas."""
    if settings.deployment_mode == "single_user" and not settings.auth_enabled:
        return PLAN_PRO
    return PLAN_FREE


async def get_or_create_settings(db: AsyncSession, user_id: uuid.UUID) -> UserAISettings:
    result = await db.execute(
        select(UserAISettings).where(UserAISettings.user_id == user_id)
    )
    row = result.scalar_one_or_none()
    if row is not None:
        return row

    row = UserAISettings(user_id=user_id, plan=_default_plan())
    db.add(row)
    try:
        await db.commit()
        await db.refresh(row)
    except IntegrityError:
        # Another request created it first.
        await db.rollback()
        result = await db.execute(
            select(UserAISettings).where(UserAISettings.user_id == user_id)
        )
        row = result.scalar_one()
    return row


async def get_or_create_usage(
    db: AsyncSession, user_id: uuid.UUID, period: str | None = None
) -> AIUsageCounter:
    period = period or _today()
    result = await db.execute(
        select(AIUsageCounter).where(
            AIUsageCounter.user_id == user_id,
            AIUsageCounter.period_date == period,
        )
    )
    row = result.scalar_one_or_none()
    if row is not None:
        return row

    row = AIUsageCounter(user_id=user_id, period_date=period)
    db.add(row)
    try:
        await db.commit()
        await db.refresh(row)
    except IntegrityError:
        await db.rollback()
        result = await db.execute(
            select(AIUsageCounter).where(
                AIUsageCounter.user_id == user_id,
                AIUsageCounter.period_date == period,
            )
        )
        row = result.scalar_one()
    return row


async def set_model_tier(db: AsyncSession, user_id: uuid.UUID, tier: str) -> UserAISettings:
    tier = (tier or "").strip().lower()
    if tier not in MODEL_TIERS:
        raise ValueError(f"Unknown model tier: {tier}")
    row = await get_or_create_settings(db, user_id)
    row.model_tier = tier
    await db.commit()
    await db.refresh(row)
    return row


def is_chat_allowed(plan: str) -> bool:
    if settings.free_chat_enabled:
        return True
    return plan == PLAN_PRO


async def check_and_reserve(db: AsyncSession, user_id: uuid.UUID) -> UserAISettings:
    """Enforce cooldown + daily caps, then reserve one request.

    Raises ``AIQuotaError`` (HTTP 429) when the user must wait or has hit a cap.
    Local single-user installs are exempt (the owner is not rate limited).
    """
    if settings.deployment_mode == "single_user" and not settings.auth_enabled:
        return await get_or_create_settings(db, user_id)

    row = await get_or_create_settings(db, user_id)
    plan = row.plan if row.plan in (PLAN_FREE, PLAN_PRO) else PLAN_FREE
    tier = row.model_tier if row.model_tier in MODEL_TIERS else DEFAULT_TIER
    limits = limits_for(plan)
    now = _utc_now()

    cooldown = max(0, settings.ai_cooldown_seconds)
    last = _as_utc(row.last_request_at)
    if last is not None and cooldown > 0:
        elapsed = (now - last).total_seconds()
        if elapsed < cooldown:
            retry = int(cooldown - elapsed) + 1
            raise AIQuotaError(
                f"Please wait {retry}s before the next AI request.",
                retry_after=retry,
            )

    usage = await get_or_create_usage(db, user_id)
    if usage.requests >= limits["daily_requests"]:
        raise AIQuotaError(
            "Daily AI limit reached on Free mode. It resets at midnight UTC.",
            retry_after=_seconds_until_midnight(),
        )
    if usage.tokens >= limits["daily_tokens"]:
        raise AIQuotaError(
            "Daily AI token budget is used up on Free mode. It resets at midnight UTC.",
            retry_after=_seconds_until_midnight(),
        )
    if tier == "smart" and usage.smart_calls >= limits["smart_calls"]:
        raise AIQuotaError(
            "Smart model limit reached for today. Switch to Fast or try tomorrow.",
            retry_after=_seconds_until_midnight(),
        )

    await db.execute(
        update(AIUsageCounter)
        .where(
            AIUsageCounter.user_id == user_id,
            AIUsageCounter.period_date == usage.period_date,
        )
        .values(
            requests=AIUsageCounter.requests + 1,
            smart_calls=AIUsageCounter.smart_calls + (1 if tier == "smart" else 0),
        )
    )
    await db.execute(
        update(UserAISettings)
        .where(UserAISettings.user_id == user_id)
        .values(last_request_at=now)
    )
    await db.commit()
    return row


async def record_tokens(db: AsyncSession, user_id: uuid.UUID, tokens: int) -> None:
    if tokens <= 0:
        return
    period = _today()
    await get_or_create_usage(db, user_id, period)
    await db.execute(
        update(AIUsageCounter)
        .where(
            AIUsageCounter.user_id == user_id,
            AIUsageCounter.period_date == period,
        )
        .values(tokens=AIUsageCounter.tokens + int(tokens))
    )
    await db.commit()


async def get_status(db: AsyncSession, user_id: uuid.UUID) -> dict[str, Any]:
    row = await get_or_create_settings(db, user_id)
    plan = row.plan if row.plan in (PLAN_FREE, PLAN_PRO) else PLAN_FREE
    tier = row.model_tier if row.model_tier in MODEL_TIERS else DEFAULT_TIER
    usage = await get_or_create_usage(db, user_id)
    limits = limits_for(plan)
    last = _as_utc(row.last_request_at)
    cooldown = max(0, settings.ai_cooldown_seconds)
    cooldown_remaining = 0
    if last is not None and cooldown > 0:
        cooldown_remaining = max(0, int(cooldown - (_utc_now() - last).total_seconds()))

    reset_at = (_utc_now() + timedelta(seconds=_seconds_until_midnight())).isoformat()
    return {
        "plan": plan,
        "model_tier": tier,
        "chat_enabled": is_chat_allowed(plan),
        "is_admin": bool(row.is_admin),
        "usage": {
            "requests": usage.requests,
            "tokens": usage.tokens,
            "smart_calls": usage.smart_calls,
        },
        "limits": limits,
        "cooldown_seconds": cooldown,
        "cooldown_remaining": cooldown_remaining,
        "reset_at": reset_at,
    }
