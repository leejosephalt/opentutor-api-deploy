"""AI model preference and usage status endpoints."""

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from database import get_db
from libs.exceptions import ValidationError
from models.user import User
from services import ai_policy
from services.auth.dependency import get_current_user

router = APIRouter()


class AIModelUpdate(BaseModel):
    model_tier: str = Field(..., max_length=16)


@router.get("/settings", summary="Get AI plan, model tier, and usage")
async def get_ai_settings(
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    return await ai_policy.get_status(db, user.id)


@router.put("/settings", summary="Choose the AI model tier (fast or smart)")
async def update_ai_settings(
    body: AIModelUpdate,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    try:
        await ai_policy.set_model_tier(db, user.id, body.model_tier)
    except ValueError as exc:
        raise ValidationError(str(exc)) from exc
    return await ai_policy.get_status(db, user.id)
