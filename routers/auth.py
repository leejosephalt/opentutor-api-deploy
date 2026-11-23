"""Authentication endpoints: register, login, refresh."""

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from config import settings
from libs.exceptions import ConflictError, PermissionDeniedError

from database import get_db
from models.user import User
from schemas.auth import RegisterRequest, LoginRequest, RefreshRequest, TokenResponse, UserResponse
from services.auth.password import hash_password, verify_password
from services.auth.jwt import JWTError, create_access_token, create_refresh_token, decode_token

router = APIRouter()


class AdminLoginRequest(BaseModel):
    username: str = Field(..., max_length=100)
    password: str = Field(..., max_length=200)


@router.post("/register", response_model=UserResponse, status_code=201)
async def register(body: RegisterRequest, db: AsyncSession = Depends(get_db)):
    """Register a new user with email and password."""
    # Check email uniqueness
    result = await db.execute(select(User).where(User.email == body.email))
    if result.scalar_one_or_none():
        raise ConflictError("Email already registered")

    user = User(
        name=body.name,
        email=body.email,
        hashed_password=hash_password(body.password),
    )
    db.add(user)
    await db.commit()
    await db.refresh(user)
    return UserResponse(id=str(user.id), email=user.email, name=user.name)


@router.post("/login", response_model=TokenResponse)
async def login(body: LoginRequest, db: AsyncSession = Depends(get_db)):
    """Login with email and password, returns JWT tokens."""
    result = await db.execute(select(User).where(User.email == body.email))
    user = result.scalar_one_or_none()

    if not user or not user.hashed_password:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    if not verify_password(body.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    if not user.is_active:
        raise PermissionDeniedError("Account disabled")

    return TokenResponse(
        access_token=create_access_token(str(user.id)),
        refresh_token=create_refresh_token(str(user.id)),
    )


@router.post("/refresh", response_model=TokenResponse)
async def refresh(body: RefreshRequest, db: AsyncSession = Depends(get_db)):
    """Refresh access token using a valid refresh token."""
    try:
        payload = decode_token(body.refresh_token)
        if payload.get("type") != "refresh":
            raise HTTPException(status_code=401, detail="Invalid token type")
    except HTTPException:
        raise
    except (JWTError, ValueError, KeyError) as exc:
        raise HTTPException(status_code=401, detail="Invalid or expired refresh token") from exc

    user_id = payload.get("sub")
    if not user_id:
        raise HTTPException(status_code=401, detail="Invalid token: missing subject")
    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()
    if not user or not user.is_active:
        raise HTTPException(status_code=401, detail="User not found")

    return TokenResponse(
        access_token=create_access_token(str(user.id)),
        refresh_token=create_refresh_token(str(user.id)),
    )


@router.post("/admin/login", response_model=TokenResponse)
async def admin_login(body: AdminLoginRequest, db: AsyncSession = Depends(get_db)):
    """Admin-only login that grants the Pro plan (chat enabled)."""
    if not settings.admin_password_hash:
        raise HTTPException(status_code=503, detail="Admin login is not configured")
    if body.username.strip().casefold() != settings.admin_username.strip().casefold():
        raise HTTPException(status_code=401, detail="Invalid credentials")
    try:
        valid = verify_password(body.password, settings.admin_password_hash)
    except ValueError:
        valid = False
    if not valid:
        raise HTTPException(status_code=401, detail="Invalid credentials")

    result = await db.execute(
        select(User).where(func.lower(User.name) == settings.admin_username.strip().lower())
    )
    user = result.scalars().first()
    if user is None:
        user = User(name=settings.admin_username)
        db.add(user)
        await db.commit()
        await db.refresh(user)
    if not user.is_active:
        raise PermissionDeniedError("Account disabled")

    from services import ai_policy

    row = await ai_policy.get_or_create_settings(db, user.id)
    row.plan = ai_policy.PLAN_PRO
    row.is_admin = True
    await db.commit()

    return TokenResponse(
        access_token=create_access_token(str(user.id)),
        refresh_token=create_refresh_token(str(user.id)),
    )
