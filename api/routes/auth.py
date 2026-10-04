"""Authentication API routes."""

from __future__ import annotations

from urllib.parse import urlencode

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from auth.email_verification import (
    create_email_verification_token,
    get_verification_token_hash,
    is_verification_token_valid,
)
from auth.hashing import hash_password, verify_password
from auth.jwt import create_access_token
from auth.schemas import (
    LoginRequest,
    RegisterRequest,
    TokenResponse,
    UserResponse,
)
from database.models import User
from database.session import get_db
from services.email import email_service, get_verification_base_url

router = APIRouter(
    prefix="/api/v1/auth",
    tags=["authentication"],
)


@router.post(
    "/register",
    response_model=UserResponse,
    status_code=status.HTTP_201_CREATED,
)
def register(
    request: RegisterRequest,
    db: Session = Depends(get_db),
) -> User:
    """Register a new application user."""

    normalized_email = request.email.strip().lower()

    existing_user = db.scalar(
        select(User).where(User.email == normalized_email)
    )

    if existing_user is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="A user with this email already exists.",
        )

    user = User(
        name=request.name.strip(),
        email=normalized_email,
        password_hash=hash_password(request.password),
    )

    verification_token = create_email_verification_token(user)

    verification_url = (
        f"{get_verification_base_url()}/api/v1/auth/verify-email?"
        f"{urlencode({'token': verification_token})}"
    )

    email_service.send_verification_email(
        recipient=user.email,
        verification_url=verification_url,
    )

    db.add(user)

    try:
        db.commit()
    except IntegrityError as exc:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="A user with this email already exists.",
        ) from exc

    db.refresh(user)

    return user


@router.post(
    "/login",
    response_model=TokenResponse,
)
def login(
    request: LoginRequest,
    db: Session = Depends(get_db),
) -> TokenResponse:
    """Authenticate a user and return a JWT access token."""

    user = db.scalar(
        select(User).where(User.email == request.email)
    )

    if user is None or not verify_password(
        request.password,
        user.password_hash,
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if not user.email_verified:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Email address must be verified before login.",
        )

    access_token = create_access_token(user.id)

    return TokenResponse(
        access_token=access_token,
    )


@router.get("/verify-email")
def verify_email(
    token: str,
    db: Session = Depends(get_db),
) -> dict[str, str]:
    """Verify a user's email address using a verification token."""

    token_hash = get_verification_token_hash(token)

    user = db.scalar(
        select(User).where(
            User.verification_token_hash == token_hash,
        )
    )

    if user is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid or expired verification token.",
        )

    if not is_verification_token_valid(
        token,
        user.verification_token_hash,
        user.verification_token_expires_at,
    ):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid or expired verification token.",
        )

    user.email_verified = True
    user.verification_token_hash = None
    user.verification_token_expires_at = None

    db.commit()

    return {
        "message": "Email address verified successfully.",
    }