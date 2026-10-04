"""Dependencies for the WasteVision browser UI."""

from __future__ import annotations

from config.settings import settings

from fastapi import Depends, HTTPException, Request, status
from sqlalchemy.orm import Session

from auth.jwt import decode_access_token
from database.models import User
from database.session import get_db

ACCESS_TOKEN_COOKIE = "access_token"

MAX_UPLOAD_SIZE_BYTES = 5 * 1024 * 1024

WEB_COOKIE_SECURE = settings.web_cookie_secure


def get_current_web_user(
    request: Request,
    db: Session = Depends(get_db),
) -> User:
    """Return the authenticated browser user from the JWT cookie."""

    token = request.cookies.get(ACCESS_TOKEN_COOKIE)

    if not token:
        raise HTTPException(
            status_code=status.HTTP_303_SEE_OTHER,
            headers={"Location": "/login"},
        )

    try:
        user_id = decode_access_token(token)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_303_SEE_OTHER,
            headers={"Location": "/login"},
        ) from exc

    user = db.get(User, user_id)

    if user is None:
        raise HTTPException(
            status_code=status.HTTP_303_SEE_OTHER,
            headers={"Location": "/login"},
        )

    return user


def get_optional_web_user(
    request: Request,
    db: Session = Depends(get_db),
) -> User | None:
    """Return the authenticated browser user, if one exists."""

    token = request.cookies.get(ACCESS_TOKEN_COOKIE)

    if not token:
        return None

    try:
        user_id = decode_access_token(token)
    except ValueError:
        return None

    return db.get(User, user_id)