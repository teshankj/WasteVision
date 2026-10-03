"""JWT creation and validation utilities."""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone

import jwt
from jwt import InvalidTokenError


JWT_ALGORITHM = "HS256"
JWT_ACCESS_TOKEN_EXPIRE_MINUTES = 60


def _get_jwt_secret() -> str:
    """Return the JWT signing secret from the environment."""
    secret = os.environ.get("JWT_SECRET_KEY")

    if not secret:
        raise RuntimeError("JWT_SECRET_KEY environment variable is not set.")

    return secret


def create_access_token(user_id: int) -> str:
    """Create a JWT access token for a user."""
    now = datetime.now(timezone.utc)
    expires_at = now + timedelta(minutes=JWT_ACCESS_TOKEN_EXPIRE_MINUTES)

    payload = {
        "sub": str(user_id),
        "iat": now,
        "exp": expires_at,
    }

    return jwt.encode(
        payload,
        _get_jwt_secret(),
        algorithm=JWT_ALGORITHM,
    )


def decode_access_token(token: str) -> int:
    """Decode a JWT and return the authenticated user ID."""
    try:
        payload = jwt.decode(
            token,
            _get_jwt_secret(),
            algorithms=[JWT_ALGORITHM],
        )
    except InvalidTokenError as exc:
        raise ValueError("Invalid or expired access token.") from exc

    subject = payload.get("sub")

    if subject is None:
        raise ValueError("Access token does not contain a user ID.")

    try:
        return int(subject)
    except (TypeError, ValueError) as exc:
        raise ValueError("Invalid user ID in access token.") from exc