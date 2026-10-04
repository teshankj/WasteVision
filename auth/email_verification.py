"""Email verification token utilities."""

from __future__ import annotations

import hashlib
import secrets
from datetime import datetime, timedelta, timezone

from database.models import User


VERIFICATION_TOKEN_EXPIRE_HOURS = 24
VERIFICATION_TOKEN_BYTES = 32


def generate_verification_token() -> str:
    """Generate a cryptographically secure email verification token."""
    return secrets.token_urlsafe(VERIFICATION_TOKEN_BYTES)


def hash_verification_token(token: str) -> str:
    """Hash a verification token before storing it."""
    return hashlib.sha256(token.encode("utf-8")).hexdigest()

def get_verification_token_hash(token: str) -> str:
    """Return the hash used for database lookup."""
    return hash_verification_token(token)


def get_verification_expiry() -> datetime:
    """Return the expiration time for a new verification token."""
    return datetime.now(timezone.utc) + timedelta(
        hours=VERIFICATION_TOKEN_EXPIRE_HOURS,
    )


def is_verification_token_valid(
    token: str,
    stored_token_hash: str | None,
    expires_at: datetime | None,
) -> bool:
    """Check whether a verification token is valid and unexpired."""
    if stored_token_hash is None or expires_at is None:
        return False

    if datetime.now(timezone.utc) >= expires_at:
        return False

    supplied_token_hash = hash_verification_token(token)

    return secrets.compare_digest(
        supplied_token_hash,
        stored_token_hash,
    )


def create_email_verification_token(user: User) -> str:
    """Create and store a new verification token for a user."""
    token = generate_verification_token()

    user.verification_token_hash = hash_verification_token(token)
    user.verification_token_expires_at = get_verification_expiry()

    return token

