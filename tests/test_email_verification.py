"""Tests for email verification token utilities."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from auth.email_verification import (
    generate_verification_token,
    get_verification_expiry,
    hash_verification_token,
    is_verification_token_valid,
)


def test_verification_token_is_random() -> None:
    """Verification tokens should be non-empty and unpredictable."""
    first_token = generate_verification_token()
    second_token = generate_verification_token()

    assert first_token
    assert second_token
    assert first_token != second_token


def test_verification_token_hash_is_not_plaintext() -> None:
    """The stored token representation should not equal the raw token."""
    token = generate_verification_token()

    hashed_token = hash_verification_token(token)

    assert hashed_token != token
    assert len(hashed_token) == 64


def test_valid_verification_token() -> None:
    """A matching unexpired token should be accepted."""
    token = generate_verification_token()
    token_hash = hash_verification_token(token)
    expires_at = get_verification_expiry()

    assert is_verification_token_valid(
        token,
        token_hash,
        expires_at,
    )


def test_wrong_verification_token_is_rejected() -> None:
    """A different token should be rejected."""
    token = generate_verification_token()
    wrong_token = generate_verification_token()

    token_hash = hash_verification_token(token)
    expires_at = get_verification_expiry()

    assert not is_verification_token_valid(
        wrong_token,
        token_hash,
        expires_at,
    )


def test_expired_verification_token_is_rejected() -> None:
    """An expired token should be rejected."""
    token = generate_verification_token()
    token_hash = hash_verification_token(token)

    expires_at = datetime.now(timezone.utc) - timedelta(
        minutes=1,
    )

    assert not is_verification_token_valid(
        token,
        token_hash,
        expires_at,
    )


def test_missing_token_data_is_rejected() -> None:
    """Missing token information should be rejected."""
    token = generate_verification_token()

    assert not is_verification_token_valid(
        token,
        None,
        None,
    )