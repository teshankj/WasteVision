"""Tests for WasteVision authentication utilities."""

from __future__ import annotations

import pytest

from auth.hashing import hash_password, verify_password
from auth.jwt import create_access_token, decode_access_token


def test_password_hash_is_not_plaintext() -> None:
    """Passwords should be stored as hashes."""
    password = "WasteVisionTest123!"

    hashed_password = hash_password(password)

    assert hashed_password != password
    assert len(hashed_password) > 0


def test_password_verification_succeeds() -> None:
    """The original password should verify against its hash."""
    password = "WasteVisionTest123!"

    hashed_password = hash_password(password)

    assert verify_password(password, hashed_password)


def test_password_verification_fails_for_wrong_password() -> None:
    """An incorrect password should not verify."""
    password = "WasteVisionTest123!"
    wrong_password = "WrongPassword123!"

    hashed_password = hash_password(password)

    assert not verify_password(wrong_password, hashed_password)


def test_jwt_round_trip(monkeypatch: pytest.MonkeyPatch) -> None:
    """A created JWT should decode to the original user ID."""

    monkeypatch.setenv(
        "JWT_SECRET_KEY",
        "test-secret-key-for-wastevision-32b",
    )

    user_id = 123

    token = create_access_token(user_id)
    decoded_user_id = decode_access_token(token)

    assert decoded_user_id == user_id


def test_invalid_jwt_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An invalid JWT should raise ValueError."""

    monkeypatch.setenv(
        "JWT_SECRET_KEY",
        "test-secret-key-for-wastevision-32b",
    )

    with pytest.raises(ValueError, match="Invalid or expired access token"):
        decode_access_token("invalid-token")


def test_jwt_requires_secret_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """JWT creation should fail when the secret is missing."""

    monkeypatch.delenv("JWT_SECRET_KEY", raising=False)

    with pytest.raises(
        RuntimeError,
        match="JWT_SECRET_KEY environment variable is not set",
    ):
        create_access_token(1)