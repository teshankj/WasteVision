"""Shared API dependencies and application configuration."""

from __future__ import annotations

import logging

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session

from auth.jwt import decode_access_token
from config.settings import settings
from database.models import User
from database.session import get_db
from wastevision.inference.predictor import WastePredictor


APP_NAME = settings.app_name
APP_VERSION = settings.app_version
MODEL_PATH = settings.model_path

logger = logging.getLogger(__name__)

bearer_scheme = HTTPBearer()

_predictor: WastePredictor | None = None


def initialize_predictor() -> None:
    """Load the WasteVision model into application memory."""
    global _predictor

    if _predictor is not None:
        logger.info("WasteVision predictor is already initialized.")
        return

    logger.info("Loading WasteVision model from %s", MODEL_PATH)

    _predictor = WastePredictor(MODEL_PATH)

    logger.info("WasteVision predictor initialized successfully.")


def get_predictor() -> WastePredictor:
    """Return the initialized WasteVision predictor."""
    if _predictor is None:
        raise RuntimeError(
            "WasteVision predictor has not been initialized."
        )

    return _predictor


def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme),
    db: Session = Depends(get_db),
) -> User:
    """Return the authenticated user from a JWT bearer token."""
    try:
        user_id = decode_access_token(credentials.credentials)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired access token.",
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc

    user = db.get(User, user_id)

    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authenticated user not found.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return user