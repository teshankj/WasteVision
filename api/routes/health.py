"""Health and application readiness endpoints."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, Response, status
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from api.dependencies import (
    APP_NAME,
    APP_VERSION,
    get_db,
    get_predictor,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["health"])


@router.get("/health")
def health_check() -> dict[str, str]:
    """Return the application's liveness status."""

    return {
        "status": "healthy",
        "service": APP_NAME,
    }


@router.get("/ready")
def readiness_check(
    response: Response,
    db: Session = Depends(get_db),
) -> dict[str, str]:
    """Check whether the application is ready to serve requests."""

    try:
        db.execute(text("SELECT 1"))
        get_predictor()

    except (SQLAlchemyError, RuntimeError) as exc:
        logger.warning(
            "Readiness check failed: %s",
            type(exc).__name__,
        )

        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE

        return {
            "status": "not_ready",
            "service": APP_NAME,
        }

    return {
        "status": "ready",
        "service": APP_NAME,
    }


@router.get("/version")
def version() -> dict[str, str]:
    """Return the running application version."""

    return {
        "service": APP_NAME,
        "version": APP_VERSION,
    }