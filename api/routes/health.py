"""Health and application version endpoints."""

from __future__ import annotations

from fastapi import APIRouter

from api.dependencies import APP_NAME, APP_VERSION

router = APIRouter(tags=["health"])


@router.get("/health")
def health_check() -> dict[str, str]:
    """Return the application's health status."""
    return {
        "status": "healthy",
        "service": APP_NAME,
    }


@router.get("/version")
def version() -> dict[str, str]:
    """Return the running application version."""
    return {
        "service": APP_NAME,
        "version": APP_VERSION,
    }