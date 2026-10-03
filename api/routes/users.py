"""User-related API routes."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from api.dependencies import get_current_user
from auth.schemas import UserResponse
from database.models import User


router = APIRouter(
    prefix="/api/v1/users",
    tags=["users"],
)


@router.get(
    "/me",
    response_model=UserResponse,
)
def get_current_user_profile(
    current_user: User = Depends(get_current_user),
) -> User:
    """Return the profile of the authenticated user."""
    return current_user