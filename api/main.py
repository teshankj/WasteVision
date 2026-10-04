"""WasteVision FastAPI application entry point."""

from __future__ import annotations

import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi import Request
from fastapi.responses import Response

from api.dependencies import (
    APP_NAME,
    APP_VERSION,
    initialize_predictor,
)
from api.routes import auth, health, prediction, users

from web import routes as web_routes

from fastapi.staticfiles import StaticFiles

from config.settings import settings


logging.basicConfig(
    level=getattr(
        logging,
        settings.log_level.upper(),
        logging.INFO,
    ),
    format=(
        "%(asctime)s | %(levelname)s | "
        "%(name)s | %(message)s"
    ),
)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncGenerator[None, None]:
    """Manage application startup and shutdown resources."""
    logger.info(
        "Starting %s API version %s",
        APP_NAME,
        APP_VERSION,
    )

    initialize_predictor()

    yield

    logger.info("Shutting down %s API", APP_NAME)


app = FastAPI(
    title=APP_NAME,
    description="Computer vision MLOps API for waste classification.",
    version=APP_VERSION,
    lifespan=lifespan,
)

@app.middleware("http")
async def prevent_authenticated_page_caching(
    request: Request,
    call_next,
) -> Response:
    response = await call_next(request)

    if request.url.path.startswith(
        ("/dashboard", "/predictions")
    ):
        response.headers["Cache-Control"] = "no-store"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"

    return response

app.mount("/static", StaticFiles(directory="static"), name="static")


app.include_router(health.router)
app.include_router(auth.router)
app.include_router(prediction.router)
app.include_router(users.router)
app.include_router(web_routes.router)