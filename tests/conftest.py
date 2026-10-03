"""Shared pytest fixtures for WasteVision API tests."""

from __future__ import annotations

import os
from collections.abc import Generator

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from api.dependencies import initialize_predictor
from api.main import app
from database.models import Base
from database.session import get_db


TEST_DATABASE_URL = os.getenv(
    "TEST_DATABASE_URL",
    "postgresql+psycopg://wastevision:wastevision@localhost:5432/wastevision_test",
)

test_engine = create_engine(
    TEST_DATABASE_URL,
    pool_pre_ping=True,
)

TestSessionLocal = sessionmaker(
    bind=test_engine,
    class_=Session,
    autoflush=False,
    autocommit=False,
)


@pytest.fixture(scope="session", autouse=True)
def setup_test_database() -> Generator[None, None, None]:
    """Create and remove all test database tables."""

    Base.metadata.create_all(bind=test_engine)

    yield

    Base.metadata.drop_all(bind=test_engine)


@pytest.fixture()
def db_session() -> Generator[Session, None, None]:
    """Provide a database session for an individual test."""

    db = TestSessionLocal()

    try:
        yield db
    finally:
        db.close()


@pytest.fixture()
def client(
    db_session: Session,
) -> Generator[TestClient, None, None]:
    """Provide a FastAPI test client using the test database."""

    def override_get_db() -> Generator[Session, None, None]:
        yield db_session

    app.dependency_overrides[get_db] = override_get_db

    # The application normally initializes the model during startup.
    # Ensure the predictor exists for tests.
    initialize_predictor()

    with TestClient(app) as test_client:
        yield test_client

    app.dependency_overrides.clear()