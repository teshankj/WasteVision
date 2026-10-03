"""Integration tests for the WasteVision API."""

from __future__ import annotations

import io

from fastapi.testclient import TestClient
from PIL import Image


TEST_EMAIL = "api-test@wastevision.example.com"
TEST_PASSWORD = "WasteVisionTest123!"


def create_test_image() -> io.BytesIO:
    """Create a small valid JPEG image for API tests."""

    image = Image.new(
        "RGB",
        (224, 224),
        color=(128, 128, 128),
    )

    image_bytes = io.BytesIO()
    image.save(image_bytes, format="JPEG")
    image_bytes.seek(0)

    return image_bytes


def register_user(
    client: TestClient,
    name: str,
    email: str,
) -> dict:
    """Register a test user."""

    response = client.post(
        "/api/v1/auth/register",
        json={
            "name": name,
            "email": email,
            "password": TEST_PASSWORD,
        },
    )

    assert response.status_code == 201

    return response.json()


def login_user(
    client: TestClient,
    email: str,
) -> str:
    """Log in a test user and return the JWT."""

    response = client.post(
        "/api/v1/auth/login",
        json={
            "email": email,
            "password": TEST_PASSWORD,
        },
    )

    assert response.status_code == 200

    return response.json()["access_token"]


def test_health(client: TestClient) -> None:
    """Health endpoint should report a healthy service."""

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "healthy",
        "service": "WasteVision",
    }


def test_version(client: TestClient) -> None:
    """Version endpoint should return service information."""

    response = client.get("/version")

    assert response.status_code == 200

    data = response.json()

    assert data["service"] == "WasteVision"
    assert data["version"] == "0.1.0"


def test_register_user(client: TestClient) -> None:
    """A new user should be registered successfully."""

    email = "register-api@wastevision.example.com"

    data = register_user(
        client,
        "API Test User",
        email,
    )

    assert data["name"] == "API Test User"
    assert data["email"] == email
    assert "password_hash" not in data


def test_duplicate_registration_is_rejected(
    client: TestClient,
) -> None:
    """Registering the same email twice should return 409."""

    email = "duplicate-api@wastevision.example.com"

    register_user(
        client,
        "Duplicate Test",
        email,
    )

    response = client.post(
        "/api/v1/auth/register",
        json={
            "name": "Duplicate Test",
            "email": email,
            "password": TEST_PASSWORD,
        },
    )

    assert response.status_code == 409


def test_invalid_login_is_rejected(
    client: TestClient,
) -> None:
    """Invalid credentials should return 401."""

    email = "invalid-login@wastevision.example.com"

    register_user(
        client,
        "Invalid Login",
        email,
    )

    response = client.post(
        "/api/v1/auth/login",
        json={
            "email": email,
            "password": "WrongPassword123!",
        },
    )

    assert response.status_code == 401


def test_authenticated_user_profile(
    client: TestClient,
) -> None:
    """An authenticated user should access their profile."""

    email = "profile-api@wastevision.example.com"

    register_user(
        client,
        "Profile User",
        email,
    )

    token = login_user(client, email)

    response = client.get(
        "/api/v1/users/me",
        headers={
            "Authorization": f"Bearer {token}",
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["name"] == "Profile User"
    assert data["email"] == email


def test_profile_requires_authentication(
    client: TestClient,
) -> None:
    """The profile endpoint should require authentication."""

    response = client.get("/api/v1/users/me")

    assert response.status_code == 401


def test_prediction_requires_authentication(
    client: TestClient,
) -> None:
    """Prediction endpoint should require authentication."""

    image = create_test_image()

    response = client.post(
        "/api/v1/predict",
        files={
            "image": (
                "test.jpg",
                image,
                "image/jpeg",
            )
        },
    )

    assert response.status_code == 401


def test_prediction_and_history(
    client: TestClient,
) -> None:
    """Authenticated prediction should be persisted and retrievable."""

    email = "prediction-api@wastevision.example.com"

    register_user(
        client,
        "Prediction User",
        email,
    )

    token = login_user(client, email)

    headers = {
        "Authorization": f"Bearer {token}",
    }

    image = create_test_image()

    response = client.post(
        "/api/v1/predict",
        headers=headers,
        files={
            "image": (
                "test.jpg",
                image,
                "image/jpeg",
            )
        },
    )

    assert response.status_code == 200

    prediction = response.json()

    assert prediction["prediction"] in {
        "cardboard",
        "glass",
        "metal",
        "paper",
        "plastic",
        "trash",
    }

    assert 0.0 <= prediction["confidence"] <= 1.0
    assert prediction["model"] == "wastevision"
    assert prediction["version"] == "1"
    assert prediction["inference_time_ms"] >= 0.0
    assert prediction["prediction_id"] > 0

    prediction_id = prediction["prediction_id"]

    history_response = client.get(
        "/api/v1/predictions",
        headers=headers,
    )

    assert history_response.status_code == 200

    history = history_response.json()

    assert len(history) == 1
    assert history[0]["id"] == prediction_id

    detail_response = client.get(
        f"/api/v1/predictions/{prediction_id}",
        headers=headers,
    )

    assert detail_response.status_code == 200
    assert detail_response.json()["id"] == prediction_id


def test_prediction_user_isolation(
    client: TestClient,
) -> None:
    """A user must not access another user's prediction."""

    first_email = "isolation-one@wastevision.example.com"
    second_email = "isolation-two@wastevision.example.com"

    register_user(
        client,
        "Isolation User One",
        first_email,
    )

    register_user(
        client,
        "Isolation User Two",
        second_email,
    )

    token_one = login_user(client, first_email)
    token_two = login_user(client, second_email)

    image = create_test_image()

    response = client.post(
        "/api/v1/predict",
        headers={
            "Authorization": f"Bearer {token_one}",
        },
        files={
            "image": (
                "test.jpg",
                image,
                "image/jpeg",
            )
        },
    )

    assert response.status_code == 200

    prediction_id = response.json()["prediction_id"]

    response = client.get(
        f"/api/v1/predictions/{prediction_id}",
        headers={
            "Authorization": f"Bearer {token_two}",
        },
    )

    assert response.status_code == 404

    history_response = client.get(
        "/api/v1/predictions",
        headers={
            "Authorization": f"Bearer {token_two}",
        },
    )

    assert history_response.status_code == 200
    assert history_response.json() == []