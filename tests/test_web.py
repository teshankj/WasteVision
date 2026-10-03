"""Integration tests for the WasteVision browser UI."""

from __future__ import annotations

import io

from fastapi.testclient import TestClient
from PIL import Image


TEST_PASSWORD = "WasteVisionTest123!"


def create_test_image() -> io.BytesIO:
    """Create a small valid JPEG image."""

    image = Image.new(
        "RGB",
        (224, 224),
        color=(128, 128, 128),
    )

    image_bytes = io.BytesIO()
    image.save(image_bytes, format="JPEG")
    image_bytes.seek(0)

    return image_bytes


def test_landing_page(client: TestClient) -> None:
    """Landing page should render successfully."""

    response = client.get("/")

    assert response.status_code == 200
    assert "WasteVision" in response.text
    assert "Login" in response.text
    assert "Register" in response.text


def test_login_page(client: TestClient) -> None:
    """Login page should render successfully."""

    response = client.get("/login")

    assert response.status_code == 200
    assert "Login" in response.text


def test_register_page(client: TestClient) -> None:
    """Registration page should render successfully."""

    response = client.get("/register")

    assert response.status_code == 200
    assert "Create Account" in response.text


def test_dashboard_requires_authentication(
    client: TestClient,
) -> None:
    """Unauthenticated users should be redirected to login."""

    response = client.get(
        "/dashboard",
        follow_redirects=False,
    )

    assert response.status_code == 303
    assert response.headers["location"] == "/login"


def test_web_registration(
    client: TestClient,
) -> None:
    """Browser registration should create an authenticated session."""

    response = client.post(
        "/register",
        data={
            "name": "Web Test User",
            "email": "web-register@wastevision.example.com",
            "password": TEST_PASSWORD,
        },
        follow_redirects=False,
    )

    assert response.status_code == 303
    assert response.headers["location"] == "/dashboard"
    assert "access_token" in response.cookies


def test_web_login(
    client: TestClient,
) -> None:
    """Browser login should create an authenticated session."""

    client.post(
        "/register",
        data={
            "name": "Web Login User",
            "email": "web-login@wastevision.example.com",
            "password": TEST_PASSWORD,
        },
    )

    client.cookies.clear()

    response = client.post(
        "/login",
        data={
            "email": "web-login@wastevision.example.com",
            "password": TEST_PASSWORD,
        },
        follow_redirects=False,
    )

    assert response.status_code == 303
    assert response.headers["location"] == "/dashboard"
    assert "access_token" in response.cookies


def test_invalid_web_login(
    client: TestClient,
) -> None:
    """Invalid browser credentials should return an error page."""

    client.post(
        "/register",
        data={
            "name": "Invalid Web Login",
            "email": "web-invalid@wastevision.example.com",
            "password": TEST_PASSWORD,
        },
    )

    client.cookies.clear()

    response = client.post(
        "/login",
        data={
            "email": "web-invalid@wastevision.example.com",
            "password": "WrongPassword123!",
        },
    )

    assert response.status_code == 401
    assert "Invalid email or password" in response.text


def test_authenticated_dashboard(
    client: TestClient,
) -> None:
    """Authenticated users should access the dashboard."""

    client.post(
        "/register",
        data={
            "name": "Dashboard User",
            "email": "web-dashboard@wastevision.example.com",
            "password": TEST_PASSWORD,
        },
    )

    response = client.get("/dashboard")

    assert response.status_code == 200
    assert "WasteVision Dashboard" in response.text
    assert "Dashboard User" in response.text


def test_logout(
    client: TestClient,
) -> None:
    """Logout should invalidate browser access."""

    client.post(
        "/register",
        data={
            "name": "Logout User",
            "email": "web-logout@wastevision.example.com",
            "password": TEST_PASSWORD,
        },
    )

    response = client.post(
        "/logout",
        follow_redirects=False,
    )

    assert response.status_code == 303
    assert response.headers["location"] == "/"

    dashboard_response = client.get(
        "/dashboard",
        follow_redirects=False,
    )

    assert dashboard_response.status_code == 303
    assert dashboard_response.headers["location"] == "/login"


def test_web_prediction(
    client: TestClient,
) -> None:
    """Authenticated browser users should be able to classify images."""

    client.post(
        "/register",
        data={
            "name": "Web Prediction User",
            "email": "web-prediction@wastevision.example.com",
            "password": TEST_PASSWORD,
        },
    )

    image = create_test_image()

    response = client.post(
        "/dashboard/predict",
        files={
            "image": (
                "test.jpg",
                image,
                "image/jpeg",
            )
        },
        follow_redirects=False,
    )

    assert response.status_code == 303

    location = response.headers["location"]

    assert location.startswith("/predictions/")

    result_response = client.get(location)

    assert result_response.status_code == 200
    assert "Prediction Result" in result_response.text