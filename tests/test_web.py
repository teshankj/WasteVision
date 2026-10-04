"""Integration tests for the WasteVision browser UI."""

from __future__ import annotations

import io
from urllib.parse import parse_qs, urlparse

from fastapi.testclient import TestClient
from PIL import Image

from services.email import email_service


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


def verify_web_user(
    client: TestClient,
    email: str,
) -> None:
    """Verify a browser user's email using the development email service."""

    verification_email = email_service.get_verification_email(email)

    assert verification_email is not None

    parsed_url = urlparse(verification_email.verification_url)
    token_values = parse_qs(parsed_url.query).get("token")

    assert token_values
    token = token_values[0]

    response = client.get(
        "/verify-email",
        params={"token": token},
    )

    assert response.status_code == 200


def login_web_user(
    client: TestClient,
    email: str,
) -> None:
    """Verify and log in a browser user."""

    verify_web_user(client, email)

    client.cookies.clear()

    response = client.post(
        "/login",
        data={
            "email": email,
            "password": TEST_PASSWORD,
        },
        follow_redirects=False,
    )

    assert response.status_code == 303
    assert response.headers["location"] == "/dashboard"
    assert "access_token" in response.cookies


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
    """Browser registration should require email verification."""

    email = "web-register@wastevision.example.com"

    response = client.post(
        "/register",
        data={
            "name": "Web Test User",
            "email": email,
            "password": TEST_PASSWORD,
        },
        follow_redirects=False,
    )

    assert response.status_code == 303
    assert response.headers["location"] == "/registration-success"
    assert "access_token" not in response.cookies

    verification_email = email_service.get_verification_email(email)

    assert verification_email is not None
    assert verification_email.recipient == email


def test_web_login(
    client: TestClient,
) -> None:
    """Browser login should work after email verification."""

    email = "web-login@wastevision.example.com"

    client.post(
        "/register",
        data={
            "name": "Web Login User",
            "email": email,
            "password": TEST_PASSWORD,
        },
    )

    login_web_user(client, email)


def test_invalid_web_login(
    client: TestClient,
) -> None:
    """Invalid browser credentials should return an error page."""

    email = "web-invalid@wastevision.example.com"

    client.post(
        "/register",
        data={
            "name": "Invalid Web Login",
            "email": email,
            "password": TEST_PASSWORD,
        },
    )

    verify_web_user(client, email)
    client.cookies.clear()

    response = client.post(
        "/login",
        data={
            "email": email,
            "password": "WrongPassword123!",
        },
    )

    assert response.status_code == 401
    assert "Invalid email or password" in response.text


def test_unverified_web_login(
    client: TestClient,
) -> None:
    """Unverified browser users should not be allowed to log in."""

    email = "web-unverified@wastevision.example.com"

    client.post(
        "/register",
        data={
            "name": "Unverified Web User",
            "email": email,
            "password": TEST_PASSWORD,
        },
    )

    client.cookies.clear()

    response = client.post(
        "/login",
        data={
            "email": email,
            "password": TEST_PASSWORD,
        },
        follow_redirects=False,
    )

    assert response.status_code == 403
    assert "verify your email" in response.text.lower()


def test_authenticated_dashboard(
    client: TestClient,
) -> None:
    """Authenticated users should access the dashboard."""

    email = "web-dashboard@wastevision.example.com"

    client.post(
        "/register",
        data={
            "name": "Dashboard User",
            "email": email,
            "password": TEST_PASSWORD,
        },
    )

    login_web_user(client, email)

    response = client.get("/dashboard")

    assert response.status_code == 200
    assert "WasteVision Dashboard" in response.text
    assert "Dashboard User" in response.text


def test_logout(
    client: TestClient,
) -> None:
    """Logout should invalidate browser access."""

    email = "web-logout@wastevision.example.com"

    client.post(
        "/register",
        data={
            "name": "Logout User",
            "email": email,
            "password": TEST_PASSWORD,
        },
    )

    login_web_user(client, email)

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

    email = "web-prediction@wastevision.example.com"

    client.post(
        "/register",
        data={
            "name": "Web Prediction User",
            "email": email,
            "password": TEST_PASSWORD,
        },
    )

    login_web_user(client, email)

    image = create_test_image()

    response = client.post(
        "/dashboard/predict",
        files={
            "image": (
                "test.jpg",
                image,
                "image/jpeg",
            ),
        },
        follow_redirects=False,
    )

    assert response.status_code == 303

    location = response.headers["location"]

    assert location.startswith("/predictions/")

    result_response = client.get(location)

    assert result_response.status_code == 200
    assert "Prediction Result" in result_response.text