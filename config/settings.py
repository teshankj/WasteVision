"""Centralized application settings for WasteVision."""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    """Application configuration loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

    app_name: str = "WasteVision"
    app_version: str = "0.1.0"
    log_level: str = "INFO"

    database_url: str = Field(
        ...,
        validation_alias="DATABASE_URL",
    )

    jwt_secret_key: str = Field(
        ...,
        min_length=32,
        validation_alias="JWT_SECRET_KEY",
    )

    email_backend: str = Field(
        default="development",
        validation_alias="EMAIL_BACKEND",
    )

    email_verification_base_url: str = Field(
        default="http://127.0.0.1:8000",
        validation_alias="EMAIL_VERIFICATION_BASE_URL",
    )

    smtp_host: str | None = Field(
        default=None,
        validation_alias="SMTP_HOST",
    )

    smtp_port: int = Field(
        default=587,
        validation_alias="SMTP_PORT",
    )

    smtp_username: str | None = Field(
        default=None,
        validation_alias="SMTP_USERNAME",
    )

    smtp_password: str | None = Field(
        default=None,
        validation_alias="SMTP_PASSWORD",
    )

    smtp_use_tls: bool = Field(
        default=True,
        validation_alias="SMTP_USE_TLS",
    )

    email_from: str | None = Field(
        default=None,
        validation_alias="EMAIL_FROM",
    )

    web_cookie_secure: bool = Field(
        default=False,
        validation_alias="WEB_COOKIE_SECURE",
    )

    model_path: Path = Field(
        default=(
            PROJECT_ROOT
            / "artifacts"
            / "mlflow_baseline"
            / "best_model.pt"
        ),
        validation_alias="MODEL_PATH",
    )


settings = Settings()