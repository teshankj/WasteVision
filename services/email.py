"""Email service for WasteVision."""

from __future__ import annotations

import logging
import smtplib
from dataclasses import dataclass
from email.message import EmailMessage

from config.settings import settings


logger = logging.getLogger(__name__)

DEFAULT_VERIFICATION_BASE_URL = "http://127.0.0.1:8000"


@dataclass(frozen=True)
class VerificationEmail:
    """Captured verification email."""

    recipient: str
    verification_url: str


class EmailService:
    """Email service supporting development and SMTP backends."""

    def __init__(self) -> None:
        self.backend = settings.email_backend

        self._verification_emails: dict[
            str,
            VerificationEmail,
        ] = {}

        self._last_verification_email: (
            VerificationEmail | None
        ) = None

    def send_verification_email(
        self,
        recipient: str,
        verification_url: str,
    ) -> None:
        """Send a verification email using the configured backend."""

        if self.backend == "development":
            self._capture_verification_email(
                recipient,
                verification_url,
            )
            return

        if self.backend == "smtp":
            self._send_smtp_verification_email(
                recipient,
                verification_url,
            )
            return

        raise RuntimeError(
            f"Unsupported email backend: {self.backend}"
        )

    def _capture_verification_email(
        self,
        recipient: str,
        verification_url: str,
    ) -> None:
        """Capture a verification email for development and tests."""

        verification_email = VerificationEmail(
            recipient=recipient,
            verification_url=verification_url,
        )

        self._last_verification_email = verification_email
        self._verification_emails[recipient] = verification_email

        logger.info(
            "Verification email generated for %s.",
            recipient,
        )

    def _send_smtp_verification_email(
        self,
        recipient: str,
        verification_url: str,
    ) -> None:
        """Send a verification email through SMTP."""

        self._validate_smtp_configuration()

        message = EmailMessage()

        message["Subject"] = "Verify your WasteVision account"
        message["From"] = settings.email_from
        message["To"] = recipient

        message.set_content(
            "Welcome to WasteVision.\n\n"
            "Please verify your email address by opening "
            "the following link:\n\n"
            f"{verification_url}\n\n"
            "This verification link expires in 24 hours.\n\n"
            "If you did not create this account, you can "
            "safely ignore this email."
        )

        try:
            with smtplib.SMTP(
                settings.smtp_host,
                settings.smtp_port,
                timeout=30,
            ) as smtp:
                smtp.ehlo()

                if settings.smtp_use_tls:
                    smtp.starttls()
                    smtp.ehlo()

                if (
                    settings.smtp_username
                    and settings.smtp_password
                ):
                    smtp.login(
                        settings.smtp_username,
                        settings.smtp_password,
                    )

                smtp.send_message(message)

        except (OSError, smtplib.SMTPException) as exc:
            logger.exception(
                "Failed to send verification email to %s.",
                recipient,
            )
            raise RuntimeError(
                "Failed to send verification email."
            ) from exc

        logger.info(
            "Verification email sent to %s.",
            recipient,
        )

    @staticmethod
    def _validate_smtp_configuration() -> None:
        """Validate required SMTP configuration."""

        missing: list[str] = []

        if not settings.smtp_host:
            missing.append("SMTP_HOST")

        if not settings.email_from:
            missing.append("EMAIL_FROM")

        if missing:
            raise RuntimeError(
                "Missing SMTP configuration: "
                + ", ".join(missing)
            )

        if settings.smtp_port <= 0:
            raise RuntimeError(
                "SMTP_PORT must be greater than zero."
            )

    def get_verification_email(
        self,
        recipient: str,
    ) -> VerificationEmail | None:
        """Return the captured development email."""

        return self._verification_emails.get(recipient)

    def get_last_verification_email(
        self,
    ) -> VerificationEmail | None:
        """Return the latest captured development email."""

        return self._last_verification_email

    def clear_last_verification_email(self) -> None:
        """Clear the captured development email."""

        self._last_verification_email = None


def get_verification_base_url() -> str:
    """Return the base URL used in verification links."""

    return settings.email_verification_base_url.rstrip("/")


email_service = EmailService()