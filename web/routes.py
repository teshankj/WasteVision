"""Jinja2 browser routes for WasteVision."""

from __future__ import annotations

from io import BytesIO
from urllib.parse import urlencode

from fastapi import APIRouter, Depends, File, Form, Request, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from PIL import Image, UnidentifiedImageError
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from api.dependencies import get_predictor
from auth.email_verification import (
    create_email_verification_token,
    get_verification_token_hash,
    is_verification_token_valid,
)
from auth.hashing import hash_password, verify_password
from auth.jwt import create_access_token
from database.models import PredictionRecord, User
from database.session import get_db
from services.email import email_service, get_verification_base_url
from wastevision.inference.predictor import (
    MODEL_NAME,
    MODEL_VERSION,
    WastePredictor,
)
from web.dependencies import (
    ACCESS_TOKEN_COOKIE,
    MAX_UPLOAD_SIZE_BYTES,
    WEB_COOKIE_SECURE,
    get_current_web_user,
    get_optional_web_user,
)

router = APIRouter(tags=["web"])

templates = Jinja2Templates(directory="templates")


@router.get("/", response_class=HTMLResponse)
def index(
    request: Request,
    current_user: User | None = Depends(get_optional_web_user),
) -> HTMLResponse:
    """Render the landing page."""

    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "user": current_user,
        },
    )


@router.get(
    "/login",
    response_class=HTMLResponse,
    response_model=None,
)
def login_page(
    request: Request,
    current_user: User | None = Depends(get_optional_web_user),
) -> HTMLResponse | RedirectResponse:
    """Render the login page."""

    if current_user is not None:
        return RedirectResponse(
            url="/dashboard",
            status_code=303,
        )

    return templates.TemplateResponse(
        request=request,
        name="login.html",
    )


@router.post("/login", response_model=None)
def login(
    request: Request,
    email: str = Form(...),
    password: str = Form(...),
    db: Session = Depends(get_db),
) -> RedirectResponse | HTMLResponse:
    """Authenticate a browser user."""

    email = email.strip().lower()

    user = db.scalar(
        select(User).where(User.email == email)
    )

    if user is None or not verify_password(
        password,
        user.password_hash,
    ):
        return templates.TemplateResponse(
            request=request,
            name="login.html",
            context={
                "error": "Invalid email or password.",
                "email": email,
            },
            status_code=401,
        )

    if not user.email_verified:
        return templates.TemplateResponse(
            request=request,
            name="login.html",
            context={
                "error": (
                    "Please verify your email address before logging in."
                ),
                "email": email,
            },
            status_code=403,
        )

    token = create_access_token(user.id)

    response = RedirectResponse(
        url="/dashboard",
        status_code=303,
    )

    response.set_cookie(
        key=ACCESS_TOKEN_COOKIE,
        value=token,
        httponly=True,
        samesite="lax",
        secure=WEB_COOKIE_SECURE,
        max_age=60 * 60,
    )

    return response


@router.get(
    "/register",
    response_class=HTMLResponse,
    response_model=None,
)
def register_page(
    request: Request,
    current_user: User | None = Depends(get_optional_web_user),
) -> HTMLResponse | RedirectResponse:
    """Render the registration page."""

    if current_user is not None:
        return RedirectResponse(
            url="/dashboard",
            status_code=303,
        )

    return templates.TemplateResponse(
        request=request,
        name="register.html",
    )


@router.post("/register", response_model=None)
def register(
    request: Request,
    name: str = Form(...),
    email: str = Form(...),
    password: str = Form(...),
    db: Session = Depends(get_db),
) -> RedirectResponse | HTMLResponse:
    """Register a browser user."""

    name = name.strip()
    email = email.strip().lower()

    if not name:
        return templates.TemplateResponse(
            request=request,
            name="register.html",
            context={"error": "Name is required."},
            status_code=400,
        )

    if len(password) < 8:
        return templates.TemplateResponse(
            request=request,
            name="register.html",
            context={
                "error": "Password must contain at least 8 characters.",
                "name": name,
                "email": email,
            },
            status_code=400,
        )

    existing_user = db.scalar(
        select(User).where(User.email == email)
    )

    if existing_user is not None:
        return templates.TemplateResponse(
            request=request,
            name="register.html",
            context={
                "error": "A user with this email already exists.",
                "name": name,
                "email": email,
            },
            status_code=409,
        )

    user = User(
        name=name,
        email=email,
        password_hash=hash_password(password),
    )

    verification_token = create_email_verification_token(user)

    verification_url = (
        f"{get_verification_base_url()}/verify-email?"
        f"{urlencode({'token': verification_token})}"
    )

    email_service.send_verification_email(
        recipient=user.email,
        verification_url=verification_url,
    )

    db.add(user)

    try:
        db.commit()
    except IntegrityError:
        db.rollback()

        return templates.TemplateResponse(
            request=request,
            name="register.html",
            context={
                "error": "A user with this email already exists.",
                "name": name,
                "email": email,
            },
            status_code=409,
        )

    db.refresh(user)

    return RedirectResponse(
        url="/registration-success",
        status_code=303,
    )


@router.get(
    "/registration-success",
    response_class=HTMLResponse,
)
def registration_success(
    request: Request,
) -> HTMLResponse:
    """Show the email verification instruction page."""

    return templates.TemplateResponse(
        request=request,
        name="registration_success.html",
    )


@router.get(
    "/verify-email",
    response_class=HTMLResponse,
)
def verify_email(
    request: Request,
    token: str,
    db: Session = Depends(get_db),
) -> HTMLResponse:
    """Verify a browser user's email address."""

    token_hash = get_verification_token_hash(token)

    user = db.scalar(
        select(User).where(
            User.verification_token_hash == token_hash,
        )
    )

    if user is None:
        return templates.TemplateResponse(
            request=request,
            name="verification_result.html",
            context={
                "success": False,
                "message": "Invalid or expired verification link.",
            },
            status_code=400,
        )

    if not is_verification_token_valid(
        token,
        user.verification_token_hash,
        user.verification_token_expires_at,
    ):
        return templates.TemplateResponse(
            request=request,
            name="verification_result.html",
            context={
                "success": False,
                "message": "Invalid or expired verification link.",
            },
            status_code=400,
        )

    user.email_verified = True
    user.verification_token_hash = None
    user.verification_token_expires_at = None

    db.commit()

    return templates.TemplateResponse(
        request=request,
        name="verification_result.html",
        context={
            "success": True,
            "message": "Your email has been verified successfully.",
        },
    )


@router.post("/logout")
def logout() -> RedirectResponse:
    """Log out the browser user."""

    response = RedirectResponse(
        url="/",
        status_code=303,
    )

    response.delete_cookie(
        key=ACCESS_TOKEN_COOKIE,
    )

    return response


@router.get("/dashboard", response_class=HTMLResponse)
def dashboard(
    request: Request,
    current_user: User = Depends(get_current_web_user),
    db: Session = Depends(get_db),
) -> HTMLResponse:
    """Render the authenticated user's dashboard."""

    statement = (
        select(PredictionRecord)
        .where(PredictionRecord.user_id == current_user.id)
        .order_by(PredictionRecord.created_at.desc())
    )

    predictions = list(db.scalars(statement).all())

    total_predictions = len(predictions)

    average_confidence = (
        sum(prediction.confidence for prediction in predictions)
        / total_predictions
        if total_predictions
        else 0.0
    )

    latest_prediction = predictions[0] if predictions else None

    return templates.TemplateResponse(
        request=request,
        name="dashboard.html",
        context={
            "user": current_user,
            "predictions": predictions,
            "total_predictions": total_predictions,
            "average_confidence": average_confidence,
            "latest_prediction": latest_prediction,
        },
    )


@router.post("/dashboard/predict", response_model=None)
def dashboard_predict(
    request: Request,
    image: UploadFile = File(...),
    current_user: User = Depends(get_current_web_user),
    predictor: WastePredictor = Depends(get_predictor),
    db: Session = Depends(get_db),
) -> RedirectResponse | HTMLResponse:
    """Run an image prediction from the browser dashboard."""

    if not image.content_type or not image.content_type.startswith("image/"):
        return templates.TemplateResponse(
            request=request,
            name="dashboard.html",
            context={
                "user": current_user,
                "predictions": [],
                "error": "Uploaded file must be an image.",
            },
            status_code=415,
        )

    image_bytes = image.file.read()

    if len(image_bytes) > MAX_UPLOAD_SIZE_BYTES:
        return templates.TemplateResponse(
            request=request,
            name="dashboard.html",
            context={
                "user": current_user,
                "predictions": [],
                "error": "Uploaded image is too large.",
            },
            status_code=413,
        )

    try:
        image_object = Image.open(BytesIO(image_bytes)).convert("RGB")
    except UnidentifiedImageError:
        return templates.TemplateResponse(
            request=request,
            name="dashboard.html",
            context={
                "user": current_user,
                "predictions": [],
                "error": "Uploaded file is not a valid image.",
            },
            status_code=415,
        )

    predicted_class, confidence, inference_time_ms = predictor.predict(
        image_object
    )

    prediction = PredictionRecord(
        user_id=current_user.id,
        predicted_class=predicted_class,
        confidence=confidence,
        model_name=MODEL_NAME,
        model_version=MODEL_VERSION,
        inference_time_ms=inference_time_ms,
    )

    db.add(prediction)
    db.commit()
    db.refresh(prediction)

    return RedirectResponse(
        url=f"/predictions/{prediction.id}",
        status_code=303,
    )


@router.get(
    "/predictions/{prediction_id}",
    response_class=HTMLResponse,
)
def prediction_detail(
    request: Request,
    prediction_id: int,
    current_user: User = Depends(get_current_web_user),
    db: Session = Depends(get_db),
) -> HTMLResponse:
    """Render a single prediction result."""

    prediction = db.scalar(
        select(PredictionRecord).where(
            PredictionRecord.id == prediction_id,
            PredictionRecord.user_id == current_user.id,
        )
    )

    if prediction is None:
        return templates.TemplateResponse(
            request=request,
            name="404.html",
            context={
                "user": current_user,
            },
            status_code=404,
        )

    return templates.TemplateResponse(
        request=request,
        name="prediction.html",
        context={
            "user": current_user,
            "prediction": prediction,
        },
    )