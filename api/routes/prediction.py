"""WasteVision prediction API routes."""

from __future__ import annotations

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from PIL import Image, UnidentifiedImageError
from sqlalchemy import select
from sqlalchemy.orm import Session

from api.dependencies import get_current_user, get_db, get_predictor
from api.schemas import PredictionRecordResponse, PredictionResponse
from database.models import PredictionRecord, User
from wastevision.inference.predictor import (
    MODEL_NAME,
    MODEL_VERSION,
    WastePredictor,
)


router = APIRouter(
    prefix="/api/v1",
    tags=["predictions"],
)


@router.post(
    "/predict",
    response_model=PredictionResponse,
)
def predict(
    image: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
    predictor: WastePredictor = Depends(get_predictor),
    db: Session = Depends(get_db),
) -> PredictionResponse:
    """Classify an uploaded waste image and persist the result."""
    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Uploaded file must be an image.",
        )

    try:
        uploaded_image = Image.open(image.file)
    except (UnidentifiedImageError, OSError) as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is not a valid image.",
        ) from exc

    try:
        prediction, confidence, inference_time_ms = predictor.predict(
            uploaded_image
        )
    finally:
        uploaded_image.close()

    prediction_record = PredictionRecord(
        user_id=current_user.id,
        predicted_class=prediction,
        confidence=confidence,
        model_name=MODEL_NAME,
        model_version=MODEL_VERSION,
        inference_time_ms=inference_time_ms,
    )

    db.add(prediction_record)
    db.commit()
    db.refresh(prediction_record)

    return PredictionResponse(
        prediction_id=prediction_record.id,
        prediction=prediction,
        confidence=confidence,
        model=MODEL_NAME,
        version=MODEL_VERSION,
        inference_time_ms=inference_time_ms,
    )


@router.get(
    "/predictions",
    response_model=list[PredictionRecordResponse],
)
def list_predictions(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> list[PredictionRecord]:
    """List predictions belonging to the authenticated user."""
    statement = (
        select(PredictionRecord)
        .where(PredictionRecord.user_id == current_user.id)
        .order_by(PredictionRecord.created_at.desc())
    )

    return list(db.scalars(statement).all())


@router.get(
    "/predictions/{prediction_id}",
    response_model=PredictionRecordResponse,
)
def get_prediction(
    prediction_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> PredictionRecord:
    """Retrieve one prediction belonging to the authenticated user."""
    statement = select(PredictionRecord).where(
        PredictionRecord.id == prediction_id,
        PredictionRecord.user_id == current_user.id,
    )

    prediction_record = db.scalar(statement)

    if prediction_record is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Prediction not found.",
        )

    return prediction_record