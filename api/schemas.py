"""Pydantic schemas for the WasteVision API."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class PredictionResponse(BaseModel):
    """Response returned after an image prediction."""

    prediction_id: int
    prediction: str
    confidence: float = Field(ge=0.0, le=1.0)
    model: str
    version: str
    inference_time_ms: float = Field(ge=0.0)


class PredictionRecordResponse(BaseModel):
    """Stored prediction record returned by the API."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    predicted_class: str
    confidence: float = Field(ge=0.0, le=1.0)
    model_name: str
    model_version: str
    inference_time_ms: float = Field(ge=0.0)
    created_at: datetime