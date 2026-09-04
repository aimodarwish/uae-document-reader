from typing import Any, Optional

from pydantic import BaseModel, Field


class MulkiyaData(BaseModel):
    plate_source: Optional[str] = None
    plate_category: Optional[str] = None
    plate_code: Optional[str] = None
    plate_number: Optional[str] = None
    vin: Optional[str] = None

    make: Optional[str] = None
    model: Optional[str] = None
    year: Optional[int] = None
    color: Optional[str] = None

    insurance_company: Optional[str] = None
    policy_number: Optional[str] = None
    insurance_expiry: Optional[str] = None

    registration_expiry: Optional[str] = None
    registration_issuance: Optional[str] = None


class ExtractResponse(BaseModel):
    success: bool
    filename: str
    processing_time_ms: int
    data: MulkiyaData = Field(default_factory=MulkiyaData)

    # Per-field OCR confidence, discounted when a value was found by scanning the
    # whole card instead of next to its own label.
    confidence: dict[str, Optional[float]] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)

    # Set only when this file failed; the batch endpoint uses it so one bad
    # upload cannot discard the other results.
    error: Optional[str] = None

    # Debug only (INCLUDE_RAW_OCR=true).
    raw_ocr: Optional[list[dict[str, Any]]] = None
    timings: Optional[dict[str, float]] = None


class BatchResponse(BaseModel):
    success: bool
    total: int
    succeeded: int
    failed: int
    processing_time_ms: int
    results: list[ExtractResponse]
