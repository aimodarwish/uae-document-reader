"""Accuracy-first, privacy-conscious identity-document extraction POC."""

from .config import AppConfig, RuntimeInfo, detect_runtime, select_vlm
from .pipeline import DocumentReader, ProcessingSession
from .schemas import CustomerType, ExtractionResult, FieldCandidate, FieldStatus

__all__ = [
    "AppConfig",
    "CustomerType",
    "DocumentReader",
    "ExtractionResult",
    "FieldCandidate",
    "FieldStatus",
    "ProcessingSession",
    "RuntimeInfo",
    "detect_runtime",
    "select_vlm",
]

