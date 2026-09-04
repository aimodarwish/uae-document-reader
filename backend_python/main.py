from __future__ import annotations

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.formparsers import MultiPartParser

from .config import settings
from .extractor import extract_fields
from .image_utils import decode_upload
from .models import BatchResponse, ExtractResponse
from .ocr_engine import get_ocr_engine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"

# PRIVACY: Starlette spools an upload to a real file on disk once it exceeds
# 1 MB, and most Mulkiya photos do. Raising the threshold above our own upload
# limit keeps every accepted document in RAM for its whole life, so nothing is
# written to disk even temporarily.
MultiPartParser.spool_max_size = settings.max_upload_bytes + (1024 * 1024)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the OCR models once, at startup, so the first request does not pay
    # for it and no request ever reloads them.
    await asyncio.to_thread(get_ocr_engine)
    logger.info("%s ready on :8000", settings.app_name)
    yield


app = FastAPI(
    title=settings.app_name,
    version="0.2.0",
    description="Local/AWS OCR API for UAE Vehicle Licences (Mulkiya).",
    lifespan=lifespan,
)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", response_class=HTMLResponse)
def home() -> HTMLResponse:
    return HTMLResponse((STATIC_DIR / "index.html").read_text(encoding="utf-8"))


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "service": settings.app_name,
        "ocr_languages": settings.languages,
        "debug_raw_ocr": settings.include_raw_ocr,
    }


async def _read_limited(file: UploadFile) -> bytes:
    """Read an upload, refusing anything over the configured size."""
    try:
        data = await file.read(settings.max_upload_bytes + 1)
    finally:
        await file.close()

    if len(data) > settings.max_upload_bytes:
        raise ValueError(f"File is larger than {settings.max_upload_mb} MB.")
    if not data:
        raise ValueError("File is empty.")
    return data


def _process_sync(data: bytes, content_type: str | None, filename: str) -> ExtractResponse:
    """Decode, OCR and extract. Timed exactly as specified: this covers image
    decoding, preprocessing, OCR and extraction, and excludes model startup."""
    started = time.perf_counter()

    try:
        pages = decode_upload(data, content_type, filename)
    except ValueError:
        # validate_upload's messages are written for the caller; pass them through.
        raise
    except Exception as exc:
        # Anything else is a decoder internal (Pillow/pdfium object reprs and
        # memory addresses). Log it, return something a user can act on.
        logger.warning("Could not decode %s: %s", filename, exc)
        raise ValueError(
            "Could not read this file. It may be corrupted, or not a real "
            "JPG/PNG/WEBP/PDF despite its extension."
        ) from exc

    lines, timings = get_ocr_engine().read_pages(pages)
    extracted, confidence, warnings = extract_fields(lines)

    elapsed = int((time.perf_counter() - started) * 1000)
    return ExtractResponse(
        success=True,
        filename=filename,
        processing_time_ms=elapsed,
        data=extracted,
        confidence=confidence,
        warnings=warnings,
        raw_ocr=[x.to_dict() for x in lines] if settings.include_raw_ocr else None,
        timings=timings if settings.include_raw_ocr else None,
    )


@app.post("/api/v1/mulkiya/extract", response_model=ExtractResponse)
async def extract_mulkiya(file: UploadFile = File(...)) -> ExtractResponse:
    try:
        data = await _read_limited(file)
        return await asyncio.to_thread(
            _process_sync, data, file.content_type, file.filename or "upload"
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/v1/mulkiya/extract/batch", response_model=BatchResponse)
async def extract_batch(files: list[UploadFile] = File(...)) -> BatchResponse:
    if not files:
        raise HTTPException(status_code=400, detail="No files supplied.")
    if len(files) > settings.max_batch_files:
        raise HTTPException(
            status_code=400,
            detail=f"Maximum {settings.max_batch_files} files per batch.",
        )

    started = time.perf_counter()
    results: list[ExtractResponse] = []

    for file in files:
        name = file.filename or "upload"
        file_started = time.perf_counter()
        try:
            data = await _read_limited(file)
            results.append(
                await asyncio.to_thread(_process_sync, data, file.content_type, name)
            )
        except ValueError as exc:
            # One unreadable file must not discard the other results.
            results.append(
                ExtractResponse(
                    success=False,
                    filename=name,
                    processing_time_ms=int((time.perf_counter() - file_started) * 1000),
                    error=str(exc),
                    warnings=[str(exc)],
                )
            )
        except Exception:
            logger.exception("Unhandled error processing %s", name)
            results.append(
                ExtractResponse(
                    success=False,
                    filename=name,
                    processing_time_ms=int((time.perf_counter() - file_started) * 1000),
                    error="Internal processing error.",
                    warnings=["Internal processing error."],
                )
            )

    succeeded = sum(1 for r in results if r.success)
    return BatchResponse(
        success=succeeded > 0,
        total=len(results),
        succeeded=succeeded,
        failed=len(results) - succeeded,
        processing_time_ms=int((time.perf_counter() - started) * 1000),
        results=results,
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    # Log the detail server-side; do not return it. The message can contain
    # internal paths and uploaded filenames.
    logger.exception("Unhandled error on %s", request.url.path)
    return JSONResponse(
        status_code=500,
        content={"success": False, "error": "Internal processing error"},
    )
