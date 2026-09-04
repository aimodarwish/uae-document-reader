from __future__ import annotations

import hashlib
import io
import json
import os
import re
import threading
import time
from collections.abc import Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Mapping

from PIL import Image

from .ocr import OCRLine, OCRResult, merge_ocr_lines
from .privacy import logger


_TRUE_VALUES = frozenset({"1", "true", "yes", "on"})
_LOCATION_PATTERN = re.compile(r"[a-z][a-z0-9-]{0,31}")


class GoogleDocumentAIConfigurationError(ValueError):
    """Raised when the explicit Google OCR opt-in is incomplete or invalid."""


@dataclass(frozen=True)
class GoogleDocumentAISettings:
    """Credentials and routing for one synchronous Document AI processor.

    The credential mapping is deliberately excluded from representations and
    comparisons so a traceback, debug print, or test assertion cannot expose
    the service-account private key.
    """

    project_id: str
    location: str
    processor_id: str
    service_account_info: Mapping[str, Any] | None = field(
        default=None, repr=False, compare=False,
    )
    timeout_seconds: float = 20.0

    @property
    def processor_name(self) -> str:
        return (
            f"projects/{self.project_id}/locations/{self.location}/"
            f"processors/{self.processor_id}"
        )

    @property
    def api_endpoint(self) -> str:
        return f"{self.location}-documentai.googleapis.com"

    @classmethod
    def from_environment(
        cls, environment: Mapping[str, str] | None = None,
    ) -> GoogleDocumentAISettings | None:
        """Return settings only after an explicit, fully configured opt-in.

        A JSON credential held in the process environment is used instead of a
        credential file. That keeps the Modal secret in memory and avoids
        writing key material to the container filesystem.
        """

        env = os.environ if environment is None else environment
        enabled = env.get("GOOGLE_DOCUMENT_AI_SHADOW_ENABLED", "").strip().lower()
        if enabled not in _TRUE_VALUES:
            return None
        return cls._configured_from_environment(
            env, require_inline_credentials=True,
            missing_code="GOOGLE_DOCUMENT_AI_SHADOW_CONFIG_MISSING",
        )

    @classmethod
    def for_localhost(
        cls, environment: Mapping[str, str] | None = None,
    ) -> GoogleDocumentAISettings:
        """Build primary-OCR settings for the explicit localhost launcher.

        Localhost may use an inline JSON credential or standard Google
        Application Default Credentials. Invoking the dedicated launcher is
        itself the opt-in, so the production shadow flag is not consulted.
        """

        env = os.environ if environment is None else environment
        return cls._configured_from_environment(
            env, require_inline_credentials=False,
            missing_code="GOOGLE_DOCUMENT_AI_LOCAL_CONFIG_MISSING",
        )

    @classmethod
    def _configured_from_environment(
        cls, env: Mapping[str, str], *, require_inline_credentials: bool,
        missing_code: str,
    ) -> GoogleDocumentAISettings:

        names = {
            "project_id": "GOOGLE_DOCUMENT_AI_PROJECT_ID",
            "location": "GOOGLE_DOCUMENT_AI_LOCATION",
            "processor_id": "GOOGLE_DOCUMENT_AI_PROCESSOR_ID",
            "service_account_json": "GOOGLE_DOCUMENT_AI_SERVICE_ACCOUNT_JSON",
        }
        values = {key: env.get(name, "").strip() for key, name in names.items()}
        required = {"project_id", "location", "processor_id"}
        if require_inline_credentials:
            required.add("service_account_json")
        missing = [names[key] for key in required if not values[key]]
        if missing:
            raise GoogleDocumentAIConfigurationError(
                missing_code + ":" + ",".join(sorted(missing))
            )
        if not _LOCATION_PATTERN.fullmatch(values["location"]):
            raise GoogleDocumentAIConfigurationError(
                "GOOGLE_DOCUMENT_AI_LOCATION_INVALID"
            )
        service_account_info = None
        if values["service_account_json"]:
            try:
                service_account_info = json.loads(values["service_account_json"])
            except json.JSONDecodeError as exc:
                raise GoogleDocumentAIConfigurationError(
                    "GOOGLE_DOCUMENT_AI_SERVICE_ACCOUNT_JSON_INVALID"
                ) from exc
            if not isinstance(service_account_info, dict):
                raise GoogleDocumentAIConfigurationError(
                    "GOOGLE_DOCUMENT_AI_SERVICE_ACCOUNT_JSON_INVALID"
                )

        try:
            timeout_seconds = float(
                env.get("GOOGLE_DOCUMENT_AI_TIMEOUT_SECONDS", "20")
            )
        except ValueError as exc:
            raise GoogleDocumentAIConfigurationError(
                "GOOGLE_DOCUMENT_AI_TIMEOUT_INVALID"
            ) from exc
        if not 1.0 <= timeout_seconds <= 60.0:
            raise GoogleDocumentAIConfigurationError(
                "GOOGLE_DOCUMENT_AI_TIMEOUT_INVALID"
            )
        return cls(
            project_id=values["project_id"],
            location=values["location"],
            processor_id=values["processor_id"],
            service_account_info=service_account_info,
            timeout_seconds=timeout_seconds,
        )


def _attribute(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _anchor_text(anchor: Any, document_text: str) -> str:
    content = _attribute(anchor, "content", "")
    if content:
        return str(content).strip()
    pieces: list[str] = []
    for segment in _attribute(anchor, "text_segments", ()) or ():
        start = int(_attribute(segment, "start_index", 0) or 0)
        end = int(_attribute(segment, "end_index", start) or start)
        pieces.append(document_text[start:end])
    return "".join(pieces).strip()


def _language(line: Any, fallback: str) -> str:
    # Languages belong to Page.Line, not Page.Layout in the v1 API.
    detected = _attribute(line, "detected_languages", ()) or ()
    code = (
        str(_attribute(detected[0], "language_code", ""))
        if detected else ""
    ).lower().split("-", 1)[0]
    if code in {"ar", "fa", "ur"}:
        return "ar"
    if code in {"ru", "uk", "be", "bg", "mk", "sr"}:
        return "ru"
    if code == "en":
        return "en"
    return code or fallback


def _bounding_box(layout: Any, image: Image.Image, page: Any) -> list[list[float]]:
    polygon = _attribute(layout, "bounding_poly")
    normalized = _attribute(polygon, "normalized_vertices", ()) or ()
    if normalized:
        return [
            [
                float(_attribute(vertex, "x", 0.0) or 0.0) * image.width,
                float(_attribute(vertex, "y", 0.0) or 0.0) * image.height,
            ]
            for vertex in normalized
        ]

    vertices = _attribute(polygon, "vertices", ()) or ()
    dimension = _attribute(page, "dimension")
    raster = _attribute(page, "image")
    page_width = float(
        _attribute(raster, "width", 0)
        or _attribute(dimension, "width", image.width) or image.width
    )
    page_height = float(
        _attribute(raster, "height", 0)
        or _attribute(dimension, "height", image.height) or image.height
    )
    if vertices:
        return [
            [
                float(_attribute(vertex, "x", 0.0) or 0.0)
                * image.width / page_width,
                float(_attribute(vertex, "y", 0.0) or 0.0)
                * image.height / page_height,
            ]
            for vertex in vertices
        ]
    return [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]


# The pages one read started, belonging to that read alone. Held per call
# rather than on the engine: a worker that ever serves two customers at once
# must not let one read cancel, or take, a page the other started. A read that
# recognizes from another thread simply finds nothing here and fetches its page
# itself, which is the behaviour this replaced.
_PREFETCHED: ContextVar[dict[str, Future[tuple[Any, Image.Image | None]]] | None] = (
    ContextVar("google_prefetched_pages", default=None)
)


def _page_key(image: Image.Image) -> str:
    """Identify a page by its pixels, so a prefetched read is served to the
    exact page it was started for.

    Not the object's identity: the pipeline hands recognition a view taken from
    the preprocessed page rather than the object the prefetch was given, and two
    equal views must be one page. Not the upload identifier either -- this
    engine is given images, and an operator who attaches the same photograph
    twice should pay the provider once.
    """
    digest = hashlib.blake2b(image.tobytes(), digest_size=16).hexdigest()
    return f"{image.mode}:{image.width}x{image.height}:{digest}"


class GoogleDocumentAIOCREngine:
    """Map synchronous Google Document AI OCR into the local OCR contract.

    Only ``process_document`` with an inline ``RawDocument`` is implemented.
    There is intentionally no batch method, GCS source, GCS destination, or
    filesystem-backed upload path in this adapter.
    """

    model_name = "Google-Document-AI-OCR"
    # Document AI recognizes all configured scripts in one request. Repeating
    # the page for a language or image-repair pass only adds latency and cost.
    auto_detects_languages = True
    supports_repair_passes = False
    returns_canonical_image = True

    def __init__(
        self, settings: GoogleDocumentAISettings, *, client: Any | None = None,
        documentai_module: Any | None = None,
    ) -> None:
        self.settings = settings
        self._client = client
        self._documentai = documentai_module
        self.initialization_warnings: list[str] = []
        self.languages = ("en", "ar", "ru", "latin")
        self.engines: dict[str, Any] = {"google": client} if client else {}
        self.document_vl = None
        # Pages whose request was started before the reader reached them are
        # held per read, in ``_PREFETCHED``; only the worker pool is shared.
        self._prefetch_executor: ThreadPoolExecutor | None = None

    def _ensure_client(self) -> Any:
        if self._client is not None:
            return self._client
        try:
            from google.api_core.client_options import ClientOptions
            from google.cloud import documentai
            from google.oauth2 import service_account
        except ImportError as exc:
            raise RuntimeError("GOOGLE_DOCUMENT_AI_DEPENDENCY_MISSING") from exc

        credentials = (
            service_account.Credentials.from_service_account_info(
                dict(self.settings.service_account_info),
                scopes=["https://www.googleapis.com/auth/cloud-platform"],
            )
            if self.settings.service_account_info is not None else None
        )
        self._documentai = documentai
        self._client = documentai.DocumentProcessorServiceClient(
            credentials=credentials,
            client_options=ClientOptions(api_endpoint=self.settings.api_endpoint),
        )
        self.engines = {"google": self._client}
        return self._client

    def initialize(self, languages: tuple[str, ...] | None = None) -> None:
        try:
            self._ensure_client()
        except Exception as exc:
            warning = f"GOOGLE_DOCUMENT_AI_INITIALIZATION_FAILED:{type(exc).__name__}"
            if warning not in self.initialization_warnings:
                self.initialization_warnings.append(warning)

    # What the page is encoded as before it is sent. PNG is lossless and was
    # the obvious choice, but it is the wrong one for a photograph: sensor
    # noise is incompressible, so a preprocessed 2402x1613 capture encodes to
    # about 8 MB in 0.73 s, against 2.7 MB in 0.03 s as JPEG at this quality --
    # 0.7 s of processor time per page spent making the request three times
    # larger to upload. Quality 92 with no chroma subsampling keeps every
    # luminance sample, which is the channel character shapes live in; the
    # recognizer sees the same strokes, and both encodings are resampled by
    # the provider anyway.
    _UPLOAD_MIME_TYPE = "image/jpeg"
    _UPLOAD_JPEG_QUALITY = 92

    @classmethod
    def _encode_upload(cls, image: Image.Image) -> bytes:
        # A new image has an empty ``info`` mapping, so EXIF/ICC/text chunks
        # from the upload cannot be copied into the provider request.
        buffer = io.BytesIO()
        converted = image.convert("RGB")
        sanitized = Image.new("RGB", image.size)
        try:
            sanitized.paste(converted)
            sanitized.save(
                buffer, format="JPEG", quality=cls._UPLOAD_JPEG_QUALITY,
                subsampling=0, optimize=False,
            )
            return buffer.getvalue()
        finally:
            converted.close()
            sanitized.close()

    def _request(self, content: bytes) -> Any:
        if self._documentai is None:
            # Used by unit-test clients without importing the optional SDK.
            return {
                "name": self.settings.processor_name,
                "raw_document": {
                    "content": content, "mime_type": self._UPLOAD_MIME_TYPE,
                },
                "skip_human_review": True,
            }
        raw_document = self._documentai.RawDocument(
            content=content, mime_type=self._UPLOAD_MIME_TYPE,
        )
        return self._documentai.ProcessRequest(
            name=self.settings.processor_name,
            raw_document=raw_document,
            skip_human_review=True,
        )

    def _parse_document(
        self, document: Any, image: Image.Image, variant: str,
        fallback_language: str,
    ) -> list[OCRLine]:
        document_text = str(_attribute(document, "text", "") or "")
        parsed: list[OCRLine] = []
        for page in _attribute(document, "pages", ()) or ():
            for line in _attribute(page, "lines", ()) or ():
                layout = _attribute(line, "layout")
                text = _anchor_text(_attribute(layout, "text_anchor"), document_text)
                if not text:
                    continue
                parsed.append(OCRLine(
                    text=text,
                    confidence=float(_attribute(layout, "confidence", 0.0) or 0.0),
                    bounding_box=_bounding_box(layout, image, page),
                    language=_language(line, fallback_language),
                    variant=variant,
                    model_name=self.model_name,
                ))
        return parsed

    # One request per page, and no more than this many waiting on the provider
    # at once. Four covers the largest bundle this workflow accepts -- passport,
    # licence front and back, permit -- without turning a burst of customers
    # into a burst of concurrent provider requests.
    _PREFETCH_WORKERS = 4
    _PREFETCH_LIMIT = 8

    def _fetch(self, image: Image.Image) -> tuple[Any, Image.Image | None]:
        """One synchronous page read: the request, and nothing interpreted yet.

        The provider's document is returned unparsed because parsing needs the
        caller's language context, which a prefetch started before the page was
        classified does not have. Separating the wait from the reading is what
        lets the wait happen early and the reading happen in its own time.
        """
        client = self._ensure_client()
        response = client.process_document(
            request=self._request(self._encode_upload(image)),
            timeout=self.settings.timeout_seconds,
            retry=None,
        )
        document = _attribute(response, "document")
        pages = _attribute(document, "pages", ()) or ()
        if len(pages) != 1:
            raise ValueError("GOOGLE_DOCUMENT_AI_EXPECTED_SINGLE_IMAGE_PAGE")
        # Google boxes refer to its deskewed Page.image. Keep that image and
        # the OCR evidence in the same coordinate system.
        content = _attribute(_attribute(pages[0], "image"), "content", b"")
        canonical: Image.Image | None = None
        if content:
            with Image.open(io.BytesIO(content)) as opened:
                canonical = opened.convert("RGB")
        return document, canonical

    def prefetch_pages(self, images: Sequence[Image.Image]) -> None:
        """Start every page of one bundle now, rather than one at a time.

        A page read is almost entirely a wait on the provider, and the pages of
        a bundle do not depend on each other, so reading them one after another
        made a four-page bundle cost four round trips end to end. Started
        together they cost about one. Extraction still runs page by page, in
        order: only the waiting overlaps.

        The request count is unchanged -- each page is still read once, and the
        reader takes the started result instead of making its own. A page the
        reader never reaches (an interrupted read) is the one case where a
        started request goes unused, which is why the count is bounded here.
        """
        self.discard_prefetched()
        if len(images) < 2:
            # One page has nothing to overlap with, so it is read in place.
            return
        try:
            self._ensure_client()
        except Exception:
            # An unusable client is reported through the customer's result by
            # the ordinary read path, which says which page it happened on.
            return
        if self._prefetch_executor is None:
            self._prefetch_executor = ThreadPoolExecutor(
                max_workers=self._PREFETCH_WORKERS, thread_name_prefix="gda-page",
            )
        started: dict[str, Future[tuple[Any, Image.Image | None]]] = {}
        for image in images[: self._PREFETCH_LIMIT]:
            key = _page_key(image)
            if key in started:
                continue
            # A copy, because the page this engine was handed goes on being
            # cropped and converted by the reader while the request runs.
            started[key] = self._prefetch_executor.submit(self._fetch, image.copy())
        _PREFETCHED.set(started)

    def discard_prefetched(self) -> None:
        """Drop whatever this read started and did not consume."""
        pending = _PREFETCHED.get()
        _PREFETCHED.set(None)
        for future in (pending or {}).values():
            future.cancel()

    def _take_prefetched(
        self, image: Image.Image,
    ) -> tuple[Any, Image.Image | None] | None:
        """This read's started request for this exact page, waited on here.

        ``None`` when nothing was started for these pixels, which is the whole
        of the fallback: the page is then fetched in place, as it always was.
        """
        started = _PREFETCHED.get()
        if not started:
            return None
        future = started.pop(_page_key(image), None)
        if future is None or future.cancelled():
            return None
        return future.result()

    def run_languages(
        self, variants: dict[str, Image.Image], languages: tuple[str, ...], **_: Any,
    ) -> OCRResult:
        lines: list[OCRLine] = []
        warnings = list(self.initialization_warnings)
        corrected_images: dict[str, Image.Image] = {}
        fallback_language = (
            "ar" if languages and all(language == "ar" for language in languages)
            else "ru" if languages and all(language == "ru" for language in languages)
            else "en"
        )
        try:
            client = self._ensure_client()
        except Exception as exc:
            warnings.append(
                f"GOOGLE_DOCUMENT_AI_INITIALIZATION_FAILED:{type(exc).__name__}"
            )
            return OCRResult(warnings=warnings)

        for variant, image in variants.items():
            try:
                started = self._take_prefetched(image)
                document, provider_image = (
                    started if started is not None else self._fetch(image)
                )
                canonical = image
                if provider_image is not None:
                    canonical = provider_image
                    corrected_images[variant] = provider_image
                lines.extend(self._parse_document(
                    document, canonical, variant, fallback_language,
                ))
            except Exception as exc:
                # Never include an exception message: provider errors can echo
                # request metadata. The type is enough for an operational alert.
                warnings.append(
                    f"GOOGLE_DOCUMENT_AI_PROCESS_FAILED:{type(exc).__name__}"
                )
        merged = merge_ocr_lines(lines)
        if not merged and not warnings:
            warnings.append("GOOGLE_DOCUMENT_AI_EMPTY_RESULT")
        return OCRResult(
            lines=merged,
            model_names=[self.model_name] if merged else [],
            warnings=warnings,
            corrected_images=corrected_images,
        )

    def run(self, variants: dict[str, Image.Image]) -> OCRResult:
        return self.run_languages(variants, self.languages)


@dataclass(frozen=True)
class ShadowOCRComparison:
    """Aggregate-only comparison; no OCR text or document identifier."""

    duration_seconds: float
    primary_line_count: int
    shadow_line_count: int
    exact_line_overlap_ratio: float
    succeeded: bool
    warning_count: int


def _normalized_line_set(result: OCRResult) -> set[str]:
    # Text is used transiently for comparison and is never retained or logged.
    return {
        re.sub(r"\W+", "", line.text, flags=re.UNICODE).casefold()
        for line in result.lines
        if re.sub(r"\W+", "", line.text, flags=re.UNICODE)
    }


class GoogleDocumentAIShadowOCR:
    """Observe Google OCR without changing the authoritative Paddle result."""

    def __init__(self, engine: GoogleDocumentAIOCREngine) -> None:
        self.engine = engine
        self.comparisons: list[ShadowOCRComparison] = []
        # Network OCR can overlap Paddle's later extraction work. A small,
        # bounded pool prevents the shadow route from extending the customer's
        # response time or retaining an unbounded queue of in-memory images.
        self._executor = ThreadPoolExecutor(
            max_workers=2, thread_name_prefix="gda-shadow",
        )
        self._pending: set[Future[ShadowOCRComparison | None]] = set()
        self._lock = threading.Lock()

    def submit(
        self, primary: OCRResult, image: Image.Image,
        languages: tuple[str, ...],
    ) -> bool:
        """Queue one in-memory page without waiting on Google in the user path."""

        with self._lock:
            self._pending = {future for future in self._pending if not future.done()}
            if len(self._pending) >= 4:
                logger.info("GDA shadow skip=busy")
                return False
            image_copy = image.copy()
            future = self._executor.submit(
                self._observe_copy, primary, image_copy, languages,
            )
            self._pending.add(future)
        return True

    def _observe_copy(
        self, primary: OCRResult, image: Image.Image,
        languages: tuple[str, ...],
    ) -> ShadowOCRComparison | None:
        try:
            return self.observe(primary, image, languages)
        except Exception as exc:
            logger.warning("GDA shadow failed kind=%s", type(exc).__name__)
            return None
        finally:
            image.close()

    def observe(
        self, primary: OCRResult, image: Image.Image,
        languages: tuple[str, ...],
    ) -> ShadowOCRComparison:
        started = time.perf_counter()
        shadow = self.engine.run_languages(
            {"original_normalized": image}, languages,
        )
        primary_lines = _normalized_line_set(primary)
        shadow_lines = _normalized_line_set(shadow)
        ratio = (
            len(primary_lines & shadow_lines) / len(primary_lines)
            if primary_lines else (1.0 if not shadow_lines else 0.0)
        )
        comparison = ShadowOCRComparison(
            duration_seconds=round(time.perf_counter() - started, 3),
            primary_line_count=len(primary.lines),
            shadow_line_count=len(shadow.lines),
            exact_line_overlap_ratio=round(ratio, 4),
            succeeded=bool(shadow.lines) and not shadow.warnings,
            warning_count=len(shadow.warnings),
        )
        # Bound in-memory metrics for a long-lived Modal worker. They contain
        # no OCR text, image, customer key, upload ID, or source filename.
        self.comparisons.append(comparison)
        del self.comparisons[:-100]
        logger.info(
            "GDA shadow ok=%s p=%d g=%d ov=%.3f sec=%.3f warn=%d",
            comparison.succeeded,
            comparison.primary_line_count,
            comparison.shadow_line_count,
            comparison.exact_line_overlap_ratio,
            comparison.duration_seconds,
            comparison.warning_count,
        )
        return comparison


def google_shadow_from_environment() -> GoogleDocumentAIShadowOCR | None:
    settings = GoogleDocumentAISettings.from_environment()
    if settings is None:
        return None
    return GoogleDocumentAIShadowOCR(GoogleDocumentAIOCREngine(settings))
