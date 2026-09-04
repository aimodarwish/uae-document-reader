from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

from PIL import Image, ImageOps, UnidentifiedImageError

from .config import AppConfig


class DocumentInputError(ValueError):
    """Raised for invalid, corrupt, encrypted, or unsupported user input."""


@dataclass(frozen=True)
class LoadedPage:
    image: Image.Image
    mime_type: str
    source_name: str
    page_index: int = 0


def read_input_bytes(source: str | Path | bytes | bytearray | BinaryIO) -> tuple[bytes, str]:
    if isinstance(source, (bytes, bytearray)):
        return bytes(source), "memory-upload"
    if isinstance(source, (str, Path)):
        path = Path(source)
        return path.read_bytes(), path.name
    if hasattr(source, "read"):
        current = source.tell() if hasattr(source, "tell") else None
        data = source.read()
        if current is not None and hasattr(source, "seek"):
            source.seek(current)
        return data, Path(getattr(source, "name", "memory-upload")).name
    raise DocumentInputError("Unsupported upload object")


def detect_mime(data: bytes) -> str:
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if data[:4] in (b"II*\x00", b"MM\x00*"):
        return "image/tiff"
    if data.startswith(b"RIFF") and data[8:12] == b"WEBP":
        return "image/webp"
    if data.startswith(b"%PDF-"):
        return "application/pdf"
    if len(data) >= 12 and data[4:8] == b"ftyp" and data[8:12] in {
        b"heic", b"heix", b"hevc", b"hevx", b"mif1", b"msf1", b"avif"
    }:
        return "image/heif"
    raise DocumentInputError("File signature is not a supported image or PDF")


def _open_image(data: bytes, mime: str) -> Image.Image:
    if mime == "image/heif":
        try:
            from pillow_heif import register_heif_opener
            register_heif_opener()
        except ImportError as exc:
            raise DocumentInputError("HEIC support is unavailable") from exc
    try:
        with Image.open(io.BytesIO(data)) as opened:
            opened.verify()
        image = Image.open(io.BytesIO(data))
        image = ImageOps.exif_transpose(image).convert("RGB")
        image.load()
        return image
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise DocumentInputError("Image is corrupt or unreadable") from exc


def _render_pdf(data: bytes, name: str, config: AppConfig) -> list[LoadedPage]:
    try:
        import pypdfium2 as pdfium
        document = pdfium.PdfDocument(data)
    except Exception as exc:
        raise DocumentInputError("PDF is corrupt, encrypted, or unreadable") from exc
    try:
        if len(document) == 0:
            raise DocumentInputError("PDF contains no pages")
        if len(document) > config.max_pdf_pages:
            raise DocumentInputError(f"PDF exceeds the {config.max_pdf_pages}-page limit")
        scale = config.pdf_dpi / 72.0
        pages: list[LoadedPage] = []
        for index in range(len(document)):
            try:
                bitmap = document[index].render(scale=scale, rotation=0)
                image = bitmap.to_pil().convert("RGB")
                pages.append(LoadedPage(image, "application/pdf", name, index))
            except Exception as exc:
                raise DocumentInputError(f"PDF page {index + 1} could not be rendered") from exc
        return pages
    finally:
        document.close()


def load_document(source: str | Path | bytes | bytearray | BinaryIO, config: AppConfig | None = None) -> list[LoadedPage]:
    config = config or AppConfig()
    data, name = read_input_bytes(source)
    if not data:
        raise DocumentInputError("Upload is empty")
    if len(data) > config.max_file_mb * 1024 * 1024:
        raise DocumentInputError(f"Upload exceeds {config.max_file_mb} MB")
    mime = detect_mime(data)
    if mime == "application/pdf":
        return _render_pdf(data, name, config)
    return [LoadedPage(_open_image(data, mime), mime, name)]

