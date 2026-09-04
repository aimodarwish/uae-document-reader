from io import BytesIO

import cv2
import numpy as np
import pypdfium2 as pdfium
from PIL import Image

from .config import settings


ALLOWED_IMAGE_TYPES = {
    "image/jpeg",
    "image/jpg",
    "image/png",
    "image/webp",
}
ALLOWED_PDF_TYPES = {"application/pdf"}


def validate_upload(content_type: str | None, filename: str) -> None:
    content_type = (content_type or "").lower()
    suffix = filename.lower().rsplit(".", 1)[-1] if "." in filename else ""

    valid_suffixes = {"jpg", "jpeg", "png", "webp", "pdf"}
    if suffix not in valid_suffixes and content_type not in ALLOWED_IMAGE_TYPES | ALLOWED_PDF_TYPES:
        raise ValueError("Unsupported file type. Use JPG, JPEG, PNG, WEBP or PDF.")


def _pil_to_bgr(image: Image.Image) -> np.ndarray:
    rgb = np.asarray(image.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def decode_upload(data: bytes, content_type: str | None, filename: str) -> list[np.ndarray]:
    validate_upload(content_type, filename)

    is_pdf = (content_type or "").lower() == "application/pdf" or filename.lower().endswith(".pdf")
    if is_pdf:
        return _decode_pdf(data)

    image = Image.open(BytesIO(data))
    return [_pil_to_bgr(image)]


def _decode_pdf(data: bytes) -> list[np.ndarray]:
    pdf = pdfium.PdfDocument(data)
    pages: list[np.ndarray] = []

    page_count = min(len(pdf), settings.max_pdf_pages)
    for index in range(page_count):
        page = pdf[index]
        # 150 DPI loses strokes on a dense bilingual table; 200 is the
        # default and is configurable via PDF_RENDER_DPI.
        bitmap = page.render(scale=settings.pdf_render_dpi / 72)
        pil = bitmap.to_pil()
        pages.append(_pil_to_bgr(pil))

    if not pages:
        raise ValueError("PDF contains no readable pages.")
    return pages


def preprocess(image: np.ndarray) -> np.ndarray:
    """Conservative preprocessing for bilingual UAE Mulkiya documents."""
    h, w = image.shape[:2]

    if w > settings.ocr_max_width:
        ratio = settings.ocr_max_width / w
        image = cv2.resize(
            image,
            (settings.ocr_max_width, max(1, int(h * ratio))),
            interpolation=cv2.INTER_AREA,
        )
    elif settings.ocr_min_width and w < settings.ocr_min_width:
        # Upscale small scans: recognition crops come from this image, so a
        # 2-letter Arabic word in a 1000px-wide photo is only a few dozen
        # pixels across. INTER_CUBIC keeps the strokes clean.
        ratio = settings.ocr_min_width / w
        image = cv2.resize(
            image,
            (settings.ocr_min_width, max(1, int(h * ratio))),
            interpolation=cv2.INTER_CUBIC,
        )

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
    l = clahe.apply(l)
    enhanced = cv2.merge((l, a, b))
    return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
