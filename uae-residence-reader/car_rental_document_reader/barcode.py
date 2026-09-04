from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date

import numpy as np
from PIL import Image


@dataclass
class BarcodeCandidate:
    barcode_type: str
    raw_value: str
    structured_candidate: dict[str, str] = field(default_factory=dict)
    source_image: str = ""
    confidence: float | None = None
    warnings: list[str] = field(default_factory=list)


def _aamva_date(value: str) -> str:
    digits = re.sub(r"\D", "", value)
    if len(digits) != 8:
        return value.strip()
    options = []
    if 1900 <= int(digits[:4]) <= 2200:
        options.append((int(digits[:4]), int(digits[4:6]), int(digits[6:])))
    options.append((int(digits[4:]), int(digits[:2]), int(digits[2:4])))
    for year, month, day in options:
        try:
            return date(year, month, day).isoformat()
        except ValueError:
            continue
    return value.strip()


def _aamva_fields(raw: str) -> dict[str, str]:
    """Parse the standard PDF417 data elements used by US/Canadian licences.

    DAJ is the jurisdiction -- the province or state that issued the card --
    and it is the one element that names which of thirteen Canadian documents
    is on the page. Canada prints no federal licence, so without it a decoded
    barcode still left the issuer to be guessed from whatever wording OCR had
    managed to read off the front.
    """
    element_map = {
        "DAQ": "license_number", "DCS": "last_name", "DAC": "first_name",
        "DAD": "middle_name", "DBB": "date_of_birth", "DBD": "issue_date",
        "DBA": "expiry_date", "DCG": "issuing_country", "DAJ": "jurisdiction",
        "DBC": "sex", "DCA": "vehicle_class", "DCB": "restrictions",
        "DCD": "endorsements",
    }
    structured: dict[str, str] = {}
    codes = "|".join(element_map)
    # Elements normally begin on a new line. The first can follow the ANSI
    # header directly as DLDAQ..., so accept a DL/ID subfile prefix as well.
    for match in re.finditer(rf"(?:^|[\r\n]|DL|ID)({codes})([^\r\n]*)", raw):
        code, value = match.group(1), match.group(2).strip()
        if not value:
            continue
        key = element_map[code]
        structured[key] = _aamva_date(value) if key in {
            "date_of_birth", "issue_date", "expiry_date",
        } else value
    return structured


def _sanitize(raw: str) -> tuple[dict[str, str], list[str]]:
    warnings: list[str] = []
    # ZXing renders non-printing AAMVA separators as literal tokens on some
    # platforms.  An Ontario PDF417 supplied with this project came back as
    # ``DCA...<LF>DBA...<LF>DAQ...`` rather than with real newlines.  Leaving
    # those tokens in place makes the first data element consume the complete
    # payload, so the licence number is present in the barcode but invisible
    # to the parser.  Restore only the well-known control names; ordinary text
    # in a QR code is otherwise untouched.
    normalized_raw = re.sub(
        r"<(?:LF|CR|RS|GS|US)>", "\n", raw, flags=re.I,
    )
    normalized_raw = re.sub(r"<EOT>", "", normalized_raw, flags=re.I)
    if normalized_raw.lstrip().lower().startswith(("http://", "https://")):
        warnings.append("URL_NOT_OPENED")
        return {}, warnings
    structured: dict[str, str] = (
        _aamva_fields(normalized_raw)
        if "ANSI " in normalized_raw or "AAMVA" in normalized_raw else {}
    )
    for segment in normalized_raw.replace("\r", "\n").split("\n"):
        if "=" in segment:
            key, value = segment.split("=", 1)
        elif ":" in segment:
            key, value = segment.split(":", 1)
        else:
            continue
        key = "_".join(key.strip().lower().split())
        value = value.strip()
        if key and value and len(key) <= 50 and len(value) <= 500:
            structured.setdefault(key, value)
    if not structured: warnings.append("UNSTRUCTURED_BARCODE_VALUE")
    return structured, warnings


def _read_barcodes(image: Image.Image):
    """The replaceable boundary around ZXing used by the regression tests."""
    try:
        import zxingcpp
        return zxingcpp.read_barcodes(np.asarray(image.convert("RGB")))
    except (ImportError, OSError, RuntimeError, ValueError):
        # A barcode is optional evidence. A native decoder failure must not
        # abort OCR and MRZ extraction for the rest of the document.
        return None


def _composite_card_regions(image: Image.Image) -> list[Image.Image]:
    """Return the two panels of a likely front-and-back licence photograph.

    Rental desks commonly receive one JPEG with both card sides stacked.  A
    PDF417 that occupies a useful fraction of the lower card becomes too small
    relative to the combined canvas for ZXing to find, even though decoding
    that lower half alone succeeds.  The geometry gates keep ordinary passport
    pages and single landscape cards on the one-scan fast path.
    """
    width, height = image.size
    if height >= width * 1.15:
        split = height // 2
        return [image.crop((0, 0, width, split)), image.crop((0, split, width, height))]
    if width >= height * 2.2:
        split = width // 2
        return [image.crop((0, 0, split, height)), image.crop((split, 0, width, height))]
    return []


def decode_barcodes(
    image: Image.Image, source_image: str, *, scan_composite: bool = False,
) -> list[BarcodeCandidate]:
    """Decode a page, with a bounded second look at combined tourist cards.

    The extra two small scans are opt-in so the UAE-resident and GCC routes
    retain their existing barcode behaviour and latency.  A successful full
    page scan also remains the one-scan path.
    """
    results = _read_barcodes(image)
    if results is None:
        return []
    decoded: list[BarcodeCandidate] = []
    seen: set[tuple[str, str]] = set()

    def append(results_to_add) -> None:
        for result in results_to_add:
            raw = str(result.text)
            barcode_type = str(result.format)
            identity = (barcode_type, raw)
            if identity in seen:
                continue
            seen.add(identity)
            structured, warnings = _sanitize(raw)
            decoded.append(BarcodeCandidate(
                barcode_type, raw, structured, source_image, None, warnings,
            ))

    append(results)
    has_licence_payload = any(
        item.structured_candidate.get("license_number") for item in decoded
    )
    if scan_composite and not has_licence_payload:
        for region in _composite_card_regions(image):
            regional_results = _read_barcodes(region)
            if regional_results is not None:
                append(regional_results)
    return decoded
