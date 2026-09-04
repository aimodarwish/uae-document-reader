from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable

import numpy as np
from PIL import Image

from .mrz import mrz_row_shape, validate_check


# Fast recognition profiles used by the tourist router.  ``latin`` is the
# important one: one PP-OCRv5 mobile model covers the Latin alphabets used by
# most driving licences, so French, Spanish, Portuguese, German, Turkish and
# dozens of other languages do not require separate full-page passes.
_OCR_PROFILES: dict[str, tuple[str, str, str | None]] = {
    # Any supported Latin language selects the shared model; ``fr`` keeps the
    # Paddle pipeline's language metadata aligned with that model rather than
    # the narrower English recognizer.
    "latin": ("fr", "PP-OCRv5", "latin_PP-OCRv5_mobile_rec"),
    "en": ("en", "PP-OCRv5", "en_PP-OCRv5_mobile_rec"),
    "ar": ("ar", "PP-OCRv5", "arabic_PP-OCRv5_mobile_rec"),
    "ru": ("ru", "PP-OCRv5", "eslav_PP-OCRv5_mobile_rec"),
    "uk": ("uk", "PP-OCRv5", "eslav_PP-OCRv5_mobile_rec"),
    "be": ("be", "PP-OCRv5", "eslav_PP-OCRv5_mobile_rec"),
    # The general PP-OCRv5 recognizer covers Simplified/Traditional Chinese,
    # Japanese and English. Korean has its own compact v5 recognizer.
    "ch": ("ch", "PP-OCRv5", "PP-OCRv5_mobile_rec"),
    "chinese_cht": ("chinese_cht", "PP-OCRv5", "PP-OCRv5_mobile_rec"),
    "japan": ("japan", "PP-OCRv5", "PP-OCRv5_mobile_rec"),
    "korean": ("korean", "PP-OCRv5", "korean_PP-OCRv5_mobile_rec"),
}


# Languages with native PP-OCRv5 support. A requested long-tail language that
# is not in this set is loaded through Paddle's broader PP-OCRv3 catalogue.
# That path is lazy and is only used after the tourist router has evidence for
# the script/country; it is never paid for on ordinary Latin documents.
_PP_OCR_V5_LANGUAGES = frozenset({
    "af", "be", "bs", "ch", "chinese_cht", "cs", "cy", "da", "de",
    "en", "es", "et", "fr", "ga", "hr", "hu", "id", "is", "it",
    "japan", "korean", "la", "latin", "lt", "mi", "ms", "nl", "no",
    "oc", "pl", "pt", "ru", "sk", "sl", "sq", "sv", "sw", "tl",
    "tr", "uk", "uz",
})


@dataclass
class OCRLine:
    text: str
    confidence: float
    bounding_box: list[list[float]]
    language: str
    variant: str
    model_name: str


@dataclass
class OCRResult:
    lines: list[OCRLine] = field(default_factory=list)
    model_names: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    corrected_images: dict[str, Image.Image] = field(default_factory=dict)
    orientation_angles: dict[str, int] = field(default_factory=dict)


def _iou(a: list[list[float]], b: list[list[float]]) -> float:
    ax = [point[0] for point in a]; ay = [point[1] for point in a]
    bx = [point[0] for point in b]; by = [point[1] for point in b]
    left, top = max(min(ax), min(bx)), max(min(ay), min(by))
    right, bottom = min(max(ax), max(bx)), min(max(ay), max(by))
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    union = (max(ax) - min(ax)) * (max(ay) - min(ay)) + (max(bx) - min(bx)) * (max(by) - min(by)) - intersection
    return intersection / union if union else 0.0


def _vertical_span(line: OCRLine) -> tuple[float, float]:
    ys = [point[1] for point in line.bounding_box]
    return min(ys), max(ys)


def reading_order(lines: Iterable[OCRLine]) -> list[OCRLine]:
    """Order boxes the way a person reads the card: row by row, left to right.

    Ordering by the top edge alone is only correct on a card that prints one
    column. Every card that prints two interleaves them, because the top edges
    of the two columns' rows do not line up to the pixel -- and the boxes then
    come back in an order that is not the order anything was written in.

    The Quebec licence shows both halves of the damage. Its title is set as two
    boxes, "Permis" at x=1115 and "de conduire" at x=1470, and the second box's
    top edge is seven pixels higher than the first: joined for classification
    the page read "de conduire Permis", the title "PERMIS DE CONDUIRE" matched
    nothing, and the front of the licence was filed as its back -- which cost
    the holder's name and date of birth, since those are not read off a back.
    The same reversal hit its right-hand column, where "Taille" and its value
    "(cm) : 169" came back in that order.

    Boxes that overlap vertically by at least half the shorter of the two are
    one printed row. Measuring against the row's first box rather than against
    a running span is deliberate: a span that grows as boxes join it drifts
    down a card whose rows are close together, and swallows the row beneath.

    A box several times taller than the page's own line height neither joins a
    row nor starts one that others join. Such a box is a rotated capture's
    vertical strip or a hologram read as text, and because "half the shorter of
    the two" is satisfied by any short box falling anywhere inside it, one of
    them left to seed a row would have collected every row it spans and
    flattened the card into a single line ordered across the page.
    """
    ordered = sorted(lines, key=lambda item: _vertical_span(item)[0])
    heights = sorted(
        max(1.0, bottom - top)
        for top, bottom in (_vertical_span(line) for line in ordered)
    )
    if not heights:
        return []
    typical = heights[len(heights) // 2]
    rows: list[list[OCRLine]] = []
    for line in ordered:
        top, bottom = _vertical_span(line)
        height = max(1.0, bottom - top)
        if height <= 3.0 * typical:
            for row in rows:
                seed_top, seed_bottom = _vertical_span(row[0])
                seed_height = max(1.0, seed_bottom - seed_top)
                if seed_height > 3.0 * typical:
                    continue
                overlap = min(bottom, seed_bottom) - max(top, seed_top)
                if overlap >= 0.5 * min(height, seed_height):
                    row.append(line)
                    break
            else:
                rows.append([line])
            continue
        rows.append([line])
    return [
        line for row in rows
        for line in sorted(row, key=lambda item: min(p[0] for p in item.bounding_box))
    ]


def merge_ocr_lines(lines: Iterable[OCRLine]) -> list[OCRLine]:
    merged: list[OCRLine] = []
    for line in reading_order(lines):
        duplicate_index = next((i for i, existing in enumerate(merged) if existing.text.strip().casefold() == line.text.strip().casefold() and _iou(existing.bounding_box, line.bounding_box) >= 0.55), None)
        if duplicate_index is None:
            merged.append(line)
        elif line.confidence > merged[duplicate_index].confidence:
            merged[duplicate_index] = line
    return merged


def merge_ocr_results(*results: OCRResult) -> OCRResult:
    """Merge independent OCR engines while preserving correction metadata."""
    corrected_images: dict[str, Image.Image] = {}
    orientation_angles: dict[str, int] = {}
    for result in reversed(results):
        corrected_images.update(result.corrected_images)
        orientation_angles.update(result.orientation_angles)
    return OCRResult(
        lines=merge_ocr_lines(
            line for result in results for line in result.lines
        ),
        model_names=sorted({
            model for result in results for model in result.model_names
        }),
        warnings=list(dict.fromkeys(
            warning for result in results for warning in result.warnings
        )),
        corrected_images=corrected_images,
        orientation_angles=orientation_angles,
    )


class PaddleOCRVLEngine:
    """Local PaddleOCR-VL 1.6 spotting wrapper for multilingual document cards."""

    def __init__(
        self,
        pipeline_version: str = "v1.6",
        max_new_tokens: int = 1536,
        detail_crops: bool = True,
    ):
        self.pipeline_version = pipeline_version
        self.max_new_tokens = max_new_tokens
        self.detail_crops = detail_crops
        self.engine: Any = None
        self.loaded = False
        self.initialization_warnings: list[str] = []

    def initialize(self) -> None:
        if self.loaded or self.initialization_warnings:
            return
        try:
            from paddleocr import PaddleOCRVL

            self.engine = PaddleOCRVL(
                pipeline_version=self.pipeline_version,
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_layout_detection=False,
                use_chart_recognition=False,
                use_seal_recognition=False,
                use_queues=False,
            )
            self.loaded = True
        except Exception as exc:
            self.engine = None
            detail = " ".join(str(exc).split())[:400]
            self.initialization_warnings.append(
                f"PADDLEOCR_VL_LOAD_FAILED:{type(exc).__name__}:{detail}"
            )

    @staticmethod
    def _payload(prediction: Any) -> dict[str, Any]:
        payload = getattr(prediction, "json", prediction)
        if callable(payload):
            payload = payload()
        if isinstance(payload, dict) and isinstance(payload.get("res"), dict):
            payload = payload["res"]
        return payload if isinstance(payload, dict) else {}

    @staticmethod
    def _language(text: str) -> str:
        if re.search(r"[\u0600-\u06ff]", text):
            return "ar"
        if re.search(r"[А-ЯЁа-яё]", text):
            return "ru"
        return "en"

    @staticmethod
    def _polygon(
        raw: Any,
        offset_x: float,
        offset_y: float,
    ) -> list[list[float]]:
        try:
            points = np.asarray(raw, dtype=float).reshape(-1, 2)
        except (TypeError, ValueError):
            points = np.empty((0, 2), dtype=float)
        if len(points) >= 4:
            return [
                [float(x + offset_x), float(y + offset_y)]
                for x, y in points[:4]
            ]
        return [
            [offset_x, offset_y], [offset_x + 1, offset_y],
            [offset_x + 1, offset_y + 1], [offset_x, offset_y + 1],
        ]

    @classmethod
    def _parse_prediction(
        cls,
        prediction: Any,
        variant: str,
        offset_x: int = 0,
        offset_y: int = 0,
        detail_crop: bool = False,
    ) -> list[OCRLine]:
        payload = cls._payload(prediction)
        model_name = "PaddleOCR-VL-1.6"
        confidence = 0.92 if detail_crop else 0.90
        parsed: list[OCRLine] = []
        spotting = payload.get("spotting_res")
        if isinstance(spotting, list):
            spotting = next((item for item in spotting if isinstance(item, dict)), {})
        if isinstance(spotting, dict):
            texts = spotting.get("rec_texts", [])
            polygons = spotting.get("rec_polys", [])
            for text, polygon in zip(texts, polygons):
                cleaned = " ".join(str(text).split())
                if not cleaned:
                    continue
                parsed.append(OCRLine(
                    cleaned, confidence,
                    cls._polygon(polygon, offset_x, offset_y),
                    cls._language(cleaned), variant, model_name,
                ))
        if parsed:
            return parsed

        # Some runtimes return only parsing blocks. Preserve their reading
        # order and synthesize a narrow row per line so the labelled extractor
        # can still bind inline labels and values.
        for block in payload.get("parsing_res_list", []):
            if not isinstance(block, dict):
                continue
            content = str(block.get("block_content", ""))
            lines = [" ".join(line.split()) for line in content.splitlines() if line.strip()]
            if not lines:
                continue
            bbox = block.get("block_bbox", [0, 0, 1, max(1, len(lines))])
            try:
                x1, y1, x2, y2 = [float(value) for value in bbox[:4]]
            except (TypeError, ValueError):
                x1, y1, x2, y2 = 0.0, 0.0, 1.0, float(len(lines))
            row_height = max((y2 - y1) / len(lines), 1.0)
            for index, text in enumerate(lines):
                top = y1 + index * row_height
                polygon = [
                    [x1 + offset_x, top + offset_y],
                    [x2 + offset_x, top + offset_y],
                    [x2 + offset_x, top + row_height + offset_y],
                    [x1 + offset_x, top + row_height + offset_y],
                ]
                parsed.append(OCRLine(
                    text, confidence, polygon, cls._language(text),
                    variant, model_name,
                ))
        return parsed

    def run(self, variants: dict[str, Image.Image]) -> OCRResult:
        self.initialize()
        if not self.loaded or self.engine is None:
            return OCRResult(warnings=(
                self.initialization_warnings or ["PADDLEOCR_VL_UNAVAILABLE"]
            ))
        lines: list[OCRLine] = []
        warnings: list[str] = []
        for variant, image in variants.items():
            rgb = image.convert("RGB")
            width, height = rgb.size
            regions: list[tuple[Image.Image, int, int, bool]]
            if self.detail_crops and width >= height * 1.10 and width >= 900:
                # GCC cards place the portrait at the left and the dense field
                # rows to its right. Use one generous text crop instead of
                # generating twice over the full card and an overlapping crop.
                left = round(width * 0.10)
                regions = [(rgb.crop((left, 0, width, height)), left, 0, True)]
            else:
                regions = [(rgb, 0, 0, False)]
            arrays = [np.asarray(region.convert("RGB")) for region, _, _, _ in regions]
            try:
                predictions = list(self.engine.predict(
                    arrays,
                    use_doc_orientation_classify=False,
                    use_doc_unwarping=False,
                    use_layout_detection=False,
                    prompt_label="spotting",
                    temperature=0.0,
                    top_p=1.0,
                    max_new_tokens=self.max_new_tokens,
                ))
                for prediction, (_, offset_x, offset_y, detail) in zip(predictions, regions):
                    lines.extend(self._parse_prediction(
                        prediction,
                        f"{variant}:paddleocr_vl_detail" if detail else f"{variant}:paddleocr_vl",
                        offset_x, offset_y, detail,
                    ))
            except Exception as exc:
                warnings.append(
                    f"PADDLEOCR_VL_VARIANT_FAILED:{variant}:{type(exc).__name__}"
                )
        merged = merge_ocr_lines(lines)
        if not merged:
            warnings.append("PADDLEOCR_VL_EMPTY_RESULT")
        return OCRResult(
            lines=merged,
            model_names=["PaddleOCR-VL-1.6"] if merged else [],
            warnings=warnings,
        )


class PaddleOCREngine:
    """Lazy, local-only PP-OCRv5 wrapper supporting PaddleOCR 3.x and legacy outputs."""

    def __init__(
        self,
        languages: tuple[str, ...] = ("en", "ar"),
        enable_document_vl: bool = False,
        document_vl_pipeline_version: str = "v1.6",
        document_vl_max_new_tokens: int = 1536,
        document_vl_detail_crops: bool = True,
    ):
        self.languages = languages
        self.engines: dict[str, Any] = {}
        self.failed_languages: set[str] = set()
        self._paddle_ocr_class: Any = None
        self.initialization_warnings: list[str] = []
        self.document_vl = PaddleOCRVLEngine(
            pipeline_version=document_vl_pipeline_version,
            max_new_tokens=document_vl_max_new_tokens,
            detail_crops=document_vl_detail_crops,
        ) if enable_document_vl else None

    def initialize(self, languages: tuple[str, ...] | None = None) -> None:
        """Load only the recognizers this request can use, and cache them.

        Loading every language Paddle supports would make the service slow and
        memory-heavy. The tourist path starts with the one broad Latin model;
        any additional script requested by routing is loaded once and retained
        for later jobs in the warm container.
        """
        requested = tuple(dict.fromkeys(languages or self.languages))
        if self._paddle_ocr_class is None and "PADDLEOCR_UNAVAILABLE" not in self.initialization_warnings:
            try:
                from paddleocr import PaddleOCR
            except ImportError:
                self.initialization_warnings.append("PADDLEOCR_UNAVAILABLE")
            else:
                self._paddle_ocr_class = PaddleOCR
        if self._paddle_ocr_class is not None:
            for language in requested:
                if language in self.engines or language in self.failed_languages:
                    continue
                paddle_language, version, recognition_model = _OCR_PROFILES.get(
                    language,
                    (
                        language,
                        "PP-OCRv5" if language in _PP_OCR_V5_LANGUAGES else "PP-OCRv3",
                        None,
                    ),
                )
                kwargs: dict[str, Any] = {
                    "lang": paddle_language,
                    "ocr_version": version,
                    "use_doc_orientation_classify": True,
                    "use_doc_unwarping": False,
                    "use_textline_orientation": False,
                }
                if recognition_model is not None:
                    kwargs.update({
                        "text_detection_model_name": "PP-OCRv5_mobile_det",
                        "text_recognition_model_name": recognition_model,
                    })
                try:
                    self.engines[language] = self._paddle_ocr_class(**kwargs)
                except Exception as exc:
                    self.failed_languages.add(language)
                    warning = (
                        f"PADDLEOCR_{language.upper()}_LOAD_FAILED:"
                        f"{type(exc).__name__}"
                    )
                    if warning not in self.initialization_warnings:
                        self.initialization_warnings.append(warning)
        if self.document_vl is not None:
            self.document_vl.initialize()

    @staticmethod
    def _needs_orientation_classification(image: Image.Image) -> bool:
        # A landscape input is already in the expected geometry for cards and
        # passport biodata pages. Restricting orientation classification to
        # portrait captures avoids rotating an already-upright landscape card
        # because of a low-confidence classifier decision.
        return image.height > image.width * 1.10

    @staticmethod
    def _parse_prediction(prediction: Any, language: str, variant: str) -> list[OCRLine]:
        parsed: list[OCRLine] = []
        payload = getattr(prediction, "json", prediction)
        if callable(payload): payload = payload()
        if isinstance(payload, dict) and "res" in payload: payload = payload["res"]
        if isinstance(payload, dict) and "rec_texts" in payload:
            texts = payload.get("rec_texts", [])
            scores = payload.get("rec_scores", [0.0] * len(texts))
            boxes = payload.get("rec_polys", payload.get("dt_polys", []))
            for text, score, box in zip(texts, scores, boxes):
                parsed.append(OCRLine(str(text), float(score), np.asarray(box).astype(float).tolist(), language, variant, f"PP-OCRv5-{language}"))
            return parsed
        # PaddleOCR 2.x compatibility is retained for controlled Colab variations.
        rows = payload
        if isinstance(rows, list) and len(rows) == 1 and isinstance(rows[0], list): rows = rows[0]
        if isinstance(rows, list):
            for row in rows:
                if not isinstance(row, (list, tuple)) or len(row) != 2: continue
                box, recognition = row
                if isinstance(recognition, (list, tuple)) and len(recognition) >= 2:
                    parsed.append(OCRLine(str(recognition[0]), float(recognition[1]), np.asarray(box).astype(float).tolist(), language, variant, f"PP-OCRv5-{language}"))
        return parsed

    @staticmethod
    def _orientation_output(prediction: Any) -> tuple[int, Image.Image] | None:
        """Return Paddle's content-based orientation correction for the preview."""
        try:
            preprocessed = prediction["doc_preprocessor_res"]
            angle = int(preprocessed["angle"])
            output = np.asarray(preprocessed["output_img"])
        except (KeyError, TypeError, ValueError, AttributeError):
            return None
        if output.ndim != 3 or output.shape[2] < 3:
            return None
        # PaddleX keeps pipeline images in BGR order.
        corrected = Image.fromarray(output[:, :, :3][:, :, ::-1].copy()).convert("RGB")
        return angle, corrected

    def run_languages(
        self, variants: dict[str, Image.Image], languages: tuple[str, ...], *,
        merge_variants: bool = True,
        use_orientation_classifier: bool = True,
    ) -> OCRResult:
        self.initialize(languages)
        selected_engines = {
            language: engine for language, engine in self.engines.items()
            if language in languages
        }
        if not selected_engines:
            return OCRResult(warnings=self.initialization_warnings or ["PADDLEOCR_UNAVAILABLE"])
        lines: list[OCRLine] = []
        warnings = list(self.initialization_warnings)
        corrected_images: dict[str, Image.Image] = {}
        orientation_angles: dict[str, int] = {}

        def consume(
            prediction: Any, language: str, variant_name: str,
        ) -> None:
            orientation = self._orientation_output(prediction)
            if orientation is not None and variant_name not in corrected_images:
                angle, corrected_images[variant_name] = orientation
                orientation_angles[variant_name] = angle
            lines.extend(self._parse_prediction(prediction, language, variant_name))

        for language, engine in selected_engines.items():
            variant_items = list(variants.items())
            if hasattr(engine, "predict"):
                # PaddleX accepts a list of independent images. Sending the
                # repair views together keeps every image and every prediction
                # unchanged, but avoids restarting the detection/recognition
                # pipeline for each fallback or zoom crop. Group by orientation
                # because that classifier is a per-call option.
                groups = {
                    orientation: [
                        (name, image) for name, image in variant_items
                        if (
                            use_orientation_classifier
                            and self._needs_orientation_classification(image)
                        ) is orientation
                    ]
                    for orientation in (False, True)
                }
                for orientation, items in groups.items():
                    if not items:
                        continue
                    arrays = [
                        np.asarray(image.convert("RGB")) for _, image in items
                    ]
                    try:
                        predictions = engine.predict(
                            arrays if len(arrays) > 1 else arrays[0],
                            use_doc_orientation_classify=orientation,
                        )
                        predictions = (
                            predictions if isinstance(predictions, list)
                            else [predictions]
                        )
                        if len(predictions) != len(items):
                            raise ValueError("OCR_BATCH_RESULT_COUNT_MISMATCH")
                    except Exception:
                        # A provider/version that does not support list input
                        # retains the prior one-image-at-a-time behavior.
                        for (variant_name, _), array in zip(items, arrays):
                            try:
                                individual = engine.predict(
                                    array,
                                    use_doc_orientation_classify=orientation,
                                )
                                for prediction in (
                                    individual if isinstance(individual, list)
                                    else [individual]
                                ):
                                    consume(prediction, language, variant_name)
                            except Exception as exc:
                                warnings.append(
                                    f"OCR_VARIANT_FAILED:{language}:{variant_name}:"
                                    f"{type(exc).__name__}"
                                )
                    else:
                        for (variant_name, _), prediction in zip(items, predictions):
                            consume(prediction, language, variant_name)
                continue

            for variant_name, image in variant_items:
                array = np.asarray(image.convert("RGB"))
                try:
                    predictions = engine.ocr(array, cls=True)
                    for prediction in predictions if isinstance(predictions, list) else [predictions]:
                        consume(prediction, language, variant_name)
                except Exception as exc:
                    warnings.append(f"OCR_VARIANT_FAILED:{language}:{variant_name}:{type(exc).__name__}")
        # Page-scale variants are alternate readings of the same coordinates
        # and therefore deduplicate. Batched zoom crops use coordinates local
        # to different crops; their rows must first be mapped back to the page.
        merged = merge_ocr_lines(lines) if merge_variants else lines
        if not merged: warnings.append("OCR_EMPTY_RESULT")
        return OCRResult(
            merged, sorted({line.model_name for line in merged}), warnings,
            corrected_images, orientation_angles,
        )

    def run(self, variants: dict[str, Image.Image]) -> OCRResult:
        return self.run_languages(variants, self.languages)

    def run_document_vl(self, variants: dict[str, Image.Image]) -> OCRResult:
        if self.document_vl is None:
            return OCRResult()
        return self.document_vl.run(variants)


# The chevron is a filled triangle, and the stroke a recogniser keeps when the
# ink is thin is its middle bar: an Indian passport's data row came back as
# "U4595089-8IND...". Dropped, the row lost a character and every field behind
# it shifted one place left -- the zone still assembled, every check digit
# failed, and the page was not recognised as a passport at all. Read as the
# filler it is, the same row checks out exactly. Nothing is taken on trust:
# these characters only ever stand where the format has a chevron, and the
# check digits decide.
_MRZ_FILLER_MARKS = str.maketrans({mark: "<" for mark in "-\u2010\u2011\u2012\u2013\u2014\u2015_~"})


def _mrz_compact(text: str) -> str:
    upper = text.upper().replace(" ", "<").translate(_MRZ_FILLER_MARKS)
    return "".join(char for char in upper if char.isalnum() or char == "<")


def find_clipped_mrz_lines(lines: list[OCRLine]) -> list[str]:
    """Return rows that look like a machine-readable zone with filler clipped.

    Kept separate from ``find_mrz_lines`` so a well-formed zone is never
    outranked by a near-miss row. Only the repair path, which validates its
    result against the row check digits, consults this.
    """
    rows: list[tuple[float, str]] = []
    for line in lines:
        compact = _mrz_compact(line.text)
        if 24 <= len(compact) < 30 and compact.count("<") >= 1:
            rows.append((min(point[1] for point in line.bounding_box), compact))
    return [text for _, text in sorted(rows)]


MRZ_ROW_LENGTHS = (30, 36, 44)


def _complete_td1_code_row_without_filler(compact: str) -> bool:
    """Recognize a complete TD1 first row whose optional area has no ``<``.

    Emirates IDs can fill every byte of TD1 row one with their ID/card
    numbers. The standard MRZ finder used filler as its first cheap signal, so
    it discarded that row and paired the valid name/data rows with a visible
    ``ID Number`` label instead. The document-number check digit distinguishes
    a genuine 30-character code row from that label without relaxing the
    general candidate filter.
    """
    return bool(
        re.fullmatch(r"[IACV][A-Z<][A-Z]{3}[A-Z0-9<]{9}\d[A-Z0-9<]{15}", compact)
        and validate_check(compact[5:14], compact[14])
    )


def _complete_passport_data_row_without_filler(compact: str) -> bool:
    """Recognize a complete TD2/TD3 data row whose optional area has no ``<``.

    Filler is the finder's first cheap signal, and a passport whose optional
    data fills its field to the last byte prints a second row without a single
    chevron in it. The Uzbek passport in this project's bug report ends
    "...30604986080046 4 8" with fourteen characters of personal number, so its
    zone was rejected for want of a mark it had no room for. Only the name row
    survived, no zone was parsed, and the page fell back to reading its own
    printed rows: the birth date then came from the licence beside it and was
    reported as conflicting, and the one date the zone brackets -- the date of
    issue -- was reported missing.

    The row proves itself instead of being recognised by its punctuation: the
    document number, the birth date and the expiry each carry their own check
    digit, and a line of page text does not satisfy three of them at once.
    """
    if "<" in compact or len(compact) not in {36, 44}:
        return False
    if not re.fullmatch(
        r"[A-Z0-9]{9}\d[A-Z]{3}\d{6}\d[MFX]\d{6}\d[A-Z0-9]+\d", compact,
    ):
        return False
    return (
        validate_check(compact[0:9], compact[9])
        and validate_check(compact[13:19], compact[19])
        and validate_check(compact[21:27], compact[27])
    )


# How much trailing filler a capture may have clipped. A phone photograph of a
# passport routinely loses several: the chevrons at the end of a row are the
# faintest ink on the page and the first thing an edge crop or a glare band
# takes.
_MAX_RESTORED_FILLER = 8


def _mrz_row_restorations(compact: str) -> list[str]:
    """Every standard row length this text could be, filler restored.

    Detection used to demand the exact length, then a length within one or two.
    Both are guesses made one row at a time, and on a Belgian passport the
    guess went wrong in the most damaging way available: the genuine name row
    had lost seven of its trailing chevrons, so at thirty-seven characters it
    was trimmed to a thirty-six character TD2 row rather than padded back to
    the forty-four it was, and the zone never assembled.

    A row cannot be told apart from its neighbours in isolation, so no choice
    is made here. Every plausible reading is offered, and the caller assembles
    them into zones whose check digits decide which one was right. That is the
    only arbiter in the format that cannot be talked into a wrong answer.
    """
    restorations: list[str] = []
    for length in MRZ_ROW_LENGTHS:
        if mrz_row_shape(compact, length) is None:
            continue
        if len(compact) == length:
            restorations.append(compact)
        elif length - _MAX_RESTORED_FILLER <= len(compact) < length:
            restorations.append(compact.ljust(length, "<"))
        elif (
            length < len(compact) <= length + 2 and compact.endswith("<")
            and compact[length:].strip("<") == ""
        ):
            restorations.append(compact[:length])
        # A passport name row whose leading P was the character the capture
        # lost. Padded at the end like any other short row it stays shifted one
        # place left, and the document code then reads as the filler beside it:
        # an Australian zone parsed with every check digit passing, its names
        # correct, and a code of "A" that the identity-row guard rightly
        # refuses. The shape is what identifies it -- filler, then a three
        # letter issuing state, on a row one character short -- and, as with
        # every reading offered here, the zone's check digits decide.
        if (
            len(compact) == length - 1
            and compact.startswith("<")
            and compact[1:4].isalpha()
            and mrz_row_shape("P" + compact, length) is not None
        ):
            restorations.append("P" + compact)
        # The eight-character bound exists because a data row's own length is
        # part of what its check digits cover, so a long guess there can be
        # wrong in a way nothing catches. A name row is different: its
        # trailing chevrons are padding and carry nothing, and a row that is
        # only letters and filler cannot be a data row at all. So its length
        # is restored however short the capture left it. An Indian passport's
        # name row came back twelve chevrons short; no reading of it was
        # offered at forty-four, the zone never assembled, and a page whose
        # second row was read whole and at full confidence produced neither
        # the passport number nor the date it expires.
        if (
            len(compact) < length - _MAX_RESTORED_FILLER
            and compact.endswith("<")
            and re.fullmatch(r"[A-Z<]+", compact)
            and mrz_row_shape(compact.ljust(length, "<"), length) is not None
        ):
            restorations.append(compact.ljust(length, "<"))
    restorations.extend(_sex_column_restorations(compact))
    return list(dict.fromkeys(restorations))


# The one column of a machine-readable zone that no check digit covers. ICAO
# leaves the sex character out of every checksum in all three formats, so it is
# also the only column where a dropped character can hide from them.
_UNCHECKED_SEX_COLUMN = {30: 7, 36: 20, 44: 20}

# What must already be present before that column for the row to be the data
# row: TD1 carries the birth date and its check digit, TD2 and TD3 the document
# number, its check digit, the nationality and the birth date with its own.
_DATA_ROW_BEFORE_SEX = {
    30: re.compile(r"\d{7}"),
    36: re.compile(r"[A-Z0-9<]{9}\d[A-Z<]{3}\d{6}\d"),
    44: re.compile(r"[A-Z0-9<]{9}\d[A-Z<]{3}\d{6}\d"),
}


def _sex_column_restorations(compact: str) -> list[str]:
    """Reopen the column a lost character can vanish into without a trace.

    A row one character short is padded on its right, and everything the
    capture dropped from the middle stays one place out of position. A
    Bulgarian passport lost the single letter of its sex field: the expiry
    date then read as 30.11.61, the sex as "2", the personal number lost its
    leading digit, and the expiry, optional-data and composite check digits
    all failed at once -- which says the row is wrong without saying where.

    Reopening this one column is safe precisely because the checksums ignore
    it: filler put back anywhere else in the row would be caught by them, so
    the repair can only ever produce a zone that validates. The row must
    already be a data row whose document-number and birth-date check digits
    pass, so a name row -- which no checksum protects -- is never split open.

    Filler is what goes back, not a letter. The check digits prove a character
    is missing there and are silent about which one it was; the printed sex
    field on the page is the evidence for that, and it is read separately.
    """
    restorations: list[str] = []
    for length, column in _UNCHECKED_SEX_COLUMN.items():
        if len(compact) != length - 1:
            continue
        head = compact[:column]
        if _DATA_ROW_BEFORE_SEX[length].fullmatch(head) is None:
            continue
        if length == 30:
            checked = validate_check(head[0:6], head[6])
        else:
            checked = (
                validate_check(head[0:9], head[9])
                and validate_check(head[13:19], head[19])
            )
        if not checked:
            continue
        restorations.append(compact[:column] + "<" + compact[column:])
    return restorations


def _filler_completions(compact: str, filler_lengths: set[int]) -> list[str]:
    """Complete a truncated row with a chevron run OCR returned on its own.

    ``_mrz_row_restorations`` restores at most a few chevrons, because padding
    a row by guesswork is how a genuine forty-four character row was once cut
    down to thirty-six. A run of chevrons the recognizer returned as its own
    box is not guesswork: where its length is exactly what the row is short of,
    it is the tail of that row. The box it carries cannot be used to place it --
    Google returned this Austrian passport's tail with the *preceding* row's
    bounding box, identical to the character -- so the length is the evidence,
    and the zone's check digits remain the arbiter.
    """
    if not compact or compact.strip("<") == "":
        return []
    completions: list[str] = []
    for length in MRZ_ROW_LENGTHS:
        if not 0 < len(compact) < length:
            continue
        if length - len(compact) not in filler_lengths:
            continue
        if mrz_row_shape(compact, length) is None:
            continue
        completions.append(compact.ljust(length, "<"))
    return completions


def find_mrz_lines(lines: list[OCRLine]) -> list[str]:
    candidates = []
    fragments = []
    compacted = [
        _mrz_compact(line.text)
        for line in lines
    ]
    filler_lengths = {
        len(compact) for compact in compacted
        if compact and compact.strip("<") == ""
    }
    for line, compact in zip(lines, compacted):
        if (
            _complete_td1_code_row_without_filler(compact)
            or _complete_passport_data_row_without_filler(compact)
        ):
            restored = [compact]
        else:
            restored = [row for row in _mrz_row_restorations(compact) if row.count("<") >= 1]
            restored.extend(
                row for row in _filler_completions(compact, filler_lengths)
                if row not in restored
            )
        if restored:
            top = min(point[1] for point in line.bounding_box)
            candidates.extend((top, row) for row in restored)
        elif len(compact) >= 8 and compact.count("<") >= 1:
            xs = [point[0] for point in line.bounding_box]
            ys = [point[1] for point in line.bounding_box]
            fragments.append((sum(ys) / len(ys), min(xs), compact, max(ys) - min(ys)))
    if len(candidates) >= 2:
        return [text for _, text in sorted(candidates)]
    # Some OCR runs split an MRZ row into two boxes. Rejoin fragments sharing
    # the same baseline before enforcing TD1/TD2/TD3 lengths.
    rows: list[list[tuple[float, str]]] = []
    row_centers: list[float] = []
    for center_y, x, text, height in sorted(fragments):
        index = next((i for i, existing_y in enumerate(row_centers) if abs(center_y - existing_y) <= max(12.0, height * 0.7)), None)
        if index is None:
            row_centers.append(center_y)
            rows.append([(x, text)])
        else:
            rows[index].append((x, text))
    for center_y, row in zip(row_centers, rows):
        joined = "".join(text for _, text in sorted(row))
        candidates.extend(
            (center_y, restored)
            for restored in _mrz_row_restorations(joined)
            if restored.count("<") >= 1
        )
    return [text for _, text in sorted(candidates)]
