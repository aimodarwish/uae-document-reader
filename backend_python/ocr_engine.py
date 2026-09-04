from __future__ import annotations

import logging
import platform
import re
import threading
import time

import cv2
import numpy as np
from paddleocr import TextDetection, TextRecognition

from .config import settings
from .image_utils import preprocess
from .ocr_types import ARABIC_RE, OCRLine

logger = logging.getLogger(__name__)


def _mkldnn_enabled() -> bool:
    """oneDNN accelerates x86 only; on arm64 it is at best inert."""
    if settings.enable_mkldnn is not None:
        return settings.enable_mkldnn
    return platform.machine().lower() in {"x86_64", "amd64"}


_VIN_SHAPED_RE = re.compile(r"[A-Z0-9]{14,}")


def _vin_shaped(text: str) -> bool:
    """Could this box hold a chassis number? Cheap, deliberately over-inclusive."""
    return bool(_VIN_SHAPED_RE.search(re.sub(r"[^A-Za-z0-9]", "", text).upper()))


def _crop_quad(image: np.ndarray, box: np.ndarray) -> np.ndarray:
    """Perspective-correct crop of one detected text quad.

    This is the step PaddleOCR normally performs internally between detection
    and recognition. We do it ourselves because we detect once and then feed the
    same crops to several recognisers.
    """
    box = np.array(box, dtype=np.float32).reshape(4, 2)
    width = int(max(np.linalg.norm(box[0] - box[1]), np.linalg.norm(box[2] - box[3])))
    height = int(max(np.linalg.norm(box[0] - box[3]), np.linalg.norm(box[1] - box[2])))
    width, height = max(width, 1), max(height, 1)

    dst = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(box, dst)
    crop = cv2.warpPerspective(
        image, matrix, (width, height),
        borderMode=cv2.BORDER_REPLICATE, flags=cv2.INTER_CUBIC,
    )

    # Tall-and-narrow crops are rotated upright before recognition.
    if height / max(width, 1) >= 1.5:
        crop = np.rot90(crop)
    return np.ascontiguousarray(crop)


class OCREngine:
    """PP-OCRv5 detection + English/Arabic recognition, loaded once, kept warm.

    Detection runs ONCE per page and every recogniser then reads the same crops.
    The original implementation ran a full detect+recognise pass per language,
    paying for detection twice (roughly half of total OCR time) and producing two
    competing line sets that had to be de-duplicated. It also cost accuracy: the
    English recogniser turns Arabic into latin noise, and that noise survived
    de-duplication and reached the extractor.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()

        common = {
            "device": "cpu",
            "cpu_threads": settings.ocr_cpu_threads,
            "enable_mkldnn": _mkldnn_enabled(),
        }
        rec_models = {"en": settings.rec_model_en, "ar": settings.rec_model_ar}

        started = time.perf_counter()
        self.detector = TextDetection(model_name=settings.det_model, **common)

        self.recognisers: dict[str, TextRecognition] = {}
        for lang in settings.languages:
            name = rec_models.get(lang)
            if not name:
                logger.warning("No recognition model configured for %r - skipping", lang)
                continue
            rec_args = dict(common)
            if settings.rec_input_shape:
                rec_args["input_shape"] = settings.rec_input_shape
            self.recognisers[lang] = TextRecognition(model_name=name, **rec_args)

        if not self.recognisers:
            raise RuntimeError(f"No usable recognition models for langs={settings.languages}")

        logger.info(
            "OCR ready in %.1fs | det=%s | rec=%s | mkldnn=%s | threads=%d",
            time.perf_counter() - started, settings.det_model,
            {k: rec_models[k] for k in self.recognisers},
            common["enable_mkldnn"], settings.ocr_cpu_threads,
        )

    def _detect(self, image: np.ndarray) -> list[np.ndarray]:
        results = self.detector.predict(image)
        if not results:
            return []
        polys = results[0].get("dt_polys")
        if polys is None or len(polys) == 0:
            return []
        return [np.array(p, dtype=np.float32) for p in polys]

    def _recognise(self, lang: str, crops: list[np.ndarray]) -> list[tuple[str, float]]:
        if not crops:
            return []

        # PaddleX pads every crop in a batch to the widest aspect ratio in that
        # batch -- it computes a width ordering but never applies it. Feeding
        # crops pre-sorted by aspect ratio keeps each batch uniform, which is
        # what makes batching a win instead of a 5x regression.
        order = sorted(
            range(len(crops)),
            key=lambda i: crops[i].shape[1] / max(crops[i].shape[0], 1),
        )
        results = self.recognisers[lang].predict(
            [crops[i] for i in order], batch_size=settings.ocr_rec_batch_size
        )

        readings: list[tuple[str, float]] = [("", 0.0)] * len(crops)
        for position, item in enumerate(results):
            if position >= len(order):
                break
            try:
                text = str(item["rec_text"]).strip()
                score = float(item["rec_score"])
            except (KeyError, TypeError, ValueError):
                continue
            readings[order[position]] = (text, score)
        return readings

    def _recognise_all(self, crops: list[np.ndarray]) -> dict[str, list[tuple[str, float]]]:
        """Read every crop, spending the English model only where it can help.

        The Arabic model is the generalist: its charset covers latin and digits,
        so it reads the entire bilingual card on its own. The English model is a
        specialist that cannot produce Arabic at all.

        Measured on a full bilingual card: Arabic alone recovered every field
        in 3.3 s; running both recognisers over every box took 6.0 s for the
        same result. So English is now reserved for the boxes where it can
        actually change the answer -- latin text the Arabic model was unsure
        about, plus anything shaped like a VIN, which is the field least
        tolerant of a single wrong character.
        """
        if not crops:
            return {lang: [] for lang in self.recognisers}

        per_lang: dict[str, list[tuple[str, float]]] = {}
        needs_english = list(range(len(crops)))

        if "ar" in self.recognisers:
            per_lang["ar"] = self._recognise("ar", crops)
            # Any weakly-read box gets a second opinion, whatever script the
            # Arabic model thought it saw. Detection sometimes merges several
            # table cells into one box; the Arabic model then returns confident-
            # looking noise for a cell that actually holds a date. On a real
            # card that lost Reg. Date entirely -- Arabic read it 0.55 as
            # 'تاريخ الترخا الر', English read it 0.87 as 'Reg.Dat22-12-2025'.
            needs_english = [
                index
                for index, (text, score) in enumerate(per_lang["ar"])
                if score < settings.ocr_second_pass_threshold or _vin_shaped(text)
            ]

        if "en" in self.recognisers:
            readings: list[tuple[str, float]] = [("", 0.0)] * len(crops)
            if needs_english:
                subset = self._recognise("en", [crops[i] for i in needs_english])
                for position, index in enumerate(needs_english):
                    if position < len(subset):
                        readings[index] = subset[position]
            per_lang["en"] = readings

        # NB: must not be dict.setdefault -- Python evaluates the default
        # argument even when the key is already present, which silently ran a
        # second full recognition pass per language over every crop and threw
        # the result away. It cost ~6 s of the ~9 s per card.
        for lang in self.recognisers:
            if lang not in per_lang:
                per_lang[lang] = self._recognise(lang, crops)
        return per_lang

    def _pick(self, per_lang: dict[str, tuple[str, float]]) -> tuple[str, float, str] | None:
        """Choose the better reading of one box across the recognisers.

        The Arabic model's charset covers latin and digits too, so it can read
        either script; the English model cannot produce Arabic at all and emits
        latin noise when pointed at it. So any genuine Arabic output wins, and
        otherwise the specialised English model is preferred for latin/numeric
        text unless it is clearly less certain.
        """
        arabic = [
            (lang, text, score)
            for lang, (text, score) in per_lang.items()
            if text and ARABIC_RE.search(text)
        ]
        if arabic:
            lang, text, score = max(arabic, key=lambda x: x[2])

            # Arabic script normally settles it -- but a box the Arabic model
            # was unsure about, which the English model read confidently as
            # latin, is a merged cell whose real content is a date or a code.
            english = per_lang.get("en")
            if (
                english
                and english[0]
                and not ARABIC_RE.search(english[0])
                and english[1] - score >= settings.ocr_script_override_margin
            ):
                return english[0], english[1], "en"
            return text, score, lang

        english = per_lang.get("en")
        if english and english[0]:
            best_other = max(
                ((l, t, s) for l, (t, s) in per_lang.items() if l != "en" and t),
                key=lambda x: x[2], default=None,
            )
            if best_other is None or english[1] >= best_other[2] - 0.15:
                return english[0], english[1], "en"
            return best_other[1], best_other[2], best_other[0]

        candidates = [(l, t, s) for l, (t, s) in per_lang.items() if t]
        if not candidates:
            return None
        lang, text, score = max(candidates, key=lambda x: x[2])
        return text, score, lang

    def read_pages(self, pages: list[np.ndarray]) -> tuple[list[OCRLine], dict[str, float]]:
        lines: list[OCRLine] = []
        timings: dict[str, float] = {
            "preprocess_ms": 0.0, "detect_ms": 0.0, "recognise_ms": 0.0, "boxes": 0,
        }

        with self._lock:
            for page_no, original in enumerate(pages):
                t0 = time.perf_counter()
                image = preprocess(original)
                t1 = time.perf_counter()

                boxes = self._detect(image)
                t2 = time.perf_counter()

                crops = [_crop_quad(image, box) for box in boxes]
                per_lang = self._recognise_all(crops)
                t3 = time.perf_counter()

                timings["preprocess_ms"] += (t1 - t0) * 1000
                timings["detect_ms"] += (t2 - t1) * 1000
                timings["recognise_ms"] += (t3 - t2) * 1000
                timings["boxes"] += len(boxes)

                for index, box in enumerate(boxes):
                    readings = {
                        lang: results[index]
                        for lang, results in per_lang.items()
                        if index < len(results)
                    }
                    winner = self._pick(readings)
                    if not winner:
                        continue

                    text, score, lang = winner
                    if text and score >= settings.ocr_min_confidence:
                        lines.append(
                            OCRLine(
                                text=text, score=score,
                                box=[[float(x), float(y)] for x, y in box],
                                lang=lang, page=page_no,
                            )
                        )

        timings = {k: (round(v, 1) if isinstance(v, float) else v) for k, v in timings.items()}
        return sorted(lines, key=lambda x: (x.page, x.cy, x.cx)), timings


ocr_engine: OCREngine | None = None
_engine_lock = threading.Lock()


def get_ocr_engine() -> OCREngine:
    global ocr_engine
    if ocr_engine is None:
        with _engine_lock:
            if ocr_engine is None:
                ocr_engine = OCREngine()
    return ocr_engine
