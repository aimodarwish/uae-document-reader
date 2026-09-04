"""OCR data types, kept free of any PaddleOCR import.

The extractor consumes OCRLine and nothing else from the OCR layer. Keeping the
type here means app.extractor can be imported -- and unit-tested -- without
pulling in paddlepaddle, so extraction logic can be iterated on in milliseconds
instead of waiting on a container.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any

ARABIC_RE = re.compile(r"[؀-ۿ]")


@dataclass
class OCRLine:
    text: str
    score: float
    box: list[list[float]]
    lang: str
    page: int = 0

    @property
    def cx(self) -> float:
        return sum(p[0] for p in self.box) / len(self.box)

    @property
    def cy(self) -> float:
        return sum(p[1] for p in self.box) / len(self.box)

    @property
    def width(self) -> float:
        xs = [p[0] for p in self.box]
        return max(xs) - min(xs)

    @property
    def height(self) -> float:
        ys = [p[1] for p in self.box]
        return max(ys) - min(ys)

    @property
    def left(self) -> float:
        return min(p[0] for p in self.box)

    @property
    def right(self) -> float:
        return max(p[0] for p in self.box)

    @property
    def top(self) -> float:
        return min(p[1] for p in self.box)

    @property
    def bottom(self) -> float:
        return max(p[1] for p in self.box)

    @property
    def has_arabic(self) -> bool:
        return bool(ARABIC_RE.search(self.text))

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["box"] = [[round(x, 1), round(y, 1)] for x, y in self.box]
        result["score"] = round(self.score, 4)
        result.update({"cx": round(self.cx, 1), "cy": round(self.cy, 1)})
        return result



# NOTE ON ARABIC TEXT DIRECTION
#
# PaddleX 3.5.2 contains a python-bidi get_display() call for the Arabic
# recognition models, which would hand back VISUAL order. Measured against a
# real RTA card, the installed stack does NOT apply it -- output arrives in
# LOGICAL order, which is what we want.
#
# Rather than hard-code either behaviour and have a paddlex upgrade silently
# reverse every Arabic field, the extractor calibrates direction per document
# against the card's own known labels. See _orientation_is_reversed there.
