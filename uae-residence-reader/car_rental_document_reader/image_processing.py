from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np
from PIL import Image

from .config import AppConfig
from .schemas import QualityInfo


@dataclass
class PreprocessedImage:
    original: Image.Image
    normalized: Image.Image
    variants: dict[str, Image.Image]
    quality: QualityInfo
    transformations: list[dict[str, Any]] = field(default_factory=list)


# Six-row punched/dot-matrix numerals used on association-issued booklet
# serials.  Keeping this as pixels rather than an OCR model makes the targeted
# read deterministic and effectively free once the page is already in memory.
_DOT_MATRIX_DIGITS: dict[str, tuple[str, ...]] = {
    "0": (".##.", "#..#", "#..#", "#..#", "#..#", ".##."),
    "1": (".#.", "##.", ".#.", ".#.", ".#.", "###"),
    "2": (".##.", "#..#", "...#", "..#.", ".#..", "####"),
    # The Touring Club d'Algerie impact head uses a rounded lower bowl: its
    # photographed 3 is ``#### / ...# / ..#. / ...# / #..# / .##.``.  The
    # previous generic terminal-style 3 was far enough away that this exact
    # glyph scored as a 5.  Keep that other head below as an alternate.
    "3": ("####", "...#", "..#.", "...#", "#..#", ".##."),
    "4": ("..#.", ".##.", "#.#.", "####", "..#.", "..#."),
    "5": ("####", "#...", "###.", "...#", "#..#", ".##."),
    "6": (".##.", "#...", "###.", "#..#", "#..#", ".##."),
    "7": ("####", "...#", "..#.", ".#..", ".#..", ".#.."),
    "8": (".##.", "#..#", "#..#", ".##.", "#..#", ".##."),
    "9": (".##.", "#..#", "#..#", ".###", "...#", ".##."),
}


_DOT_MATRIX_DIGIT_ALTERNATES: dict[str, tuple[tuple[str, ...], ...]] = {
    "3": (("###.", "...#", "...#", ".##.", "...#", "###."),),
}


def _cluster_axis(values: list[float], tolerance: float) -> list[float]:
    clusters: list[list[float]] = []
    for value in sorted(values):
        if not clusters or value - sum(clusters[-1]) / len(clusters[-1]) > tolerance:
            clusters.append([value])
        else:
            clusters[-1].append(value)
    return [sum(cluster) / len(cluster) for cluster in clusters]


def _decode_six_row_dot_number(
    points: list[tuple[float, float]], dot_size: float,
) -> tuple[str, float] | None:
    if len(points) < 24:
        return None
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.05)
    # A photographed booklet is often keystoned: the rightmost copy of the
    # serial can climb a full dot row from left to right.  Fit that baseline
    # before clustering, otherwise bottom dots from one digit are assigned to
    # the penultimate row of the next and a 3 can look like a 5.
    mean_x = sum(point[0] for point in points) / len(points)
    best_rows: tuple[float, list[float], float] | None = None
    for slope in np.linspace(-0.30, 0.30, 25):
        corrected = [y - float(slope) * (x - mean_x) for x, y in points]
        samples = np.float32([[value] for value in corrected])
        compactness, _, centers = cv2.kmeans(
            samples, 6, None, criteria, 4, cv2.KMEANS_PP_CENTERS,
        )
        candidate_rows = sorted(float(value) for value in centers[:, 0])
        steps = np.diff(candidate_rows)
        if len(steps) != 5 or min(steps) <= max(2.0, dot_size * 0.55):
            continue
        metric = float(compactness) / max(1.0, len(points) * float(np.median(steps)) ** 2)
        metric += float(np.std(steps) / max(float(np.mean(steps)), 1.0)) * 0.25
        if best_rows is None or metric < best_rows[0]:
            best_rows = (metric, candidate_rows, float(slope))
    if best_rows is None:
        return None
    _, rows, slope = best_rows
    row_steps = np.diff(rows)
    if (
        len(row_steps) != 5
        or min(row_steps) <= max(2.0, dot_size * 0.55)
        or max(row_steps) / min(row_steps) > 1.9
    ):
        return None
    row_step = float(np.median(row_steps))
    assigned = [
        (x, min(range(6), key=lambda index: abs(y - rows[index])))
        for x, raw_y in points
        for y in (raw_y - slope * (x - mean_x),)
        if min(abs(y - row) for row in rows) <= row_step * 0.48
    ]
    x_columns = _cluster_axis([x for x, _ in assigned], max(2.0, dot_size * 0.72))
    if len(x_columns) < 15:
        return None
    gaps = np.diff(x_columns)
    small_gaps = sorted(float(gap) for gap in gaps)[:max(1, int(len(gaps) * 0.65))]
    column_step = float(np.median(small_gaps))
    if column_step <= dot_size * 0.55:
        return None
    groups: list[list[float]] = [[x_columns[0]]]
    for previous, column in zip(x_columns, x_columns[1:]):
        if column - previous > column_step * 1.34:
            groups.append([column])
        else:
            groups[-1].append(column)
    if not 5 <= len(groups) <= 12 or any(not 3 <= len(group) <= 4 for group in groups):
        return None

    decoded: list[str] = []
    digit_scores: list[float] = []
    for group in groups:
        observed = [[False] * len(group) for _ in range(6)]
        left, right = group[0] - column_step * 0.5, group[-1] + column_step * 0.5
        for x, row_index in assigned:
            if not left <= x <= right:
                continue
            column_index = min(
                range(len(group)), key=lambda index: abs(x - group[index]),
            )
            if abs(x - group[column_index]) <= column_step * 0.48:
                observed[row_index][column_index] = True
        observed_set = {
            (row, column)
            for row in range(6) for column in range(len(group))
            if observed[row][column]
        }
        options: list[tuple[float, str]] = []
        for digit, canonical in _DOT_MATRIX_DIGITS.items():
            for pattern in (
                canonical, *_DOT_MATRIX_DIGIT_ALTERNATES.get(digit, ()),
            ):
                if len(pattern[0]) != len(group):
                    continue
                expected = {
                    (row, column)
                    for row, pattern_row in enumerate(pattern)
                    for column, value in enumerate(pattern_row)
                    if value == "#"
                }
                union = expected | observed_set
                score = 1.0 - len(expected ^ observed_set) / max(1, len(union))
                options.append((score, digit))
        if not options:
            return None
        score, digit = max(options)
        if score < 0.58:
            return None
        decoded.append(digit)
        digit_scores.append(score)
    return "".join(decoded), float(sum(digit_scores) / len(digit_scores))


def dot_matrix_number(
    image: Image.Image,
) -> tuple[str, tuple[float, float, float, float], float] | None:
    """Decode a repeated six-row dot serial near the foot of an IDP page.

    The Algerian 1968-convention booklet prints its permit number twice as
    large isolated dots.  Text OCR ignores those disconnected circles and, in
    the reported sample, selected ``196808`` from the convention sentence
    instead.  This pass examines only the bottom quarter, groups equal-sized
    circular components into six-row grids, and accepts either two agreeing
    copies or one high-quality grid.  It runs in a few milliseconds and is
    invoked only after tourist routing has established an Algerian IDP.
    """
    gray = cv2.cvtColor(_as_bgr(image), cv2.COLOR_BGR2GRAY)
    height, width = gray.shape[:2]
    # How far down the *capture* the serial sits is not how far down the page
    # it is printed. A photograph that keeps the desk below the booklet pushes
    # everything up the frame: on the Constantine permit in this project's bug
    # report the serial's rows began at 0.714 of the image height, so a band
    # starting at 0.72 sliced the top off its glyphs and the region test then
    # refused what was left for sitting above 0.78. Both bounds are moved to
    # the middle of the frame. What keeps this from reading something else is
    # the grid itself -- six rows, three-or-four-column glyphs, a per-digit
    # match score -- not where the band happens to start.
    y_offset = int(height * 0.55)
    lower = gray[y_offset:, :]
    block = max(15, int(round(min(height, width) * 0.035)) | 1)
    binary = cv2.adaptiveThreshold(
        lower, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, block, 9,
    )
    count, _, stats, centroids = cv2.connectedComponentsWithStats(binary, 8)
    minimum = max(2, int(round(min(height, width) * 0.0035)))
    maximum = max(minimum + 1, int(round(min(height, width) * 0.022)))
    dots: list[tuple[float, float, int, int]] = []
    for index in range(1, count):
        x, y, component_width, component_height, area = map(int, stats[index])
        if not (
            minimum <= component_width <= maximum
            and minimum <= component_height <= maximum
            and 0.55 <= component_width / max(component_height, 1) <= 1.75
            and 0.30 <= area / max(component_width * component_height, 1) <= 0.92
        ):
            continue
        center_x, center_y = centroids[index]
        dots.append((float(center_x), float(center_y + y_offset), component_width, component_height))
    if len(dots) < 24:
        return None
    dot_size = float(np.median([max(item[2], item[3]) for item in dots]))
    mask = np.zeros((height, width), dtype=np.uint8)
    for center_x, center_y, _, _ in dots:
        cv2.circle(mask, (round(center_x), round(center_y)), max(1, round(dot_size * 0.55)), 255, -1)
    # The narrow glyph for ``1`` leaves a wider inter-character gap before the
    # following digit than the four-column glyphs do.  Four dot diameters close
    # that gap while remaining far below the space between two printed copies.
    join = max(5, round(dot_size * 4.0))
    joined = cv2.dilate(
        mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (join, join)),
    )
    region_count, _, region_stats, _ = cv2.connectedComponentsWithStats(joined, 8)
    readings: list[tuple[str, tuple[float, float, float, float], float]] = []
    for index in range(1, region_count):
        x, y, region_width, region_height, _ = map(int, region_stats[index])
        if (
            y < height * 0.58
            or region_width < dot_size * 15
            or region_height < dot_size * 5
            or region_width / max(region_height, 1) < 2.4
        ):
            continue
        region_points = [
            (center_x, center_y)
            for center_x, center_y, _, _ in dots
            if x <= center_x <= x + region_width and y <= center_y <= y + region_height
        ]
        decoded = _decode_six_row_dot_number(region_points, dot_size)
        if decoded is None:
            continue
        value, score = decoded
        readings.append((
            value, (float(x), float(y), float(x + region_width), float(y + region_height)), score,
        ))
    if not readings:
        return None
    by_value: dict[str, list[tuple[str, tuple[float, float, float, float], float]]] = {}
    for reading in readings:
        by_value.setdefault(reading[0], []).append(reading)
    value, agreeing = max(by_value.items(), key=lambda item: (len(item[1]), max(one[2] for one in item[1])))
    if len(agreeing) >= 2:
        best = max(agreeing, key=lambda item: item[2])
        return value, best[1], min(0.96, 0.90 + best[2] * 0.06)
    best = agreeing[0]
    if best[2] < 0.76:
        return None
    return value, best[1], min(0.86, 0.72 + best[2] * 0.14)


# Card borders are found just as reliably on a downscaled copy, and bounding
# the contour search keeps a 12-megapixel phone photograph from dominating the
# preprocessing budget.
_CONTOUR_DETECTION_MAX_SIDE = 1400


def _as_bgr(image: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.asarray(image.convert("RGB")), cv2.COLOR_RGB2BGR)


# Sharpness is measured at one fixed size, whatever size the capture arrived
# at. The Laplacian's variance is per pixel, so spreading a photograph's edges
# over more pixels lowers it: the same picture of the same card scored 5.20 at
# 900 pixels wide and 0.16 at 2400, a factor of thirty-two for holding the
# camera still and using a better phone. Every threshold built on that number
# -- the blur warning, the unreadable flag, the low-quality penalty that forces
# a field to review -- was therefore penalising resolution, and a Queensland
# licence photographed at 1620 pixels was ruled unreadable while a smaller,
# genuinely blurrier copy of it was not.
_BLUR_REFERENCE_LONG_SIDE = 1000


def _gray_blur_score(gray: np.ndarray) -> float:
    height, width = gray.shape[:2]
    longest = max(height, width)
    if longest > _BLUR_REFERENCE_LONG_SIDE:
        scale = _BLUR_REFERENCE_LONG_SIDE / longest
        gray = cv2.resize(
            gray, (max(1, round(width * scale)), max(1, round(height * scale))),
            interpolation=cv2.INTER_AREA,
        )
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def blur_score(image: Image.Image) -> float:
    return _gray_blur_score(cv2.cvtColor(_as_bgr(image), cv2.COLOR_BGR2GRAY))


def _glare_fraction(bgr: np.ndarray) -> float:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask = ((hsv[:, :, 2] >= 245) & (hsv[:, :, 1] <= 40)).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if count <= 1:
        return 0.0
    significant = sum(int(stats[i, cv2.CC_STAT_AREA]) for i in range(1, count) if stats[i, cv2.CC_STAT_AREA] >= 64)
    return significant / mask.size


def red_ink_boxes(image: Image.Image) -> list[tuple[float, float, float, float]]:
    """Every patch of red print on the page, as ``(x1, y1, x2, y2)``.

    An international driving permit prints one thing in red -- its own number,
    beside the numero sign on the left of the holder page. Everything else on
    the booklet, including the national licence number at the foot of the same
    column that a digit search keeps choosing instead, is black.

    Colour is the cue that survives the capture. The permit number is set in a
    small face over a patterned blue guilloche, so contrast alone is weak and
    the red channel is exactly where the guilloche is brightest -- which is why
    a greyscale variant flattens the number into its background. Hue does not
    flatten: a phone photograph moves red's saturation and value, not its
    position on the wheel.

    Both ends of the hue wheel are matched because red straddles the wrap. The
    saturation floor keeps the page's own warm paper tone out, and the area
    floor drops the speckle the guilloche leaves behind. The issuing
    association's round red seal also matches, and is left in: callers keep
    only boxes that a recognised text row already overlaps, and the seal
    carries no digits to be mistaken for a number.
    """
    bgr = _as_bgr(image.convert("RGB"))
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    hue, saturation, value = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    red = (
        ((hue <= 12) | (hue >= 168)) & (saturation >= 90) & (value >= 70)
    ).astype(np.uint8)
    if not int(np.count_nonzero(red)):
        return []
    # Close along the row so the separate strokes of "01 EA 044761" join into
    # one box instead of returning eleven glyph-sized fragments.
    height, width = red.shape[:2]
    kernel_width = max(3, round(width * 0.012))
    red = cv2.morphologyEx(
        red, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_width, 3)),
    )
    count, _, stats, _ = cv2.connectedComponentsWithStats(red, 8)
    minimum_area = max(40, round(height * width * 0.00002))
    boxes = [
        (
            float(stats[index, cv2.CC_STAT_LEFT]), float(stats[index, cv2.CC_STAT_TOP]),
            float(stats[index, cv2.CC_STAT_LEFT] + stats[index, cv2.CC_STAT_WIDTH]),
            float(stats[index, cv2.CC_STAT_TOP] + stats[index, cv2.CC_STAT_HEIGHT]),
        )
        for index in range(1, count)
        if stats[index, cv2.CC_STAT_AREA] >= minimum_area
    ]
    return sorted(boxes, key=lambda box: (box[2] - box[0]) * (box[3] - box[1]), reverse=True)


def _illumination_flattened(bgr: np.ndarray) -> Image.Image:
    """Remove the lighting from the page and leave the ink.

    A laminated card under a counter light throws a broad bright band across
    its own print. That band is low-frequency -- it varies over centimetres,
    while characters vary over fractions of a millimetre -- so dividing the
    page by a heavily blurred copy of itself cancels the lighting and leaves
    the strokes. This is what the previous fallback could not do: CLAHE
    equalises a neighbourhood, so inside a glare band it raises the noise
    floor and lowers the letters, which is the wrong way round.

    Done per channel so the result is still colour, because the permit
    number's red ink is read off this same page.
    """
    # The background is by definition the part of the page that survives
    # throwing most of the pixels away, so it is computed on a small copy and
    # stretched back. A sigma wide enough to erase characters at full
    # resolution costs seconds per page; the same field costs milliseconds at
    # an eighth the size, and the two are indistinguishable once divided out.
    height, width = bgr.shape[:2]
    small = cv2.resize(
        bgr, (max(8, width // 8), max(8, height // 8)), interpolation=cv2.INTER_AREA,
    )
    background = cv2.resize(
        cv2.GaussianBlur(small, (0, 0), max(2.0, max(small.shape[:2]) / 20.0)),
        (width, height), interpolation=cv2.INTER_LINEAR,
    )
    # 1 rather than 0 in the divisor: a genuinely black pixel must not divide
    # by zero and come back white.
    flattened = cv2.divide(bgr, np.maximum(background, 1), scale=255)
    channels = cv2.split(flattened)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    balanced = cv2.merge([clahe.apply(channel) for channel in channels])
    return Image.fromarray(cv2.cvtColor(balanced, cv2.COLOR_BGR2RGB))


def _deblurred(bgr: np.ndarray) -> Image.Image:
    """Restore stroke edges a hand-held capture smeared.

    An unsharp mask at two radii rather than the single 3x3 kernel the
    "sharpened" variant applies. A 3x3 kernel sharpens detail one pixel wide,
    which on a blurred capture is noise; the strokes that were lost are two to
    four pixels wide and need a radius to match. The wider pass recovers the
    stroke, the narrow one its edge, and the weights stay modest because an
    over-sharpened character grows halos that a recogniser reads as extra
    marks.
    """
    scale = max(bgr.shape[:2]) / 1600.0
    wide = cv2.GaussianBlur(bgr, (0, 0), max(1.2, 2.4 * scale))
    narrow = cv2.GaussianBlur(bgr, (0, 0), max(0.6, 1.1 * scale))
    sharpened = cv2.addWeighted(bgr, 1.7, wide, -0.7, 0)
    sharpened = cv2.addWeighted(sharpened, 1.4, narrow, -0.4, 0)
    return Image.fromarray(cv2.cvtColor(sharpened, cv2.COLOR_BGR2RGB))


def zoom_repair(image: Image.Image) -> Image.Image:
    """Flatten the lighting and restore the edges of one enlarged crop.

    The same two repairs the whole-page fallbacks apply, run on a region a few
    hundred pixels across instead. At that size both are effectively free, and
    a glare band that covers a third of a page usually covers all of one row.
    """
    flattened = _illumination_flattened(_as_bgr(image))
    return _deblurred(_as_bgr(flattened))


def _build_ocr_variants(
    normalized: Image.Image, names: set[str],
) -> dict[str, Image.Image]:
    """Render only the OCR views the current recognition stage will consume."""
    bgr = _as_bgr(normalized)
    luminance = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    builders: dict[str, Any] = {
        "original_normalized": lambda: normalized,
        "contrast_enhanced": lambda: Image.fromarray(
            cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(luminance),
        ).convert("RGB"),
        "grayscale": lambda: Image.fromarray(luminance).convert("RGB"),
        "adaptive_threshold": lambda: Image.fromarray(cv2.adaptiveThreshold(
            luminance, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 11,
        )).convert("RGB"),
        "sharpened": lambda: Image.fromarray(cv2.cvtColor(
            cv2.filter2D(bgr, -1, np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])),
            cv2.COLOR_BGR2RGB,
        )),
        "illumination_flattened": lambda: _illumination_flattened(bgr),
        "deblurred": lambda: _deblurred(bgr),
    }
    return {
        name: build() for name, build in builders.items() if name in names
    }


def ensure_ocr_variants(
    preprocessed: PreprocessedImage, names: tuple[str, ...] | set[str],
) -> dict[str, Image.Image]:
    """Build requested fallback views lazily, preserving their existing pixels.

    Clean pages finish after the normalized read, so creating three additional
    full-resolution images before knowing whether they are needed wastes CPU
    and memory. A difficult page still receives the exact same repair views;
    they are simply rendered at the point the fallback recognizer is invoked.
    """
    wanted = set(names)
    missing = wanted - set(preprocessed.variants)
    if missing:
        preprocessed.variants.update(
            _build_ocr_variants(preprocessed.normalized, missing),
        )
    # Keep the established builder ordering. Some OCR engines use insertion
    # order for deterministic variant processing and evidence tie-breaking.
    return {
        name: image for name, image in preprocessed.variants.items()
        if name in wanted
    }


def _estimate_skew(gray: np.ndarray) -> float:
    binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    points = np.column_stack(np.where(binary > 0))
    if len(points) < 100:
        return 0.0
    angle = float(cv2.minAreaRect(points[:, ::-1].astype(np.float32))[-1])
    if angle > 45:
        angle -= 90
    return angle if abs(angle) <= 15 else 0.0


def _rotate_bound(bgr: np.ndarray, angle: float) -> np.ndarray:
    if abs(angle) < 0.25:
        return bgr
    h, w = bgr.shape[:2]
    center = (w / 2, h / 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    cos, sin = abs(matrix[0, 0]), abs(matrix[0, 1])
    nw, nh = int(h * sin + w * cos), int(h * cos + w * sin)
    matrix[0, 2] += nw / 2 - center[0]
    matrix[1, 2] += nh / 2 - center[1]
    return cv2.warpAffine(bgr, matrix, (nw, nh), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


def _order_quad(points: np.ndarray) -> np.ndarray:
    result = np.zeros((4, 2), dtype=np.float32)
    sums, diffs = points.sum(axis=1), np.diff(points, axis=1).ravel()
    result[0], result[2] = points[np.argmin(sums)], points[np.argmax(sums)]
    result[1], result[3] = points[np.argmin(diffs)], points[np.argmax(diffs)]
    return result


def _ink_outside(bgr: np.ndarray, quad: np.ndarray) -> float:
    """The share of the frame's edge detail that a crop to this quad discards.

    An outline is not always found around both halves of a two-sided capture:
    on an Italian licence only the reverse had a border the contour search
    could close, so the frame was cropped to it and the front -- the name, the
    licence number, the dates, none of which the back carries -- was thrown
    away before OCR ran. What was left behind is measurable, and a crop that
    discards a quarter of the page's detail is not a crop of the page.
    """
    edges = cv2.Canny(cv2.GaussianBlur(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), (5, 5), 0), 60, 180)
    total = int(np.count_nonzero(edges))
    if total == 0:
        return 0.0
    mask = np.zeros(edges.shape, np.uint8)
    cv2.fillConvexPoly(mask, quad.astype(np.int32), 255)
    return 1.0 - int(np.count_nonzero(cv2.bitwise_and(edges, edges, mask=mask))) / total


def _card_quads(bgr: np.ndarray, minimum_area_ratio: float) -> list[np.ndarray]:
    """Every card-shaped quadrilateral in the frame, largest first."""
    h, w = bgr.shape[:2]
    # Card edges are a large-scale feature, so the contour search runs on a
    # bounded-size copy and the corners it finds are scaled back up. The warp
    # itself still uses the full-resolution pixels.
    detection_scale = min(1.0, _CONTOUR_DETECTION_MAX_SIDE / max(h, w))
    if detection_scale < 1.0:
        detection = cv2.resize(
            bgr, (max(1, round(w * detection_scale)), max(1, round(h * detection_scale))),
            interpolation=cv2.INTER_AREA,
        )
    else:
        detection = bgr
    gray = cv2.cvtColor(detection, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 60, 180)
    detection_h, detection_w = detection.shape[:2]
    contours = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:12]
    quads: list[np.ndarray] = []
    for contour in contours:
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        area = cv2.contourArea(approx)
        if len(approx) != 4 or not (
            minimum_area_ratio * detection_h * detection_w
            <= area <= 0.995 * detection_h * detection_w
        ):
            continue
        quad = _order_quad(approx.reshape(4, 2).astype(np.float32)) / detection_scale
        widths = [np.linalg.norm(quad[1] - quad[0]), np.linalg.norm(quad[2] - quad[3])]
        heights = [np.linalg.norm(quad[3] - quad[0]), np.linalg.norm(quad[2] - quad[1])]
        if int(max(widths)) < 400 or int(max(heights)) < 250:
            continue
        if any(_quads_overlap(quad, existing) for existing in quads):
            continue
        quads.append(quad)
    return quads


def _quad_bounds(quad: np.ndarray) -> tuple[float, float, float, float]:
    return (
        float(quad[:, 0].min()), float(quad[:, 1].min()),
        float(quad[:, 0].max()), float(quad[:, 1].max()),
    )


def _quads_overlap(a: np.ndarray, b: np.ndarray) -> bool:
    ax1, ay1, ax2, ay2 = _quad_bounds(a)
    bx1, by1, bx2, by2 = _quad_bounds(b)
    overlap = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(0.0, min(ay2, by2) - max(ay1, by1))
    smaller = min((ax2 - ax1) * (ay2 - ay1), (bx2 - bx1) * (by2 - by1))
    return smaller <= 0 or overlap / smaller > 0.20


def split_card_sides(image: Image.Image) -> list[Image.Image]:
    """Cut a capture holding both sides of one card into two pages.

    Photographing the front and the back together, one above the other, is how
    these documents arrive: three of the first three real bundles were captured
    that way. The contour search saw two cards and cropped to whichever was
    larger, so an Italian licence reached OCR as its reverse alone -- no name,
    no licence number, no dates, none of which are printed on the back. Nothing
    downstream could recover them, because they were no longer in the picture.

    The quads decide only where to cut. The image is divided along the gap
    between them and every pixel is kept, so each half then goes through the
    ordinary single-card path and is deskewed there.
    """
    bgr = _as_bgr(image.convert("RGB"))
    height, width = bgr.shape[:2]
    quads = _card_quads(bgr, minimum_area_ratio=0.18)
    if len(quads) < 2:
        return [image]
    first, second = (_quad_bounds(quad) for quad in quads[:2])
    for axis, size in ((1, height), (0, width)):
        low, high = sorted((first, second), key=lambda bounds: bounds[axis])
        gap_start, gap_end = high[axis], low[axis + 2]
        if gap_start <= gap_end:
            continue                      # they overlap along this axis
        split = int(round((gap_start + gap_end) / 2))
        if not 0.25 * size <= split <= 0.75 * size:
            continue                      # one part would be too small to be a card
        return [
            image.crop((0, 0, width, split)), image.crop((0, split, width, height)),
        ] if axis == 1 else [
            image.crop((0, 0, split, height)), image.crop((split, 0, width, height)),
        ]
    return [image]


def _perspective_if_reliable(bgr: np.ndarray) -> tuple[np.ndarray, bool, bool]:
    h, w = bgr.shape[:2]
    quads = _card_quads(bgr, minimum_area_ratio=0.30)
    if not quads:
        return bgr, False, True
    # A second card beside the first means cropping to either one throws the
    # other away. Splitting happens before this point; if it declined, the
    # frame is left whole rather than half of it discarded.
    if len(quads) > 1:
        return bgr, False, True
    quad = quads[0]
    if _ink_outside(bgr, quad) > 0.25:
        # Another document is in the frame whose own outline was not closed
        # enough to be found. Keep the whole capture rather than crop half of
        # it away.
        return bgr, False, True
    widths = [np.linalg.norm(quad[1] - quad[0]), np.linalg.norm(quad[2] - quad[3])]
    heights = [np.linalg.norm(quad[3] - quad[0]), np.linalg.norm(quad[2] - quad[1])]
    tw, th = int(max(widths)), int(max(heights))
    destination = np.array([[0, 0], [tw - 1, 0], [tw - 1, th - 1], [0, th - 1]], np.float32)
    warped = cv2.warpPerspective(bgr, cv2.getPerspectiveTransform(quad, destination), (tw, th))
    touches = bool(np.any(quad[:, 0] < 8) or np.any(quad[:, 1] < 8) or np.any(quad[:, 0] > w - 9) or np.any(quad[:, 1] > h - 9))
    return warped, True, touches


def _trim_dark_screenshot_margins(
    bgr: np.ndarray,
) -> tuple[np.ndarray, tuple[int, int, int, int] | None]:
    """Remove large, solid black screenshot gutters before quality analysis.

    Phone screenshots often place a white document page between black app
    margins.  Counting those margins as document pixels makes the same image
    simultaneously "over" and "under" exposed and can mark a readable
    passport as unreadable.  Trim only edge-connected near-black bands that
    discard a material part of the frame and leave a bright document-sized
    region; a dark passport cover or a genuinely dark photograph is retained.
    """
    height, width = bgr.shape[:2]
    luminance = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    dark = luminance <= 12

    def edge_limit(values: np.ndarray, *, start: bool) -> int:
        indices = range(len(values)) if start else range(len(values) - 1, -1, -1)
        count = 0
        for index in indices:
            if not values[index]:
                break
            count += 1
        return count

    # A gutter fills virtually all of an edge row/column.  Allow a small
    # amount of phone UI antialiasing while refusing a black document edge.
    dark_rows = np.mean(dark, axis=1) >= 0.985
    dark_columns = np.mean(dark, axis=0) >= 0.985
    top = edge_limit(dark_rows, start=True)
    bottom = height - edge_limit(dark_rows, start=False)
    left = edge_limit(dark_columns, start=True)
    right = width - edge_limit(dark_columns, start=False)
    if right - left < 250 or bottom - top < 250:
        return bgr, None
    kept_area = (right - left) * (bottom - top)
    if 1.0 - kept_area / (height * width) < 0.08:
        return bgr, None
    candidate = bgr[top:bottom, left:right]
    candidate_luminance = luminance[top:bottom, left:right]
    # A real document/screen page has substantial non-black content.  This
    # guard prevents turning the white title on a black cover into a crop cue.
    if float(np.mean(candidate_luminance > 32)) < 0.35:
        return bgr, None
    return candidate, (left, top, right, bottom)


def analyze_and_preprocess(image: Image.Image, config: AppConfig | None = None) -> PreprocessedImage:
    config = config or AppConfig()
    original = image.convert("RGB")
    bgr = _as_bgr(original)
    transformations: list[dict[str, Any]] = []
    bgr, dark_margin_bounds = _trim_dark_screenshot_margins(bgr)
    if dark_margin_bounds is not None:
        left, top, right, bottom = dark_margin_bounds
        transformations.append({
            "operation": "trim_dark_screenshot_margins",
            "left": left, "top": top, "right": right, "bottom": bottom,
        })
    h0, w0 = bgr.shape[:2]
    if max(h0, w0) > config.max_image_dimension:
        scale = config.max_image_dimension / max(h0, w0)
        bgr = cv2.resize(bgr, (round(w0 * scale), round(h0 * scale)), interpolation=cv2.INTER_AREA)
        transformations.append({"operation": "resize", "scale": scale})
    # The phone frame is not the document orientation. A portrait photograph can
    # contain a perfectly upright landscape card, so rotating solely from the
    # outer image dimensions can turn that card sideways. Right-angle correction
    # is delegated to OCR's document-orientation classifier, which looks at the
    # text/content and can distinguish 0/90/180/270 degrees.
    bgr, perspective, crop_warning = _perspective_if_reliable(bgr)
    if perspective:
        transformations.append({"operation": "perspective_correction", "reliable": True})
    # Small cards embedded in phone photos can be perspective-cropped to only
    # a few hundred pixels even when their text is visibly clear. Upscale the
    # normalized OCR input before recognition; retain the source-resolution
    # warning because interpolation does not create new document evidence.
    source_h, source_w = bgr.shape[:2]
    source_low_res = min(source_h, source_w) < config.min_image_side
    # Recognition needs pixels per character, not pixels per page. A licence
    # photographed with both sides in one 1000x1277 frame leaves each side about
    # 1000x600, and 8pt print in 600 rows is three or four pixels of x-height --
    # below what any recogniser resolves. Interpolation invents no evidence, but
    # it does give the recogniser's receptive field something to work with, and
    # it is the difference between reading the row and not seeing it.
    ocr_target_min_side = config.ocr_target_min_side
    if min(source_h, source_w) < ocr_target_min_side:
        scale = min(
            ocr_target_min_side / min(source_h, source_w),
            config.max_image_dimension / max(source_h, source_w),
        )
        if scale > 1.05:
            bgr = cv2.resize(
                bgr,
                (round(source_w * scale), round(source_h * scale)),
                interpolation=cv2.INTER_CUBIC,
            )
            transformations.append({
                "operation": "ocr_upscale", "scale": round(scale, 4),
                "source_width": source_w, "source_height": source_h,
            })
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    angle = _estimate_skew(gray)
    if angle:
        bgr = _rotate_bound(bgr, angle)
        transformations.append({"operation": "deskew", "degrees": angle})
    h, w = bgr.shape[:2]
    # One greyscale conversion feeds the blur score, the exposure checks and
    # every greyscale-derived OCR variant below.
    luminance = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    score = _gray_blur_score(luminance)
    glare_fraction = _glare_fraction(bgr)
    over = float(np.mean(luminance > 248)) > 0.22
    under = float(np.mean(luminance < 18)) > 0.22
    low_res = source_low_res
    warnings: list[str] = []
    if score < config.blur_threshold: warnings.append("BLUR_WARNING")
    if glare_fraction >= config.glare_fraction_threshold: warnings.append("GLARE_WARNING")
    if crop_warning: warnings.append("CROP_WARNING")
    if over: warnings.append("OVEREXPOSURE_WARNING")
    if under: warnings.append("UNDEREXPOSURE_WARNING")
    if low_res: warnings.append("LOW_RESOLUTION")
    unreadable = (score < config.blur_threshold * 0.25) or (over and under) or min(h, w) < 250
    if unreadable: warnings.append("UNREADABLE_WARNING")
    total_rotation = angle
    quality = QualityInfo(
        blur_score=round(score, 2), glare_detected=glare_fraction >= config.glare_fraction_threshold,
        low_resolution=low_res, rotation_degrees=round(total_rotation, 2), width=w, height=h,
        orientation="LANDSCAPE" if w >= h else "PORTRAIT", crop_warning=crop_warning,
        overexposed=over, underexposed=under, unreadable=unreadable, warnings=warnings,
    )
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    normalized = Image.fromarray(rgb)
    # The first recognition stage consumes only these views. Expensive repair
    # variants are generated by ``ensure_ocr_variants`` if, and only if, this
    # read leaves a critical field missing.
    wanted = {*config.ocr_variant_names, "original_normalized"}
    variants = _build_ocr_variants(normalized, wanted)
    return PreprocessedImage(original, normalized, variants, quality, transformations)
