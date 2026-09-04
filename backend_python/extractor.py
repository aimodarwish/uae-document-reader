from __future__ import annotations

import difflib
import re
import unicodedata
from dataclasses import dataclass, field, replace
from datetime import datetime
from typing import Callable, Iterable

from .models import MulkiyaData
from .ocr_types import ARABIC_RE as _AR, OCRLine

# --------------------------------------------------------------------------
# Text normalisation
# --------------------------------------------------------------------------

# UAE documents frequently render numbers in Arabic-Indic or Persian digits.
# Fold them to ASCII before any numeric regex runs.
_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")

# Tashkeel, superscript alef and tatweel carry no meaning for matching and OCR
# emits them inconsistently.
_ARABIC_MARKS = re.compile(r"[ً-ْٰـ]")

ARABIC_RE = _AR
# Bounded by "not a digit" rather than \b: OCR glues a label onto its value
# often enough that a leading \b loses the date entirely. A real card produced
# 'Reg.Dat22-12-2025' -- the 't' before '22' is a word character, so \b never
# matched and registration_issuance came back null.
DATE_RE = re.compile(r"(?<!\d)([0-3]?\d)[/\-.]([01]?\d)[/\-.]((?:19|20)?\d{2})(?!\d)")
YEAR_RE = re.compile(r"\b(19[89]\d|20[0-5]\d)\b")

# VIN: exactly 17 chars, and I/O/Q are not valid VIN characters.
VIN_RUN_RE = re.compile(r"[A-HJ-NPR-Z0-9]{17,}")
VIN_EXACT_RE = re.compile(r"^[A-HJ-NPR-Z0-9]{17}$")
VIN_LABEL_WORDS = ("CHASSISNUMBER", "CHASSISNO", "CHASSIS", "VINNUMBER", "VINNO", "VIN", "FRAMENO")
_VIN_LABEL_RE = re.compile(r"^(?:" + "|".join(VIN_LABEL_WORDS) + r")\.?")
_VIN_FIX = str.maketrans("IOQ", "100")  # I->1, O->0, Q->0


def _fold_arabic(text: str) -> str:
    """Collapse the Arabic letter variants OCR flips between."""
    text = _ARABIC_MARKS.sub("", text)
    text = re.sub(r"[أإآٱ]", "ا", text)
    text = text.replace("ى", "ي").replace("ة", "ه")
    text = text.replace("ؤ", "و").replace("ئ", "ي")
    return text


def _norm(text: str) -> str:
    text = unicodedata.normalize("NFKC", str(text)).translate(_DIGITS)
    text = text.replace("ـ", "")
    return re.sub(r"\s+", " ", text).strip()


def _norm_key(text: str) -> str:
    """Aggressive key used for label matching and dictionary lookups."""
    text = _fold_arabic(_norm(text).lower())
    return re.sub(r"[^a-z0-9؀-ۿ]+", " ", text).strip()


# --------------------------------------------------------------------------
# UAE dictionaries (keys normalised once, at import)
# --------------------------------------------------------------------------

def _build(mapping: dict[str, str]) -> dict[str, str]:
    return {_norm_key(k): v for k, v in mapping.items()}


SOURCE_MAP = _build({
    "دبي": "Dubai", "دبى": "Dubai", "dubai": "Dubai",
    "ابوظبي": "Abu Dhabi", "أبوظبي": "Abu Dhabi", "أبو ظبي": "Abu Dhabi",
    "ابو ظبي": "Abu Dhabi", "abu dhabi": "Abu Dhabi", "abudhabi": "Abu Dhabi",
    "الشارقة": "Sharjah", "شارقة": "Sharjah", "sharjah": "Sharjah",
    "عجمان": "Ajman", "ajman": "Ajman",
    "راس الخيمة": "Ras Al Khaimah", "رأس الخيمة": "Ras Al Khaimah",
    "ras al khaimah": "Ras Al Khaimah", "ras alkhaimah": "Ras Al Khaimah",
    "الفجيرة": "Fujairah", "فجيرة": "Fujairah", "fujairah": "Fujairah",
    "ام القيوين": "Umm Al Quwain", "أم القيوين": "Umm Al Quwain",
    "umm al quwain": "Umm Al Quwain", "umm alquwain": "Umm Al Quwain",
})

CATEGORY_MAP = _build({
    "خصوصي": "Private", "خصوصى": "Private", "private": "Private",
    "تجاري": "Commercial", "تجارى": "Commercial", "commercial": "Commercial",
    "تأجير": "Rental", "تاجير": "Rental", "ايجار": "Rental", "إيجار": "Rental",
    "rental": "Rental", "rent a car": "Rental",
    "نقل": "Transport", "transport": "Transport",
    "حكومي": "Government", "government": "Government",
    "دراجة": "Motorcycle", "motorcycle": "Motorcycle",
    "كلاسيكي": "Classic", "classic": "Classic",
})

COLOR_MAP = _build({
    "رمادي": "Grey", "رصاصي": "Grey", "سكني": "Grey", "grey": "Grey", "gray": "Grey",
    "فضي": "Silver", "silver": "Silver",
    "ابيض": "White", "أبيض": "White", "white": "White",
    "اسود": "Black", "أسود": "Black", "black": "Black",
    "احمر": "Red", "أحمر": "Red", "red": "Red",
    "ازرق": "Blue", "أزرق": "Blue", "blue": "Blue",
    "اخضر": "Green", "أخضر": "Green", "green": "Green",
    "اصفر": "Yellow", "أصفر": "Yellow", "yellow": "Yellow",
    "بني": "Brown", "brown": "Brown",
    "برتقالي": "Orange", "orange": "Orange",
    "بيج": "Beige", "beige": "Beige",
    "ذهبي": "Gold", "gold": "Gold",
    "بنفسجي": "Purple", "purple": "Purple",
    "وردي": "Pink", "pink": "Pink",
    "نحاسي": "Bronze", "bronze": "Bronze",
    "كحلي": "Navy", "navy": "Navy",
})

# Multi-word makes must be tried before their single-word prefixes, so the list
# is sorted longest-first at use. Hyphens are normalised away for matching and
# restored for display.
KNOWN_MAKES = [
    "RANGE ROVER", "LAND ROVER", "ROLLS ROYCE", "MERCEDES BENZ", "ASTON MARTIN",
    "ALFA ROMEO", "GREAT WALL", "MERCEDES", "LAMBORGHINI", "VOLKSWAGEN",
    "MITSUBISHI", "CHEVROLET", "MASERATI", "CADILLAC", "CHRYSLER", "INFINITI",
    "PORSCHE", "BENTLEY", "FERRARI", "MCLAREN", "HYUNDAI", "PEUGEOT", "RENAULT",
    "SUZUKI", "SUBARU", "TOYOTA", "NISSAN", "BUGATTI", "CITROEN", "DODGE",
    "HONDA", "LEXUS", "MAZDA", "SKODA", "TESLA", "VOLVO", "LOTUS", "JAGUAR",
    "GENESIS", "LINCOLN", "PAGANI", "KOENIGSEGG", "AUDI", "BMW", "GMC", "FORD",
    "JEEP", "KIA", "MINI", "OPEL", "SEAT", "FIAT", "RAM", "HUMMER", "ISUZU",
    "DAIHATSU", "CHERY", "CHANGAN", "HAVAL", "MG", "BYD",
]
_MAKES_BY_LENGTH = sorted({m: None for m in KNOWN_MAKES}, key=len, reverse=True)

_MAKE_DISPLAY = {
    "MERCEDES BENZ": "MERCEDES-BENZ",
    "ROLLS ROYCE": "ROLLS-ROYCE",
}


# --------------------------------------------------------------------------
# Arabic text direction
# --------------------------------------------------------------------------

# Labels that appear on essentially every UAE Mulkiya. They are used only as a
# direction probe -- never as data.
ORIENTATION_PROBES = [
    "رقم اللوحة", "جهة الترخيص", "صنف اللوحة", "تاريخ الترخيص",
    "انتهاء الترخيص", "مؤمنة لدى", "نوع التأمين", "رقم الوثيقة",
    "سنة الصنع", "بلد الصنع", "لون المركبة", "نوع المركبة",
    "رقم القاعدة", "رقم المحرك", "عدد الركاب", "جهة الرهن",
]


def _orientation_is_reversed(lines: list[OCRLine]) -> bool:
    """Does this document's Arabic arrive in visual (reversed) order?

    Recognition stacks disagree on whether to hand back logical or visual order
    -- PaddleX carries a python-bidi call for the Arabic models that the version
    we ship does not actually apply, and an upgrade could change that silently.
    Hard-coding either answer means one library bump reverses every Arabic field
    with no error anywhere.

    So we measure instead: count how many of the card's own known labels match
    as-read versus reversed. The document answers the question about itself.
    """
    arabic = [line for line in lines if ARABIC_RE.search(line.text)]
    if not arabic:
        return False

    as_read = sum(1 for line in arabic if _contains_any(line.text, ORIENTATION_PROBES))
    flipped = sum(1 for line in arabic if _contains_any(line.text[::-1], ORIENTATION_PROBES))
    return flipped > as_read


def _apply_orientation(lines: list[OCRLine]) -> tuple[list[OCRLine], bool]:
    if not _orientation_is_reversed(lines):
        return lines, False

    corrected = [
        replace(line, text=line.text[::-1]) if ARABIC_RE.search(line.text) else line
        for line in lines
    ]
    return corrected, True


# --------------------------------------------------------------------------
# Field plumbing
# --------------------------------------------------------------------------

@dataclass
class FieldValue:
    value: object | None = None
    confidence: float | None = None
    # "anchored" = found next to its label, "global" = found by scanning the
    # whole card. Global hits are discounted because they are more guessy.
    source: str = "anchored"
    notes: list[str] = field(default_factory=list)

    @property
    def scored(self) -> float | None:
        if self.confidence is None:
            return None
        penalty = 0.85 if self.source == "global" else 1.0
        return round(self.confidence * penalty, 4)


def _compact(text: str) -> str:
    """Key with ALL whitespace removed.

    OCR regularly splits an Arabic word ('خصوص ي' for 'خصوصي') because the
    glyphs are spaced. Matching on the space-free form recovers those, and is
    only ever used as a fallback so it cannot loosen an exact match.
    """
    return _norm_key(text).replace(" ", "")


def _contains_any(text: str, keys: Iterable[str]) -> bool:
    key = _norm_key(text)
    if any(_norm_key(k) in key for k in keys):
        return True
    compact = _compact(text)
    return any(_compact(k) in compact for k in keys)


def _map_lookup(text: str, mapping: dict[str, str]) -> str | None:
    key = _norm_key(text)
    if not key:
        return None
    if key in mapping:
        return mapping[key]
    # Longest key first so "ضد الغير فقط" beats "ضد الغير".
    for source in sorted(mapping, key=len, reverse=True):
        if source and source in key:
            return mapping[source]

    # Fallback: ignore spaces, for words OCR split mid-token.
    compact = key.replace(" ", "")
    for source in sorted(mapping, key=len, reverse=True):
        squashed = source.replace(" ", "")
        if squashed and squashed in compact:
            return mapping[source]
    return None


def _same_row(anchor: OCRLine, candidate: OCRLine) -> bool:
    if anchor.page != candidate.page:
        return False
    tolerance = max(anchor.height, candidate.height, 12.0) * 0.7
    return abs(anchor.cy - candidate.cy) <= tolerance


def _row_text(lines: list[OCRLine], anchor: OCRLine) -> str:
    """Left-to-right text of the whole row an anchor sits on.

    Needed because OCR often splits a value across boxes (a VIN broken in two,
    a plate rendered as separate 'AA' and '88271' cells).
    """
    row = [x for x in lines if _same_row(anchor, x)]
    return " ".join(x.text for x in sorted(row, key=lambda x: x.left))


def _nearest_value(
    lines: list[OCRLine],
    keys: Iterable[str],
    predicate: Callable[[str], bool] | None = None,
    direction: str = "any",
    allow_anchor_text: bool = True,
) -> OCRLine | None:
    """Find the value belonging to a label, on the same row.

    A Mulkiya row reads: English label | value | Arabic label. So the value sits
    to the right of an English anchor and to the left of an Arabic one.
    `direction` biases the search rather than hard-filtering, which survives OCR
    boxes that drift a little.
    """
    anchors = [line for line in lines if _contains_any(line.text, keys)]
    candidates: list[tuple[float, OCRLine]] = []

    for anchor in anchors:
        if allow_anchor_text and predicate and predicate(anchor.text):
            candidates.append((0.0, anchor))

        for line in lines:
            if line is anchor or not _same_row(anchor, line):
                continue
            if predicate and not predicate(line.text):
                continue
            if _contains_any(line.text, keys):
                continue

            dx = line.cx - anchor.cx
            penalty = 1.0
            if direction == "right" and dx < 0:
                penalty = 3.0
            elif direction == "left" and dx > 0:
                penalty = 3.0

            candidates.append((abs(dx) * penalty, line))

    if not candidates:
        return None
    return min(candidates, key=lambda x: x[0])[1]


def _fuzzy_map_lookup(text: str, mapping: dict[str, str]) -> str | None:
    """Last-resort match against a CLOSED vocabulary.

    OCR drops a letter from short Arabic words surprisingly often -- a real card
    read 'دبي' as 'دي', losing plate_source entirely. Because these dictionaries
    are closed sets (seven emirates, a fixed list of colours), a near match can
    be resolved safely, but only under three conditions: the exact lookup has
    already failed, the word is long enough for a single error to be meaningful,
    and exactly ONE distinct value comes back. A tie is treated as unreadable.
    """
    key = _norm_key(text)
    if len(key) < 2:
        return None

    close = difflib.get_close_matches(key, list(mapping), n=4, cutoff=0.75)
    values = {mapping[k] for k in close}
    if len(values) == 1:
        return values.pop()
    return None


def _find_mapped(
    lines: list[OCRLine],
    mapping: dict[str, str],
    anchor_keys: Iterable[str],
    direction: str = "any",
) -> FieldValue:
    """Label-anchored dictionary lookup, falling back to a whole-card scan.

    The fallback matters but must stay a fallback: 'دبي' appears in the RTA
    header of every Dubai-issued card regardless of what Place of Issue says, so
    an anchored hit always wins.
    """
    candidate = _nearest_value(
        lines, keys=anchor_keys,
        predicate=lambda t: _map_lookup(t, mapping) is not None,
        direction=direction,
    )
    if candidate:
        value = _map_lookup(candidate.text, mapping)
        if value:
            return FieldValue(value, candidate.score, "anchored")

    # Nothing matched exactly. Retry against the value sitting beside the label,
    # allowing for a dropped or swapped letter -- but only there, never across
    # the whole card, and only when the answer is unambiguous.
    near_label = _nearest_value(
        lines, keys=anchor_keys,
        predicate=lambda t: _fuzzy_map_lookup(t, mapping) is not None,
        direction=direction,
    )
    if near_label:
        value = _fuzzy_map_lookup(near_label.text, mapping)
        if value:
            return FieldValue(
                value, near_label.score, "anchored",
                [f"Matched {value!r} from an imperfect OCR read: {_norm(near_label.text)!r}"],
            )

    matches = [
        (line.score, value, line)
        for line in lines
        if (value := _map_lookup(line.text, mapping)) is not None
    ]
    if matches:
        score, value, _ = max(matches, key=lambda x: x[0])
        return FieldValue(value, score, "global")
    return FieldValue()


# --------------------------------------------------------------------------
# Dates
# --------------------------------------------------------------------------

def _is_date(text: str) -> bool:
    return _format_date(text) is not None


def _format_date(text: str) -> str | None:
    match = DATE_RE.search(_norm(text))
    if not match:
        return None

    day, month, year = match.groups()
    if len(year) == 2:
        year = "20" + year
    try:
        # UAE documents are DD/MM/YYYY. An out-of-range month is a bad read, not
        # an American date -- we reject rather than silently swap.
        return datetime(int(year), int(month), int(day)).strftime("%Y-%m-%d")
    except ValueError:
        return None


REG_EXPIRY_KEYS = ["Exp. Date", "Expiry Date", "إنتهاء الترخيص", "انتهاء الترخيص"]
REG_ISSUANCE_KEYS = ["Reg. Date", "Registration Date", "تاريخ الترخيص", "تاريخ التسجيل"]


def _find_registration_dates(lines: list[OCRLine]) -> tuple[FieldValue, FieldValue]:
    """Registration expiry and issuance, which share ONE row on a Mulkiya:

        Exp. Date | 19/06/2026 | إنتهاء الترخيص | Reg. Date | 28/04/2025 | تاريخ الترخيص

    registration_issuance IS the Reg. Date column -- the two are the same thing.
    Both are looked up by their own labels first, in either language.

    The fallback exists because on a real RTA card OCR swallowed the words
    "Reg. Date" completely: only 'ate' survived, glued onto the neighbouring
    Arabic label. When neither issuance label is legible we use the row's
    geometry instead -- Reg. Date is the next date to the right of Exp. Date on
    the same row -- which is exactly the column the value belongs to.
    """
    expiry_line = _nearest_value(lines, keys=REG_EXPIRY_KEYS, predicate=_is_date)
    issuance_line = _nearest_value(lines, keys=REG_ISSUANCE_KEYS, predicate=_is_date)

    expiry = (
        FieldValue(_format_date(expiry_line.text), expiry_line.score, "anchored")
        if expiry_line else FieldValue()
    )

    if issuance_line:
        issuance = FieldValue(_format_date(issuance_line.text), issuance_line.score, "anchored")
    elif expiry_line:
        expiry_date = _format_date(expiry_line.text)
        to_the_right = [
            line for line in lines
            if line is not expiry_line
            and _same_row(expiry_line, line)
            and line.cx > expiry_line.cx
            and _is_date(line.text)
            and _format_date(line.text) != expiry_date
        ]
        if to_the_right:
            nearest = min(to_the_right, key=lambda x: x.cx)
            issuance = FieldValue(
                _format_date(nearest.text), nearest.score, "global",
                ["registration_issuance taken from the Reg. Date column by position "
                 "because its label was not legible."],
            )
        else:
            issuance = FieldValue()
    else:
        issuance = FieldValue()

    # A registration cannot expire before it was issued. ISO strings compare
    # correctly, so this is a cheap sanity check on both reads.
    if issuance.value and expiry.value and str(issuance.value) >= str(expiry.value):
        issuance.notes.append(
            f"Registration issuance ({issuance.value}) is not before its expiry "
            f"({expiry.value}) - check both dates."
        )

    return expiry, issuance


def _find_date_by_label(lines: list[OCRLine], keys: Iterable[str]) -> FieldValue:
    candidate = _nearest_value(lines, keys=keys, predicate=_is_date)
    if not candidate:
        return FieldValue()
    return FieldValue(_format_date(candidate.text), candidate.score, "anchored")


# --------------------------------------------------------------------------
# VIN
# --------------------------------------------------------------------------

def _vin_from(text: str) -> tuple[str | None, list[str]]:
    """Pull a 17-character VIN out of raw OCR text.

    Deliberately conservative. A clean read is returned untouched. I/O/Q are the
    only substitutions applied, because they cannot legally appear in a VIN, so
    seeing one is proof of a misread rather than a guess -- and even then the
    change is reported. Ambiguous pairs (S/5, B/8, Z/2) are NOT rewritten: both
    characters are legal in a VIN, so any 'correction' would be invention.
    """
    compact = re.sub(r"[^A-Za-z0-9]", "", _norm(text)).upper()

    # Strip a leading label. When OCR puts label and value in one box we get
    # 'CHASSISNOSAL1P9...', and the I/O in the label would otherwise be treated
    # as part of the VIN once substitutions are applied.
    compact = _VIN_LABEL_RE.sub("", compact)
    if len(compact) < 17:
        return None, []

    # Pass 1 - untouched. Any I/O/Q in neighbouring words breaks the run here,
    # which conveniently delimits the VIN.
    runs = VIN_RUN_RE.findall(compact)
    for run in runs:
        if len(run) == 17:
            return run, []
    for run in runs:
        for start in range(len(run) - 16):
            window = run[start:start + 17]
            if VIN_EXACT_RE.match(window):
                return window, []

    # Pass 2 - substitute only the characters a VIN cannot contain, preferring
    # the window that needs the fewest of them.
    best: tuple[str, list[str]] | None = None
    for match in re.finditer(r"[A-Z0-9]{17,}", compact):
        run = match.group(0)
        for start in range(len(run) - 16):
            window = run[start:start + 17]
            fixed = window.translate(_VIN_FIX)
            if not VIN_EXACT_RE.match(fixed):
                continue
            changes = [
                f"{window[i]}->{fixed[i]}" for i in range(17) if window[i] != fixed[i]
            ]
            if best is None or len(changes) < len(best[1]):
                best = (fixed, changes)

    return best if best else (None, [])


def _find_vin(lines: list[OCRLine]) -> FieldValue:
    anchors = ["Chassis No", "Chassis", "رقم القاعدة", "رقم الهيكل", "VIN"]

    # Prefer the chassis row, joining its boxes so a VIN split across two cells
    # is still recoverable.
    for line in lines:
        if not _contains_any(line.text, anchors):
            continue
        vin, changes = _vin_from(_row_text(lines, line))
        if vin:
            notes = [f"VIN: corrected impossible character(s) {', '.join(changes)}"] if changes else []
            return FieldValue(vin, line.score, "anchored", notes)

    # Fall back to any line that yields a structurally valid VIN.
    for line in sorted(lines, key=lambda x: x.score, reverse=True):
        vin, changes = _vin_from(line.text)
        if vin:
            notes = [f"VIN: corrected impossible character(s) {', '.join(changes)}"] if changes else []
            return FieldValue(vin, line.score, "global", notes)

    return FieldValue()


# --------------------------------------------------------------------------
# Plate
# --------------------------------------------------------------------------

# Dubai issues letter codes (A, AA); Abu Dhabi, Sharjah and the northern
# emirates issue NUMERIC codes (13/12345). Both must parse.
PLATE_PAIR_RE = re.compile(r"\b([A-Z]{1,2}|\d{1,2})\s*[/\-|]\s*(\d{1,5})\b")
_PLATE_ANCHORS = ["Traffic Plate No", "Plate No", "رقم اللوحة", "Plate Number"]


def _plate_from_row(text: str) -> tuple[str, str] | None:
    text = _norm(text).upper()
    if DATE_RE.search(text):  # never mine a date row for a plate
        return None
    match = PLATE_PAIR_RE.search(text)
    if match:
        return match.group(1), match.group(2)
    return None


def _find_plate(lines: list[OCRLine]) -> tuple[FieldValue, FieldValue]:
    for line in lines:
        if not _contains_any(line.text, _PLATE_ANCHORS):
            continue

        row = [x for x in lines if _same_row(line, x)]
        joined = _row_text(lines, line)

        pair = _plate_from_row(joined)
        if pair:
            code, number = pair
            return (FieldValue(code, line.score, "anchored"),
                    FieldValue(number, line.score, "anchored"))

        # No separator: code and number sit in their own boxes on the row.
        tokens = [
            (_norm(x.text).upper().strip(" :-"), x)
            for x in sorted(row, key=lambda x: x.left)
            if not _contains_any(x.text, _PLATE_ANCHORS)
        ]
        numbers = [(t, x) for t, x in tokens if re.fullmatch(r"\d{1,5}", t)]
        codes = [(t, x) for t, x in tokens if re.fullmatch(r"[A-Z]{1,2}", t)]

        if numbers:
            number, num_line = max(numbers, key=lambda t: (len(t[0]), t[1].score))
            if codes:
                code, code_line = codes[0]
                return (FieldValue(code, code_line.score, "anchored"),
                        FieldValue(number, num_line.score, "anchored"))
            # Numeric code: the other short number on the row.
            others = [(t, x) for t, x in numbers if x is not num_line and len(t) <= 2]
            if others:
                code, code_line = others[0]
                return (FieldValue(code, code_line.score, "anchored"),
                        FieldValue(number, num_line.score, "anchored"))
            return FieldValue(), FieldValue(number, num_line.score, "anchored")

    # Global fallback, strictly guarded: must look like a plate and not a date.
    for line in sorted(lines, key=lambda x: x.score, reverse=True):
        pair = _plate_from_row(line.text)
        if pair:
            code, number = pair
            return (FieldValue(code, line.score, "global"),
                    FieldValue(number, line.score, "global"))

    return FieldValue(), FieldValue()


# --------------------------------------------------------------------------
# Vehicle
# --------------------------------------------------------------------------

def _year_in(text: str) -> int | None:
    """A model year, never a date. '19/06/2026' must not yield 2026."""
    text = _norm(text)
    if DATE_RE.search(text):
        return None
    match = YEAR_RE.search(text)
    return int(match.group(0)) if match else None


def _find_year(lines: list[OCRLine]) -> FieldValue:
    candidate = _nearest_value(
        lines,
        keys=["Model", "سنة الصنع", "سنه الصنع", "Year of Manufacture"],
        predicate=lambda t: _year_in(t) is not None,
    )
    if candidate:
        year = _year_in(candidate.text)
        if year:
            return FieldValue(year, candidate.score, "anchored")
    return FieldValue()


def _find_vehicle_type(lines: list[OCRLine]) -> FieldValue:
    candidate = _nearest_value(
        lines,
        keys=["Veh. Type", "Veh Type", "Vehicle Type", "نوع المركبة", "الصنع والطراز"],
        predicate=lambda t: (
            len(re.findall(r"[A-Za-z]", t)) >= 3
            and not _contains_any(t, ["Veh Type", "Vehicle Type", "Veh. Type"])
            and not _is_date(t)
        ),
        direction="right",
    )
    if not candidate:
        return FieldValue()
    value = re.sub(r"\s+", " ", _norm(candidate.text)).strip(" :-")
    return FieldValue(value.upper(), candidate.score, "anchored")


def _split_make_model(vehicle_type: FieldValue) -> tuple[FieldValue, FieldValue]:
    if not vehicle_type.value:
        return FieldValue(), FieldValue()

    raw = str(vehicle_type.value).upper()
    value = re.sub(r"\s+", " ", raw.replace("-", " ").replace("_", " ")).strip()
    conf, src = vehicle_type.confidence, vehicle_type.source

    for make in _MAKES_BY_LENGTH:
        display = _MAKE_DISPLAY.get(make, make)
        if value == make:
            return FieldValue(display, conf, src), FieldValue()
        if value.startswith(make + " "):
            model = value[len(make):].strip()
            return FieldValue(display, conf, src), FieldValue(model or None, conf, src)

    # OCR sometimes drops the space inside a two-word make ("RANGEROVER SPORT").
    compact = value.replace(" ", "")
    for make in _MAKES_BY_LENGTH:
        squashed = make.replace(" ", "")
        if compact.startswith(squashed):
            display = _MAKE_DISPLAY.get(make, make)
            model = compact[len(squashed):].strip()
            return FieldValue(display, conf, src), FieldValue(model or None, conf, src)

    parts = value.split()
    if len(parts) >= 2:
        return (FieldValue(parts[0], conf, "global"),
                FieldValue(" ".join(parts[1:]), conf, "global"))
    return FieldValue(value, conf, src), FieldValue()


# --------------------------------------------------------------------------
# Insurance
# --------------------------------------------------------------------------

# Arabic letters OCR and typography flip between. Used to strip a label from a
# box even when its spelling differs from ours by a hamza or a teh marbuta.
_FOLD_CLASSES = {"ا": "اأإآٱ", "ه": "هة", "ي": "يى", "و": "وؤ"}


def _fold_tolerant_pattern(label: str) -> re.Pattern[str]:
    parts = []
    for char in _fold_arabic(label):
        if char == " ":
            parts.append(r"\s*")
        elif char in _FOLD_CLASSES:
            parts.append(f"[{_FOLD_CLASSES[char]}]")
        else:
            parts.append(re.escape(char))
    return re.compile("".join(parts), re.IGNORECASE)


def _strip_known_labels(text: str, labels: Iterable[str]) -> str:
    """Remove label text OCR merged into the same box as its value.

    On a real card the insurer and its Arabic label frequently land in ONE
    detected box. Rejecting such a box loses the company entirely; stripping the
    label keeps it.
    """
    result = _norm(text)
    for label in sorted(labels, key=len, reverse=True):
        result = _fold_tolerant_pattern(label).sub(" ", result)
    return re.sub(r"\s+", " ", result).strip(" :-()\u060c")


def _find_policy_number(lines: list[OCRLine]) -> FieldValue:
    candidate = _nearest_value(
        lines,
        keys=["Policy No", "Policy Number", "رقم الوثيقة", "رقم البوليصة"],
        predicate=lambda t: (
            not _is_date(t) and 6 <= len(re.sub(r"\D", "", _norm(t))) <= 20
        ),
    )
    if candidate:
        digits = re.sub(r"\D", "", _norm(candidate.text))
        if 6 <= len(digits) <= 20:
            return FieldValue(digits, candidate.score, "anchored")
    return FieldValue()


# Labels that share or neighbour the insurer's row. A box containing only these
# is a label; a box containing these PLUS other words is a merged label+value,
# and the label part is stripped off rather than the whole box discarded.
# Labels whose ROW the insurer must share. Restricting candidates to this row
# is what stops the card's footer ("سلطة الترخيص" / Licensing Authority, which
# OCR mangles into things like "سلفلة الترقيس") from scoring as a company name.
INSURANCE_ROW_ANCHORS = [
    "مؤمنة لدى", "Ins. Exp", "انتهاء التأمين", "Insurance Company", "Insured By",
]

INSURER_ROW_LABELS = [
    "مؤمنة لدى", "نوع التأمين", "انتهاء التأمين", "رقم الوثيقة",
    "جهة الرهن", "المالك", "الجنسية", "ملاحظات",
    "جهة الترخيص", "تاريخ الترخيص", "انتهاء الترخيص", "صنف اللوحة",
    "صنف المركبة", "نوع المركبة", "لون المركبة",
    "Ins. Exp", "Policy No", "Insurance Type", "Mortgage By", "Owner",
    "Nationality", "Exp. Date", "Reg. Date",
]


def _is_probable_company(text: str) -> bool:
    text = _norm(text)
    if len(text) < 5 or DATE_RE.search(text):
        return False
    if _contains_any(text, INSURER_ROW_LABELS) and not _strip_known_labels(text, INSURER_ROW_LABELS):
        # Nothing but label text -- not a company.
        return False
    if re.fullmatch(r"[\d\s/\-.]+", text):
        return False
    return bool(ARABIC_RE.search(text) or len(re.findall(r"[A-Za-z]", text)) >= 5)


def _value_column(lines: list[OCRLine]) -> float | None:
    """Horizontal centre of the card's value column.

    Dates and long digit runs are the values we read most reliably, so their
    median x marks where values live. Arabic LABELS sit in a different column
    entirely, which is what lets us reject a misread label that would otherwise
    look like a plausible company name.
    """
    values = [
        line for line in lines
        if _is_date(line.text) or re.fullmatch(r"\d{6,}", _norm(line.text))
    ]
    if len(values) < 2:
        return None
    xs = sorted(line.cx for line in values)
    return xs[len(xs) // 2]


def _find_insurance_company(lines: list[OCRLine]) -> FieldValue:
    # allow_anchor_text is deliberately True: OCR routinely lands the label and
    # the company in ONE box ("مؤمنة لدى ادمجى انشورنس كومباني إنتهاء التأمين"),
    # and excluding the anchor discarded the only box holding the company.
    candidate = _nearest_value(
        lines,
        keys=["مؤمنة لدى", "Insurance Company", "Insured By", "شركة التأمين"],
        predicate=lambda t: len(_strip_known_labels(t, INSURER_ROW_LABELS)) >= 5
        and not DATE_RE.search(_norm(t)),
        direction="left",
        allow_anchor_text=True,
    )
    if candidate:
        cleaned = _strip_known_labels(candidate.text, INSURER_ROW_LABELS)
        if cleaned:
            return FieldValue(cleaned, candidate.score, "anchored")

    # Otherwise: the longest plausible Arabic string that sits in the VALUE
    # column. Without the column test a misread Arabic label (OCR turns
    # 'انتهاء التأمين' into things like 'ازتهاع الأأَمين') scores as a company
    # name, because it is long, Arabic, and matches no known label.
    # Structural constraint first: the insurer shares a row with an insurance
    # label. That is far more reliable than column position, which shifts
    # whenever OCR merges values into the left-hand label boxes.
    anchors = [line for line in lines if _contains_any(line.text, INSURANCE_ROW_ANCHORS)]
    possible = [
        line for line in lines
        if ARABIC_RE.search(line.text)
        and _is_probable_company(line.text)
        and any(_same_row(anchor, line) for anchor in anchors)
    ]

    cleaned_options = [
        (_strip_known_labels(line.text, INSURER_ROW_LABELS), line) for line in possible
    ]
    cleaned_options = [(t, line) for t, line in cleaned_options if len(t) >= 5]

    if cleaned_options:
        text, line = max(cleaned_options, key=lambda x: (len(x[0]), x[1].score))
        return FieldValue(text, line.score, "global")
    return FieldValue()


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------

# Every field is reported when missing. The spec is explicit: a field that
# cannot be read reliably returns null AND raises a warning -- never a guess.
REQUIRED = [
    "plate_source", "plate_category", "plate_code", "plate_number", "vin",
    "make", "model", "year", "color",
    "insurance_company", "policy_number", "insurance_expiry",
    "registration_expiry", "registration_issuance",
]


def extract_fields(
    lines: list[OCRLine],
) -> tuple[MulkiyaData, dict[str, float | None], list[str]]:
    lines, flipped = _apply_orientation(lines)

    plate_code, plate_number = _find_plate(lines)
    registration_expiry, registration_issuance = _find_registration_dates(lines)
    vehicle_type = _find_vehicle_type(lines)
    make, model = _split_make_model(vehicle_type)

    fields: dict[str, FieldValue] = {
        "plate_source": _find_mapped(
            lines, SOURCE_MAP,
            [
                "Place of Issue", "جهة الترخيص", "مصدر اللوحة",
                "Source", "مكان الإصدار",
            ],
        ),
        "plate_category": _find_mapped(
            lines, CATEGORY_MAP,
            ["Plate Category", "صنف اللوحة", "فئة اللوحة", "Category"],
            direction="left",
        ),
        "plate_code": plate_code,
        "plate_number": plate_number,
        "vin": _find_vin(lines),
        "make": make,
        "model": model,
        "year": _find_year(lines),
        "color": _find_mapped(
            lines, COLOR_MAP,
            ["Colour", "Color", "لون المركبة", "اللون"],
            direction="left",
        ),
        "insurance_company": _find_insurance_company(lines),
        "policy_number": _find_policy_number(lines),
        "insurance_expiry": _find_date_by_label(
            lines, ["Ins. Exp", "Insurance Expiry", "انتهاء التأمين"]
        ),
        "registration_expiry": registration_expiry,
        "registration_issuance": registration_issuance,
    }

    data = MulkiyaData(**{key: fv.value for key, fv in fields.items()})
    confidence = {key: fv.scored for key, fv in fields.items()}

    warnings = [note for fv in fields.values() for note in fv.notes]
    if flipped:
        warnings.append(
            "Arabic text arrived in visual order and was reversed to logical order."
        )
    warnings += [
        f"Could not confidently extract: {key}"
        for key in REQUIRED if getattr(data, key) is None
    ]
    return data, confidence, warnings
