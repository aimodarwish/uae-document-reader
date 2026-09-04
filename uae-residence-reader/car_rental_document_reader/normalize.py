from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from datetime import date, datetime
from itertools import permutations

import pycountry
from functools import lru_cache
from rapidfuzz.fuzz import ratio, token_set_ratio


@dataclass(frozen=True)
class NormalizedValue:
    value: str | None
    warnings: tuple[str, ...] = ()


# Letters that are drawn identically in Latin, Greek and Cyrillic. An OCR engine
# picks whichever script its language model favours, so the same printed word
# comes back spelled differently from one capture to the next. This is not a
# theoretical concern: the EU Council's own PRADO register transcribes the
# Kazakh licence title as "ЖΥΡΓΙЗУШІ ΚУƏЛІГІ" with Greek Υ, Ρ, Γ, Ι and Κ mixed
# into Cyrillic, and the Bosnian one as "ДOЗBOЛA" with Latin O and B. Folding
# every confusable to its Latin shape makes a match independent of which script
# the recogniser chose.
_CONFUSABLES = str.maketrans({
    # Cyrillic capitals
    "А": "A", "В": "B", "Е": "E", "З": "3", "И": "N", "К": "K", "М": "M",
    "Н": "H", "О": "O", "Р": "P", "С": "C", "Т": "T", "У": "Y", "Х": "X",
    "Ѕ": "S", "І": "I", "Ј": "J", "Ү": "Y", "Ө": "O", "Ә": "A", "Қ": "K",
    "Ұ": "Y", "Һ": "H",
    # Greek capitals
    "Α": "A", "Β": "B", "Ε": "E", "Ζ": "Z", "Η": "H", "Ι": "I", "Κ": "K",
    "Μ": "M", "Ν": "N", "Ο": "O", "Ρ": "P", "Τ": "T", "Υ": "Y", "Χ": "X",
    "Θ": "O",
    # Γ (Greek gamma), Г (Cyrillic ghe) and Ғ (Kazakh ghe with stroke) are the
    # same shape with no Latin counterpart, so they need a shared target of
    # their own rather than each being folded somewhere different.
    "Γ": "Г", "Ғ": "Г",
    # Λ (Greek lambda) and Л (Cyrillic el); Ə (Latin schwa, used by Azerbaijani)
    # and Ә (Cyrillic schwa, used by Kazakh) are likewise one shape apiece.
    "Λ": "Л", "Ə": "A",
})


# Recognisers routinely return a Greek or Cyrillic capital in place of the
# Latin letter it is drawn identically to. A UK licence number came back as
# "ΙΜΑAN010124Z99RY" -- iota, mu and alpha for I, M and A -- and no pattern
# expecting Latin letters could match it. Only the characters that are visually
# the same letter are mapped, so genuine Greek or Cyrillic text is untouched
# except where it was never Greek or Cyrillic to begin with.
_LATIN_LOOKALIKES = str.maketrans({
    "Α": "A", "Β": "B", "Ε": "E", "Ζ": "Z", "Η": "H", "Ι": "I", "Κ": "K",
    "Μ": "M", "Ν": "N", "Ο": "O", "Ρ": "P", "Τ": "T", "Υ": "Y", "Χ": "X",
    "А": "A", "В": "B", "Е": "E", "К": "K", "М": "M", "Н": "H", "О": "O",
    "Р": "P", "С": "C", "Т": "T", "У": "Y", "Х": "X",
    "α": "a", "ο": "o", "ρ": "p", "ν": "v", "а": "a", "е": "e", "о": "o",
    "р": "p", "с": "c", "х": "x", "у": "y",
})


def latinize_lookalikes(text: str) -> str:
    """Replace Greek and Cyrillic characters drawn as their Latin twins."""
    return text.translate(_LATIN_LOOKALIKES)


def fold_for_match(text: str) -> str:
    """Upper-case and strip accents so a document title matches its own wording.

    Two scripts break naive ``.upper()`` comparison outright:

    * Turkish. ``"Sürücü belgesi".upper()`` yields ``BELGESI`` with a dotless I,
      while the card and every reference print ``BELGESİ`` with the dotted one.
      The Turkish licence title therefore never matched, and Turkey is a country
      whose own licence is accepted without a permit.
    * Greek. ``"Άδεια οδήγησης".upper()`` keeps its accents (``ΆΔΕΙΑ ΟΔΉΓΗΣΗΣ``)
      where printed Greek drops them (``ΑΔΕΙΑ ΟΔΗΓΗΣΗΣ``).

    Folding both sides through NFKD and dropping combining marks fixes those two
    and every accented Latin and Vietnamese title at the same time, instead of
    storing an accented and an unaccented spelling of every marker. Cyrillic,
    Hebrew and CJK are unaffected; Arabic loses only its optional harakat, which
    is what a printed card omits anyway.
    """
    decomposed = unicodedata.normalize("NFKD", text.upper())
    stripped = "".join(char for char in decomposed if not unicodedata.combining(char))
    return stripped.translate(_CONFUSABLES)


_MONTHS = {
    "JAN": 1, "JANUARY": 1, "FEB": 2, "FEBRUARY": 2, "MAR": 3, "MARCH": 3,
    # Francophone documents often print February as ``FÉV/FEB``.  Accents are
    # folded before lookup, leaving ``FEV`` or ``FEVRIER`` as the visible
    # spelling on a French-only row.
    "FEV": 2, "FEVRIER": 2,
    "APR": 4, "APRIL": 4, "MAY": 5, "JUN": 6, "JUNE": 6,
    "JUL": 7, "JULY": 7, "AUG": 8, "AUGUST": 8, "SEP": 9, "SEPT": 9,
    "SEPTEMBER": 9, "OCT": 10, "OCTOBER": 10, "NOV": 11, "NOVEMBER": 11,
    "DEC": 12, "DECEMBER": 12,
}


def _month_number(token: str) -> int | None:
    """Resolve a month after a conservative OCR-confusion repair.

    Month words form a closed vocabulary, so a repaired token still has to
    equal a real month exactly.  This recovers Canadian passport dates such as
    ``31 0CT /0CT 23`` without loosening arbitrary text or identifier parsing.
    """
    upper = token.upper()
    direct = _MONTHS.get(upper)
    if direct is not None:
        return direct
    repaired = upper.translate(str.maketrans({"0": "O", "1": "I", "5": "S"}))
    return _MONTHS.get(repaired)


# Arabic month names, in the three vocabularies a passport actually prints. The
# Maghreb kept the French calendar (جانفي، فيفري، أفريل، ماي، جوان، جويلية، أوت),
# the Gulf transliterates the English one (يناير … ديسمبر) and the Levant uses
# the Syriac names (كانون الثاني … كانون الأول). Which set appears is a property
# of the issuing state, so all three are read rather than guessed at.
#
# The month is the only part of an Arabic-printed date that is a word, and on a
# passport it is the one row no machine-readable zone repeats: the Algerian
# booklet in this project's bug report printed "30 جانفي 2024" beside "صادر في",
# and with no month vocabulary to match, its issue date was reported missing.
_ARABIC_MONTH_NAMES: dict[int, tuple[str, ...]] = {
    1: ("يناير", "جانفي", "كانون الثاني"),
    2: ("فبراير", "فيفري", "شباط"),
    3: ("مارس", "آذار"),
    4: ("أبريل", "إبريل", "أفريل", "نيسان"),
    5: ("مايو", "ماي", "أيار"),
    6: ("يونيو", "يونية", "جوان", "حزيران"),
    7: ("يوليو", "يولية", "جويلية", "تموز"),
    8: ("أغسطس", "أوت", "آب"),
    9: ("سبتمبر", "أيلول"),
    10: ("أكتوبر", "تشرين الأول"),
    11: ("نوفمبر", "تشرين الثاني"),
    12: ("ديسمبر", "كانون الأول"),
}
# Folding a name strips the hamza carriers, so أفريل and افريل, آب and اب are one
# entry apiece rather than a spelling each. Both the printed form and its folded
# form are matchable, because the scanner that reads a whole OCR row applies
# this pattern to raw text while ``normalize_date`` applies it to folded text.
_ARABIC_MONTHS: dict[str, int] = {
    fold_for_match(name): number
    for number, names in _ARABIC_MONTH_NAMES.items() for name in names
}
_ARABIC_MONTH_SPELLINGS = sorted(
    {
        spelling
        for names in _ARABIC_MONTH_NAMES.values() for name in names
        for spelling in (name, fold_for_match(name))
    },
    key=len, reverse=True,
)
ARABIC_MONTH_PATTERN = "|".join(
    r"\s+".join(re.escape(word) for word in spelling.split())
    for spelling in _ARABIC_MONTH_SPELLINGS
)
_ARABIC_MONTH_RE = re.compile(ARABIC_MONTH_PATTERN)
# Every order the three parts arrive in, because bidirectional text has no one
# order. Google's OCR returns a mixed Arabic/Latin row in visual order, and the
# Algerian passport row above came back as "30 2024 جانفي" -- day, year, month --
# after the right-hand column was joined to the place-of-birth row beside it.
ARABIC_MONTH_DATE_PATTERN = (
    r"(?<!\d)(?:"
    + "|".join(
        r"[\s.,\-/]{0,3}".join(parts)
        for parts in permutations(
            (r"\d{1,2}", f"(?:{ARABIC_MONTH_PATTERN})", r"\d{4}"),
        )
    )
    + r")(?!\d)"
)


def _arabic_month_date(raw: str) -> NormalizedValue | None:
    """Parse a date whose month is an Arabic word, in whatever order it arrives.

    Anchoring on the month word and reading the two numbers left over is what
    makes this independent of the visual reordering: the day is the short one
    and the year is the four-digit one, whichever side of the word they landed
    on. Anything else in the row -- a second year, a stray figure from a joined
    column -- means the row is not one date, and nothing is claimed.

    ``None`` for a row with no Arabic month word at all, so that every existing
    spelling reaches the parsers below unchanged.
    """
    match = _ARABIC_MONTH_RE.search(raw)
    if match is None:
        return None
    numbers = re.findall(r"\d{1,4}", _ARABIC_MONTH_RE.sub(" ", raw))
    years = [number for number in numbers if len(number) == 4]
    days = [number for number in numbers if len(number) <= 2]
    if len(numbers) != 2 or len(years) != 1 or len(days) != 1:
        return None
    month = _ARABIC_MONTHS[fold_for_match(re.sub(r"\s+", " ", match.group(0)))]
    try:
        return NormalizedValue(
            date(int(years[0]), month, int(days[0])).isoformat(),
            ("ARABIC_MONTH_NAME_DATE",),
        )
    except ValueError:
        return NormalizedValue(None, ("INVALID_DATE",))


# The year is the last part a date prints, so it is closed only where nothing
# dated follows it: "19.01.11" is a two-figure year with dots between its own
# parts, and reading its first four characters as a year would destroy it.
_SPLIT_YEAR = re.compile(
    r"(?<=[\s.\-/])((?:19|20))[.,](\d{2})(?!\d)(?![.,\-/]\d)"
)


def close_split_year(text: str) -> str:
    """Close a mark a recogniser put inside a four-figure year.

    A French passport's issue row came back as "24 09 20.24": every figure
    present and in order, with a mark inside the year. Nothing reads that as a
    date, so the row bound to no field at all. Only 19xx and 20xx are closed,
    and only where the split is a single mark, so a pair of loose numbers does
    not become a year by being written next to one another.
    """
    return _SPLIT_YEAR.sub(r"\1\2", text)


# A numeric date carries one mark, repeated: "05.09.2022" or "05 09 2022".
# Where the two differ, one of them was not printed as it is read -- a dot
# thinned to nothing, a hyphen taken for the gap around it -- so the run is
# closed onto the mark it still shows. Only a four-figure 19xx/20xx year
# qualifies, and only a run standing clear of the figures around it, so a pair
# of loose numbers beside a serial is never joined into a date.
_MIXED_DATE_MARKS = re.compile(
    r"(?<![\d.,\-/])(\d{1,2})([.\-/]|\s)(\d{1,2})([.\-/]|\s)((?:19|20)\d{2})(?!\d)"
)


def _unify_marks(match: re.Match[str]) -> str:
    day, first, month, second, year = match.groups()
    if first == second or (first.strip() and second.strip()):
        return match.group(0)
    mark = first if first.strip() else second
    return f"{day}{mark}{month}{mark}{year}"


def close_mixed_date_marks(text: str) -> str:
    """Close a numeric date whose two separators disagree.

    A French passport's issue row came back as "05.09 2022": every figure in
    order, with one of the two marks missing. No date form spells its parts
    with two different separators, so the row matched nothing and bound to no
    field -- the same silent loss as a mark inside the year, and repaired the
    same way.
    """
    return _MIXED_DATE_MARKS.sub(_unify_marks, text)


def month_and_year(text: str) -> tuple[int, int] | None:
    """The month and year a row names when it carries no readable day.

    A passport prints its issue date as a day, a named month and a year, and a
    glare band or a fold takes the day off the row far more often than the two
    wider fields: the Algerian booklet in this project's bug report returned
    "أفريل 2019" with nothing before the month. That row is not a date and must
    never be normalised as one -- but it does state a month and a year, which
    is enough for a caller holding a proven expiry to finish.

    ``None`` unless the row names exactly one month and exactly one plausible
    year, so a bilingual row spelling one month twice still counts once and a
    row with two years counts for nothing.
    """
    folded = fold_for_match(text)
    months = {
        _ARABIC_MONTHS[fold_for_match(re.sub(r"\s+", " ", match.group(0)))]
        for match in _ARABIC_MONTH_RE.finditer(folded)
    }
    months.update(
        number
        for word in re.findall(r"[A-Z]{3,9}", folded)
        for number in (_month_number(word),)
        if number is not None
    )
    years = set(re.findall(r"(?<!\d)(?:19|20)\d{2}(?!\d)", folded))
    if len(months) != 1 or len(years) != 1:
        return None
    return next(iter(months)), int(next(iter(years)))


_COLON_DATE_SEPARATOR = re.compile(
    r"(?<!\d)(\d{1,2})\s*:\s*(\d{1,2})\s*[:\s]\s*((?:19|20)\d{2})(?!\d)"
)


# A bilingual passport prints its month twice around a slash -- "30 JUL /JUIL
# 17" on the British page -- and the slash is one thin stroke between two words
# in the smallest type on the card. Lost, the row reads as four separate tokens
# and matches no date at all: the British passport in this project's bug report
# reported no issue date while printing it plainly, and the zone cannot supply
# that field.
#
# The stroke is restored only where one of the two words is a month this
# reader knows. Four tokens in a row are not a date on their own.
_BILINGUAL_MONTH_GAP = re.compile(
    r"(?<!\d)(\d{1,2})\s+([A-Z]{3,9})\s+([A-Z]{3,9})\s+(\d{2,4})(?!\d)", re.I,
)


def close_bilingual_month_gap(text: str) -> str:
    """Restore the slash a recogniser lost between two spellings of a month."""
    def repair(match: re.Match[str]) -> str:
        if not (_month_number(match.group(2)) or _month_number(match.group(3))):
            return match.group(0)
        return (
            f"{match.group(1)} {match.group(2)}/{match.group(3)} {match.group(4)}"
        )
    return _BILINGUAL_MONTH_GAP.sub(repair, text)


def normalize_date(value: str | None, day_first_hint: bool | None = None) -> NormalizedValue:
    if not value or not value.strip():
        return NormalizedValue(None, ("MISSING_DATE",))
    # Date words are a closed vocabulary.  Folding their accents is therefore
    # safe and lets French ``FÉV/FEB`` use the same calendar parser as the
    # English half printed beside it, without altering names or identifiers.
    raw = re.sub(r"\s+", " ", fold_for_match(value).strip())
    # A colon is the separator some biodata pages print between the parts of
    # a numeric date -- the Zimbabwean passport in this project's bug report
    # carries "17:02:2020" for issue and "16:02 2030" for expiry -- and every
    # numeric form below expects a dot, a dash or a slash. Folding it here
    # keeps the repair in one place for each caller that isolated the token.
    # A four-figure 19xx/20xx year is required, so a time of day is left as
    # the time it is.
    raw = _COLON_DATE_SEPARATOR.sub(r"\1.\2.\3", raw)
    # Before every Latin form below: a named month in Arabic is a closed
    # vocabulary of its own, and the row carrying it may have been reordered.
    arabic = _arabic_month_date(raw)
    if arabic is not None:
        return arabic
    iso_match = re.fullmatch(r"(\d{4})[-/.](\d{1,2})[-/.](\d{1,2})", raw)
    if iso_match:
        parts = tuple(map(int, iso_match.groups()))
        try: return NormalizedValue(date(*parts).isoformat())
        except ValueError: return NormalizedValue(None, ("INVALID_DATE",))
    # A Romanian passport writes the month twice, in Romanian and English:
    # "01 IUL/JUL 26". Unparsed, the issue date was lost -- it is the one date
    # no machine-readable zone carries -- and the row was mistaken for a place
    # of birth by the label above it.
    bilingual = re.fullmatch(
        r"(\d{1,2})[\s\-/.]*([A-Z0-9\u0370-\u03FF\u0400-\u04FF\u0530-\u058F]{3,9})"
        r"\s*/\s*"
        r"([A-Z0-9\u0370-\u03FF\u0400-\u04FF\u0530-\u058F]{3,9})"
        r"[\s\-/.]*(\d{2,4})", raw,
    )
    if bilingual:
        month = _month_number(bilingual.group(3)) or _month_number(bilingual.group(2))
        if month:
            return normalize_date(
                f"{bilingual.group(1)}.{month}.{bilingual.group(4)}", day_first_hint=True,
            )
    # Canada writes the year first and names the month: "2023-Mar-03" on a
    # British Columbia licence, "1989-Sep-02" for the birth date beside it.
    # Every form below expects the day in front, so this one parsed as nothing
    # and the card reported no issue date, no expiry and no birth date.
    year_first = re.fullmatch(
        r"(\d{4})[\s\-/.]+([A-Z0-9]+)[\s\-/.]+(\d{1,2})", raw,
    )
    year_first_month = _month_number(year_first.group(2)) if year_first else None
    if year_first and year_first_month:
        try:
            return NormalizedValue(date(
                int(year_first.group(1)), year_first_month,
                int(year_first.group(3)),
            ).isoformat())
        except ValueError:
            return NormalizedValue(None, ("INVALID_DATE",))
    # Some North-American and association-issued documents print a date as
    # ``JAN 20, 2019``.  It must be recognized as a date before a generic
    # identifier fallback can mistake its digits for a permit number.
    month_first = re.fullmatch(
        r"([A-Z0-9]+)[\s\-/.]+(\d{1,2})(?:,\s*|[\s\-/.]+)(\d{2,4})", raw,
    )
    month_first_month = _month_number(month_first.group(1)) if month_first else None
    if month_first and month_first_month:
        try:
            year = int(month_first.group(3))
            if year < 100:
                pivot = (date.today().year + 20) % 100
                year += 2000 if year <= pivot else 1900
            return NormalizedValue(date(
                year, month_first_month, int(month_first.group(2)),
            ).isoformat())
        except ValueError:
            return NormalizedValue(None, ("INVALID_DATE",))
    text_match = re.fullmatch(
        r"(\d{1,2})[\s\-/.]+([A-Z0-9]+)[\s\-/.]+(\d{2,4})", raw,
    )
    text_month = _month_number(text_match.group(2)) if text_match else None
    if text_match and text_month:
        try:
            year = int(text_match.group(3))
            if year < 100:
                pivot = (date.today().year + 20) % 100
                year += 2000 if year <= pivot else 1900
            result = date(year, text_month, int(text_match.group(1)))
            return NormalizedValue(result.isoformat())
        except ValueError:
            return NormalizedValue(None, ("INVALID_DATE",))
    # A Belgian passport separates the parts with spaces -- "09 07 2021" --
    # and nothing else on the page carries the issue date, so refusing that
    # spelling lost the field outright. Both parts are required to be two
    # digits: a single space between shorter runs of figures is a much weaker
    # signal that a date is what is being read.
    numeric = re.fullmatch(r"(\d{1,2})[-/.](\d{1,2})[-/.](\d{4})", raw) or re.fullmatch(
        r"(\d{2}) (\d{2}) (\d{4})", raw,
    )
    extra: tuple[str, ...] = ()
    if not numeric:
        # A two-digit year, as printed on the Queensland licence ("Effective
        # 19.01.11"). These were not parsed at all, so the field came back empty
        # with no explanation rather than as a value someone could check. The
        # century is genuinely not on the card, so it is inferred and the
        # candidate is marked for review rather than presented as read.
        short = re.fullmatch(r"(\d{1,2})[-/.](\d{1,2})[-/.](\d{2})", raw) or re.fullmatch(
            # Denmark prints passport dates as ``14 11 22``. Keep this out of
            # the generic date scanner (three short number groups are weak on
            # an arbitrary page), but allow a caller that has already isolated
            # the token to normalize it with the same explicit century warning
            # as dotted two-digit dates.
            r"(\d{2}) (\d{2}) (\d{2})", raw,
        )
        if not short:
            # A recogniser can drop a mark inside the year itself: a French
            # passport's issue row came back as "24 09 20.24". Every figure is
            # there and in order, and closing that one split is the whole
            # repair -- the result still has to be a real calendar date, and
            # the year still has to read 19xx or 20xx.
            closed = close_split_year(raw)
            if closed != raw:
                retry = normalize_date(closed, day_first_hint)
                if retry.value:
                    return NormalizedValue(
                        retry.value, (*retry.warnings, "YEAR_SPLIT_BY_A_MARK"),
                    )
            # The same loss one place earlier: a mark between two of the
            # date's own parts, rather than inside the year.
            unified = close_mixed_date_marks(closed)
            if unified != closed:
                retry = normalize_date(unified, day_first_hint)
                if retry.value:
                    return NormalizedValue(
                        retry.value, (*retry.warnings, "DATE_MARKS_DISAGREED"),
                    )
            return NormalizedValue(None, ("UNRECOGNIZED_DATE_FORMAT",))
        numeric, extra = short, ("TWO_DIGIT_YEAR_CENTURY_INFERRED",)
    first, second, year = map(int, numeric.groups())
    if year < 100:
        # Cards in hand carry dates near the present; a year more than twenty
        # ahead is a past century, not a future one.
        pivot = (date.today().year + 20) % 100
        year += 2000 if year <= pivot else 1900
    if first <= 12 and second <= 12 and first != second and day_first_hint is None:
        return NormalizedValue(None, ("AMBIGUOUS_DAY_MONTH",))
    day_first = day_first_hint if day_first_hint is not None else first > 12
    day, month = (first, second) if day_first else (second, first)
    try:
        return NormalizedValue(date(year, month, day).isoformat(), extra)
    except ValueError:
        return NormalizedValue(None, ("INVALID_DATE",))


# No jurisdiction licenses a driver below this age, counting the lowest moped
# and farm-permit ages rather than the ordinary car-driving age, so that a real
# holder is never rejected.
MINIMUM_DRIVING_AGE_YEARS = 14


def implausible_birth_date(
    iso_value: str | None, today: date | None = None,
    minimum_age: int | None = None,
) -> bool:
    """True when an ISO date cannot be anyone's date of birth.

    Applied when a candidate is built rather than after reconciliation: a
    card-expiry row read against a birth-date label is a well-formed date, so it
    survives format validation and then competes with the real birth date. That
    leaves the field CONFLICTING and empty instead of simply unmatched.

    ``minimum_age`` adds the rule that only applies on a driving document: its
    holder had to be old enough to drive. Without it, a Vietnamese licence whose
    "Năm sinh" row carries a year alone let the birth-date label reach the
    "Có giá trị đến" row instead, and an expiry of 2021 was accepted as the
    birth date of someone who would have been five. It is not passed for
    passports, which children hold legitimately.
    """
    if not iso_value: return False
    today = today or date.today()
    if iso_value > today.isoformat() or int(iso_value[:4]) < today.year - 120:
        return True
    if minimum_age is None:
        return False
    try:
        born = date.fromisoformat(iso_value)
    except ValueError:
        return False
    latest_allowed = date(today.year - minimum_age, today.month, today.day)
    return born > latest_allowed


def validate_date_relationships(birth: str | None = None, issue: str | None = None, expiry: str | None = None, today: date | None = None) -> list[str]:
    today = today or date.today()
    warnings: list[str] = []
    parsed: dict[str, date | None] = {}
    for key, value in {"birth": birth, "issue": issue, "expiry": expiry}.items():
        if not value:
            parsed[key] = None
            continue
        try:
            parsed[key] = datetime.strptime(value, "%Y-%m-%d").date()
        except (TypeError, ValueError):
            parsed[key] = None
            warnings.append(f"INVALID_{key.upper()}_DATE_FORMAT")
    if parsed["birth"] and parsed["birth"] > today: warnings.append("BIRTH_DATE_IN_FUTURE")
    if parsed["birth"] and parsed["birth"].year < today.year - 120: warnings.append("IMPLAUSIBLE_BIRTH_YEAR")
    if parsed["issue"] and parsed["expiry"] and parsed["issue"] > parsed["expiry"]: warnings.append("ISSUE_AFTER_EXPIRY")
    if parsed["expiry"] and parsed["expiry"] < today: warnings.append("DOCUMENT_EXPIRED")
    return warnings


def normalize_emirates_id(value: str | None) -> NormalizedValue:
    if not value:
        return NormalizedValue(None, ("MISSING_EMIRATES_ID",))
    digits = re.sub(r"\D", "", value)
    if len(digits) != 15 or not digits.startswith("784"):
        return NormalizedValue(None, ("INVALID_EMIRATES_ID_PUBLIC_FORMAT",))
    return NormalizedValue(f"{digits[:3]}-{digits[3:7]}-{digits[7:14]}-{digits[14]}")


_COUNTRY_ALIASES = {
    "UAE": "ARE", "UNITED ARAB EMIRATES": "ARE", "EMIRATES": "ARE",
    "UK": "GBR", "UNITED KINGDOM": "GBR", "GREAT BRITAIN": "GBR",
    "US": "USA", "U.S.A": "USA", "UNITED STATES": "USA", "AMERICA": "USA",
    "SYRIA": "SYR", "SYRIAN ARAB REPUBLIC": "SYR",
    "RUSSIA": "RUS", "RUSSIAN FEDERATION": "RUS",
    "РОССИЯ": "RUS", "РОССИЙСКАЯ ФЕДЕРАЦИЯ": "RUS",
    "SAUDI": "SAU", "SAUDI ARABIA": "SAU",
    "KINGDOM OF SAUDI ARABIA": "SAU", "المملكة العربية السعودية": "SAU",
    "السعودية": "SAU", "سعودي": "SAU", "سعودية": "SAU",
    "KUWAIT": "KWT", "STATE OF KUWAIT": "KWT", "الكويت": "KWT",
    "كويتي": "KWT", "كويتية": "KWT",
    "BAHRAIN": "BHR", "KINGDOM OF BAHRAIN": "BHR", "مملكة البحرين": "BHR",
    "البحرين": "BHR",
    "بحريني": "BHR", "بحرينية": "BHR",
    "QATAR": "QAT", "STATE OF QATAR": "QAT", "قطر": "QAT",
    "قطري": "QAT", "قطرية": "QAT",
    "OMAN": "OMN", "SULTANATE OF OMAN": "OMN", "سلطنة عمان": "OMN",
    "عمان": "OMN", "عُمان": "OMN", "عماني": "OMN", "عمانية": "OMN",
}


def normalize_country(value: str | None) -> tuple[str | None, str | None, list[str]]:
    if not value:
        return None, None, ["MISSING_COUNTRY"]
    cleaned = re.sub(r"[^\w .'-]", "", unicodedata.normalize("NFKC", value).upper(), flags=re.UNICODE).strip()
    code = _COUNTRY_ALIASES.get(cleaned, cleaned if len(cleaned) == 3 else None)
    country = pycountry.countries.get(alpha_3=code) if code else None
    if country is None:
        try: country = pycountry.countries.lookup(cleaned)
        except LookupError: return None, None, ["UNKNOWN_COUNTRY"]
    return country.alpha_3, country.name, []


# A licence states the holder's nationality as an adjective -- MEXICANA,
# ITALIANA, IRANIAN -- while every table here is keyed on the country's name.
# ``normalize_country`` therefore refused the Mexico City card's "Nacionalidad
# / MEXICANA" outright. An adjective is the country's own stem plus one of a
# closed set of endings, so it is read that way: the stem must match a country
# in full (its final vowel aside), the ending must be one of these, and exactly
# one country may answer. NIGERIEN is the reason the ending list is closed and
# the match must be unique -- it belongs to Niger, and a looser rule hands it
# to Nigeria.
_NATIONALITY_ENDINGS = (
    "AISE", "AIS", "ACA", "ACO", "ANA", "ANO", "ANE", "AN",
    "ENSE", "ESA", "ESE", "ESI", "ES",
    "IANA", "IANO", "IAN", "IENNE", "IEN", "ICA", "ICO",
    "ISCHE", "ISCH", "ISH", "ITA", "ITE",
    "A", "O", "E", "I",
)
_NATIONALITY_MINIMUM_STEM = 3


@lru_cache(maxsize=1)
def _country_stems() -> tuple[tuple[str, str], ...]:
    """Every country name and alias, trimmed to the stem an adjective uses."""
    stems: dict[str, set[str]] = {}
    names: list[tuple[str, str]] = [
        (alias, code) for alias, code in _COUNTRY_ALIASES.items()
    ]
    for country in pycountry.countries:
        for attribute in ("name", "official_name", "common_name"):
            spelling = getattr(country, attribute, None)
            if spelling:
                names.append((spelling, country.alpha_3))
    for spelling, code in names:
        folded = re.sub(r"[^A-Z]", "", _ascii_folded(spelling).upper())
        if len(folded) < _NATIONALITY_MINIMUM_STEM + 1:
            continue
        for stem in {folded, folded[:-1] if folded[-1] in "AEIOUY" else folded}:
            if len(stem) >= _NATIONALITY_MINIMUM_STEM:
                stems.setdefault(stem, set()).add(code)
    return tuple(
        (stem, next(iter(codes))) for stem, codes in stems.items()
        if len(codes) == 1
    )


def _ascii_folded(value: str) -> str:
    decomposed = unicodedata.normalize("NFKD", value)
    return "".join(c for c in decomposed if not unicodedata.combining(c))


def nationality_country(value: str | None) -> tuple[str | None, str | None]:
    """Resolve a nationality, written as a country or as its adjective."""
    code, name, _warnings = normalize_country(value)
    if code is not None:
        return code, name
    if not value:
        return None, None
    folded = re.sub(r"[^A-Z]", "", _ascii_folded(value).upper())
    if len(folded) < _NATIONALITY_MINIMUM_STEM + 2:
        return None, None
    matched: set[str] = set()
    for stem, alpha_3 in _country_stems():
        if not folded.startswith(stem) or folded == stem:
            continue
        if folded[len(stem):] in _NATIONALITY_ENDINGS:
            matched.add(alpha_3)
    if len(matched) != 1:
        return None, None
    country = pycountry.countries.get(alpha_3=next(iter(matched)))
    return (country.alpha_3, country.name) if country else (None, None)


def normalize_name(value: str | None) -> str:
    if not value:
        return ""
    value = unicodedata.normalize("NFKD", value)
    value = "".join(c for c in value if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", re.sub(r"[^A-Z0-9 ]", " ", value.upper())).strip()


def name_similarity(a: str | None, b: str | None) -> float | None:
    na, nb = normalize_name(a), normalize_name(b)
    if not na or not nb:
        return None
    ordered = ratio(na, nb)
    tokens = token_set_ratio(na, nb)
    return round((0.6 * ordered + 0.4 * tokens) / 100.0, 4)


def display_date(iso_value: str | None) -> str:
    if not iso_value:
        return ""
    try: return datetime.strptime(iso_value, "%Y-%m-%d").strftime("%d-%m-%Y")
    except ValueError: return ""
