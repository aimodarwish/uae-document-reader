from __future__ import annotations

import itertools
import re
from dataclasses import dataclass, field
from datetime import date
from typing import Any


WEIGHTS = (7, 3, 1)
CONFUSIONS = {"O": "0", "I": "1", "B": "8", "S": "5", "Z": "2", "G": "6"}


def character_value(char: str) -> int:
    if char == "<": return 0
    if char.isdigit(): return int(char)
    if "A" <= char <= "Z": return ord(char) - 55
    raise ValueError(f"Invalid MRZ character {char!r}")


def check_digit(data: str) -> str:
    return str(sum(character_value(c) * WEIGHTS[i % 3] for i, c in enumerate(data)) % 10)


def validate_check(data: str, digit: str) -> bool:
    return digit.isdigit() and check_digit(data) == digit


_PASSPORT_MRZ_NAME_ROW = re.compile(r"P[A-Z<][A-Z]{3}[A-Z]+<<[A-Z<]+")
_PASSPORT_MRZ_ROW_LENGTHS = (36, 44)


def passport_name_row_present(rows: list[str] | None) -> bool:
    """Whether a restored MRZ row is a passport's name row.

    This answers "which document is this", not "are its fields trustworthy" --
    two questions the reader had joined. An Austrian passport whose zone lost
    two printed check digits to OCR parsed as a TD3 with every field correct
    and ``valid`` False, so the page was routed as an unknown card and then
    labelled a second driving licence; the holder's name, which a tourist may
    only take from an identity document, was reported as absent.

    Deliberately not a substring test. A loose "P<" occurs in the multilingual
    wording of an IDP booklet, and matching it there once cost a Russian permit
    its own document type. What is matched here is a row the zone restorer
    already produced at a standard length, whose whole shape is the passport
    name row: the P code, an issuing state, a surname, and the double chevron
    that separates it from the given names.
    """
    return any(
        len(row) in _PASSPORT_MRZ_ROW_LENGTHS
        and _PASSPORT_MRZ_NAME_ROW.fullmatch(row) is not None
        for row in rows or ()
    )


def normalize_mrz_line(line: str) -> str:
    return re.sub(r"[^A-Z0-9<]", "", line.upper().replace(" ", "<"))


def mrz_row_shape(compact: str, length: int) -> str | None:
    """Whether a row is shaped like an MRZ data row, a name row, or neither.

    Length alone does not identify a machine readable zone, and treating it as
    if it did read page furniture as one. A Belgian passport photographed with
    the facing page in frame produced "AUSSTELLUNGSLAND / ISSUING COUNTRY" and
    "AUSSTELLUNGSDATUM / DATE OF ISSUE", both exactly thirty-six characters
    once spaces became filler, and they parsed as a TD2 zone: issuing state
    SST, surname ELLUNGSLAND. That surname conflicted with the real one and
    emptied the holder's name, while the genuine zone at the foot of the page
    was rejected for having lost the tail of its filler.

    A data row is mostly digits by construction -- two dates, three check
    digits and a document number -- and that is the property no line of prose
    has. The name row is left deliberately loose, because a zone is only ever
    accepted as a pair and the data row is what proves it is one.
    """
    if not compact:
        return None
    digits = sum(char.isdigit() for char in compact)
    head = compact[:28]
    if len(head) >= 20 and sum(char.isdigit() for char in head) >= len(head) * 0.4:
        return "data"
    # The opening row of a TD1 card: a document code, the document number and
    # whatever the issuer puts in the optional field beside it.
    if compact[0].isalpha() and digits >= 5:
        return "code"
    if "<<" in compact and digits <= 2:
        return "name"
    return None


def zone_rows_have_shape(normalized: list[str], length: int) -> bool:
    """True when the rows carry the roles their format assigns them.

    TD2 and TD3 are a name row above a data row; TD1 opens with the document
    number, then the dates, then the name. Checking the roles is what stops two
    lines of ordinary text that happen to share a row length from being read as
    a zone -- the data row is the one no line of prose can imitate.
    """
    shapes = [mrz_row_shape(row, length) for row in normalized]
    if len(normalized) == 2:
        return shapes[0] in {"name", "code"} and shapes[1] == "data"
    return (
        shapes[0] in {"code", "data"}
        and shapes[1] == "data"
        and shapes[2] == "name"
    )


@dataclass
class ParsedMRZ:
    mrz_type: str | None = None
    raw_lines: list[str] = field(default_factory=list)
    normalized_lines: list[str] = field(default_factory=list)
    fields: dict[str, str | None] = field(default_factory=dict)
    checks: dict[str, bool | None] = field(default_factory=dict)
    corrections: list[dict[str, Any]] = field(default_factory=list)
    valid: bool = False
    warnings: list[str] = field(default_factory=list)


def _name_fields(raw: str) -> dict[str, str | None]:
    surname_raw, _, given_raw = raw.partition("<<")
    surname = re.sub(r"<+", " ", surname_raw).strip() or None
    given = [part for part in re.split(r"<+", given_raw) if part]
    return {
        "last_name": surname,
        # The business field is the passport's complete GIVEN NAMES value, not
        # merely its first token.  A holder whose printed given names are
        # "AMINA NOOR" must therefore reach first_name as "AMINA NOOR"; the
        # optional middle-name field remains available internally for callers
        # that still need the component after the first token.
        "first_name": " ".join(given) or None,
        "middle_name": " ".join(given[1:]) or None,
        "given_names": " ".join(given) or None,
        "full_name": " ".join([*(given or []), *([surname] if surname else [])]) or None,
    }


def _short_date(raw: str, kind: str, today: date) -> tuple[str | None, str | None]:
    if not re.fullmatch(r"\d{6}", raw): return None, "INVALID_MRZ_DATE"
    yy, mm, dd = int(raw[:2]), int(raw[2:4]), int(raw[4:6])
    candidates: list[date] = []
    for century in (1900, 2000, 2100):
        try: candidates.append(date(century + yy, mm, dd))
        except ValueError: pass
    if kind == "birth":
        plausible = [d for d in candidates if d <= today and 0 <= (today - d).days / 365.2425 <= 120]
        if len(plausible) == 1: return plausible[0].isoformat(), None
        # Two centuries fit inside a 120-year window for every year from 2006
        # on: "060823" is 2006 or 1906, and in 2026 both are "under 120". They
        # are exactly a century apart, so whichever pair survives, the older
        # one belongs to somebody past 100 and the recent one is the holder.
        # Taking the earliest dated a French customer born on 23 August 2006 to
        # 1906 -- with the zone's own check digit passing, because no check
        # digit covers the century. The most recent past reading is also how
        # every border system resolves a two-digit year.
        if plausible: return max(plausible).isoformat(), None
    else:
        plausible = [d for d in candidates if today.year - 20 <= d.year <= today.year + 30]
        if len(plausible) == 1: return plausible[0].isoformat(), None
        if plausible: return min(plausible, key=lambda d: abs((d - today).days)).isoformat(), "AMBIGUOUS_MRZ_CENTURY"
    return None, "IMPLAUSIBLE_MRZ_DATE"


def _field_with_repair(value: str, expected_digit: str, numeric: bool, label: str) -> tuple[str, list[dict[str, Any]]]:
    if validate_check(value, expected_digit): return value, []
    candidates: list[tuple[str, list[dict[str, Any]]]] = []
    options: list[list[str]] = []
    for c in value:
        current = [c]
        if c in CONFUSIONS and (numeric or CONFUSIONS[c].isdigit()): current.append(CONFUSIONS[c])
        reverse = [k for k, v in CONFUSIONS.items() if v == c and not numeric]
        current.extend(reverse)
        options.append(list(dict.fromkeys(current)))
    for combination in itertools.product(*options):
        changed = sum(a != b for a, b in zip(value, combination))
        if not 1 <= changed <= 2: continue
        candidate = "".join(combination)
        if numeric and not candidate.isdigit(): continue
        if validate_check(candidate, expected_digit):
            changes = [{"field": label, "position": i, "from": a, "to": b, "reason": "field class and check digit"} for i, (a, b) in enumerate(zip(value, candidate)) if a != b]
            candidates.append((candidate, changes))
    unique = {candidate: changes for candidate, changes in candidates}
    if len(unique) == 1:
        repaired, changes = next(iter(unique.items()))
        return repaired, changes
    return value, []


# The reverse of the confusion table: what a digit standing in a letters-only
# field was meant to be.
_DIGIT_TO_LETTER = {digit: letter for letter, digit in CONFUSIONS.items()}

# Germany is the ICAO passport exception: the three-character issuing-state
# and nationality slots contain ``D<<`` rather than ISO alpha-3 ``DEU``. The
# rest of the system stores ISO alpha-3, so filler is removed and the one-letter
# German designator is translated at the MRZ boundary. Keeping this here makes
# every consumer -- passport issuer, nationality and cross-document checks --
# see one canonical value.
_MRZ_COUNTRY_CODE_ALIASES = {"D": "DEU"}


def normalize_mrz_country_code(code: str | None) -> str | None:
    if not code:
        return None
    raw = code.strip().upper()
    compact = raw.replace("<", "")
    if compact in _MRZ_COUNTRY_CODE_ALIASES:
        return _MRZ_COUNTRY_CODE_ALIASES[compact]
    # Preserve an unrepairable OCR value (for example R0U) so downstream
    # reconciliation can surface it as unverified evidence instead of silently
    # dropping it. Only a structurally valid code, or an explicit ICAO alias
    # above, is canonicalized.
    return compact if re.fullmatch(r"[A-Z]{3}", compact) else raw


def repair_country_code(
    code: str | None, issuing_country_code: str | None,
) -> tuple[str | None, list[dict[str, Any]]]:
    """Put a letter back in a nationality field that OCR read as a digit.

    No check digit covers the nationality. The composite digit of a TD3 zone
    is computed over the document number, the dates and the optional data --
    not over those three characters -- so a Romanian passport read as R0U was
    reported as verified with every check passing, and the field then matched
    no country at all.

    A country code is three letters, so a digit there is not a value: it is a
    misreading with one obvious correction. The repair is accepted only when
    it agrees with the issuing state printed on the row above, which is the
    same three letters on a passport and is what makes this evidence rather
    than a guess.
    """
    if not code or not any(character.isdigit() for character in code):
        return code, []
    repaired = "".join(_DIGIT_TO_LETTER.get(character, character) for character in code)
    if repaired != issuing_country_code or not repaired.isalpha():
        return code, []
    return repaired, [{
        "field": "nationality_code", "position": 10, "from": code, "to": repaired,
        "reason": "letters-only field agreed with the issuing state",
    }]


def realign_name_line(line: str, nationality: str) -> tuple[str, list[dict[str, Any]]]:
    """Restore a character dropped from the issuing-state field of row one.

    Row one of a passport zone carries no check digit of its own, so a
    character lost there is silent: every field after it shifts one place left.
    On this Albanian page the "L" of ALB went missing and the row read
    ``P<ABGECAJ<<DANILO``, which stored the issuer as ABG and -- far worse --
    the surname as ECAJ, presented as verified because the rest of the zone
    checked out.

    Row two states the nationality inside the span its composite check digit
    covers, and on a passport that is the same three letters as the issuing
    state. That makes the loss detectable and the repair determined: there is
    one place the missing character can go that restores those three letters.
    Nothing is assumed about the name, which simply stops being misaligned.
    """
    if len(nationality) != 3 or len(line) < 6 or line[2:5] == nationality:
        return line, []
    if not line.endswith("<"):
        return line, []                 # no filler to give back the character
    repairs = {
        line[:position] + nationality[position - 2] + line[position:-1]
        for position in (2, 3, 4)
        if (line[:position] + nationality[position - 2] + line[position:])[2:5]
        == nationality
    }
    if len(repairs) != 1:
        return line, []
    repaired = next(iter(repairs))
    return repaired, [{
        "field": "name_line", "position": 2, "from": line[2:5], "to": nationality,
        "reason": "issuing state realigned to the checksummed nationality",
    }]


def td1_filler_repairs(lines: list[str]) -> list[list[str]]:
    """Return TD1 readings with cropped trailing filler restored.

    A TD1 zone is three rows of exactly thirty characters, and the tail of each
    row is filler ``<``. Scanners and phone captures routinely clip one or two
    of those filler characters, which used to make the whole zone unparseable
    and throw away the checksummed birth date, expiry and sex printed on the
    reverse of most GCC identity cards.

    Restoring filler is a guess about layout, never about data, and every
    reading it produces is still subject to the row's own check digits, so a
    wrong repair simply fails validation instead of inventing a value. The
    middle row ends with its composite check digit, so filler is reinserted
    before that digit rather than after it.
    """
    if len(lines) != 3:
        return []
    normalized = [normalize_mrz_line(line) for line in lines]
    if not all(24 <= len(line) < 30 for line in normalized):
        return []
    options: list[list[str]] = []
    for index, line in enumerate(normalized):
        missing = 30 - len(line)
        variants = [line + "<" * missing]
        if index == 1 and line and line[-1].isdigit():
            variants.append(line[:-1] + "<" * missing + line[-1])
        options.append(variants)
    return [
        [first, second, third]
        for first in options[0] for second in options[1] for third in options[2]
    ]


def pad_short_name_row(rows: list[str]) -> list[str]:
    """Restore filler a recogniser dropped from the end of a name row.

    The upper row of a zone carries no check digit at all: it is the document
    code, the issuing state, the names, and filler to the end of the line. A
    French passport in this project's bug report returned that row as
    "P<FRADUHO<<ABRANI<<" followed by katakana -- the recogniser read part of
    the filler as another script -- and once those characters were dropped the
    row was ten short of its length. The zone was then refused whole, taking
    with it a lower row whose every check digit was intact and which alone
    proves the document number, the birth date, the sex and the expiry.

    Only a row that already ends in filler is padded, so a name cut off
    mid-word is never quietly completed, and only ever with the filler
    character, which is what those positions carried on the page.
    """
    if len(rows) != 2:
        return rows
    length = next((len(row) for row in rows if len(row) in {44, 36}), None)
    if length is None:
        return rows
    return [
        row.ljust(length, "<")
        if (
            length - 12 <= len(row) < length
            and row.endswith("<")
            # A name row has no digits, and a row that does carries the check
            # digits this padding must never invent.
            and re.fullmatch(r"[A-Z<]+", row)
        )
        else row
        for row in rows
    ]


def parse_mrz(lines: list[str], today: date | None = None, allow_correction: bool = True) -> ParsedMRZ:
    today = today or date.today()
    normalized = pad_short_name_row(
        [normalize_mrz_line(line) for line in lines if normalize_mrz_line(line)],
    )
    result = ParsedMRZ(raw_lines=list(lines), normalized_lines=normalized)
    if len(normalized) == 2 and all(len(line) == 44 for line in normalized):
        if not zone_rows_have_shape(normalized, 44):
            result.warnings.append("UNSUPPORTED_OR_INCOMPLETE_MRZ")
            return result
        result.mrz_type = "TD3"
        l1, l2 = normalized
        document_number = l2[0:9]
        if allow_correction:
            document_number, changes = _field_with_repair(document_number, l2[9], False, "document_number")
            result.corrections.extend(changes)
        checks = {
            "document_number": validate_check(document_number, l2[9]),
            "date_of_birth": validate_check(l2[13:19], l2[19]),
            "expiry_date": validate_check(l2[21:27], l2[27]),
            "optional_data": validate_check(l2[28:42], l2[42]),
            "composite": validate_check(l2[0:10] + l2[13:20] + l2[21:43], l2[43]),
        }
        birth, bw = _short_date(l2[13:19], "birth", today)
        expiry, ew = _short_date(l2[21:27], "expiry", today)
        if allow_correction and checks["composite"]:
            l1, realignment = realign_name_line(l1, l2[10:13])
            result.corrections.extend(realignment)
        result.fields = {
            "document_code": l1[0:2].replace("<", ""), "issuing_country_code": l1[2:5],
            **_name_fields(l1[5:]), "document_number": document_number.replace("<", ""),
            "nationality_code": l2[10:13], "date_of_birth": birth,
            "gender": None if l2[20] == "<" else l2[20], "expiry_date": expiry,
            "optional_data": l2[28:42].rstrip("<") or None,
        }
        result.checks = checks
        result.warnings.extend([w for w in (bw, ew) if w])
    elif len(normalized) == 2 and all(len(line) == 36 for line in normalized):
        if not zone_rows_have_shape(normalized, 36):
            result.warnings.append("UNSUPPORTED_OR_INCOMPLETE_MRZ")
            return result
        result.mrz_type = "TD2"
        l1, l2 = normalized
        checks = {
            "document_number": validate_check(l2[0:9], l2[9]),
            "date_of_birth": validate_check(l2[13:19], l2[19]),
            "expiry_date": validate_check(l2[21:27], l2[27]),
            "optional_data": None,
            "composite": validate_check(l2[0:10] + l2[13:20] + l2[21:35], l2[35]),
        }
        birth, bw = _short_date(l2[13:19], "birth", today)
        expiry, ew = _short_date(l2[21:27], "expiry", today)
        if allow_correction and checks["composite"]:
            l1, realignment = realign_name_line(l1, l2[10:13])
            result.corrections.extend(realignment)
        result.fields = {
            "document_code": l1[0:2].replace("<", ""), "issuing_country_code": l1[2:5],
            **_name_fields(l1[5:]), "document_number": l2[0:9].replace("<", ""),
            "nationality_code": l2[10:13], "date_of_birth": birth,
            "gender": None if l2[20] == "<" else l2[20], "expiry_date": expiry,
            "optional_data": l2[28:35].rstrip("<") or None,
        }
        result.checks = checks
        result.warnings.extend([w for w in (bw, ew) if w])
    elif len(normalized) == 3 and all(len(line) == 30 for line in normalized):
        if not zone_rows_have_shape(normalized, 30):
            result.warnings.append("UNSUPPORTED_OR_INCOMPLETE_MRZ")
            return result
        result.mrz_type = "TD1"
        l1, l2, l3 = normalized
        checks = {
            "document_number": validate_check(l1[5:14], l1[14]),
            "date_of_birth": validate_check(l2[0:6], l2[6]),
            "expiry_date": validate_check(l2[8:14], l2[14]),
            "optional_data": None,
            "composite": validate_check(l1[5:30] + l2[0:7] + l2[8:15] + l2[18:29], l2[29]),
        }
        birth, bw = _short_date(l2[0:6], "birth", today)
        expiry, ew = _short_date(l2[8:14], "expiry", today)
        result.fields = {
            "document_code": l1[0:2].replace("<", ""), "issuing_country_code": l1[2:5],
            **_name_fields(l3), "document_number": l1[5:14].replace("<", ""),
            "nationality_code": l2[15:18], "date_of_birth": birth,
            "gender": None if l2[7] == "<" else l2[7], "expiry_date": expiry,
            "optional_data": (l1[15:30] + l2[18:29]).rstrip("<") or None,
        }
        result.checks = checks
        result.warnings.extend([w for w in (bw, ew) if w])
    else:
        result.warnings.append("UNSUPPORTED_OR_INCOMPLETE_MRZ")
        return result
    nationality, repair = repair_country_code(
        result.fields.get("nationality_code"), result.fields.get("issuing_country_code"),
    )
    if repair:
        result.fields["nationality_code"] = nationality
        result.corrections.extend(repair)
    for field_name in ("issuing_country_code", "nationality_code"):
        raw_code = result.fields.get(field_name)
        normalized_code = normalize_mrz_country_code(raw_code)
        result.fields[field_name] = normalized_code
        if raw_code and normalized_code and raw_code != normalized_code:
            result.corrections.append({
                "field": field_name, "position": 2 if field_name == "issuing_country_code" else 10,
                "from": raw_code, "to": normalized_code,
                "reason": "MRZ country designator normalized to ISO alpha-3",
            })
    required = [value for key, value in result.checks.items() if key != "optional_data" and value is not None]
    result.valid = bool(required) and all(required)
    if not result.valid: result.warnings.append("MRZ_CHECKSUM_FAILURE")
    return result


def make_td3(document_number: str, nationality: str, birth_yymmdd: str, sex: str, expiry_yymmdd: str, surname: str, given_names: str, issuer: str = "UTO") -> list[str]:
    document_number = document_number.upper().ljust(9, "<")[:9]
    name = f"{surname.upper()}<<{given_names.upper().replace(' ', '<')}".ljust(39, "<")[:39]
    line1 = f"P<{issuer[:3].upper()}{name}"
    optional = "SYNTHETIC".ljust(14, "<")
    core = f"{document_number}{check_digit(document_number)}{nationality[:3].upper()}{birth_yymmdd}{check_digit(birth_yymmdd)}{sex[:1].upper()}{expiry_yymmdd}{check_digit(expiry_yymmdd)}{optional}{check_digit(optional)}"
    line2 = core + check_digit(core[0:10] + core[13:20] + core[21:43])
    return [line1, line2]
