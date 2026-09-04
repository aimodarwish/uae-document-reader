"""The thirteen Canadian licence issuers, and the shape each one's number has.

Canada issues no federal driving licence. Every card is a provincial or
territorial document, printed to that jurisdiction's own layout, and the only
thing they share is the numbered field scheme -- which three provinces do not
use either. Reading a Canadian licence therefore means knowing which of
thirteen documents is on the page, and that is what this module is for.

Two consequences drove the numbers recorded here rather than only the layouts.

The first is that a Canadian card need not label its licence number at all.
The Quebec licence prints "4d" against it in grey four-point type and nothing
else -- no "Permis n°", no "Licence No." -- so on a photograph of a real card
the recogniser returned that designator as the letters "BL" and the number
G3006-140404-00, read at 0.9997 confidence and plainly visible, bound to no
field and was reported as missing. A number no label can reach has to be
recognised by its shape.

The second is that shape alone is weak evidence, and the field it would fill is
the one the rental contract is keyed on. So where a jurisdiction builds the
holder's date of birth into the number -- Quebec and Ontario both do -- the
number is checked against the date of birth read off the same card. When the
two agree, a shape match stops being a guess: an eleven-digit run that happens
to encode exactly this holder's birthday is that holder's licence number.

Sources
-------
Field designators and which provinces use them:
    https://en.wikipedia.org/wiki/Driver%27s_licences_in_Canada
Number formats per jurisdiction:
    https://learn.microsoft.com/en-us/purview/sit-defn-canada-drivers-license-number
Quebec's internal structure (surname letter, consonant code, given-name digit,
JJMMAA birth date, two trailing digits):
    https://github.com/LexMajor/permisdeconduireqc
Card design standard the provinces build to:
    https://www.aamva.org/assets/best-practices%2C-guides%2C-standards%2C-manuals%2C-whitepapers/aamva-dl-id-card-design-standard-%282020%29
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date


@dataclass(frozen=True)
class ProvinceProfile:
    """One Canadian jurisdiction's licence document.

    ``number_pattern`` matches the number as it is printed, hyphens included.
    ``numbered_fields`` records whether the card labels its rows with the
    designators 1, 2, 3, 4a, 4b, 4d, 5; the three provinces that do not print
    them label their rows in words instead, and are read by the labelled pass.
    ``licence_number_designator`` is which designator carries the number on the
    cards that do print them. It is 4d throughout, which is the AAMVA standard
    the Canadian cards are built to: 4d is the customer identifier and 5 is the
    document discriminator, a serial for the piece of plastic that changes with
    every replacement card.

    Wikipedia's summary of the Canadian layout states the opposite -- 5 for the
    licence number, 4d for "a different number for administrative purposes" --
    which is the European model, not AAMVA. Both specimens this reader has been
    shown contradict it: the Quebec card prints "4d G3006-140404-00" against a
    "N° de référence", and the Ontario card "4d NUMBER/NUMÉRO A1059-35419-80608"
    directly above "5 DD/RÉF IS0831624". Where a document and a summary of it
    disagree, the document decides.
    """

    name: str
    code: str
    markers: tuple[str, ...]
    number_pattern: re.Pattern[str] | None
    numbered_fields: bool
    licence_number_designator: str
    # Set where the number encodes the holder's birth date, and readable by
    # ``birth_date_in_number`` below. That check is what lets a number be bound
    # from its shape alone.
    encodes_birth_date: bool = False
    # Set where the shape is particular enough that a run of characters on a
    # card matching it is the licence number rather than a coincidence.
    #
    # It is false for exactly the jurisdictions whose number is a bare run of
    # digits: British Columbia's seven, Saskatchewan's eight, New Brunswick's
    # five to seven, Prince Edward Island's five or six. A postal code, a
    # telephone number, a street number and a height all produce runs that
    # long, so binding on shape there would be picking one number off the card
    # and calling it the licence. Those four are read from their printed
    # labels instead -- which is how the British Columbia card, whose number is
    # headed "DL:", has been read since that card was worked on.
    bindable_by_shape: bool = False


# ``markers`` holds only wordings that name an issuer and nothing else: an
# authority ("ServiceOntario", "SAAQ", "ICBC") or a province name with no twin
# anywhere else. The bare names that do have twins are in WEAK_MARKERS below.
PROVINCE_PROFILES: tuple[ProvinceProfile, ...] = (
    ProvinceProfile(
        "Quebec", "QC",
        ("QUÉBEC", "QUEBEC", "SAAQ", "SOCIÉTÉ DE L'ASSURANCE AUTOMOBILE",
         "SOCIETE DE L'ASSURANCE AUTOMOBILE", "SAAQ.GOUV.QC.CA"),
        # One letter and twelve digits, hyphenated 1+4 / 6 / 2 on the card.
        re.compile(r"(?<![A-Z0-9])[A-Z]\d{4}-?\d{6}-?\d{2}(?![A-Z0-9])", re.I),
        numbered_fields=True, licence_number_designator="4D",
        encodes_birth_date=True,
        bindable_by_shape=True,
    ),
    ProvinceProfile(
        "Ontario", "ON",
        ("SERVICEONTARIO", "SERVICE ONTARIO", "ONTARIO.CA",
         "MINISTRY OF TRANSPORTATION"),
        # One letter and fourteen digits, hyphenated 1+4 / 5 / 5.
        re.compile(r"(?<![A-Z0-9])[A-Z]\d{4}-?\d{5}-?\d{5}(?![A-Z0-9])", re.I),
        numbered_fields=True, licence_number_designator="4D",
        encodes_birth_date=True,
        bindable_by_shape=True,
    ),
    ProvinceProfile(
        "British Columbia", "BC",
        ("BRITISH COLUMBIA", "COLOMBIE-BRITANNIQUE", "ICBC"),
        re.compile(r"(?<![A-Z0-9])\d{7}(?![A-Z0-9])", re.I),
        numbered_fields=False, licence_number_designator="",
    ),
    ProvinceProfile(
        "Alberta", "AB",
        ("ALBERTA",),
        # Alberta issues both a hyphenated 6-3 number and a plain run of five
        # to nine digits. Only the hyphenated form is recorded, because only it
        # is particular enough to recognise on sight; the plain form is a run
        # of digits like any other on the card and is read from the "Licence
        # No." the Alberta card prints beside it.
        re.compile(r"(?<![A-Z0-9])\d{6}-\d{3}(?![A-Z0-9])", re.I),
        numbered_fields=False, licence_number_designator="",
        bindable_by_shape=True,
    ),
    ProvinceProfile(
        "Saskatchewan", "SK",
        ("SASKATCHEWAN", "SGI"),
        re.compile(r"(?<![A-Z0-9])\d{8}(?![A-Z0-9])", re.I),
        numbered_fields=False, licence_number_designator="",
    ),
    ProvinceProfile(
        "Manitoba", "MB",
        ("MANITOBA", "MPI", "MANITOBA PUBLIC INSURANCE"),
        re.compile(
            r"(?<![A-Z0-9])[A-Z]{2}-?[A-Z]{2}-?[A-Z]{2}-?[A-Z]\d{3}[A-Z]{2}"
            r"(?![A-Z0-9])", re.I,
        ),
        numbered_fields=True, licence_number_designator="4D",
        bindable_by_shape=True,
    ),
    ProvinceProfile(
        "Nova Scotia", "NS",
        ("NOVA SCOTIA", "NOUVELLE-ÉCOSSE", "NOUVELLE-ECOSSE"),
        re.compile(r"(?<![A-Z0-9])[A-Z]{5}-?[0-3]\d[01]\d{6}(?![A-Z0-9])", re.I),
        numbered_fields=True, licence_number_designator="4D",
        bindable_by_shape=True,
    ),
    ProvinceProfile(
        "New Brunswick", "NB",
        ("SERVICE NEW BRUNSWICK", "SERVICE NOUVEAU-BRUNSWICK",
         "NOUVEAU-BRUNSWICK"),
        re.compile(r"(?<![A-Z0-9])\d{5,7}(?![A-Z0-9])", re.I),
        numbered_fields=True, licence_number_designator="4D",
    ),
    ProvinceProfile(
        "Newfoundland and Labrador", "NL",
        ("NEWFOUNDLAND", "LABRADOR", "TERRE-NEUVE"),
        re.compile(r"(?<![A-Z0-9])[A-Z]\d{9}(?![A-Z0-9])", re.I),
        numbered_fields=True, licence_number_designator="4D",
        bindable_by_shape=True,
    ),
    ProvinceProfile(
        "Prince Edward Island", "PE",
        ("PRINCE EDWARD ISLAND", "ÎLE-DU-PRINCE-ÉDOUARD",
         "ILE-DU-PRINCE-EDOUARD"),
        re.compile(r"(?<![A-Z0-9])\d{5,6}(?![A-Z0-9])", re.I),
        numbered_fields=True, licence_number_designator="4D",
    ),
    ProvinceProfile(
        "Yukon", "YT",
        ("YUKON",),
        None,
        numbered_fields=True, licence_number_designator="4D",
    ),
    ProvinceProfile(
        "Northwest Territories", "NT",
        ("NORTHWEST TERRITORIES", "TERRITOIRES DU NORD-OUEST"),
        None,
        numbered_fields=True, licence_number_designator="4D",
    ),
    ProvinceProfile(
        "Nunavut", "NU",
        ("NUNAVUT",),
        None,
        numbered_fields=True, licence_number_designator="4D",
    ),
)


# Province names that also name somewhere else. There is an Ontario in
# California and a New Brunswick in New Jersey, and the country table leaves
# both out for that reason: an address line naming one would hand an American
# licence to Canada, and a wrong country silently selects the wrong acceptance
# rule.
#
# Deciding which Canadian province issued a card is a narrower question, asked
# only once the bundle is already established as Canadian, so the twin cities
# are not in play. It still cannot be answered by these alone: the Ontario card
# reads "Ontario" at the top and "OAKVILLE, ON" in the address, and a card from
# a holder who has moved would read both its own province and another. So a
# weak marker decides only when it is the only province the page names, or when
# the licence number printed on the card is shaped like that province's.
WEAK_MARKERS: dict[str, tuple[str, ...]] = {
    "ON": ("ONTARIO",),
    "NB": ("NEW BRUNSWICK",),
}


_BY_CODE = {profile.code: profile for profile in PROVINCE_PROFILES}
_BY_NAME = {profile.name.upper(): profile for profile in PROVINCE_PROFILES}


def _fold(value: str) -> str:
    return " ".join(value.upper().split())


_MARKER_PATTERNS: tuple[tuple[re.Pattern[str], ProvinceProfile], ...] = tuple(
    (re.compile(rf"(?<![A-Z0-9]){re.escape(_fold(marker))}(?![A-Z0-9])"), profile)
    for profile in PROVINCE_PROFILES
    for marker in profile.markers
)


# What a page must carry before a bare province name is allowed to name its
# issuer: the country, its ISO code, or a Canadian postal code -- "L6K 3K7" on
# the Ontario card. A five-digit American ZIP matches none of these.
_CANADIAN_CORROBORATION = re.compile(
    r"(?<![A-Z0-9])(?:CANADA|CAN|[A-Z]\d[A-Z]\s?\d[A-Z]\d)(?![A-Z0-9])",
)


_WEAK_PATTERNS: tuple[tuple[re.Pattern[str], ProvinceProfile], ...] = tuple(
    (re.compile(rf"(?<![A-Z0-9]){re.escape(_fold(marker))}(?![A-Z0-9])"),
     _BY_CODE[code])
    for code, markers in WEAK_MARKERS.items()
    for marker in markers
)


def province_from_text(text: str) -> ProvinceProfile | None:
    """The jurisdiction a Canadian card names as its issuer.

    An authority name settles it outright, longest first so that "SERVICE NEW
    BRUNSWICK" is not decided by a shorter marker sitting inside it. Only when
    the page names no authority does a bare province name get a say, and then
    only if it is the single province named or its own number shape is printed
    on the card as well.

    That second test is what reads a front-only Ontario upload. The card heads
    itself "Ontario" and names no authority -- "ServiceOntario.ca" is printed
    on the reverse -- so on a photograph of the front alone the issuer came
    back as the bare country, "CANADA", and no province profile was selected to
    read the number with.
    """
    folded = _fold(text)
    best: tuple[int, ProvinceProfile] | None = None
    for pattern, profile in _MARKER_PATTERNS:
        match = pattern.search(folded)
        if match is None:
            continue
        length = len(match.group(0))
        if best is None or length > best[0]:
            best = (length, profile)
    if best is not None:
        return best[1]
    if not _CANADIAN_CORROBORATION.search(folded):
        # A weak marker is a place name, and the places it names are not all in
        # Canada. The page has to say it is a Canadian document by some other
        # means first -- the word itself, the ISO code, or a Canadian postal
        # code, all three of which a real card prints. Without this, the
        # American licence reading "CALIFORNIA USA ... ONTARIO CA 91761" names
        # an Ontario province, and the caller's country gate would be the only
        # thing between that and a Canadian acceptance rule.
        return None
    named = {
        profile.code: profile
        for pattern, profile in _WEAK_PATTERNS if pattern.search(folded)
    }
    if not named:
        return None
    if len(named) == 1:
        return next(iter(named.values()))
    shaped = [
        profile for profile in named.values()
        if profile.number_pattern is not None
        and profile.number_pattern.search(compact_identifiers(folded))
    ]
    return shaped[0] if len(shaped) == 1 else None


def compact_identifiers(text: str) -> str:
    """Close the gaps a recogniser leaves inside a printed serial.

    The Ontario card sets its number with air around the hyphens -- OCR returns
    "A1059 - 35419 - 80608" -- and a pattern written for the number as the
    standard defines it matches none of that. The same card prints the number a
    second time in smaller type without the spaces, which is the only reason
    the first Ontario upload read at all; a card without that second printing
    would have given up nothing.
    """
    closed = re.sub(r"\s*-\s*", "-", text)
    return re.sub(r"(?<=[A-Z0-9])\s+(?=[A-Z0-9])", "", closed)


def province_for(value: str | None) -> ProvinceProfile | None:
    """Look a jurisdiction up by its name or its two-letter code."""
    if not value:
        return None
    cleaned = _fold(value)
    return _BY_NAME.get(cleaned) or _BY_CODE.get(cleaned)


def _quebec_birth_date(digits: str) -> date | None:
    """The JJMMAA the Quebec number carries in positions 6 to 11.

    G3006-140404-00 for a holder born on 14 April 2004: G, the consonant code
    300 for GOYETTE, 6 for the given name MEDERIK, then 140404, then 00.
    """
    day, month, year = int(digits[4:6]), int(digits[6:8]), int(digits[8:10])
    return _resolve(year, month, day)


def _ontario_birth_date(digits: str) -> date | None:
    """The birth date the Ontario number carries in its last six digits.

    Written YYMMDD, with 50 added to the month on a female holder's card, so
    both readings are tried and either one matching is a match.
    """
    year, month, day = int(digits[8:10]), int(digits[10:12]), int(digits[12:14])
    return _resolve(year, month - 50 if month > 50 else month, day)


def _resolve(year: int, month: int, day: int) -> date | None:
    """A two-digit year as a real date, reading it as the recent past.

    A licence holder is alive and old enough to drive, so a two-digit year is
    this century where that lands in the past and the last one otherwise.
    """
    if not 1 <= month <= 12 or not 1 <= day <= 31:
        return None
    today = date.today()
    for century in (2000, 1900):
        try:
            candidate = date(century + year, month, day)
        except ValueError:
            continue
        if candidate <= today:
            return candidate
    return None


_BIRTH_DATE_READERS = {
    "QC": (12, _quebec_birth_date),
    "ON": (14, _ontario_birth_date),
}


def birth_date_in_number(profile: ProvinceProfile, number: str) -> str | None:
    """The holder's birth date as the licence number itself states it.

    Returns None where this jurisdiction does not build one in, or where the
    digits do not spell a real date -- so a caller can only ever use this to
    confirm a number, never to invent a date the card does not carry.
    """
    reader = _BIRTH_DATE_READERS.get(profile.code)
    if reader is None:
        return None
    expected_digits, parse = reader
    digits = re.sub(r"\D", "", number)
    if len(digits) != expected_digits:
        return None
    parsed = parse(digits)
    return parsed.isoformat() if parsed else None
