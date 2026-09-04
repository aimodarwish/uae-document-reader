from __future__ import annotations

import math
import re
import unicodedata
from collections.abc import Iterable
from collections.abc import Callable
from dataclasses import dataclass, replace
from difflib import SequenceMatcher
from functools import lru_cache
from datetime import date, timedelta
from typing import Any

from .barcode import BarcodeCandidate
from .canada_profiles import (
    birth_date_in_number, compact_identifiers, province_for, province_from_text,
)
from .country_documents import LICENCE_TITLES
from .language_labels import LANGUAGE_FIELD_LABELS
from .gcc_profiles import (
    ascii_numerals, gcc_labels_for_path, identity_issue_date_printed,
    normalize_gcc_number, profile_for_gcc_country,
)
from .classify import page_licence_title_present
from .licence_profiles import COMMON_NATIONAL_LABELS
from .tourist_detect import us_states_named
from .mrz import CONFUSIONS, ParsedMRZ, normalize_mrz_line, validate_check
from .normalize import (
    ARABIC_MONTH_DATE_PATTERN, MINIMUM_DRIVING_AGE_YEARS,
    close_bilingual_month_gap, fold_for_match,
    latinize_lookalikes,
    close_split_year, implausible_birth_date, month_and_year, normalize_country,
    normalize_date,
    nationality_country, normalize_emirates_id,
)
from .ocr import OCRLine
from .schemas import DocumentType, FieldCandidate


# Two readings of the same printed date differ by the punctuation a recogniser
# hallucinates between the glyphs, so the separators are written as a class
# rather than as a space. A Queensland card returned "DOB 15.Oct 2004": the
# month-name form accepted a space or a hyphen before the month and not the dot
# the recogniser put there, so the holder's birth date matched nothing at all.
_DATE_SEPARATOR = r"[ .\-]"
# The two halves of a bilingual month are not both Latin. A Ukrainian passport
# prints "24 ЧЕР/JUN 22" and an Armenian one its own script beside the English,
# and a Latin-only token class matched neither -- so the one date a passport's
# zone never carries was reported missing from a page that prints it plainly.
# Only one half has to be a month this reader knows; the other is carried along.
_DATE_MONTH_TOKEN = r"[A-ZÀ-ÖØ-Ý0-9\u0370-\u03FF\u0400-\u04FF\u0530-\u058F]"
DATE_PATTERN = (
    # A colon separates the parts of a numeric date on some biodata pages:
    # the Zimbabwean passport in this project's bug report prints
    # "17:02:2020" against its issue label and "16:02 2030" against expiry.
    # No numeric form here accepted that mark, so the one date a
    # machine-readable zone never carries was reported missing from a page
    # that prints it plainly. Admitted only where a four-figure year closes
    # the run, which is what keeps a time of day ("14:30", "08:30:15") from
    # reading as a date.
    r"(?:\d{1,2}[-/.:]\d{1,2}[-/.:]\d{4}"
    r"|\d{2} \d{2} \d{4}"
    # One of the two marks a numeric date prints can be lost -- thinned to
    # nothing, or taken for the gap around it -- and a French passport's issue
    # row arrived as "05.09 2022". The figures are all there and in order, and
    # the surviving mark still says the row is one date, so it is read as one;
    # normalization closes the run onto that mark. The all-space form stays
    # restricted to two-figure day and month above, so three loose short
    # numbers on a line are no more a date than they were.
    r"|\d{1,2}[-/.:]\d{1,2} \d{4}"
    r"|\d{1,2} \d{1,2}[-/.:]\d{4}"
    # A bilingual month row prints the name twice around a slash, and the
    # spacing either side of it is whatever the card's typesetter chose: a
    # Canadian passport prints "03 JAN /JAN  23", with a space before the
    # slash and a two-digit year, and matched nothing at all.
    # A zero in place of the O in OCT is a common OCR result; date
    # normalization still requires the repaired token to be an exact month.
    r"|\d{1,2} ?" + _DATE_MONTH_TOKEN + r"{3,9}\s?/\s?"
    + _DATE_MONTH_TOKEN + r"{3,9}\s{0,3}\d{2,4}"
    # Year first with the month named, which is how Canada writes a date:
    # "2023-Mar-03" on a British Columbia licence, "1989-Sep-02" for a birth
    # date. Every branch above expects either a numeric month or the day in
    # front, so this form was not a date as far as this reader was concerned.
    r"|\d{4}[-/. ](?:JAN(?:UARY)?|FEB(?:RUARY)?|MAR(?:CH)?|APR(?:IL)?|MAY"
    r"|JUN(?:E)?|JUL(?:Y)?|AUG(?:UST)?|SEP(?:T(?:EMBER)?)?|OCT(?:OBER)?"
    r"|NOV(?:EMBER)?|DEC(?:EMBER)?)[-/. ]\d{1,2}(?!\d)"
    r"|\d{4}[-/.]\d{1,2}[-/.]\d{1,2}"
    # Association cards occasionally use the North-American month-first form
    # ``JAN 20, 2019``.  Treat it as a date, never a permissive IDP number.
    r"|(?:JAN(?:UARY)?|FEB(?:RUARY)?|MAR(?:CH)?|APR(?:IL)?|MAY|JUN(?:E)?|"
    r"JUL(?:Y)?|AUG(?:UST)?|SEP(?:T(?:EMBER)?)?|OCT(?:OBER)?|NOV(?:EMBER)?|"
    r"DEC(?:EMBER)?)[ .\-]+\d{1,2}(?:,\s*|[ .\-]+)\d{2,4}"
    r"|\d{1,2}" + _DATE_SEPARATOR + r"?(?:JAN(?:UARY)?|FEB(?:RUARY)?|MAR(?:CH)?"
    r"|APR(?:IL)?|MAY|JUN(?:E)?|JUL(?:Y)?|AUG(?:UST)?|SEP(?:T(?:EMBER)?)?"
    r"|OCT(?:OBER)?|NOV(?:EMBER)?|DEC(?:EMBER)?)" + _DATE_SEPARATOR + r"?\d{4}"
    # Compact passport VIZ rows use a two-digit year even when the month is
    # spelled out: Greek ``10 Apr 23``, beside ``Iss. date``.  Numeric dates
    # and bilingual named-month dates already accepted two digits; the single
    # named-month form was the inconsistent exception.
    r"|\d{1,2}" + _DATE_SEPARATOR + r"?(?:JAN(?:UARY)?|FEB(?:RUARY)?|MAR(?:CH)?"
    r"|APR(?:IL)?|MAY|JUN(?:E)?|JUL(?:Y)?|AUG(?:UST)?|SEP(?:T(?:EMBER)?)?"
    r"|OCT(?:OBER)?|NOV(?:EMBER)?|DEC(?:EMBER)?)" + _DATE_SEPARATOR + r"?\d{2}(?!\d)"
    # A two-figure year is the end of the date, so nothing dated may follow
    # it. Without that, the first three parts of a longer dotted run were read
    # as a whole date and the real one behind them was never seen: an Israeli
    # licence prints "4a. 07.03.2021" and returned "4.07.03.2021", of which
    # "4.07.03" matched here as the fourth of July 2003. The row bound to no
    # field, and the date the licence was issued on was reported missing from
    # a card that prints it plainly.
    r"|\d{1,2}[-/.]\d{1,2}[-/.]\d{2}(?!\d)(?![-/.]\d))"
)


# Ink stamps on older permit booklets often separate the two digits of the day
# and put punctuation on both sides of the abbreviated month.  Algerian IDPs
# in particular read ``2 8. SEP. 2025`` or ``2.8.SEP..2025`` after OCR.  That
# is unambiguous once a named month follows it, but it must not loosen the
# generic date scanner used on arbitrary document text.
_IDP_STAMPED_DATE = re.compile(
    r"(?<!\d)(?P<day>\d\s*[.]\s*\d|\d\s+\d|\d{1,2})"
    r"[\s./-]*(?P<month>JAN(?:UARY)?|FEB(?:RUARY)?|MAR(?:CH)?|APR(?:IL)?|MAY|"
    r"JUN(?:E)?|JUL(?:Y)?|AUG(?:UST)?|SEP(?:T(?:EMBER)?)?|OCT(?:OBER)?|"
    r"NOV(?:EMBER)?|DEC(?:EMBER)?)[\s./-]+(?P<year>\d{4})(?!\d)",
    re.I,
)


# A booklet's issue date is filled in by hand, and a hand-drawn separator is
# not a glyph any recogniser has a class for. The Constantine permit in this
# project's bug report carries "26|11|2023" in ballpoint; Document AI returned
# "26/1112023", reading one stroke as a slash and the other as a one. Every
# digit is present and in order, which is what makes this a reading rather
# than a guess: a day, a month and a 19xx/20xx year have to fall out of the
# run exactly, and the result still has to be a real calendar date.
#
# Kept out of the generic date scanner: accepting a stray "1" as a separator
# is safe beside a permit's hand-filled delivery row and nowhere else, where a
# ten-digit serial would start looking like a date.
_HANDWRITTEN_DATE_RUN = re.compile(
    r"(?<![0-9A-Za-z])[0-9/.\-|lI]{8,10}(?![0-9A-Za-z])",
)
# What a hand-drawn stroke comes back as. ``1`` is the whole reason this exists
# and also the whole danger in it, which is why a reading is only accepted when
# every way of splitting the run agrees on the same day, month and year.
_HANDWRITTEN_SEPARATORS = frozenset("/.-|lI1")


def _handwritten_readings(run: str) -> set[str]:
    """Every calendar date this run could be, separators read or missed."""
    values: set[str] = set()
    for day_length in (1, 2):
        for first in (0, 1):
            for month_length in (1, 2):
                for second in (0, 1):
                    if day_length + first + month_length + second + 4 != len(run):
                        continue
                    at = 0
                    day, at = run[at:at + day_length], at + day_length
                    separator, at = run[at:at + first], at + first
                    month, at = run[at:at + month_length], at + month_length
                    separator += run[at:at + second]
                    year = run[at + second:]
                    if not (day.isdigit() and month.isdigit() and year.isdigit()):
                        continue
                    if any(glyph not in _HANDWRITTEN_SEPARATORS for glyph in separator):
                        continue
                    if not year.startswith(("19", "20")):
                        continue
                    normalized = normalize_date(
                        f"{day}.{month}.{year}", day_first_hint=True,
                    )
                    if normalized.value:
                        values.add(normalized.value)
    return values


def handwritten_date_values(text: str) -> list[str]:
    """The calendar dates a hand-filled row spells, once its strokes are read.

    A run is claimed only where it can be read one way. "1411.2003" is a
    fourteenth of November with a separator missed and a fourteenth of January
    with one read as a one, and no evidence on the page says which -- so it is
    neither. "26/1112023", the Constantine permit's delivery row, is the
    twenty-sixth of November 2023 and nothing else, because no other split of
    those characters leaves a day, a month and a 19xx/20xx year behind.
    """
    values: list[str] = []
    for match in _HANDWRITTEN_DATE_RUN.finditer(text):
        readings = _handwritten_readings(match.group())
        if len(readings) == 1:
            values.append(next(iter(readings)))
    return values


# Two dates printed in one cell come back with the space between them lost:
# "31.05.25 31.05.27" arrives as "31.05.2531.05.27", where the greedy
# four-digit year reads 2531 and the second date disappears entirely. A
# sixteen-character run of two day.month.year groups is not one date with a
# thirty-first-century year, and separating them is what lets the validity
# range fill both fields it was printed for.
# A separator after the pair means neither half ended where this reading says.
# "29/09/2022-28/09/2027" is one validity range with four-digit years, and read
# as two two-digit-year dates it became "29/09/20 22-28/09/2027" -- a licence
# issued in the year 20.
_RUN_TOGETHER_DATES = re.compile(
    r"(?<!\d)(\d{1,2}[-/.]\d{1,2}[-/.]\d{2})(\d{1,2}[-/.]\d{1,2}[-/.]\d{2})"
    r"(?![-/.]?\d)"
)


# The same loss between a number and the date printed after it. A Michigan
# licence sets its number and its issue date on one row, and the row came back
# as "S 300 772 60902504/12/2024": the year-first branch of the date scanner
# then read "2504/12/20" out of the join, which is not a date any card printed,
# and the issue date the card states plainly was never seen.
# A validity range is not a number running into a date. "29/09/2022-28/09/2027"
# offered "22-28/09/2027" to the lookahead, and the split left the range
# starting at "29/09/20" -- a two-digit year read as the whole of it. Where the
# digits before the split already complete a date, nothing ran together.
_NUMBER_RUN_INTO_DATE = re.compile(
    r"(?<=\d)(?<!\d[-/.]\d\d)(?<!\d\d[-/.]\d\d)"
    r"(?=\d{2}[-/.]\d{2}[-/.]\d{4}(?!\d))"
)


def split_run_together_dates(text: str) -> str:
    """Restore the separators a recogniser lost inside a row of dates."""
    return close_bilingual_month_gap(
        _NUMBER_RUN_INTO_DATE.sub(" ", _RUN_TOGETHER_DATES.sub(r"\1 \2", text))
    )


# Arabic name particles bind to the family name that follows them, so the
# surname on "OMAR AHMED ABDUL RASHEED AL BALUSHI" is "AL BALUSHI". A particle
# is never a surname by itself: a name line ending in one has been truncated by
# a crop or a glare band, and deriving "AL" from it produces a bogus surname
# that then conflicts with the correct value read off the other page.
NAME_PARTICLES = frozenset({
    "AL", "EL", "BIN", "BEN", "IBN", "BINT", "ABU", "ABO", "AAL",
})


def _family_name(parts: list[str]) -> tuple[str, int]:
    """Return the surname in `parts` and how many trailing tokens it spans."""
    if len(parts) < 2:
        return "", 0
    if parts[-1].strip("-").upper() in NAME_PARTICLES:
        return "", 0
    if len(parts) >= 3 and parts[-2].strip("-").upper() in NAME_PARTICLES:
        return f"{parts[-2]} {parts[-1]}", 2
    return parts[-1], 1


FIELD_LABELS = {
    "personal_info.full_name": (
        "NAME", "FULL NAME", "NAME OF HOLDER", "ФИО",
        "ФАМИЛИЯ ИМЯ ОТЧЕСТВО", "ИМЯ ВЛАДЕЛЬЦА",
    ),
    "personal_info.first_name": ("GIVEN NAMES", "GIVEN NAME", "ИМЯ", "ИМЕНА"),
    "personal_info.middle_name": ("MIDDLE NAME", "PATRONYMIC", "ОТЧЕСТВО"),
    "personal_info.last_name": ("SURNAME", "ФАМИЛИЯ"),
    "personal_info.date_of_birth": ("DATE OF BIRTH", "BIRTH DATE", "DOB", "ДАТА РОЖДЕНИЯ"),
    "personal_info.nationality_name": ("NATIONALITY", "ГРАЖДАНСТВО"),
    "personal_info.gender": ("SEX", "GENDER", "ПОЛ"),
    # A bilingual passport prints this row twice, and either half can be the
    # one OCR reads cleanly: on the Albanian page "vendlindja" came through
    # exactly while its English twin came back as "place of birch", so the
    # field was reported missing even though the city was on the page and
    # legible. Listing the local wording alongside the English costs nothing --
    # a label a document never prints simply never matches.
    "personal_info.place_of_birth": (
        "PLACE OF BIRTH", "BIRTH PLACE", "مكان الميلاد", "محل الميلاد",
        "VENDLINDJA", "LIEU DE NAISSANCE", "LUGAR DE NACIMIENTO",
        "LOCAL DE NASCIMENTO", "LUOGO DI NASCITA", "GEBURTSORT",
        "GEBOORTEPLAATS", "GEBUERTSUERT", "МЕСТО РОЖДЕНИЯ", "МІСЦЕ НАРОДЖЕННЯ",
        "МЯСТО НА РАЖДАНЕ", "МЕСТО РОЂЕЊА", "MJESTO ROĐENJA", "MÍSTO NAROZENÍ",
        "MIESTO NARODENIA", "MIEJSCE URODZENIA", "SZÜLETÉSI HELY",
        "LOCUL NAȘTERII", "DOĞUM YERI", "DZIMŠANAS VIETA", "GIMIMO VIETA",
        "SÜNNIKOHT", "SYNNYINPAIKKA", "FÖDELSEORT", "FØDESTED", "FÆÐINGARSTAÐUR",
        "ΤΟΠΟΣ ΓΕΝΝΗΣΗΣ", "TEMPAT LAHIR", "NƠI SINH", "สถานที่เกิด",
        "מקום לידה", "出生地", "출생지",
    ),
    "emirates_id.number": ("ID NUMBER", "IDENTITY NUMBER", "ID NO"),
    "emirates_id.issue_date": (
        "ISSUE DATE", "DATE OF ISSUE", "ISSUING DATE", "تاريخ الإصدار", "تاريخ إصدار البطاقة",
    ),
    "emirates_id.expiry_date": ("EXPIRY DATE", "DATE OF EXPIRY"),
    "passport.number": ("PASSPORT NO", "PASSPORT NUMBER", "DOCUMENT NO", "НОМЕР ПАСПОРТА"),
    "passport.issued_by_code": (
        "ISSUING COUNTRY", "COUNTRY CODE", "PAYS ÉMETTEUR", "PAYS EMETTEUR",
        "PEYI KI FÈ LI", "PEYI KI FE LI",
    ),
    "passport.issue_date": ("DATE OF ISSUE", "ISSUE DATE", "ДАТА ВЫДАЧИ"),
    "passport.expiry_date": ("DATE OF EXPIRY", "EXPIRY DATE", "ДАТА ОКОНЧАНИЯ СРОКА"),
    "uae_driving_licence.number": (
        "LICENCE NO", "LICENSE NO", "LICENCE NUMBER", "LICENSE NUMBER",
        # PP-OCR can confuse the narrow C in LICENCE with O on the coloured
        # UAE card. Keep this tightly scoped to the licence-number label.
        "LIOENSE NO",
        "رقم الرخصة", "رقم رخصة القيادة",
    ),
    "uae_driving_licence.issued_by_name": (
        "ISSUED BY", "PLACE OF ISSUE", "LICENSING AUTHORITY", "ISSUING AUTHORITY",
        "جهة الإصدار", "مكان الإصدار",
    ),
    "uae_driving_licence.issue_date": (
        "ISSUE DATE", "DATE OF ISSUE", "تاريخ الإصدار",
    ),
    "uae_driving_licence.expiry_date": (
        "EXPIRY DATE", "DATE OF EXPIRY", "تاريخ الانتهاء", "تاريخ انتهاء الرخصة",
        # The printed Y and DA can join into ``Explry Pete`` on low-contrast
        # licence captures; its adjacent value remains independently checked.
        "EXPLRY PETE",
    ),
    "gcc_identity.number": (
        "ID NUMBER", "ID NO", "NATIONAL ID", "CIVIL ID", "CIVIL NUMBER",
        "PERSONAL NUMBER", "CPR NUMBER", "رقم الهوية", "الرقم المدني", "الرقم الشخصي",
    ),
    "gcc_identity.issue_date": (
        "ISSUING DATE", "ISSUE DATE", "DATE OF ISSUE", "تاريخ الإصدار",
    ),
    "gcc_identity.expiry_date": (
        "EXPIRY DATE", "DATE OF EXPIRY", "VALID UNTIL", "تاريخ الانتهاء",
    ),
    "gcc_driving_licence.number": (
        "LICENCE NO", "LICENSE NO", "LICENCE NUMBER", "LICENSE NUMBER",
        "رقم الرخصة", "رقم رخصة القيادة",
    ),
    "gcc_driving_licence.issued_by_name": (
        "ISSUED BY", "PLACE OF ISSUE", "ISSUING AUTHORITY", "جهة الإصدار", "مكان الإصدار",
    ),
    "gcc_driving_licence.issue_date": (
        "ISSUE DATE", "DATE OF ISSUE", "FIRST ISSUE", "تاريخ الإصدار",
    ),
    "gcc_driving_licence.expiry_date": (
        "EXPIRY DATE", "DATE OF EXPIRY", "VALID UNTIL", "تاريخ الانتهاء",
    ),
    "international_driving_permit.number": (
        "PERMIT NO", "IDP NO", "PERMIT NUMBER", "NO. DU PERMIS",
        "НОМЕР МВУ", "НОМЕР УДОСТОВЕРЕНИЯ", "УДОСТОВЕРЕНИЕ №",
    ),
    "international_driving_permit.issued_by_name": (
        "ISSUED BY", "PLACE OF ISSUE", "ISSUING AUTHORITY", "КЕМ ВЫДАНО",
        "ОРГАН ВЫДАЧИ", "ОРГАН, ВЫДАВШИЙ УДОСТОВЕРЕНИЕ",
    ),
    "international_driving_permit.issue_date": ("DATE OF ISSUE", "ISSUE DATE", "ДАТА ВЫДАЧИ"),
    "international_driving_permit.expiry_date": ("VALID UNTIL", "EXPIRY DATE", "DATE OF EXPIRY", "ДЕЙСТВИТЕЛЬНО ДО"),
    "national_driving_licence.number": (
        "LICENCE NO", "LICENSE NO", "LICENCE #", "LICENSE #", "DL NO",
        # The ordinal is part of the printed label and survives compaction, so
        # the spelling the card actually uses has to be listed beside the plain
        # one. An Argentine card heads the row "5. N° Licencia / License N°";
        # against "LICENSE NO" alone it matched neither half, and the licence
        # number -- the field the rental is keyed on -- was read as nothing.
        "LICENSE N°", "LICENCE N°", "N° LICENCIA", "N LICENCIA",
        "NUMERO DE LICENCIA", "NÚMERO DE LICENCIA",
        # A South African card abbreviates the caption and prints it in both
        # official languages -- "Lic. No./Lisensienr.: 402800063077" -- and
        # against the spelt-out forms alone it matched neither half. The
        # Afrikaans word is listed first because it is the longer of the two
        # and so is tried first, which leaves the value and not the rest of
        # the caption after the match.
        "LISENSIENR", "LIC NO",
    ),
    "national_driving_licence.issued_by_name": ("ISSUED BY", "PLACE OF ISSUE", "AUTHORITY"),
    "national_driving_licence.issue_date": ("ISSUE DATE", "DATE OF ISSUE"),
    "national_driving_licence.expiry_date": ("EXPIRY DATE", "DATE OF EXPIRY"),
}


# What a European passport prints beside each row, in the issuing state's own
# language.
#
# ICAO 9303 standardises the machine-readable zone and nothing else: the printed
# rows above it are labelled in the state's language, and the English half that
# is supposed to sit beside them is small, grey, and the first thing a phone
# photo loses to glare. That is not an edge case -- it is how the French
# passport in this project's own bug reports failed. With only English and
# Russian listed, two European passports in thirty-four gave up their name and
# birth date once the zone was unreadable.
#
# Grouped by language rather than by country, because the wording follows the
# language: an Austrian and a German passport print the same rows, as do a
# French and a Luxembourgish one. Given-name rows are plural on a passport
# ("Prénoms", "Vornamen", "Imiona") where a licence prints the singular, which
# is why the generated licence table does not cover them.
PASSPORT_VIZ_LABELS: dict[str, tuple[str, ...]] = {
    "personal_info.last_name": (
        # Romance
        "NOM", "COGNOME", "APELLIDOS", "APELIDO", "APELIDOS", "NUME", "NUMELE",
        # Germanic
        "NAAM", "ACHTERNAAM", "EFTERNAMN", "EFTERNAVN", "ETTERNAVN", "SUKUNIMI",
        "EFTIRNAFN", "SLOINNE",
        # Slavic and Baltic
        "NAZWISKO", "PŘÍJMENÍ", "PRIEZVISKO", "PRIIMEK", "PREZIME", "ПРЕЗИМЕ",
        "ФАМИЛИЯ", "ПРІЗВИЩЕ", "UZVĀRDS", "PAVARDĖ", "PEREKONNANIMI",
        # Greek, Hungarian, Maltese, Albanian, Turkish
        "ΕΠΩΝΥΜΟ", "ΕΠΏΝΥΜΟ", "VEZETÉKNÉV", "CSALÁDI NÉV", "KUNJOM", "MBIEMRI",
        "SOYADI", "SOYADI/SURNAME",
    ),
    "personal_info.first_name": (
        # A passport names every given name, so the row is plural.
        "PRÉNOMS", "PRENOMS", "NON/PRÉNOM", "NON/PRENOM", "NOME", "NOMBRE", "NOMBRES", "NOME PRÓPRIO",
        "NOME PROPRIO", "PRENUME",
        "VORNAMEN", "VOORNAMEN", "FÖRNAMN", "FORNAVN", "ETUNIMET", "EIGINNAFN",
        "CÉADAINM",
        "IMIONA", "JMÉNO", "MENO", "IME", "ИМЕ", "ИМЯ", "ІМ'Я", "VĀRDS",
        "VARDAS", "EESNIMED",
        "ΟΝΟΜΑ", "ΌΝΟΜΑ", "UTÓNÉV", "ISEM", "EMRI", "ADI", "ADI/GIVEN NAMES",
    ),
    "personal_info.date_of_birth": (
        "DATE DE NAISSANCE", "DATA DI NASCITA", "FECHA DE NACIMIENTO",
        "DATA DE NASCIMENTO", "DATA NAȘTERII", "DATA NASTERII",
        "GEBURTSDATUM", "GEBOORTEDATUM", "FÖDELSEDATUM", "FØDSELSDATO",
        "SYNTYMÄAIKA", "FÆÐINGARDAGUR", "DÁTA BREITHE",
        "DATA URODZENIA", "DATUM NAROZENÍ", "DÁTUM NARODENIA", "DATUM ROJSTVA",
        "DATUM ROĐENJA", "ДАТУМ РОЂЕЊА", "ДАТА НА РАЖДАНЕ", "ДАТА НАРОДЖЕННЯ",
        "DZIMŠANAS DATUMS", "GIMIMO DATA", "SÜNNIAEG",
        "ΗΜΕΡΟΜΗΝΙΑ ΓΕΝΝΗΣΗΣ", "ΗΜΕΡΟΜΗΝΊΑ ΓΈΝΝΗΣΗΣ", "SZÜLETÉSI IDŐ",
        "DATA TAT-TWELID", "DATËLINDJA",
        # Turkish prints a dotted capital İ, and str.upper() of the printed
        # lowercase row produces a plain I. Both spellings are listed because
        # the label index compares them literally.
        "DOĞUM TARİHİ", "DOĞUM TARIHI",
    ),
    "personal_info.nationality_name": (
        "NATIONALITÉ", "NATIONALITE", "CITTADINANZA", "NACIONALIDAD",
        "NACIONALIDADE", "CETĂȚENIE", "CETATENIE", "CETĂȚENIA", "CETATENIA",
        "STAATSANGEHÖRIGKEIT", "NATIONALITEIT", "MEDBORGARSKAP",
        "STATSBORGERSKAB", "STATSBORGERSKAP", "KANSALAISUUS", "RÍKISFANG",
        "NÁISIÚNTACHT",
        "OBYWATELSTWO", "STÁTNÍ OBČANSTVÍ", "ŠTÁTNA PRÍSLUŠNOSŤ",
        "DRŽAVLJANSTVO", "ДРЖАВЉАНСТВО", "ГРАЖДАНСТВО", "ГРОМАДЯНСТВО",
        "PILSONĪBA", "PILIETYBĖ", "KODAKONDSUS",
        "ΙΘΑΓΕΝΕΙΑ", "ΙΘΑΓΈΝΕΙΑ", "ÁLLAMPOLGÁRSÁG", "ĊITTADINANZA",
        "SHTETËSIA", "UYRUĞU",
    ),
    "passport.issue_date": (
        # Greek passports print the compact bilingual English caption
        # ``Iss. date``.  The production OCR read both that caption and the
        # value ``10 Apr 23`` cleanly, but the long-only label table left the
        # date unbound.
        "ISS. DATE", "ISS DATE",
        "DATE D'ÉMISSION", "DATE D'EMISSION", "DATE D EMISSION",
        "DAT PASPÒ A FÈT", "DAT PASPO A FET",
        "DATE DE DÉLIVRANCE", "DATE DE DELIVRANCE", "DATA DI RILASCIO",
        "FECHA DE EXPEDICIÓN", "FECHA DE EXPEDICION", "DATA DE EMISSÃO",
        "DATA DE EMISSAO", "DATA ELIBERĂRII", "DATA ELIBERARII",
        "AUSSTELLUNGSDATUM", "DATUM VAN AFGIFTE", "AFGIFTEDATUM",
        "UTFÄRDANDEDATUM", "UDSTEDELSESDATO", "UTSTEDELSESDATO", "UTSTEDT",
        "MYÖNTÄMISPÄIVÄ", "ÚTGÁFUDAGUR", "DÁTA EISIÚNA",
        "DATA WYDANIA", "DATUM VYDÁNÍ", "DÁTUM VYDANIA", "DATUM IZDAJE",
        "DATUM IZDAVANJA", "ДАТУМ ИЗДАВАЊА", "ДАТА НА ИЗДАВАНЕ", "ДАТА ВИДАЧІ",
        "IZDOŠANAS DATUMS", "IŠDAVIMO DATA", "VÄLJAANDMISE KUUPÄEV",
        "ΗΜΕΡΟΜΗΝΙΑ ΕΚΔΟΣΗΣ", "ΗΜΕΡΟΜΗΝΊΑ ΈΚΔΟΣΗΣ", "KIÁLLÍTÁS DÁTUMA",
        "DATA TAL-ĦRUĠ", "DATA E LËSHIMIT", "VERİLİŞ TARİHİ", "VERILIŞ TARIHI",
    ),
    "passport.expiry_date": (
        "DATE D'EXPIRATION", "DATE D EXPIRATION", "DATA DI SCADENZA",
        "FECHA DE CADUCIDAD", "DATA DE VALIDADE", "VÁLIDO ATÉ",
        "DATA EXPIRĂRII", "DATA EXPIRARII",
        "GÜLTIG BIS", "GELDIG TOT", "GILTIGT TILL", "GILTIGT T.O.M",
        "UDLØBSDATO", "UTLØPSDATO", "VIIMEINEN VOIMASSAOLOPÄIVÄ",
        "GILDIR TIL", "AS BHAILÍ",
        "DATA WAŻNOŚCI", "DATA WAZNOSCI", "PLATNOST DO", "PLATNOSŤ DO",
        "VELJA DO", "VRIJEDI DO", "VAŽI DO", "ВАЖИ ДО", "ВАЛИДЕН ДО",
        "ДІЙСНИЙ ДО", "DERĪGA LĪDZ", "GALIOJA IKI", "KEHTIV KUNI",
        "ΗΜΕΡΟΜΗΝΙΑ ΛΗΞΗΣ", "ΗΜΕΡΟΜΗΝΊΑ ΛΉΞΗΣ", "ÉRVÉNYES", "VALIDA SA",
        "VLEN DERI", "GEÇERLİLİK TARİHİ", "GEÇERLILIK TARIHI",
    ),
    "passport.number": (
        "PASSEPORT N", "PASSAPORTO N", "PASAPORTE N", "PASSAPORTE N",
        "PASSNUMMER", "PASNUMMER", "PASPOORTNUMMER", "REISEPASS NR",
        "PASSIN NUMERO", "VEGABRÉFSNÚMER",
        "NR PASZPORTU", "ČÍSLO PASU", "ČÍSLO CESTOVNÉHO PASU", "ŠTEVILKA POTNEGA LISTA",
        "БРОЈ ПАСОША", "НОМЕР ПАСПОРТА", "НОМЕР ПАСПОРТА", "НОМЕР ПАСПОРТА",
        "PASES NR", "PASO NR", "PASSI NUMBER",
        "ΑΡΙΘΜΟΣ ΔΙΑΒΑΤΗΡΙΟΥ", "ΑΡΙΘ. ΔΙΑΒΑΤΗΡΙΟΥ", "ÚTLEVÉL SZÁMA",
        "NUMRU TAL-PASSAPORT", "NUMRI I PASAPORTËS", "PASAPORT NO",
    ),
}


# Labels that name the surname and nothing else. Their presence on a page means
# a neighbouring "name" row is given names, not a full name. Deliberately
# excludes bare "NOM": on a French card it labels the surname, but it is also a
# substring of "NOMBRE" and "NOME", which are given-name and full-name labels.
DEDICATED_SURNAME_LABELS = (
    "APELLIDO", "APELLIDOS", "SURNAME", "FAMILY NAME", "LAST NAME",
    "COGNOME", "NACHNAME", "SOYADI", "ФАМИЛИЯ", "ПРІЗВИЩЕ",
    # French "Nom" is the surname. It is safe to list now that the label
    # boundary counts accented letters, so it no longer matches inside "Prénom".
    "NOM",
)


# A passport names every given name, so its given-name row is plural. Where one
# of these is on the page, the bare "name" row above it is the surname.
PLURAL_GIVEN_NAME_LABELS = (
    "VORNAMEN", "VOORNAMEN", "PRÉNOMS", "PRENOMS", "GIVEN NAMES", "ETUNIMET",
    "EESNIMED", "IMIONA", "FÖRNAMN", "FORNAVN", "ΟΝΟΜΑΤΑ",
)


# The German, Austrian, Swiss and Dutch passports label the surname row with the
# bare word for "name" -- "Name / Surname" printed above "Vornamen / Given
# names". Read alone that row looks like a full name, which is what it is on
# most other documents, and the surname was being lost on exactly the three
# states whose licences this rental accepts outright. Only used where a plural
# given-name row is present to say which layout this is.
BARE_NAME_AS_SURNAME_LABELS = ("NAME", "NAAM", "NAVN", "NAMN")


def _bare_label_row(text: str) -> str:
    """The row's label content, with a leading designator and punctuation gone.

    "1. Name" and "Name:" are the bare word; "Name / Surname" is not.
    """
    return re.sub(r"^\s*\d+\s*[.):\-]*\s*", "", text).strip(" .:-").upper()


# Documents whose holder must have been old enough to drive when they were
# issued. A passport is deliberately excluded: children hold passports.
DRIVING_DOCUMENTS_REQUIRING_ADULT_HOLDER = frozenset({
    DocumentType.NATIONAL_DRIVING_LICENCE_FRONT,
    DocumentType.NATIONAL_DRIVING_LICENCE_BACK,
    DocumentType.INTERNATIONAL_DRIVING_PERMIT,
    DocumentType.UAE_DRIVING_LICENCE_FRONT,
    DocumentType.UAE_DRIVING_LICENCE_BACK,
    DocumentType.GCC_DRIVING_LICENCE_FRONT,
    DocumentType.GCC_DRIVING_LICENCE_BACK,
})


def _is_rtl(text: str) -> bool:
    """True when the text is written in a right-to-left script (Arabic here)."""
    return re.search(r"[؀-ۿݐ-ݿﭐ-﷿ﹰ-﻿]", text) is not None


def _same_language_pass(left: OCRLine, right: OCRLine) -> bool:
    # Paddle's language identifies independent recognition passes. Google
    # detects languages per line within ONE multilingual pass, so an Arabic
    # label and Latin digits can still be neighboring evidence from that pass.
    return left.language == right.language or (
        left.model_name.partition("+")[0]
        == right.model_name.partition("+")[0]
        == "Google-Document-AI-OCR"
    )


def _line_rect(line: OCRLine) -> tuple[float, float, float, float]:
    xs, ys = [p[0] for p in line.bounding_box], [p[1] for p in line.bounding_box]
    return min(xs), min(ys), max(xs), max(ys)


_BILINGUAL_DATE_WITHOUT_YEAR = re.compile(
    r"^\s*\d{1,2}\s+[A-Z0-9]{3,9}\s*/\s*[A-Z0-9]{3,9}\s*$", re.I,
)
_MONTH_HOMOGLYPHS = str.maketrans({"О": "O", "С": "C", "Т": "T"})

# A nationality the country tables cannot place is kept as printed, but only
# where the row is a bare word: one or two letters-only tokens, no caption
# riding along with them.
_BARE_NATIONALITY_WORD = re.compile(r"[^\W\d_]{3,20}(?: [^\W\d_]{2,20})?", re.UNICODE)


# A licence number is a Latin/figure identifier, but the letters in front of it
# are printed in whatever alphabet the card is set in, and a recogniser asked
# to read a Kazakh page returns them in that alphabet -- or in another one that
# draws the same shapes. This Kazakh front prints its series as Cyrillic "АН"
# and came back as Greek "ΑΝ", and because the number pattern below is written
# in Latin it matched only from the figures onward: a card numbered AN 294297
# was reported as 294297, missing the two characters the series is identified
# by. The letters below are the capitals Latin, Cyrillic and Greek draw
# identically, so each is replaced by the Latin letter of the same shape.
# Applied only where a licence number is being read. The holder identifier
# beside it is a figures-only national number introduced by a label in the
# card's own alphabet -- "4d) ЖСН/IIN ..." here -- and reading that label as
# Latin would pull it into the value, so it keeps the stricter reading.
_IDENTIFIER_HOMOGLYPHS = str.maketrans({
    # Cyrillic
    "А": "A", "В": "B", "Е": "E", "З": "3", "І": "I", "Ј": "J", "К": "K",
    "М": "M", "Н": "H", "О": "O", "Р": "P", "С": "C", "Ѕ": "S", "Т": "T",
    "У": "Y", "Х": "X",
    # Greek
    "Α": "A", "Β": "B", "Ε": "E", "Ζ": "Z", "Η": "H", "Ι": "I", "Κ": "K",
    "Μ": "M", "Ν": "N", "Ο": "O", "Ρ": "P", "Τ": "T", "Υ": "Y", "Χ": "X",
})


def _latin_identifier_glyphs(text: str) -> str:
    """Read an identifier's letters as the Latin capitals they are drawn as."""
    return text.translate(_IDENTIFIER_HOMOGLYPHS)



def _join_split_passport_dates(lines: list[OCRLine]) -> list[OCRLine]:
    """Add a date row when OCR split its trailing year into a second box.

    Canadian passports typeset ``31 OCT /OCT 23`` with a large visual gap
    before the year. Paddle detects that two-digit year as its own box. Neither
    fragment is a date by itself, so the labelled issue/expiry row otherwise
    looks empty even though both pieces were recognised with high confidence.

    The repair is geometry-bound, passport-only, and creates evidence only
    when the joined text is itself a valid normalizable date. It launches no
    additional recognition pass.
    """
    additions: list[OCRLine] = []
    for left in lines:
        # The Cyrillic recognizer often wins this otherwise-Latin row and
        # returns the lookalikes О/С/Т. Canonicalize only those three month
        # glyphs; applying broad script transliteration to names or numbers
        # would be unsafe.
        left_text = left.text.upper().translate(_MONTH_HOMOGLYPHS)
        if not _BILINGUAL_DATE_WITHOUT_YEAR.fullmatch(left_text):
            continue
        lx1, ly1, lx2, ly2 = _line_rect(left)
        left_height = max(ly2 - ly1, 1.0)
        for right in lines:
            if right is left or re.fullmatch(r"\s*\d{2,4}\s*", right.text) is None:
                continue
            if right.variant != left.variant or not _same_language_pass(left, right):
                continue
            rx1, ry1, rx2, ry2 = _line_rect(right)
            overlap = min(ly2, ry2) - max(ly1, ry1)
            right_height = max(ry2 - ry1, 1.0)
            if overlap < min(left_height, right_height) * 0.55:
                continue
            gap = rx1 - lx2
            if gap < -left_height * 0.25 or gap > max(left_height * 2.0, 180.0):
                continue
            joined = f"{left_text.strip()} {right.text.strip()}"
            if re.fullmatch(DATE_PATTERN, joined, re.I) is None:
                continue
            if normalize_date(joined, day_first_hint=True).value is None:
                continue
            additions.append(OCRLine(
                text=joined,
                confidence=min(left.confidence, right.confidence),
                bounding_box=[
                    [min(lx1, rx1), min(ly1, ry1)],
                    [max(lx2, rx2), min(ly1, ry1)],
                    [max(lx2, rx2), max(ly2, ry2)],
                    [min(lx1, rx1), max(ly2, ry2)],
                ],
                language=left.language,
                variant=left.variant,
                model_name=f"{left.model_name}+date-join",
            ))
            break
    return [*lines, *additions]


# Labels identifying numbers that must never be borrowed as a licence or ID
# value merely because OCR placed them in a nearby box. Built once: it is
# consulted for every label of every field path on every line, so rebuilding it
# per lookup dominated extraction time on dense bilingual cards.
_NON_HOLDER_LABELS = (
    "TRAFFIC NO", "TRAFFIC NUMBER", "CARD NO", "CARD NUMBER",
    "VERSION NO", "VERSION NUMBER", "رقم النسخة", "BARCODE", "QR CODE",
)


# Identity and licence values that must bind to their own label row rather
# than to the nearest plausible-looking box anywhere on the card.
_STRICT_NEIGHBOR_PATHS = frozenset({
    "emirates_id.issue_date", "emirates_id.expiry_date",
    "gcc_identity.number", "gcc_identity.issue_date", "gcc_identity.expiry_date",
    "gcc_driving_licence.number", "gcc_driving_licence.issue_date",
    "gcc_driving_licence.expiry_date",
    "personal_info.place_of_birth",
})

# On a GCC card the holder rows are stacked tightly and every one of them
# holds a date, so these must bind to their own row. A loose nearest-neighbour
# rule lets تاريخ الميلاد capture the expiry printed beneath it, which then
# competes with the checksummed birth date from the machine-readable zone and
# leaves the field empty.
_GCC_STRICT_PATHS = frozenset({
    "personal_info.date_of_birth", "personal_info.gender",
    "personal_info.nationality_name",
})

# Bilingual cards print an Arabic label beside a Latin or numeric value, so
# these lookups may cross recognizers.
#
# A line's "language" is not the document's language. It records which
# recognizer returned that box, and a page is shown to several of them, each
# winning different rows. Requiring a label and its value to carry the same one
# is therefore requiring a coincidence -- that the same model happened to read
# both rows -- and where it did not, the label skips its own value and binds
# the next row a single model did read.
#
# That is what a Canadian passport did. "Date of issue/Date de délivrance" was
# read by the Latin model and the date under it, "14 JUN/JUIN 2024", by the
# Cyrillic one; the issue label could not see it, reached past it, and took the
# value under the *expiry* label instead. The passport was reported as issued
# on 14 June 2034 -- its expiry date, ten years out, on a document whose two
# dates the recogniser had read correctly at 0.985 and 0.991.
#
# This is the same argument already recorded below for image variants, and it
# covers the same documents: the number and dates of a passport, a national
# licence and a permit. Names are deliberately left out. Two recognizers
# spelling a name differently is a genuine disagreement about what the card
# says, not two views of one unambiguous string of digits.
_CROSS_LANGUAGE_PATHS = frozenset({
    "emirates_id.issue_date", "emirates_id.expiry_date",
    "uae_driving_licence.number",
    "gcc_identity.number", "gcc_identity.issue_date", "gcc_identity.expiry_date",
    "gcc_driving_licence.number", "gcc_driving_licence.issue_date",
    "gcc_driving_licence.expiry_date",
    "personal_info.place_of_birth",
    "personal_info.date_of_birth",
    "passport.number", "passport.issued_by_code",
    "passport.issue_date", "passport.expiry_date",
    "national_driving_licence.number", "national_driving_licence.issue_date",
    "national_driving_licence.expiry_date",
    "international_driving_permit.number",
    "international_driving_permit.issue_date",
    "international_driving_permit.expiry_date",
})

# Small print is frequently recovered by only one image variant, so the label
# and its value may come from different passes over the same page.
_NAME_PATHS = frozenset({
    "personal_info.full_name", "personal_info.first_name",
    "personal_info.middle_name", "personal_info.last_name",
})


_NAME_EDGE_NOISE = " :#;,.\u2022"
_NAME_PUNCTUATION = frozenset(" -'\u2019.,\u00b7")
_PRIMARY_PASSPORT_SURNAME_DESIGNATOR = re.compile(r"^\s*\[\s*A\s*\]\s*", re.I)

# German passports print field 14 vertically on the facing page as
# "Ordens- oder Künstlername / Religious name or pseudonym / Nom de religion
# ou pseudonyme".  A recognizer can split that label into three confident
# lines; the generic NOM/NAME match then mistakes the remainder for the
# holder's surname or given name.  These are field descriptors in several
# common passport languages, never the value of the primary name fields.
_NON_PRIMARY_NAME_FIELD_MARKERS = (
    "ORDENSNAME", "ORDENS- ODER", "KUNSTLERNAME",
    "RELIGIOUS NAME", "NOM DE RELIGION", "NOME RELIGIOSO",
    "NOMBRE RELIGIOSO", "RELIGIEUZE NAAM", "STAGE NAME", "ARTIST NAME",
)

# Biographical/family pages can be captured beneath the passport biodata page.
# These rows name somebody related to the holder, never the holder themselves.
# Keeping them in the global label index also prevents their values from being
# borrowed as a neighbouring primary-name row.
_NON_HOLDER_NAME_LABELS = (
    "NAME OF FATHER", "NAME OF MOTHER", "NAME OF SPOUSE",
    "NAME OF LEGAL GUARDIAN", "FATHER/LEGAL GUARDIAN",
)
_COMPACT_NON_HOLDER_NAME_LABELS = tuple(
    re.sub(r"[^A-Z0-9]", "", label)
    for label in _NON_HOLDER_NAME_LABELS
)


def _clean_person_name(value: str) -> str:
    """Remove OCR decoration without changing punctuation used inside names."""
    return " ".join(value.strip(_NAME_EDGE_NOISE).split())


def _non_primary_name_field_label(text: str) -> bool:
    folded = fold_for_match(text)
    if any(marker in folded for marker in _NON_PRIMARY_NAME_FIELD_MARKERS):
        return True
    compact = re.sub(r"[^A-Z0-9]", "", folded)
    name_at = compact.find("NAME")
    if name_at < 0:
        return False
    relative = compact[name_at:]
    # OCR commonly closes the space (``Nameof Father``) or loses two letters
    # in the relationship (``Name of Sose`` for Spouse). Once the literal word
    # NAME is present, a close match to a known relative-field caption is safe
    # to reject and stops that relative's name entering the customer's fields.
    return any(
        SequenceMatcher(None, relative[:len(label) + 2], label).ratio() >= 0.78
        for label in _COMPACT_NON_HOLDER_NAME_LABELS
    )


def _plausible_person_name(value: str) -> bool:
    """Whether text has the shape of a name in any Unicode script.

    Parentheses, slashes, stars and other form notation are not part of a
    printed holder name. This matters for tiny captions such as ``name(x)``:
    after the label is removed, ``(x)`` has a letter and used to become a
    competing first name. Unicode letters and combining marks stay valid, as
    do the separators genuinely used by names. A clean one-letter legal name
    remains valid; a one-letter form marker wrapped in punctuation does not.
    """
    cleaned = _clean_person_name(value)
    if not cleaned or any(character.isdigit() for character in cleaned):
        return False
    letters = 0
    for character in cleaned:
        category = unicodedata.category(character)
        if category.startswith("L"):
            letters += 1
            continue
        if category.startswith("M") or category == "Pd" or character in _NAME_PUNCTUATION:
            continue
        return False
    return letters >= 1


_CROSS_VARIANT_PATHS = frozenset({
    # A holder's name is often recovered by one image variant only, and the
    # label beside it by the other. Keeping them apart left a Belgian
    # passport's SURNAME label with no name to bind in its own pass, so it
    # reached up a row and took the issuing country instead. Two variants
    # reading the same name slightly differently is not a conflict: the
    # reconciler already treats near-identical names as one value.
    "personal_info.full_name", "personal_info.first_name",
    "personal_info.middle_name", "personal_info.last_name",
    "emirates_id.number", "emirates_id.issue_date", "emirates_id.expiry_date",
    "uae_driving_licence.number",
    "gcc_identity.number", "gcc_identity.issue_date", "gcc_identity.expiry_date",
    "gcc_driving_licence.number", "gcc_driving_licence.issue_date",
    "gcc_driving_licence.expiry_date",
    "personal_info.place_of_birth",
    # The national licence and the permit, which were the only driving
    # documents left out of this and had no reason to be.
    #
    # A recogniser is shown the page several times over -- as captured, with
    # the lighting divided out, with the stroke edges restored -- and each
    # rendering wins different rows. On a Queensland licence the label came
    # back from two of them and its number from the third at 0.9997: the page
    # said "LICENCE NO / CRN" and "130 750 802" one row apart, both plainly
    # read, and the number was refused for having been recovered by a
    # different pass than the label above it. Every other document type could
    # already bind across renderings; this one reported no licence number at
    # all while five rounds went looking elsewhere for the cause.
    "national_driving_licence.number", "national_driving_licence.issue_date",
    "national_driving_licence.expiry_date",
    "international_driving_permit.number",
    "international_driving_permit.issue_date",
    "international_driving_permit.expiry_date",
    "passport.number", "passport.issue_date", "passport.expiry_date",
    # The holder's birth date, which was the one date left out and is read the
    # same way as the three above. On the Ontario licence the label "3 DOB/DDN"
    # came back from the page as captured and the date under it only from the
    # deblurred pass, so the label had nothing of its own to bind and the card
    # gave up no birth date at all. That also costs the licence number, which
    # on an Ontario or Quebec card is confirmed by checking the birth date it
    # encodes against the one the card prints.
    "personal_info.date_of_birth",
})


def _all_known_labels() -> tuple[str, ...]:
    """Every wording the reader recognises as a label rather than a value.

    A row this does not know is treated as content, so a page labelled only in
    its own language had its label rows read as values: the Turkish passport's
    "Veriliş tarihi" was a candidate value for the row it labels. The passport
    and licence tables belong here for the same reason the English ones do.

    The two national tables were left out, and a caption printed only in the
    issuing country's language was therefore invisible as a caption -- not to
    the field that reads it, which is handed its own wordings, but to every
    rule that asks whether some *other* field's caption is standing in the way.
    A Brazilian licence prints "DATA EMISSÃO" over its issue date; unknown as a
    label, it could not stop the word "validade", picked out of a paragraph of
    legal text beside the card, from reaching down and claiming that date as
    the expiry.
    """
    return (
        *(label for group in FIELD_LABELS.values() for label in group),
        *(label for group in PASSPORT_VIZ_LABELS.values() for label in group),
        *(label for group in COMMON_NATIONAL_LABELS.values() for label in group),
        *(label for group in LANGUAGE_FIELD_LABELS.values() for label in group),
        *_NON_HOLDER_NAME_LABELS,
        *_NON_HOLDER_LABELS,
    )


# A letter in any script, minus the ordinal indicators and degree sign. Unicode
# classes those as letters, but on a card they are typography: "Licencia Nº",
# "Permis N°" and "N° de référence" all end their label with one, so counting
# them as letters hid the label behind its own punctuation.
_LETTER = r"[^\W\d_ªº°]"


@lru_cache(maxsize=4096)
def label_pattern(label: str) -> re.Pattern[str]:
    """Compile a label so printed punctuation cannot hide it.

    These cards abbreviate and punctuate their labels freely: a Qatari card
    prints "ID. No" and "D.O.B", an Omani licence "D.O.B", where the profiles
    record "ID NO" and "DATE OF BIRTH". Matching the label literally missed all
    of them, and a missed label is not merely a missed field -- it used to send
    the reader into a fifteen-second generative pass hunting for a value
    printed in plain sight.

    Only dots and whitespace are tolerated, and only between the label's own
    characters, so the surrounding word boundaries still hold.

    The boundary is any letter in any script, written ``[^\\W\\d_]``. It used to
    be the ASCII-and-Cyrillic class ``[A-ZА-ЯЁ]``, which meant an accented
    letter did not count as a letter and so read as a word boundary: "NOM"
    matched inside the French "PRÉNOM", and a Moroccan licence bound the given
    name AZIZA as the surname. Every accented language was affected, not only
    French.
    """
    # Whitespace is tolerated only where the label itself has a space. Between
    # two characters the label writes together, a space on the card is not
    # spacing inside the caption -- it is the end of the caption. New Jersey
    # prints "DL N2335 15300 01035", and with whitespace allowed everywhere the
    # caption "DLN" swallowed the "N" that begins the number, so the licence
    # number came back with its first character missing.
    parts: list[str] = []
    after_space = False
    for character in label:
        if character.isspace():
            after_space = True
            continue
        if parts:
            parts.append(r"\.?\s*" if after_space else r"\.?")
        parts.append(re.escape(character))
        after_space = False
    body = "".join(parts)
    return re.compile(rf"(?<!{_LETTER}){body}(?!{_LETTER})", re.I)


def compact_label(label: str) -> str:
    return re.sub(r"[\s.]", "", label).upper()


@dataclass(frozen=True)
class _LineIndex:
    """Per-line geometry and text facts computed once for a whole document.

    Every derived value here used to be recomputed inside the label/neighbour
    loops, which made the cost of reading one card grow with
    (fields x labels x lines x lines) instead of with the number of lines.
    """

    lines: tuple[OCRLine, ...]
    rects: tuple[tuple[float, float, float, float], ...]
    uppers: tuple[str, ...]
    compacts: tuple[str, ...]
    stripped: tuple[str, ...]
    is_label: tuple[bool, ...]
    is_rtl: tuple[bool, ...]
    haystack: str


def build_line_index(
    lines: Iterable[OCRLine], extra_labels: tuple[str, ...] = (),
) -> _LineIndex:
    ordered = sorted(
        lines, key=lambda line: (_line_rect(line)[1], _line_rect(line)[0]),
    )
    rects = tuple(_line_rect(line) for line in ordered)
    uppers = tuple(line.text.upper().strip() for line in ordered)
    known = (*_all_known_labels(), *extra_labels)
    compacts = tuple(compact_label(upper) for upper in uppers)
    # A row is a label only where the wording sits on its own word boundary.
    # The plain substring test that used to decide this put "ADI" -- Turkish for
    # "name" -- inside the given name NADIR, flagged that row as a label, and so
    # refused to read it as anybody's name. The substring test stays as the
    # cheap reject: it runs in C, and only the handful of labels it admits pay
    # for the boundary check.
    compact_known = tuple((compact_label(label), label) for label in known)
    return _LineIndex(
        lines=tuple(ordered), rects=rects, uppers=uppers, compacts=compacts,
        stripped=tuple(line.text.strip() for line in ordered),
        is_label=tuple(
            any(
                folded in compact and label_pattern(label).search(upper)
                for folded, label in compact_known
            )
            for compact, upper in zip(compacts, uppers)
        ),
        is_rtl=tuple(_is_rtl(line.text) for line in ordered),
        haystack="\n".join(compacts),
    )


def _mangled_label(value: str, labels: tuple[str, ...]) -> bool:
    """True when the text is one of these labels as OCR mis-read it.

    A bilingual page prints one row as "vendlindja/place of birth", so whatever
    follows the label that matched is the same label in the other language and
    never a value. Recognising that needs a tolerance: the Albanian passport's
    English half came back as "place of birch", one character out, and an exact
    comparison let the word "birch" be stored as the holder's birthplace.
    """
    compact = compact_label(value.upper())
    if not compact:
        return False
    return any(
        abs(len(target) - len(compact)) <= 2
        and SequenceMatcher(None, compact, target).ratio() >= 0.82
        for target in (compact_label(label.upper()) for label in labels)
        if target
    )


def _mangled_parallel_label(value: str, labels: tuple[str, ...]) -> bool:
    """Recognise the damaged second half of a bilingual label row.

    This is deliberately looser than :func:`_mangled_label`, but it is only
    used for text following a slash after a label that was read exactly.  The
    Colombian passport that exposed the gap returned ``Apellidos / Sumamo``:
    the first half proves what the row is, while ``Sumamo`` is OCR damage to
    the parallel ``Surname`` label, not the holder's surname.  Applying this
    resemblance globally would be unsafe because a real surname can resemble
    a field label; the slash plus the already-confirmed first label is the
    guard that makes the weaker comparison useful.
    """
    stripped = value.strip()
    if (
        not stripped
        or any(character.isdigit() for character in stripped)
        or len(stripped.split()) > 3
    ):
        return False
    compact = compact_label(stripped.upper())
    if len(compact) < 3:
        return False
    for label in labels:
        target = compact_label(label.upper())
        # This weaker comparison is only reached after an exact first label
        # and a slash. It covers short bilingual captions such as ``Name/Nom``
        # when OCR returns the French half as ``Non``; without it, NON was
        # stored as the customer's given name. The same test is unsafe on a
        # free-standing three-letter row, which is why it lives only here.
        if 3 <= len(compact) <= 4 and 3 <= len(target) <= 5:
            if (
                compact[0] == target[0]
                and abs(len(target) - len(compact)) <= 1
                and SequenceMatcher(None, compact, target).ratio() >= 0.66
            ):
                return True
            continue
        if (
            len(target) < 5
            or compact[0] != target[0]
            or abs(len(target) - len(compact)) > 2
        ):
            continue
        if SequenceMatcher(None, compact, target).ratio() >= 0.60:
            return True
    return False


# How close a printed row has to be to a label before it is read as that label,
# and how long the label has to be for the question to be worth asking.
#
# A worn passport returns its small print a character or two out: "Date of
# issue/Date de délivrance" came back as "Date df issusDate de défveance",
# "Date of birth" as "Dats of birtih", "Given names" as "Givan names". Every
# value beside them was read cleanly -- the dates at 0.99, the name at 0.9995 --
# and all of them were reported missing, because a label is matched literally
# and each of these is one letter wrong.
#
# The length floor is what keeps this honest, and it has to be high.
#
# "Sex" came back as "Sax" on the same page, and three characters with one
# wrong is 0.67 similar -- below any threshold that could be set without
# matching things that are not labels at all. But short is not the only danger:
# at a floor of six, the title "Permis de conduire" matched the licence-number
# label "Permis n°" at 0.86, because they differ only in the character after
# the word they share. The card's own title became the label for the number,
# and bound the nearest value to it -- the holder's street address, offered as
# the number the rental is keyed on.
#
# Ten characters is what the labels this recovers actually measure: "date of
# issue", "date of birth", "given names". It excludes "permis n" and "dl
# number", which are the two that went wrong.
_FUZZY_LABEL_RATIO = 0.82
_FUZZY_LABEL_MINIMUM_LENGTH = 10


def _fuzzy_label_rows(
    index: _LineIndex, labels: tuple[str, ...],
) -> dict[int, str]:
    """Rows that are one of these labels as a worn card returned it."""
    targets = [
        (compact_label(label), label) for label in labels
        if len(compact_label(label)) >= _FUZZY_LABEL_MINIMUM_LENGTH
    ]
    if not targets:
        return {}
    matched: dict[int, str] = {}
    for position, compact in enumerate(index.compacts):
        if not compact:
            continue
        # The row naming the document is not a field on it. Without this the
        # nearest thing to a title is whatever it happens to resemble.
        folded = fold_for_match(index.stripped[position])
        if any(title in folded for title in _FOLDED_LICENCE_TITLES):
            continue
        # A row that already spells another field's caption exactly is that
        # field's, however close it looks to this one. A Mexican licence
        # prints "FECHA DE NACIMIENTO" over the birth date and "FECHA DE
        # VENCIMIENTO" over the expiry; OCR read the second as "VENDIMIENTO",
        # so the expiry caption was looked for by resemblance -- and the birth
        # caption, three characters away, answered as well. The card was
        # reported as expiring on the holder's date of birth, in conflict with
        # the expiry printed beneath its own caption.
        if any(
            compact_label(known) in compact
            and label_pattern(known).search(index.uppers[position])
            for known in _all_known_labels()
            if known not in labels
        ):
            continue
        # A bilingual caption states its field twice around a slash, and the
        # half a recogniser damages is not always the first: an Indian
        # passport's expiry caption came back as "समाप्ति की तिथि / Date of
        # Expuy", whose opening is the Hindi half, so the English half -- one
        # character from the label it is -- was never compared at all and the
        # date the passport expires was reported missing. Each half is offered
        # as well as the whole row; the ratio test still decides.
        pieces = [compact]
        if "/" in index.stripped[position]:
            pieces.extend(
                part for part in (
                    compact_label(half)
                    for half in index.stripped[position].split("/")
                ) if part and part not in pieces
            )
        best: tuple[float, str] | None = None
        for target, label in targets:
            # A passport prints its label at the head of the row and the rest
            # of the row is the same label in the other language, so the
            # comparison is against the opening of the row rather than all of
            # it. The opening is measured at a few lengths either side of the
            # label's own, because a recogniser that mis-reads a character can
            # as easily insert or drop one: "Date of birth" came back as "Dats
            # of birtih", a character longer than the label it is.
            for piece in pieces:
                for width in range(len(target) - 1, len(target) + 3):
                    if width < _FUZZY_LABEL_MINIMUM_LENGTH or len(piece) < width:
                        continue
                    ratio = SequenceMatcher(None, piece[:width], target).ratio()
                    if ratio >= _FUZZY_LABEL_RATIO and (best is None or ratio > best[0]):
                        best = (ratio, label)
        if best is not None:
            matched[position] = best[1]
    return matched


def _index_with_labels_repaired(
    index: _LineIndex, repaired: dict[int, str],
) -> _LineIndex:
    """The same page with the worn label rows spelled as the labels they are.

    Only the index is rewritten; every ``OCRLine`` keeps the text the
    recogniser actually returned, so the evidence an operator is shown is still
    what is printed on the document.
    """
    uppers = list(index.uppers)
    compacts = list(index.compacts)
    stripped = list(index.stripped)
    is_label = list(index.is_label)
    for position, label in repaired.items():
        uppers[position] = label.upper()
        compacts[position] = compact_label(label)
        stripped[position] = label
        is_label[position] = True
    return _LineIndex(
        lines=index.lines, rects=index.rects, uppers=tuple(uppers),
        compacts=tuple(compacts), stripped=tuple(stripped),
        is_label=tuple(is_label), is_rtl=index.is_rtl,
        haystack="\n".join(compacts),
    )


def _label_between(
    index: _LineIndex, label_position: int, value_position: int,
) -> bool:
    """True where another field's label stands between a label and a value.

    Across a shared printed row, and down a shared column. Both are the same
    statement -- a caption names the nearest value under or beside it, and
    another field's caption standing in the way means the value is that
    field's, however near this one is by distance.

    The column arm was added for a Brazilian licence, whose digital copy
    prints a paragraph of legal text beside the card: "Sua validade poderá ser
    confirmada..." carries the word the card uses for the expiry caption, so
    it was read as that caption, reached down the page and claimed the issue
    date -- which sits under its own "DATA EMISSÃO" caption, standing plainly
    between the two. The licence was then reported as expiring on the day it
    was issued, in conflict with the expiry the card actually prints.
    """
    x1, y1, x2, y2 = index.rects[label_position]
    cx1, cy1, cx2, cy2 = index.rects[value_position]
    if min(y2, cy2) > max(y1, cy1):
        left, right = (x2, cx1) if cx1 >= x2 else (cx2, x1)
        if right - left <= 0:
            return False
        for position, is_label in enumerate(index.is_label):
            if not is_label or position in (label_position, value_position):
                continue
            ox1, oy1, ox2, oy2 = index.rects[position]
            if min(y2, oy2) <= max(y1, oy1):
                continue                   # a different printed row
            if ox1 >= left and ox2 <= right:
                return True
        return False
    top, bottom = (y2, cy1) if cy1 >= y2 else (cy2, y1)
    if bottom - top <= 0:
        return False
    for position, is_label in enumerate(index.is_label):
        if not is_label or position in (label_position, value_position):
            continue
        ox1, oy1, ox2, oy2 = index.rects[position]
        # Wholly inside the gap: a caption that merely overlaps one of the two
        # rows is part of that row's own wording, not something in its way.
        if oy1 < top or oy2 > bottom:
            continue
        # And standing over the value, not in some other column of the card.
        if min(ox2, cx2) - max(ox1, cx1) <= 0:
            continue
        return True
    return False


# Two of the AAMVA date rows, which together no other card model prints.
_AAMVA_DATE_ROWS = (
    re.compile(r"(?<![A-Z0-9])3\s*[.):\-]?\s*DOB\b", re.I),
    re.compile(r"(?<![A-Z0-9])4\s*A\s*[.):\-]?\s*ISS\b", re.I),
    re.compile(r"(?<![A-Z0-9])4\s*B\s*[.):\-]?\s*EXP\b", re.I),
)


# An American card that captions its rows in words rather than by designator.
# New York prints "DOB", "Issued" and "Expires" against its dates and no
# numbers at all, so the designator test below does not see it.
_AMERICAN_LICENCE_CAPTIONS = (
    re.compile(r"(?<![A-Z])DOB(?![A-Z])", re.I),
    # Michigan abbreviates both of its dated rows -- "Exp 03/26/2030" -- so the
    # spelled-out captions alone left that card reading day first, and an
    # expiry in the twenty-sixth month was no date at all.
    re.compile(r"(?<![A-Z])ISS(?:UED)?(?![A-Z])", re.I),
    re.compile(r"(?<![A-Z])EXP(?:IRES)?(?![A-Z])", re.I),
)


def _prints_american_licence_captions(lines: list[OCRLine]) -> bool:
    """Whether the page is an American card, by the words it captions with.

    The designators are one way a card says it is American and the captions
    are another: New York heads itself "NEW YORK STATE USA" and writes
    "DOB 04/06/1998", "Issued 05/31/2022", "Expires 04/06/2030" -- month
    first, with no designator anywhere on it. Read day first, the issue row
    named a thirty-first month and was reported missing from a licence that
    prints it plainly, and the birth date came back a different day from the
    passport's, so the holder's own date of birth was reported CONFLICTING.

    The country must be on the page and Canada must not be, exactly as for the
    designators: an Ontario card captions its rows in English too and writes
    its dates year first.
    """
    text = " ".join(line.text for line in lines).upper()
    if "CANADA" in text or not re.search(r"(?<![A-Z])USA(?![A-Z])", text):
        return False
    return sum(
        bool(caption.search(text)) for caption in _AMERICAN_LICENCE_CAPTIONS
    ) >= 2


def _prints_aamva_field_codes(lines: list[OCRLine]) -> bool:
    """Whether the page is an American card, by the way it names its fields.

    The designators alone are not enough. A Canadian province prints the same
    ones -- Ontario sets "3 DOB", "4a ISS/DEL", "4b EXP" -- and writes its
    dates year first, so reading a Canadian card as American moved its dates
    and took its number from the caption row. The country has to be on the
    page as well.
    """
    text = " ".join(line.text for line in lines).upper()
    if "CANADA" in text or not re.search(r"(?<![A-Z])USA(?![A-Z])", text):
        return False
    return sum(bool(row.search(text)) for row in _AAMVA_DATE_ROWS) >= 2


def _value_column_under_label(
    index: _LineIndex, label_position: int, value_position: int,
) -> str | None:
    """The part of a value row that stands in this label's column.

    A card can set three fields side by side and the recogniser may return the
    whole row as one box: a Brazilian licence gave back
    "5133884349 SSPSP 723456789012" for the row under "4d CPF",
    "5 N REGISTRO" and "9 CAT. HAB". The licence-number field joins across a
    space, because a serial split in two is the usual reason for one, and
    joining three fields' values made a twenty-seven character string that no
    number could be -- so a card printing its number plainly reported none.

    The row of captions immediately above says where the columns are. That is
    geometry, not vocabulary: every box on that row delimits a column, whether
    or not the reader knows the words in it, which matters because "4d CPF"
    means nothing to it. Tokens are then kept whole and chosen by where they
    sit -- under the caption itself if any do, otherwise within its column --
    so a value is never cut through the middle.
    """
    x1, _, x2, _ = index.rects[label_position]
    cx1, cy1, cx2, cy2 = index.rects[value_position]
    height = max(cy2 - cy1, 1.0)
    span = max(cx2 - cx1, 1.0)
    # OCR boxes are generous, and a caption is measured no more exactly than
    # the row beneath it: the Massachusetts card's "4d NUMBER" begins five
    # pixels left of the row it names and ends two pixels inside it. Asking
    # for a caption wholly within the row, wholly above it, threw all three
    # columns away over those seven pixels, and the licence number -- printed
    # in the same box as the expiry and the birth date, which is what makes
    # the column split necessary at all -- came back as no number found. A
    # caption belongs to this row when it stands mostly over it.
    slack = height * 0.25
    above = [
        rect for position, rect in enumerate(index.rects)
        if position != value_position
        and min(rect[2], cx2) - max(rect[0], cx1) >= (rect[2] - rect[0]) * 0.8
        and -slack <= cy1 - rect[3] <= height
    ]
    if not above:
        return None
    # One printed row of captions, the nearest. Two captions stacked one above
    # the other are the same field named twice, not two columns -- and taken
    # for columns they cut a name in half at its first space.
    baseline = max(rect[3] for rect in above)
    columns = sorted(
        (rect[0], rect[2]) for rect in above
        if baseline - rect[3] <= height * 0.5
    )
    if len(columns) < 2 or (x1, x2) not in columns:
        return None
    if any(
        min(right, columns[position + 1][1]) > max(left, columns[position + 1][0])
        for position, (left, right) in enumerate(columns[:-1])
    ):
        return None                        # overlapping boxes are not columns
    place = columns.index((x1, x2))
    left = cx1 if place == 0 else (columns[place - 1][1] + x1) * 0.5
    right = cx2 if place == len(columns) - 1 else (x2 + columns[place + 1][0]) * 0.5

    text = index.stripped[value_position]
    tokens: list[tuple[float, str]] = []
    offset = 0
    for token in text.split():
        offset = text.index(token, offset)
        middle = cx1 + (offset + len(token) * 0.5) / len(text) * span
        tokens.append((middle, token))
        offset += len(token)
    if len(tokens) < 2:
        return None
    for low, high in ((x1, x2), (left, right)):
        chosen = [token for middle, token in tokens if low <= middle <= high]
        if chosen and len(chosen) < len(tokens):
            return " ".join(chosen)
    return None


# A caption printed inside a row ends with the mark that separates it from
# its value. Requiring that mark is what keeps an ordinary word in somebody's
# name or address from being mistaken for the start of another field.
_INLINE_CAPTION = r"(?:^|(?<=[\s,;/]))%s\s*[:.]"

# An AAMVA card names its fields with these codes and no separator at all --
# "DLN M325-666-79-471-09 CLASS E", "3 DOB 12/31/1979 15SEX M". The code may
# carry its designator glued to the front of it, as "15SEX" does. The set is
# closed and the words are specific, so a value can be ended at one of them
# without the punctuation the rule above requires.
_AAMVA_FIELD_CODE = re.compile(
    r"(?<=\s)\d{0,2}\s*(?:DLN|CLASS|DOB|SEX|HGT|WGT|EYES|HAIR|REST|END|ISS|EXP|DD)\b",
    re.I,
)


# A field named by nothing but "No.:" -- the second half of a caption whose
# first word the capture lost. A South African licence sets the licence number
# and the card's own sequence number on one printed line, and the recogniser
# returned "893900002SKB No.: 1" as a single box; the whole of it was stored as
# the number the rental is keyed on. A colon is required: "AA No 035630" on a
# permit booklet names the value that follows it and must stay whole.
_BARE_NUMBER_CAPTION = re.compile(
    r"(?<=\S)\s+(?:N[Oo]\.?|NR\.?|N[°º]|NUM\.?)\s*:", re.I,
)


def _inline_value_before_next_caption(tail: str) -> str:
    """Cut a label's own row at the next field named on it.

    A British Columbia licence prints two fields on one line -- "Issued:
    2021-Feb-10 DOB: 1991-Mar-02" -- and the whole of it followed the Issued
    caption. The date field then had two dates to choose between and took the
    earlier of them, which is the rule for a validity range printed as one
    cell; here the earlier date was the holder's birth date, offered as the
    day their licence was issued.
    """
    if not tail:
        return tail
    compact = compact_label(tail.upper())
    upper = tail.upper()
    cut = len(tail)
    for label in _all_known_labels():
        folded = compact_label(label)
        if not folded or folded not in compact:
            continue
        found = re.search(
            _INLINE_CAPTION % re.escape(label).replace(r"\ ", r"\s+"),
            upper, re.I,
        )
        if found is not None and 0 < found.start() < cut:
            cut = found.start()
    code = _AAMVA_FIELD_CODE.search(tail)
    if code is not None and 0 < code.start() < cut:
        cut = code.start()
    bare = _BARE_NUMBER_CAPTION.search(tail)
    if bare is not None and 0 < bare.start() < cut:
        cut = bare.start()
    return tail[:cut].strip() if cut < len(tail) else tail


def _caption_column(
    index: _LineIndex, position: int, start: int, end: int,
    labels: tuple[str, ...],
) -> tuple[float, float]:
    """The horizontal share of a caption box that belongs to this caption.

    A passport can print two captions in one box -- a Czech page sets
    "07 DATUM VYDANI DATE OF ISSUE/DATE DE DELIVRANCE 08 PLATNOST DO/DATE OF
    EXPIRY/DATE D'EXPIRATION" as a single line -- with each caption's value
    below its own half. Given the whole box, the issue caption reached the
    expiry's value, which sits higher on the page and so was nearer; the two
    fields then claimed the same date and the issue date, being the one that
    cannot be later than the expiry, was refused. The card states it plainly.

    A caption naming the same field again in another language is not a
    boundary -- "DATE DE DELIVRANCE" is still the issue caption -- so only a
    wording this field does not own ends the column.
    """
    x1, _, x2, _ = index.rects[position]
    upper = index.uppers[position]
    if not upper:
        return x1, x2
    own = {compact_label(label) for label in labels}
    left, right = 0, len(upper)
    for label in _all_known_labels():
        folded = compact_label(label)
        if not folded or folded in own or folded not in index.compacts[position]:
            continue
        for found in label_pattern(label).finditer(upper):
            if found.end() <= start:
                left = max(left, found.end())
            elif found.start() >= end:
                right = min(right, found.start())
    if (left, right) == (0, len(upper)):
        return x1, x2
    span = x2 - x1
    return (
        x1 + span * left / len(upper),
        x1 + span * right / len(upper),
    )


def _outsized_rows(
    rects: tuple[tuple[float, float, float, float], ...],
) -> frozenset[int]:
    """Boxes far taller than the page's own text rows, which state nothing.

    Measured against the page's median row so that a card photographed on its
    side, where every box is tall, is judged by its own proportions.
    """
    if len(rects) < 8:
        return frozenset()
    heights = sorted(rect[3] - rect[1] for rect in rects)
    typical = heights[len(heights) // 2]
    if typical <= 0:
        return frozenset()
    return frozenset(
        position for position, rect in enumerate(rects)
        if rect[3] - rect[1] > typical * 4.0
    )


def _label_value(
    index: _LineIndex,
    labels: tuple[str, ...],
    accepts: Callable[[str], bool] | None = None,
    strict_neighbor: bool = False,
    strict_next_row: bool = False,
    cross_language_values: bool = False,
    cross_variant_values: bool = False,
    avoid_labels: tuple[str, ...] = (),
    max_rows_below: float | None = None,
) -> list[tuple[OCRLine, str, float]]:
    # The pattern tolerates only dots and spaces inside the label, so testing
    # the punctuation-stripped document for the stripped label is a sound and
    # much cheaper pre-filter.
    present = [
        label for label in labels
        # The substring test is the cheap reject; the boundary test is the
        # answer. Counting "ADI" inside a longer word as a label present on the
        # page was enough to keep the recovery below from ever being asked for
        # the holder's given name.
        if compact_label(label) in index.haystack
        and any(label_pattern(label).search(upper) for upper in index.uppers)
    ]
    if not present:
        # Nothing on the page spells any of these labels. Before giving the
        # field up, ask whether a row is one of them as OCR returned it -- and
        # only then, so a page whose label was read cleanly is never decided by
        # a resemblance.
        repaired = _fuzzy_label_rows(index, labels)
        if not repaired:
            return []
        index = _index_with_labels_repaired(index, repaired)
        present = sorted(set(repaired.values()))
    ordered_labels = sorted(present, key=len, reverse=True)
    results: list[tuple[OCRLine, str, float]] = []
    ordered = index.lines
    all_labels = _all_known_labels()
    accepted: list[bool | None] = [None] * len(ordered)

    def acceptable(position: int) -> bool:
        if accepts is None:
            return True
        cached = accepted[position]
        if cached is None:
            cached = accepts(index.stripped[position])
            accepted[position] = cached
        return cached

    avoid = tuple(
        label for label in avoid_labels if compact_label(label) in index.haystack
    )
    # A box many times taller than the page's own rows is not a printed row.
    # The South African licence carries a diagonal ghost image of its own data,
    # and the recogniser returned it as one slanted box spanning five rows --
    # "05/01/2021 30/04/2003 THAE". Its dates were bound as the licence's issue
    # and expiry, from a smear that states neither.
    outsized = _outsized_rows(index.rects)
    columns: dict[int, str] = {}
    for position, line in enumerate(ordered):
        upper = index.uppers[position]
        # A row the page labels more precisely belongs to that field. "Given
        # names" contains the word "name", so the generic full-name lookup
        # bound a Belgian passport's given-name row and the split then stored
        # NADIR as the holder's surname as well.
        if avoid and any(label_pattern(label).search(upper) for label in avoid):
            continue
        for label in ordered_labels:
            if compact_label(label) not in index.compacts[position]:
                continue
            match = label_pattern(label).search(upper)
            if match is None: continue
            inline_tail = line.text[match.end():]
            inline = _inline_value_before_next_caption(
                re.sub(r"^[\s:#.\-/]+", "", inline_tail)
            )
            inline_is_another_label = any(
                re.sub(r"^[\s:#.\-/]+", "", inline).upper() == known
                for known in all_labels
            ) or _mangled_label(inline, labels) or (
                "/" in inline_tail and _mangled_parallel_label(inline, all_labels)
            ) or _is_label_only_row(inline)
            if inline and not inline_is_another_label and (accepts is None or accepts(inline)):
                results.append((line, inline, 1.0))
                break
            x1, y1, x2, y2 = index.rects[position]
            x1, x2 = _caption_column(
                index, position, match.start(), match.end(), labels,
            )
            height = max(y2 - y1, 1)
            neighbors: list[tuple[float, OCRLine, float]] = []
            for candidate_position, candidate in enumerate(ordered):
                if candidate is line:
                    continue
                if not cross_variant_values and candidate.variant != line.variant:
                    continue
                if not cross_language_values and not _same_language_pass(candidate, line):
                    continue
                if index.is_label[candidate_position]:
                    continue
                if candidate_position in outsized:
                    continue
                if not acceptable(candidate_position):
                    # A row the recogniser returned as one box may still hold
                    # this label's value in its own column, beside another
                    # field's. Judge that column, not the whole row.
                    column = _value_column_under_label(
                        index, position, candidate_position,
                    )
                    if column is None or (accepts is not None and not accepts(column)):
                        continue
                    columns[candidate_position] = column
                cx1, cy1, cx2, cy2 = index.rects[candidate_position]
                # A box sitting inside the label's own box is another pass
                # reading part of that label, not a value printed beside it.
                # The passport's "Givan namesPranom" was returned whole by one
                # pass and as the fragment "esPreom" by another; the fragment
                # lies within the label, shares its row exactly, and so beat
                # the holder's given name printed on the line below. MAKENDY
                # lost to a piece of the word "Prénoms".
                #
                # A zoom re-read is exempt, and has to be: that pass exists to
                # photograph the label's own row again at four times the size
                # when the value beside it was lost, so the value it recovers
                # maps back onto the label by design. Those rows are filed
                # under the anchor label's own variant so that they can bind at
                # all, and carry a "+zoom" marker to stay recognisable here.
                overlap_width = min(x2, cx2) - max(x1, cx1)
                overlap_height = min(y2, cy2) - max(y1, cy1)
                if (
                    not candidate.model_name.endswith("+zoom")
                    and overlap_width > 0 and overlap_height > 0
                    and overlap_width * overlap_height
                    >= 0.8 * max(1.0, (cx2 - cx1) * (cy2 - cy1))
                ):
                    continue
                if strict_neighbor:
                    label_width = max(x2 - x1, 1.0)
                    center_dx = abs((cx1 + cx2) - (x1 + x2)) * 0.5
                    center_dy = abs((cy1 + cy2) - (y1 + y2)) * 0.5
                    if height > label_width * 1.5:
                        # A 90-degree capture produces vertical text boxes. The
                        # value beside License No. must remain in its narrow
                        # x-column, not at an unrelated serial elsewhere.
                        if center_dx > max(label_width * 2.5, 80.0):
                            continue
                    else:
                        # Upright UAE cards print the numeric value on the same
                        # row immediately to the right of the License No. label.
                        label_center_y = (y1 + y2) * 0.5
                        candidate_center_y = (cy1 + cy2) * 0.5
                        # The Emirates ID prints Issuing Date and Expiry Date
                        # as a bilingual caption row with the date on the row
                        # *beneath* it, a little over one label height away.
                        # The same-row window refused both by six to nine
                        # pixels and reported no evidence for dates the card
                        # states plainly. Reaching one row down is opened only
                        # for the fields whose layout needs it: on a GCC card
                        # the holder rows are stacked tightly and every one of
                        # them holds a date, so there the window stays shut.
                        below_reach = (
                            max(height * 2.2, 72.0)
                            if strict_next_row and candidate_center_y >= label_center_y
                            else max(height, 32.0)
                        )
                        if center_dy > below_reach:
                            continue
                        if candidate_center_y < label_center_y - height * 0.25:
                            # UAE ID issue/expiry values are on the label row or
                            # directly below it. Never let Expiry Date borrow
                            # the preceding Issuing Date value above its label.
                            continue
                        reach = max(label_width * 7.0, 650.0)
                        if index.is_rtl[position]:
                            # Arabic labels on GCC cards are right-aligned, so
                            # their value is normally printed to the *left*.
                            # Accept either side: the LTR-only rule below drops
                            # the whole RTL column and leaves Saudi ID/licence
                            # numbers and dates unbound. Nearest-neighbour
                            # ranking still picks a single value per label.
                            if cx2 < x1 - reach or cx1 > x2 + reach:
                                continue
                        elif cx2 < x1 - height or cx1 > x2 + reach:
                            continue
                if height <= max(x2 - x1, 1.0) * 1.5 and cy2 <= y1:
                    # On horizontal documents a value belongs on the label row
                    # or below it. Allowing values from the previous row swaps
                    # SURNAME/GIVEN NAMES and passport dates.
                    #
                    # Sharing the row means the two boxes overlap vertically.
                    # A fixed slack of a quarter of the label's height was too
                    # generous by about six pixels on a Belgian passport, where
                    # the issuing country BEL sits in its own row just above the
                    # surname label and was stored as the holder's name.
                    continue
                # Rectangle distance works for both horizontal cards and
                # 90-degree captures. Directional "right/below" rules confuse
                # adjacent vertical date columns on rotated licences.
                dx = max(x1 - cx2, cx1 - x2, 0.0)
                dy = max(y1 - cy2, cy1 - y2, 0.0)
                if max_rows_below is not None and dy > height * max_rows_below:
                    # The value a label names is on its own row or the one
                    # under it. A Belgian passport recognised the surname only
                    # in the contrast pass, so in the primary pass the SURNAME
                    # label found nothing on its row, skipped one, and stored
                    # the given name as the surname.
                    continue
                if dx <= max(900.0, (x2 - x1) * 8) and dy <= max(500.0, height * 5):
                    center_dx = abs((cx1 + cx2) - (x1 + x2)) * 0.5
                    center_dy = abs((cy1 + cy2) - (y1 + y2)) * 0.5
                    if height > max(x2 - x1, 1.0) * 1.5:
                        # Vertical OCR line: values belonging to a label share
                        # its narrow x-column; neighboring dates differ in x.
                        distance = center_dx * 3.0 + dy + center_dy * 0.05
                    else:
                        # A label names the value on its row or the value under
                        # it, and nothing else. Distance alone did not say so:
                        # the window reaches most of a card's width, so where a
                        # blurred pass failed to read the value itself, the
                        # nearest surviving box won by default. That is how
                        # "LICENCE NO." at the top right of a Victorian licence
                        # came to name "34 KANGERONG AVE" at the bottom left,
                        # 825 pixels away and three rows down -- a street the
                        # holder lives on, offered as the number the rental is
                        # keyed on, and close enough in score to empty the field.
                        shares_row = min(y2, cy2) > max(y1, cy1)
                        shares_column = center_dx <= max((x2 - x1) * 1.5, 120.0)
                        if not shares_row and not shares_column:
                            continue
                        # A value with another field's label standing between it
                        # and this one belongs to that field, however near it is
                        # by distance. The Ontario licence sets two dated fields
                        # on a single printed row -- "4a ISS/DÉL  2025/10/22
                        # 4b EXP/ EXP.  2028/01/12" -- and only one pass over
                        # the page read the issue date, so in the passes that
                        # missed it the issue label reached straight across the
                        # card, over the expiry label, and took the expiry date.
                        # The licence was reported as issued and expiring on the
                        # same day, at 0.96, with nothing marked for review.
                        if _label_between(index, position, candidate_position):
                            continue
                        # Horizontal OCR line: values share the label's row.
                        distance = center_dy * 3.0 + dx + center_dx * 0.05
                        if cy2 < y1 or (cy1 + cy2) * 0.5 < (y1 + y2) * 0.5 - height * 0.5:
                            # Passport fields are read top-to-bottom. Without a
                            # penalty, the previous row's surname can be closer
                            # to GIVEN NAMES than its value printed just below.
                            #
                            # Measured from the centres as well as the edges,
                            # because OCR boxes are generous and two stacked
                            # rows overlap by a few pixels: "TERMILUS" ended
                            # seven pixels inside the "Given names" label and
                            # so escaped the edge test entirely, and on the
                            # same passport the issue date ended six pixels
                            # inside the expiry label and was reported as the
                            # date that document expired.
                            distance += center_dy * 4.0 + height * 3.0
                    neighbors.append((distance, candidate, 0.96))
            # One label maps to one value. Returning multiple nearby values is
            # especially harmful for rotated cards, where DOB/issue/expiry are
            # adjacent vertical columns and all look like valid dates.
            for _, candidate, proximity in sorted(neighbors, key=lambda item: item[0])[:1]:
                value_position = ordered.index(candidate)
                text = columns.get(value_position)
                if text is None:
                    text = _value_column_under_label(
                        index, position, value_position,
                    )
                    if text is not None and accepts is not None and not accepts(text):
                        text = None
                # A row printed beside a label can carry the next field's
                # caption as well as this field's value; the cut that a label's
                # own row already gets applies to it for the same reason.
                chosen = _inline_value_before_next_caption(
                    text or candidate.text.strip()
                )
                results.append((
                    candidate, chosen or (text or candidate.text.strip()), proximity,
                ))
            break
    return results


def _plausible_raw_value(path: str, value: str, licence_country: str | None = None) -> bool:
    cleaned = value.strip(" :#.-")
    if path == "personal_info.last_name":
        # Field 1(a) is the current surname on a German passport; the bracketed
        # designator is form notation, not part of the holder's name. Field
        # 1(b), the birth name, is intentionally not stripped or admitted.
        cleaned = _PRIMARY_PASSPORT_SURNAME_DESIGNATOR.sub("", cleaned)
    if not cleaned:
        return False
    if path == "emirates_id.number":
        return re.search(r"784(?:[-\s]?\d){12}", cleaned) is not None
    if path == "passport.number":
        if re.search(DATE_PATTERN, cleaned, re.I):
            return False
        tokens = re.findall(r"(?<![A-Z0-9])[A-Z0-9]{7,12}(?![A-Z0-9])", cleaned.upper())
        return any(sum(char.isdigit() for char in token) >= 5 for token in tokens)
    if path == "passport.issued_by_code":
        # A country-code field accepts only a value the country normalizer can
        # resolve. This keeps nearby Type (P) and passport-number rows from
        # being borrowed when a multilingual issuer label spans several cells.
        return normalize_country(cleaned)[0] is not None
    if path == "uae_driving_licence.number":
        if re.search(DATE_PATTERN, cleaned, re.I):
            return False
        compact = re.sub(r"[\s-]", "", cleaned)
        # UAE licence numbers in this workflow are the numeric value visibly
        # printed beside License No./Licence No. Reject card/reference codes
        # such as AJTR7477 and never salvage their numeric suffix.
        return re.fullmatch(r"\d{4,15}", compact) is not None
    if path == "gcc_identity.number":
        return normalize_gcc_number(cleaned, licence_country, identity=True) is not None
    if path == "gcc_driving_licence.number":
        return normalize_gcc_number(cleaned, licence_country, identity=False) is not None
    if path.endswith(("date_of_birth", "issue_date", "expiry_date")):
        # The same repairs the reader applies before parsing a bound value. A
        # row this test rejects is never offered to the caption at all, so a
        # British passport's "30 JUL JUIL 17" -- its slash lost -- was passed
        # over and the issue caption bound the expiry row two lines below it.
        return re.search(
            DATE_PATTERN,
            close_split_year(split_run_together_dates(ascii_numerals(cleaned))),
            re.I,
        ) is not None
    if path == "personal_info.gender":
        return (
            re.search(r"\b(MALE|FEMALE|M|F|X)\b", cleaned, re.I) is not None
            or any(marker in cleaned for marker in ("ذكر", "أنثى", "انثى"))
        )
    if path.endswith(".number"):
        if len(cleaned) < 4 or not any(char.isdigit() for char in cleaned):
            return False
        # A document number is never a date. The passport number has refused
        # one since this reader was written; every other number accepted them,
        # and a Queensland licence -- whose class table prints "CA O 31.05.25
        # 31.05.27" three rows under the number it could not read -- came back
        # keyed on "31.05.25 31.05.27". The rule is the same for all of them,
        # so it is stated once here rather than per country, per document, and
        # per bug report.
        if re.search(DATE_PATTERN, ascii_numerals(cleaned), re.I):
            return False
        # An identifier is not a word. Every licence-number format in this
        # corpus builds its letters into short groups -- a two-letter series, a
        # country prefix, the five surname letters a British number opens with
        # -- and none of them spells anything longer. A street name does: the
        # Victorian card in this project's bug report had "34 KANGERONG AVE"
        # read as "34KANGERONG" and bound as the licence number, where it tied
        # with the real one to within two thousandths of a point and emptied
        # the field the rental is keyed on.
        return not re.search(r"[A-Za-z]{6,}", cleaned)
    if path.endswith("full_name"):
        return _plausible_person_name(cleaned)
    if path.endswith("issued_by_name"):
        return len(cleaned) >= 2 and any(char.isalpha() for char in cleaned)
    if path == "personal_info.place_of_birth":
        # A Romanian passport prints "01 IUL/JUL 26" a row from the birthplace
        # label, and a town is not a date. Requiring letters alone accepted it.
        if re.search(DATE_PATTERN, ascii_numerals(cleaned), re.I):
            return False
        # Nor is a town the page's own trilingual field name. "Semnatura
        # titularului/Holder's signature/Signature du titulaire" is sixty-six
        # characters carrying two slashes; a place name is neither.
        if len(cleaned) > 40 or cleaned.count("/") > 1:
            return False
        letters = sum(char.isalpha() for char in cleaned)
        return len(cleaned) >= 2 and letters >= 2 and letters >= sum(
            char.isdigit() for char in cleaned
        )
    if path.endswith(("first_name", "middle_name", "last_name")):
        return _plausible_person_name(cleaned)
    return True


def _labels_in(text: str, extra: tuple[str, ...] = ()) -> frozenset[str]:
    compact = compact_label(text)
    return frozenset(
        label for label in (*_all_known_labels(), *extra)
        if compact_label(label) in compact
    )


def country_labels(licence_country: str | None) -> tuple[str, ...]:
    """Labels a specific GCC state prints that the shared table does not list.

    A country's own wording has to count as a label everywhere a label matters,
    or a row that only this state uses -- "No:" beside a Saudi record number --
    is treated as ordinary text and discarded as a duplicate.
    """
    profile = profile_for_gcc_country(licence_country)
    if profile is None:
        return ()
    return tuple(dict.fromkeys((
        *profile.identity_number_labels, *profile.licence_number_labels,
    )))


def _redundant_reading(
    line: OCRLine, kept: list[OCRLine], extra: tuple[str, ...] = (),
) -> bool:
    """True when a second recogniser's reading of the same box adds nothing.

    Two recognisers reading one row usually agree, and the duplicate is dropped
    so that a stray character in one of them cannot compete with the other. But
    when the duplicate is the only reading carrying the field label -- one
    engine read "No: 1234567890" while the other clipped it to "N:1234567890"
    -- the duplicate is what holds the row together, and dropping it costs the
    field entirely.
    """
    overlapping = [other for other in kept if _same_text_region(line, other)]
    if not overlapping:
        return False
    labels = _labels_in(line.text, extra)
    if not labels:
        return True
    return all(labels <= _labels_in(other.text, extra) for other in overlapping)


def _rect_iou(a: OCRLine, b: OCRLine) -> float:
    ax1, ay1, ax2, ay2 = _line_rect(a)
    bx1, by1, bx2, by2 = _line_rect(b)
    intersection = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(0.0, min(ay2, by2) - max(ay1, by1))
    union = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1) + max(0.0, bx2 - bx1) * max(0.0, by2 - by1) - intersection
    return intersection / union if union else 0.0


def _same_text_region(a: OCRLine, b: OCRLine) -> bool:
    ax1, ay1, ax2, ay2 = _line_rect(a)
    bx1, by1, bx2, by2 = _line_rect(b)
    x_overlap = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    smaller_width = max(1.0, min(ax2 - ax1, bx2 - bx1))
    a_center, b_center = (ay1 + ay2) * 0.5, (by1 + by2) * 0.5
    max_height = max(ay2 - ay1, by2 - by1, 1.0)
    return x_overlap / smaller_width >= 0.50 and abs(a_center - b_center) <= max_height * 0.85


def _duplicate_reading_of(line: OCRLine, other: OCRLine) -> bool:
    """Whether one row is a second recognizer's reading of the other.

    Two recognition passes over the same page return the same printed row
    twice, and the weaker reading has to go. That is a property of engines
    that run one pass per language: Google Document AI detects the language of
    each line inside a single pass, so two of its rows are two pieces of text,
    never two readings of one -- the distinction ``_same_language_pass``
    already draws when deciding whether a label and its value may bind.

    Asked without that distinction, the region test answers on how close two
    boxes are, and on a densely set bilingual page the row below is close
    enough. A Bulgarian passport stacks its Cyrillic captions and its Latin
    values half a row apart: the printed issue date 30.11.2021 sat twenty-six
    pixels under the "Date of expiry" caption, was discarded as a duplicate
    reading of it, and the Date of issue field then reached past the gap it
    left and reported the expiry date instead.
    """
    return not _same_language_pass(line, other) and _same_text_region(line, other)


# A passport biodata page carries two long alphanumeric identifiers: the
# document number and the holder's national number. They are not
# interchangeable -- one keys the rental contract, the other is a lifelong
# citizen identifier -- and picking whichever has more digits chose the wrong
# one on the Albanian page, where the passport number BB0075828 carries seven
# and the personal number K00721078T carries eight. The card says which is
# which, in the language it is printed in and in English beside it.
_PERSONAL_NUMBER_LABELS = (
    "PERSONAL NO", "PERSONAL NUMBER", "PERSONAL CODE", "NR PERSONAL",
    "NATIONAL ID", "NATIONAL NUMBER", "IDENTITY NUMBER", "ID NUMBER",
    "الرقم الشخصي", "الرقم الوطني", "ЛИЧНЫЙ НОМЕР",
)

# A Spanish passport's optional personal-number field is numbered ``(11)`` and
# headed ``D.N.I. No.``.  On the reported image OCR lost the ``N.I`` and left
# ``(11) d.No.``; the number below then looked like the longest visible token
# and the generic passport-layout fallback proposed it as a passport number.
# The field number plus the compressed D.N.I. shape make this recognition
# specific to that VIZ row, rather than treating an ordinary ``document no``
# label as a national identity number.
_SPANISH_PASSPORT_DNI_NUMBER_LABEL = re.compile(
    r"^\s*\(?\s*11\s*\)?\s*[.)]?\s*D\s*\.?\s*"
    r"(?:(?:N\s*\.?\s*I\s*\.?\s*)?N\s*(?:O|0)|N\s*(?:O|0))\s*\.?\s*$",
    re.I,
)


def _personal_number_rows(lines: list[OCRLine]) -> tuple[set[int], list[OCRLine]]:
    """Split the personal-number rows off from the document-number search.

    Returns the lines to keep out of that search, and the value lines
    themselves so the number can be stored in the field it belongs to.
    """
    marked: set[int] = set()
    values: list[OCRLine] = []
    for label in lines:
        is_personal_number_label = any(
            label_pattern(name).search(label.text) for name in _PERSONAL_NUMBER_LABELS
        ) or _SPANISH_PASSPORT_DNI_NUMBER_LABEL.fullmatch(label.text) is not None
        if not label.bounding_box or not is_personal_number_label:
            continue
        marked.add(id(label))
        left, top, right, bottom = _line_rect(label)
        height = max(1.0, bottom - top)
        # A label box on a passport is set in two languages and runs taller
        # than the value beneath it, so the value's top edge can sit slightly
        # above the label's bottom. Requiring a positive gap missed it.
        below = [
            line for line in lines
            if line.bounding_box and _line_rect(line)[1] > top
            and -0.5 * height <= _line_rect(line)[1] - bottom <= 1.5 * height
            and min(right, _line_rect(line)[2]) - max(left, _line_rect(line)[0]) > 0
        ]
        if below:
            value = min(below, key=lambda line: _line_rect(line)[1])
            marked.add(id(value))
            values.append(value)
    return marked, values


# A passport is issued for at most ten years, so a printed date more than
# eleven years before the expiry the zone proves is not this passport's issue
# date. The extra year is slack for a state that dates the document before it
# hands it over.
_MAX_PASSPORT_VALIDITY_YEARS = 11
# The Arabic form is admitted here and not in the generic scanner: it is the
# spelling an Arabic biodata page uses for the one date its zone omits, and the
# row it sits in may be a two-column join carrying other text. Bracketing by the
# two dates the zone proves is what makes reading it from such a row safe.
_PASSPORT_PRINTED_DATE_PATTERN = (
    rf"(?:{DATE_PATTERN}|(?<!\d)\d{{2}}\s+\d{{2}}\s+\d{{2}}(?!\d)"
    rf"|{ARABIC_MONTH_DATE_PATTERN})"
)


# A printed date can arrive as two boxes: the Uzbek passport in this project's
# bug report returned "26 01" and "2021" side by side on the issue row, and
# neither half is a date. The pieces are rejoined only where they sit on one
# baseline within a character or so of each other, which is what one printed
# row looks like after a recogniser has split it.
_SPLIT_ROW_MAXIMUM_GAP = 1.6


def _rejoined_date_rows(lines: list[OCRLine]) -> list[OCRLine]:
    """Rows a recogniser cut through the middle of a date, put back together.

    A value bound to its caption has to look like the field it fills, and half
    a date does not. The Uzbek passport in this project's bug report returned
    "26 01" and "2021" as two boxes on the issue row, so the row under "DATE
    OF ISSUE" stated nothing the date filter would accept, the search carried
    on down the column, and the issue field was filled with the expiry printed
    two rows below it -- a passport reported as issued on the day it expires.

    Only a split that a date falls out of is rejoined: where either half is
    already a date, the row was never cut and joining it would invent a second
    reading of a value the page states once.
    """
    rejoined: list[OCRLine] = []
    boxed = [line for line in lines if line.bounding_box]
    for left in boxed:
        left_rect = _line_rect(left)
        left_height = max(1.0, left_rect[3] - left_rect[1])
        if re.search(DATE_PATTERN, left.text, re.I):
            continue
        for right in boxed:
            if right is left or right.variant != left.variant:
                continue
            if re.search(DATE_PATTERN, right.text, re.I):
                continue
            right_rect = _line_rect(right)
            right_height = max(1.0, right_rect[3] - right_rect[1])
            gap = right_rect[0] - left_rect[2]
            if gap < 0 or gap > _SPLIT_ROW_MAXIMUM_GAP * min(
                left_height, right_height,
            ):
                continue
            offset = abs(
                (left_rect[1] + left_rect[3]) - (right_rect[1] + right_rect[3])
            ) * 0.5
            if offset > 0.5 * min(left_height, right_height):
                continue
            text = f"{left.text} {right.text}"
            if re.search(DATE_PATTERN, text, re.I) is None:
                continue
            top = min(left_rect[1], right_rect[1])
            bottom = max(left_rect[3], right_rect[3])
            rejoined.append(OCRLine(
                text=text,
                confidence=min(left.confidence, right.confidence),
                bounding_box=[
                    [left_rect[0], top], [right_rect[2], top],
                    [right_rect[2], bottom], [left_rect[0], bottom],
                ],
                language=left.language,
                variant=left.variant,
                model_name=left.model_name,
            ))
    return rejoined


def _same_row_joins(lines: list[OCRLine]) -> list[tuple[str, OCRLine]]:
    """Neighbouring boxes on one baseline, joined back into the row they were."""
    boxed = [line for line in lines if line.bounding_box]
    joins: list[tuple[str, OCRLine]] = []
    for left in boxed:
        left_rect = _line_rect(left)
        left_height = max(1.0, left_rect[3] - left_rect[1])
        for right in boxed:
            if right is left:
                continue
            right_rect = _line_rect(right)
            right_height = max(1.0, right_rect[3] - right_rect[1])
            gap = right_rect[0] - left_rect[2]
            if gap < 0 or gap > _SPLIT_ROW_MAXIMUM_GAP * min(
                left_height, right_height,
            ):
                continue
            offset = abs(
                (left_rect[1] + left_rect[3]) - (right_rect[1] + right_rect[3])
            ) * 0.5
            if offset > 0.5 * min(left_height, right_height):
                continue
            joins.append((
                f"{left.text} {right.text}",
                left if left.confidence <= right.confidence else right,
            ))
    return joins


def passport_issue_date_from_mrz(
    lines: list[OCRLine], source: str, mrz: ParsedMRZ | None,
) -> list[FieldCandidate]:
    """The issue date a passport prints but its machine-readable zone omits.

    ICAO 9303 gives the zone a birth date and an expiry date, each carrying its
    own check digit, and no issue date at all. That field exists only in the
    printed rows above, labelled in the issuing state's language, in the
    smallest type on the page. On the Austrian passport in this project's bug
    report the label is grey 5pt over a guilloche, and losing it lost all three
    dates at once -- the two the zone had already proven along with the one it
    had not.

    Which leaves a subtraction rather than a guess. Two of the three dates are
    known and arithmetically confirmed, so a third printed date falling between
    them is the row the page labels issue: a biodata page prints no other date.

    Where two survive the filter nothing is claimed. Two candidates mean a
    misread digit somewhere, and a wrong issue date is worse than an empty one.

    The subtraction rests on the birth and expiry check digits, so those are
    what it asks for. Demanding a wholly valid zone asked for more: the same
    Austrian passport returned its zone with the two printed digits that close
    it read as filler, which fails the optional-data and composite checks while
    leaving the birth date, the expiry date and the document number each proven
    by its own digit. The one field on the page the zone cannot supply was
    refused over two characters that say nothing about either date it uses.
    """
    if mrz is None:
        return []
    if not mrz.valid and not (
        mrz.checks.get("date_of_birth") and mrz.checks.get("expiry_date")
    ):
        return []
    birth = mrz.fields.get("date_of_birth")
    expiry = mrz.fields.get("expiry_date")
    if not birth or not expiry or birth >= expiry:
        return []
    earliest = f"{int(expiry[:4]) - _MAX_PASSPORT_VALIDITY_YEARS}{expiry[4:]}"
    today = date.today().isoformat()
    found: dict[str, tuple[OCRLine, tuple[str, ...]]] = {}
    readings: list[tuple[str, OCRLine]] = [(line.text, line) for line in lines]
    # Two halves of one printed row are one date; the bracket below is what
    # makes joining them safe, since a join that spells anything other than a
    # date between two zone-proven ones is discarded like any other reading.
    readings.extend(_same_row_joins(lines))
    for text, line in readings:
        # The zone itself repeats both proven dates in its own encoding; only
        # the printed rows above are being read here.
        if "<" in text:
            continue
        # Non-overlapping, so "14.06.2017" is one date rather than that and the
        # "4.06.2017" inside it. Two readings of one printed row look like two
        # candidates, and the rule below would decline to name either.
        for match in re.finditer(
            _PASSPORT_PRINTED_DATE_PATTERN,
            split_run_together_dates(text), re.I,
        ):
            normalized = normalize_date(match.group(0), day_first_hint=True)
            value = normalized.value
            if value is None or not birth < value < expiry:
                continue
            if value > today or value < earliest:
                continue
            if value not in found or line.confidence > found[value][0].confidence:
                found[value] = (line, normalized.warnings)
    if len(found) > 1:
        return []

    inferred_zero_day = False
    if not found:
        # On one Algerian passport the two-column OCR joined the place-of-birth
        # row to the issue-date value and dropped the narrow leading ``3`` from
        # ``30 Jan 2024``.  The surviving text was ``0Jan2024``.  Do not turn a
        # partial date into a free guess: try only days whose visible last digit
        # is zero and accept one only when the MRZ-proven expiry is exactly one
        # day before that date's anniversary.  That relationship identifies a
        # single printed value; otherwise the field remains empty.
        zero_day = re.compile(
            r"(?<!\d)0\s*[.\-/ ]*"
            r"(JAN(?:UARY)?|FEB(?:RUARY)?|MAR(?:CH)?|APR(?:IL)?|MAY|"
            r"JUN(?:E)?|JUL(?:Y)?|AUG(?:UST)?|SEP(?:T(?:EMBER)?)?|"
            r"OCT(?:OBER)?|NOV(?:EMBER)?|DEC(?:EMBER)?)"
            r"\s*[.\-/ ]*(\d{4})(?!\d)",
            re.I,
        )
        expiry_date = date.fromisoformat(expiry)
        repaired: dict[str, OCRLine] = {}
        for line in lines:
            if "<" in line.text:
                continue
            for match in zero_day.finditer(ascii_numerals(line.text)):
                year = int(match.group(2))
                validity_years = expiry_date.year - year
                # This stricter reconstruction path needs a standard whole-year
                # validity period; the broader eleven-year window above is only
                # slack for filtering complete printed dates.
                if validity_years not in {1, 2, 3, 5, 10}:
                    continue
                for day in (10, 20, 30):
                    normalized = normalize_date(
                        f"{day} {match.group(1)} {year}", day_first_hint=True,
                    )
                    if normalized.value is None:
                        continue
                    issue_date = date.fromisoformat(normalized.value)
                    try:
                        anniversary = issue_date.replace(
                            year=issue_date.year + validity_years,
                        )
                    except ValueError:
                        continue
                    if anniversary - timedelta(days=1) != expiry_date:
                        continue
                    repaired[normalized.value] = line
        if len(repaired) == 1:
            value, line = next(iter(repaired.items()))
            found[value] = (line, ())
            inferred_zero_day = True

    absent_day = False
    if not found:
        # Sometimes the day is not misread but gone: glare across the Algerian
        # booklet's issue row left "أفريل 2019", a month and a year and nothing
        # in front of them. A month and a year are not a date and must never be
        # presented as one -- but the zone proves the expiry, and a passport
        # issued for a whole term expires the day before its own anniversary.
        # That arithmetic names one day, and it is only accepted when the day
        # it names falls in the month and year the page actually prints.
        expiry_date = date.fromisoformat(expiry)
        anchor = expiry_date + timedelta(days=1)
        reconstructed: dict[str, OCRLine] = {}
        for line in lines:
            if "<" in line.text:
                continue
            named = month_and_year(line.text)
            if named is None:
                continue
            month, year = named
            validity_years = anchor.year - year
            if validity_years not in {1, 2, 3, 5, 10}:
                continue
            try:
                issued = anchor.replace(year=year)
            except ValueError:                     # 29 February
                continue
            if (issued.month, issued.year) != (month, year):
                continue
            if not birth < issued.isoformat() < expiry:
                continue
            if issued.isoformat() > today or issued.isoformat() < earliest:
                continue
            reconstructed[issued.isoformat()] = line
        if len(reconstructed) == 1:
            value, line = next(iter(reconstructed.items()))
            found[value] = (line, ())
            absent_day = True

    collapsed_row = False
    if not found:
        # A row can lose its month word altogether. This Algerian booklet
        # printed "19 ماي 2021" beside a caption that OCR merged with the
        # column to its left, and what came back was "Line192021": a day and a
        # year run together, with nothing between them.
        #
        # Digits that close up like that are not a date on their own. What
        # makes them one here is that the page proves the form: the expiry row
        # collapsed the same way, into digits that match the expiry the zone
        # has already checksummed. Once the page is shown to write its dates
        # this way, the run naming the day and year of a whole passport term
        # before that expiry is the issue row, and nothing else on the page
        # reads as one.
        expiry_date = date.fromisoformat(expiry)
        anchor = expiry_date + timedelta(days=1)
        printed = [line for line in lines if "<" not in line.text]

        def collapsed(value: date) -> re.Pattern[str]:
            """How this page would print that date once a glyph was lost.

            Two ways, and the page has to demonstrate whichever it used. The
            month word can vanish and leave the day against the year, as the
            Algerian booklet's rows did; or the year can lose its last figure
            at the edge of the print, as this French page's did -- "05 02 203"
            for an expiry the zone proves is the fifth of February 2034.
            """
            gap = r"\s*[.,/-]?\s*"
            day, month = rf"0?{value.day}", rf"0?{value.month}"
            year = str(value.year)
            return re.compile(
                rf"(?<!\d)(?:{day}{gap}{year}|{day}{gap}{month}{gap}{year[:3]})(?!\d)"
            )

        proven = collapsed(expiry_date)
        if any(proven.search(ascii_numerals(line.text)) for line in printed):
            collapsed_rows: dict[str, OCRLine] = {}
            for validity_years in (1, 2, 3, 5, 10):
                try:
                    issued = anchor.replace(year=anchor.year - validity_years)
                except ValueError:                 # 29 February
                    continue
                if not birth < issued.isoformat() < expiry:
                    continue
                if issued.isoformat() > today or issued.isoformat() < earliest:
                    continue
                pattern = collapsed(issued)
                for line in printed:
                    if pattern.search(ascii_numerals(line.text)):
                        collapsed_rows[issued.isoformat()] = line
                        break
            if len(collapsed_rows) != 1:
                return []
            value, line = next(iter(collapsed_rows.items()))
            found[value] = (line, ())
            collapsed_row = True

    if not found:
        return []

    value, (line, date_warnings) = next(iter(found.items()))
    return [FieldCandidate(
        field_path="passport.issue_date", value=value, normalized_value=value,
        source_document=source, source_method="document_parser",
        confidence=min(0.88, line.confidence * 0.94),
        evidence_text=line.text, bounding_box=line.bounding_box,
        validation_passed=True,
        warnings=[
            "PASSPORT_ISSUE_DATE_BETWEEN_MRZ_PROVEN_DATES",
            *(["PASSPORT_ISSUE_DAY_GLYPH_RECOVERED_FROM_EXPIRY_RELATION"] if inferred_zero_day else []),
            *(["PASSPORT_ISSUE_DAY_ABSENT_RECOVERED_FROM_EXPIRY_RELATION"] if absent_day else []),
            *(["PASSPORT_ISSUE_ROW_COLLAPSED_CONFIRMED_BY_EXPIRY_ROW"] if collapsed_row else []),
            *date_warnings,
        ],
    )]


# What a passport prints in its nationality row: the adjective, not the country.
# "CANADIAN/CANADIENNE" is what the Canadian booklet carries, and the ISO
# normaliser knows only "CANADA", so the row read at 0.9991 named nothing.
#
# Both halves of the bilingual pair are listed where a state prints one, and
# the feminine form too, because that is the half OCR sometimes returns cleanly.
# The list covers the countries whose licence this business accepts and the
# common tourist origins; anything outside it falls back to the state that
# issued the booklet, which a passport's nationality row almost always repeats.
_DEMONYMS: dict[str, str] = {
    "CANADIAN": "Canada", "CANADIENNE": "Canada", "CANADIEN": "Canada",
    "AMERICAN": "United States", "BRITISH": "United Kingdom",
    "BRITISH CITIZEN": "United Kingdom", "AUSTRALIAN": "Australia",
    "NEW ZEALANDER": "New Zealand", "IRISH": "Ireland", "FRENCH": "France",
    "FRANCAISE": "France", "FRANÇAISE": "France", "FRANCAIS": "France",
    "GERMAN": "Germany", "DEUTSCH": "Germany", "DEUTSCHE": "Germany",
    "ITALIAN": "Italy", "ITALIANA": "Italy", "SPANISH": "Spain",
    "ESPANOLA": "Spain", "ESPAÑOLA": "Spain", "PORTUGUESE": "Portugal",
    "PORTUGUESA": "Portugal", "DUTCH": "Netherlands",
    "NEDERLANDSE": "Netherlands", "BELGIAN": "Belgium", "BELGE": "Belgium",
    "SWISS": "Switzerland", "SUISSE": "Switzerland", "AUSTRIAN": "Austria",
    "SWEDISH": "Sweden", "SVENSK": "Sweden", "NORWEGIAN": "Norway",
    "DANISH": "Denmark", "FINNISH": "Finland", "POLISH": "Poland",
    "GREEK": "Greece", "ROMANIAN": "Romania", "ROMANA": "Romania",
    "JAPANESE": "Japan", "KOREAN": "South Korea", "CHINESE": "China",
    "INDIAN": "India", "SINGAPOREAN": "Singapore",
    "SOUTH AFRICAN": "South Africa", "TURKISH": "Turkey", "TURK": "Turkey",
    "EMIRATI": "United Arab Emirates", "SAUDI": "Saudi Arabia",
    "QATARI": "Qatar", "KUWAITI": "Kuwait", "BAHRAINI": "Bahrain",
    "OMANI": "Oman", "RUSSIAN": "Russia", "UKRAINIAN": "Ukraine",
    "BRAZILIAN": "Brazil", "BRASILEIRA": "Brazil", "MEXICAN": "Mexico",
    "ARGENTINE": "Argentina", "ARGENTINA": "Argentina",
}


def _nationality_from_demonym(text: str) -> str | None:
    """The country a printed nationality row names, adjective or not.

    A passport sets the row bilingually -- "CANADIAN/CANADIENNE",
    "ALLEMANDE/GERMAN" -- so each half is tried on its own.
    """
    for part in re.split(r"[/|,]", fold_for_match(text)):
        cleaned = " ".join(part.split())
        if not cleaned:
            continue
        if cleaned in _DEMONYMS:
            return _DEMONYMS[cleaned]
        # Spelled out only. The ISO normaliser answers two-letter codes as
        # readily as country names, and a passport page is littered with
        # two-letter fragments: the stray box "PE", returned at 0.22 from the
        # top margin of a Canadian booklet, is a valid alpha-2 code and named
        # the holder a Peruvian.
        if len(cleaned) < 4:
            continue
        named = normalize_country(cleaned)[1]
        if named:
            return named
    return None


# The sex row is three characters of label and one of value, and on a worn page
# neither survives cleanly: "Sex/Sexe" came back as "Sax/Se". Three characters
# with one wrong is 0.67 similar, far below the threshold a long label is held
# to -- but the field's whole vocabulary is M, F and X, so a loose label cannot
# produce a loose value. The worst a wrong row can do here is offer a letter
# that is not one of those three, and then nothing is bound at all.
_SEX_LABELS = ("SEX", "SEXE", "SESSO", "SEXO", "GESCHLECHT", "GENDER", "ПОЛ")
_SEX_LABEL_RATIO = 0.6
# Tajik and a number of other bilingual passports repeat the one-letter value
# on both sides of the slash (``M/M`` or ``F/F``).  It is still one value, not
# an ambiguous pair, so accept it as the equivalent of the single-letter row.
_SEX_VALUE = re.compile(r"^\s*([MFX])(?:\s*/\s*\1)?\s*$", re.I)


def passport_sex_and_nationality(
    lines: list[OCRLine], source: str, issuing_country: str | None = None,
) -> list[FieldCandidate]:
    """Read the two rows a worn passport gives up last.

    Both are inferred rather than read off a label bound to a value, so both
    carry a warning and a confidence that puts them in front of a person. An
    operator correcting a suggested value is a second of work; an empty field
    is a trip back to the customer for a document already in the system.
    """
    candidates: list[FieldCandidate] = []
    # The best-read row that names a country, not the first one encountered. A
    # page carries fragments the recogniser itself doubts, and one of them
    # naming a country is not evidence of the holder's.
    named_rows = [
        (line, named) for line, named in (
            (line, _nationality_from_demonym(line.text)) for line in lines
            if line.confidence >= 0.60
        ) if named is not None
    ]
    if named_rows:
        line, named = max(named_rows, key=lambda item: item[0].confidence)
        candidates.append(FieldCandidate(
            field_path="personal_info.nationality_name",
            value=line.text.strip(), normalized_value=named,
            source_document=source, source_method="document_parser",
            confidence=min(0.76, line.confidence * 0.78),
            evidence_text=line.text, bounding_box=line.bounding_box,
            validation_passed=True,
            warnings=["NATIONALITY_FROM_PRINTED_DEMONYM_REQUIRES_REVIEW"],
        ))
    if not candidates and issuing_country:
        # A passport's nationality row repeats the state that issued it on all
        # but the rare travel document written for someone else's national.
        named = normalize_country(issuing_country)[1]
        if named:
            candidates.append(FieldCandidate(
                field_path="personal_info.nationality_name",
                value=named, normalized_value=named,
                source_document=source, source_method="document_parser",
                confidence=0.70, evidence_text=issuing_country,
                bounding_box=None, validation_passed=True,
                warnings=["NATIONALITY_ASSUMED_FROM_ISSUING_STATE_REQUIRES_REVIEW"],
            ))
    best: tuple[float, OCRLine] | None = None
    for line in lines:
        compact = compact_label(line.text)
        if not compact or len(compact) > 8:
            continue
        # A Cyrillic recognition pass can retain the Tajik label while the
        # English half reads as ``Sех`` with Cyrillic lookalikes. Folding makes
        # that visible ``SEX`` label reliable; the compact prefix otherwise
        # starts with the Tajik word and would never be compared with it.
        folded = fold_for_match(line.text)
        if re.search(r"(?<![A-Z])(?:SEX|SEXE|SESSO|SEXO|GENDER)(?![A-Z])", folded):
            ratio = 1.0
        else:
            ratio = max(
                SequenceMatcher(None, compact[:len(label)], label).ratio()
                for label in _SEX_LABELS
            )
        if ratio >= _SEX_LABEL_RATIO and (best is None or ratio > best[0]):
            best = (ratio, line)
    if best is not None:
        label = best[1]
        left, top, right, bottom = _line_rect(label)
        height = max(1.0, bottom - top)
        nearest: tuple[float, OCRLine, str] | None = None
        for line in lines:
            match = _SEX_VALUE.match(line.text)
            if line is label or match is None or not line.bounding_box:
                continue
            value_left, value_top, _, value_bottom = _line_rect(line)
            if min(bottom, value_bottom) - max(top, value_top) <= 0:
                # The value sits on the label's own row or the one under it.
                if not top < value_top < bottom + 2.0 * height:
                    continue
            distance = abs(value_left - right) + abs(value_top - top)
            if nearest is None or distance < nearest[0]:
                nearest = (distance, line, match.group(1).upper())
        if nearest is not None:
            candidates.append(FieldCandidate(
                field_path="personal_info.gender",
                value=nearest[2], normalized_value=nearest[2],
                source_document=source, source_method="document_parser",
                confidence=min(0.76, nearest[1].confidence * 0.78),
                evidence_text=f"{label.text} {nearest[2]}",
                bounding_box=nearest[1].bounding_box, validation_passed=True,
                warnings=["SEX_FROM_RESEMBLING_LABEL_REQUIRES_REVIEW"],
            ))
    return candidates


def _passport_layout_candidates(lines: list[OCRLine], source: str) -> list[FieldCandidate]:
    """Recover standardized passport rows when tiny labels are not readable."""
    candidates: list[FieldCandidate] = []
    dated: list[tuple[float, float, float, OCRLine, str]] = []
    for line in lines:
        match = re.search(DATE_PATTERN, line.text, re.I)
        if match is None:
            continue
        normalized = normalize_date(match.group(0), day_first_hint=True)
        if normalized.value is None:
            continue
        x1, y1, x2, y2 = _line_rect(line)
        dated.append(((y1 + y2) * 0.5, (x1 + x2) * 0.5, max(y2 - y1, 1.0), line, normalized.value))
    # Russian and many bilingual passports place issue and expiry on one final
    # row, with date/place of birth on an earlier row.
    if len(dated) >= 3:
        bottom_y = max(item[0] for item in dated)
        bottom_height = max(item[2] for item in dated if item[0] == bottom_y)
        bottom = [item for item in dated if abs(item[0] - bottom_y) <= max(12.0, bottom_height * 1.2)]
        earlier = [item for item in dated if item not in bottom]
        # This fallback describes passports that print issue and expiry beside
        # one another on the same bottom row.  A Greek passport stacks them in
        # one x-column instead.  Its two boxes overlap vertically enough to
        # enter the generous row band above, and choosing min/max x (their
        # centres differ by only a pixel) swapped the labelled issue date for
        # the expiry.  Require the horizontal separation the fallback claims.
        bottom_centers = sorted(item[1] for item in bottom)
        horizontally_separated = (
            len(bottom_centers) >= 2
            and bottom_centers[-1] - bottom_centers[0] > 80.0
        )
        if len(bottom) >= 2 and earlier and horizontally_separated:
            issue = min(bottom, key=lambda item: item[1])
            expiry = max(bottom, key=lambda item: item[1])
            birth = max(earlier, key=lambda item: item[0])
            for path, item in (
                ("personal_info.date_of_birth", birth),
                ("passport.issue_date", issue),
                ("passport.expiry_date", expiry),
            ):
                _, _, _, line, value = item
                candidates.append(_candidate(
                    path, value, line, source, True, value,
                    ["PASSPORT_STANDARD_LAYOUT_FALLBACK"], 0.94,
                ))
    # Tunisia's biodata page is a vertical form: the date of birth, date of
    # issue and date of expiry sit in the same narrow column on separate rows.
    # The generic fallback above is deliberately horizontal because that is
    # what Russian and many bilingual passports print.  Applying it to the
    # Tunisian form leaves three clear dates unbound. Scope the vertical
    # sequence to the official heading and one aligned date column, so a
    # random three-date passport or a date table can never borrow it.
    tunisian_heading = " ".join(fold_for_match(
        " ".join(line.text for line in lines),
    ).split())
    is_tunisian_passport = (
        "REPUBLIQUE TUNISIENNE" in tunisian_heading
        or "REPUBLIC OF TUNISIA" in tunisian_heading
        or "الجمهورية التونسية" in tunisian_heading
    )
    if is_tunisian_passport and len(dated) >= 3:
        # One value may be read in more than one repair view. Keep the most
        # confident geometry for each date before testing the printed order.
        by_value: dict[str, tuple[float, float, float, OCRLine, str]] = {}
        for item in dated:
            current = by_value.get(item[4])
            if current is None or item[3].confidence > current[3].confidence:
                by_value[item[4]] = item
        vertical = sorted(by_value.values(), key=lambda item: item[0])
        if len(vertical) == 3:
            birth, issued, expiry = vertical
            birth_y, birth_x, birth_height, _, birth_value = birth
            # All three dates occupy the same column, top-to-bottom. On the
            # Tunisian form the nearby English text may be ``Place of birth``
            # rather than the date caption, so do not mistake the label OCR
            # gap for absent document evidence.
            aligned = all(
                abs(item[1] - birth_x) <= max(130.0, birth_height * 5.0)
                for item in vertical
            )
            issue_value, expiry_value = issued[4], expiry[4]
            if (
                aligned
                and birth_value < issue_value < expiry_value
                and int(issue_value[:4]) - int(birth_value[:4]) >= 14
            ):
                for path, item in (
                    ("personal_info.date_of_birth", birth),
                    ("passport.issue_date", issued),
                    ("passport.expiry_date", expiry),
                ):
                    _, _, _, line, value = item
                    candidates.append(_candidate(
                        path, value, line, source, True, value,
                        ["TUNISIAN_PASSPORT_VERTICAL_DATE_LAYOUT"], 0.84,
                    ))
    number_options: list[tuple[float, OCRLine, str]] = []
    personal, personal_values = _personal_number_rows(lines)
    for line in personal_values:
        token = next(
            iter(re.findall(r"(?<![A-Z0-9])[A-Z0-9]{6,15}(?![A-Z0-9])", line.text.upper())),
            None,
        )
        if token is not None and any(character.isdigit() for character in token):
            candidates.append(_candidate(
                "passport.holder_id", token, line, source, True, token,
                ["LABELLED_PERSONAL_NUMBER"], 0.95,
            ))
    for line in lines:
        if re.search(DATE_PATTERN, line.text, re.I) or id(line) in personal:
            continue
        for token in re.findall(r"(?<![A-Z0-9])[A-Z0-9]{7,12}(?![A-Z0-9])", line.text.upper()):
            digits = sum(char.isdigit() for char in token)
            if digits >= 6 and not token.isalpha():
                number_options.append((line.confidence, line, token))
    if number_options:
        _, line, token = max(number_options, key=lambda item: (item[0], sum(char.isdigit() for char in item[2])))
        candidates.append(_candidate(
            "passport.number", token, line, source, True, token,
            ["VISIBLE_PASSPORT_NUMBER_PATTERN"], 0.95,
        ))
    return candidates


def _tajik_passport_name_candidates(
    lines: list[OCRLine], source: str,
) -> list[FieldCandidate]:
    """Recover the bilingual identity rows of a Tajik passport.

    The printed Tajik surname caption is often returned as ``Sumame`` and the
    given-name caption ends in the OCR hybrid ``Nате``.  Their values remain
    sharp in a shared x-column beneath them.  This country-scoped geometry
    avoids treating the nearby ``Passport`` label or the patronymic as a name.
    """
    page_text = fold_for_match(" ".join(line.text for line in lines))
    if "TAJIKISTAN" not in page_text or not any(
        line.text.strip().upper() == "TJK" for line in lines
    ):
        return []

    def below_value(label: OCRLine, *, latin_only: bool = False) -> OCRLine | None:
        left, top, right, bottom = _line_rect(label)
        height = max(bottom - top, 1.0)
        choices: list[tuple[float, OCRLine]] = []
        for line in lines:
            if line is label or not line.bounding_box:
                continue
            value = _clean_person_name(line.text)
            if not _plausible_person_name(value) or len(value.split()) > 4:
                continue
            vleft, vtop, vright, vbottom = _line_rect(line)
            if vtop < bottom - height * 0.15 or vtop > bottom + height * 4.0:
                continue
            if min(right, vright) - max(left, vleft) <= 0:
                continue
            # The Tajik Cyrillic line is often tagged ``ru`` together with the
            # Latin transliteration directly below it, so the OCR language tag
            # alone cannot tell which printed value is the Latin field value.
            is_latin_value = bool(re.search(r"[A-Za-z]", line.text)) and not bool(
                re.search(r"[А-Яа-яЁёӢӣҶҷҲҳҚқҒғӮӯӮӯ]", line.text)
            )
            latin_bonus = 80.0 if latin_only and is_latin_value else 0.0
            score = (vtop - bottom) + abs(vleft - left) * 0.10 - latin_bonus
            choices.append((score, line))
        return min(choices, key=lambda item: item[0])[1] if choices else None

    def nationality_value(label: OCRLine) -> OCRLine | None:
        """Return the country row directly below Tajikistan's nationality label."""
        left, top, right, bottom = _line_rect(label)
        height = max(bottom - top, 1.0)
        choices: list[OCRLine] = []
        for line in lines:
            if line is label or not line.bounding_box:
                continue
            value_left, value_top, value_right, _value_bottom = _line_rect(line)
            if value_top < bottom - height * 0.15 or value_top > bottom + height * 4.0:
                continue
            if min(right, value_right) - max(left, value_left) <= 0:
                continue
            compact = re.sub(r"[^A-Z0-9]", "", fold_for_match(line.text))
            tajik_cyrillic = "ТОЧ" in line.text.upper() and "ИСТОН" in line.text.upper()
            # OCR commonly produces ``TAJIISTAN`` or loses the K in this row.
            # The row is already bound to the Tajik nationality caption and
            # the document heading/code have established the passport's state.
            tajik_latin = "TAJ" in compact and "STAN" in compact
            if tajik_cyrillic or tajik_latin:
                choices.append(line)
        return max(choices, key=lambda line: line.confidence) if choices else None

    surname_label = next((
        line for line in lines
        if "SUMAME" in fold_for_match(line.text) or "SURNAME" in fold_for_match(line.text)
    ), None)
    given_label = next((
        line for line in lines
        if "НОМ ВА НОМИ ПАДАР" in line.text.upper()
    ), None)
    candidates: list[FieldCandidate] = []
    for path, label, latin_only in (
        ("personal_info.last_name", surname_label, True),
        ("personal_info.first_name", given_label, True),
    ):
        if label is None or (value_line := below_value(label, latin_only=latin_only)) is None:
            continue
        value = _clean_person_name(value_line.text)
        candidates.append(FieldCandidate(
            field_path=path, value=value, normalized_value=value,
            source_document=source, source_method="document_parser",
            confidence=min(0.92, value_line.confidence * 0.94),
            evidence_text=value_line.text, bounding_box=value_line.bounding_box,
            validation_passed=True,
            warnings=["TAJIK_PASSPORT_BILINGUAL_NAME_LAYOUT"],
        ))
    names = {
        candidate.field_path: candidate
        for candidate in candidates
        if candidate.field_path in {
            "personal_info.first_name", "personal_info.last_name",
        }
    }
    if {
        "personal_info.first_name", "personal_info.last_name",
    } <= names.keys():
        first = names["personal_info.first_name"]
        last = names["personal_info.last_name"]
        full_name = f"{first.normalized_value} {last.normalized_value}"
        candidates.append(FieldCandidate(
            field_path="personal_info.full_name", value=full_name,
            normalized_value=full_name, source_document=source,
            source_method="document_parser",
            confidence=min(first.confidence, last.confidence),
            evidence_text=first.evidence_text, bounding_box=first.bounding_box,
            validation_passed=True,
            warnings=["TAJIK_PASSPORT_BILINGUAL_NAME_LAYOUT"],
        ))
    nationality_label = next((
        line for line in lines
        if "ШАХРВАНД" in line.text.upper()
        or "NATIONALITY" in fold_for_match(line.text)
    ), None)
    if nationality_label and (value_line := nationality_value(nationality_label)) is not None:
        candidates.append(FieldCandidate(
            field_path="personal_info.nationality_name", value="Tajikistan",
            normalized_value="Tajikistan", source_document=source,
            source_method="document_parser",
            confidence=min(0.76, value_line.confidence * 0.82),
            evidence_text=value_line.text, bounding_box=value_line.bounding_box,
            validation_passed=True,
            warnings=["TAJIK_PASSPORT_NATIONALITY_LAYOUT_REQUIRES_REVIEW"],
        ))
    return candidates


def _haitian_passport_name_candidates(
    lines: list[OCRLine], source: str,
) -> list[FieldCandidate]:
    """Read Haiti's ``Siyati/Nom`` and ``Non/Prénom`` name rows safely.

    The very small Creole/French given-name caption is repeatedly returned as
    ``Non/Pranci`` or ``Non1Prac`` while its value remains sharp. Matching each
    damaged spelling globally would turn ordinary text on other passports into
    labels. The official Haiti heading plus the undamaged ``Siyati/Nom`` row
    scopes the recovery to this layout; geometry then accepts only the nearest
    clean name directly beneath each caption.
    """
    folded = [(line, fold_for_match(line.text)) for line in lines]
    if not any("HAITI" in text or "AYITI" in text for _, text in folded):
        return []
    surname_labels = [
        line for line, text in folded
        if re.search(r"^\s*SIYATI\s*/\s*NO[MN]\b", text)
    ]
    if not surname_labels:
        return []
    given_labels = [
        line for line, text in folded
        if re.fullmatch(r"NO[MN][^A-Z0-9]*1?PR[A-Z]{2,8}", text.strip())
    ]
    if not given_labels:
        return []

    blocked_words = frozenset({
        "HAITI", "AYITI", "SIYATI", "NOM", "NON", "PRENOM", "SIGNATURE",
        "PASSEPORT", "PASSPORT", "NATIONALITE", "NATIONALITY",
    })

    def nearest_value(label: OCRLine) -> OCRLine | None:
        left, top, right, bottom = _line_rect(label)
        height = max(1.0, bottom - top)
        label_center = (top + bottom) * 0.5
        choices: list[tuple[float, OCRLine]] = []
        for line in lines:
            if line is label or not line.bounding_box:
                continue
            value_left, value_top, value_right, value_bottom = _line_rect(line)
            value_center = (value_top + value_bottom) * 0.5
            if not label_center < value_center <= bottom + 2.5 * height:
                continue
            if min(right, value_right) - max(left, value_left) <= 0:
                continue
            value = _clean_person_name(line.text)
            words = set(fold_for_match(value).replace("/", " ").split())
            if (
                not _plausible_person_name(value)
                or len(value) > 70
                or len(value.split()) > 4
                or words & blocked_words
                or normalize_country(value)[0] is not None
            ):
                continue
            horizontal = abs(value_left - left)
            score = value_center - label_center + horizontal * 0.15 - line.confidence
            choices.append((score, line))
        return min(choices, key=lambda item: item[0])[1] if choices else None

    candidates: list[FieldCandidate] = []
    for path, labels in (
        ("personal_info.last_name", surname_labels),
        ("personal_info.first_name", given_labels),
    ):
        values = [value for label in labels if (value := nearest_value(label))]
        if not values:
            continue
        line = max(values, key=lambda item: item.confidence)
        value = _clean_person_name(line.text)
        candidates.append(FieldCandidate(
            field_path=path, value=value, normalized_value=value,
            source_document=source, source_method="document_parser",
            confidence=min(0.92, line.confidence * 0.94),
            evidence_text=line.text, bounding_box=line.bounding_box,
            validation_passed=True,
            warnings=["HAITIAN_PASSPORT_NAME_LAYOUT"],
        ))
    return candidates


def _moldovan_passport_layout_candidates(
    lines: list[OCRLine], source: str,
) -> list[FieldCandidate]:
    """Read Moldova's tightly stacked Romanian VIZ fields from their layout.

    The Moldovan passport prints ``Numele``, ``Prenume`` and ``Cetățenia`` in
    a compact vertical column.  Glare commonly changes the final ``m`` in
    ``Prenume`` to ``n`` and runs the nationality caption into its English
    translation, while the values directly underneath stay clear.  Treating
    those damaged captions as names yielded the labels themselves as the
    customer's first and last names.  The national heading scopes this
    recovery to Moldova; geometry then accepts only the immediately-below
    visible value, never a free-text match elsewhere on a passport.
    """
    folded = [(line, fold_for_match(line.text)) for line in lines]
    if not any(
        "REPUBLICAMOLDOVA" in re.sub(r"[^A-Z0-9]", "", text)
        for _, text in folded
    ):
        return []

    surname_labels = [
        line for line, text in folded
        # A glared final ``e`` is often classified as ``c`` (``Numelc``).
        if re.fullmatch(r"NUMEL(?:E|EA|C)", re.sub(r"[^A-Z0-9]", "", text))
    ]
    given_labels = [
        line for line, text in folded
        # OCR's common M -> N error makes the printed ``Prenume`` ``Prenun``.
        if re.fullmatch(r"PRENU[MN][A-Z]{0,6}", re.sub(r"[^A-Z0-9]", "", text))
    ]
    nationality_labels = [
        line for line, text in folded
        # ``Cetățenia/Nationality`` is often joined into a single OCR box.  A
        # one-letter confusion in its Romanian stem is allowed only after the
        # Moldova heading above has established this exact layout.
        if re.match(r"CE[TR]ATEN", re.sub(r"[^A-Z0-9]", "", text))
    ]
    gender_labels = [
        line for line, text in folded
        # The compact bilingual caption ``Sex/Sexe`` can be joined and read
        # as ``Sealsex``. It remains a caption, not a gender value.
        if re.fullmatch(r"(?:SEX|SE[A-Z]{0,5}SEX)", re.sub(r"[^A-Z0-9]", "", text))
    ]

    blocked_words = frozenset({
        "MOLDOVA", "REPUBLICA", "NUMELE", "NUME", "PRENUME", "PRENUN",
        "CETATENIA", "CETATENIE", "NATIONALITATE", "NATIONALITY",
        "PASAPORT", "PASSPORT", "SEMNATURA", "SIGNATURE",
    })

    def nearest_value(
        label: OCRLine, accepts: Callable[[str], bool],
    ) -> OCRLine | None:
        left, top, right, bottom = _line_rect(label)
        height = max(1.0, bottom - top)
        label_center = (top + bottom) * 0.5
        choices: list[tuple[float, OCRLine]] = []
        for line in lines:
            if line is label or not line.bounding_box:
                continue
            value_left, value_top, value_right, value_bottom = _line_rect(line)
            value_center = (value_top + value_bottom) * 0.5
            # The OCR boxes for label and value can overlap by a few pixels,
            # but the value's centre remains below the caption and within two
            # printed rows of it.
            if not label_center < value_center <= bottom + 2.5 * height:
                continue
            # A second recognizer can place a different reading of the
            # caption itself a fraction of a pixel lower. It is not the value
            # below the caption, even if its centre happens to pass the check
            # above. Keep the overlap skewed stacked rows produce, but reject
            # a same-row duplicate.
            if value_top < bottom - 0.6 * height:
                continue
            if min(right, value_right) - max(left, value_left) <= 0:
                continue
            if abs(value_left - left) > max(120.0, (right - left) * 1.5):
                continue
            value = _clean_person_name(line.text)
            compact = re.sub(r"[^A-Z0-9]", "", fold_for_match(value))
            if compact in blocked_words or not accepts(value):
                continue
            # Prefer the closest value under the same left-hand field column;
            # OCR confidence only breaks otherwise equal layout matches.
            score = (value_center - label_center) + abs(value_left - left) * 0.15 - line.confidence
            choices.append((score, line))
        return min(choices, key=lambda item: item[0])[1] if choices else None

    name_accepts = lambda value: (
        _plausible_person_name(value)
        and len(value) <= 70
        and len(value.split()) <= 4
        and normalize_country(value)[0] is None
    )
    def nationality_accepts(value: str) -> bool:
        compact = re.sub(r"[^A-Z0-9]", "", fold_for_match(value))
        return (
            normalize_country(value)[0] is not None
            or compact == "REPUBLICAMOLDOVA"
        )

    def gender_accepts(value: str) -> bool:
        return re.fullmatch(r"\s*(?:M|F|MALE|FEMALE)\s*", value, re.I) is not None

    candidates: list[FieldCandidate] = []
    for path, labels, accepts in (
        ("personal_info.last_name", surname_labels, name_accepts),
        ("personal_info.first_name", given_labels, name_accepts),
        ("personal_info.nationality_name", nationality_labels, nationality_accepts),
        ("personal_info.gender", gender_labels, gender_accepts),
    ):
        values = [value for label in labels if (value := nearest_value(label, accepts))]
        if not values:
            continue
        line = max(values, key=lambda item: item.confidence)
        value = _clean_person_name(line.text)
        if path == "personal_info.nationality_name":
            _, normalized, warnings = normalize_country(value)
            # Romanian uses ``Republica Moldova`` while the country registry
            # indexes the shorter English country name. The national heading
            # and this exact nationality value make that alias unambiguous.
            if normalized is None and re.sub(
                r"[^A-Z0-9]", "", fold_for_match(value),
            ) == "REPUBLICAMOLDOVA":
                _, normalized, warnings = normalize_country("Moldova")
            if normalized is None:
                continue
        elif path == "personal_info.gender":
            value = value.upper()
            normalized = {"MALE": "M", "FEMALE": "F"}.get(value, value)
            warnings = []
        else:
            normalized, warnings = value, []
        candidates.append(FieldCandidate(
            field_path=path, value=value, normalized_value=normalized,
            source_document=source, source_method="document_parser",
            confidence=min(0.92, line.confidence * 0.94),
            evidence_text=line.text, bounding_box=line.bounding_box,
            validation_passed=True,
            warnings=["MOLDOVAN_PASSPORT_VISIBLE_LAYOUT", *warnings],
        ))
    return candidates


def idp_layout_candidates(
    lines: list[OCRLine], source: str,
    red_boxes: list[tuple[float, float, float, float]] | None = None,
) -> list[FieldCandidate]:
    """Recover numeric IDP fields when the booklet's tiny labels are lost.

    `red_boxes` are the page's patches of red print, from
    ``image_processing.red_ink_boxes``. They are optional because this parser
    also runs during classification, before a page is known to be a permit and
    where only the text is in hand.
    """
    candidates: list[FieldCandidate] = []
    date_rows: dict[str, tuple[OCRLine, str]] = {}
    for line in lines:
        if re.search(r"CONVENTION|ROAD\s+TRAFFIC|КОНВЕНЦ", line.text, re.I):
            continue
        # OCR on patterned booklet pages can prefix a valid date with zeros
        # (0001.08.2029) or change a separator to a comma. Examine overlapping
        # date substrings and keep only plausible calendar years.
        date_text = split_run_together_dates(re.sub(r"(?<=\d),(?=\d)", ".", line.text))
        tokens = [
            match.group(1)
            for match in re.finditer(rf"(?=({DATE_PATTERN}))", date_text, re.I)
        ]
        for match in _IDP_STAMPED_DATE.finditer(date_text):
            day = re.sub(r"\D", "", match.group("day"))
            tokens.append(f"{day} {match.group('month')} {match.group('year')}")
        tokens.extend(handwritten_date_values(date_text))
        plausible_tokens: list[tuple[str, str]] = []
        for token in tokens:
            normalized = normalize_date(token, day_first_hint=True)
            if normalized.value and 1900 <= int(normalized.value[:4]) <= 2100:
                plausible_tokens.append((token, normalized.value))
        if plausible_tokens:
            # Prefer 14.11.2003 over its overlapping suffix 4.11.2003.
            _, value = max(plausible_tokens, key=lambda item: len(item[0]))
            current = date_rows.get(value)
            if current is None or line.confidence > current[0].confidence:
                date_rows[value] = (line, value)
    ordered_dates = sorted(date_rows)
    if len(ordered_dates) >= 3:
        birth_value, issue_value, expiry_value = ordered_dates[0], ordered_dates[-2], ordered_dates[-1]
        birth_year, issue_year = int(birth_value[:4]), int(issue_value[:4])
        if issue_year - birth_year >= 14 and issue_value <= expiry_value:
            for path, value in (
                ("personal_info.date_of_birth", birth_value),
                ("international_driving_permit.issue_date", issue_value),
                ("international_driving_permit.expiry_date", expiry_value),
            ):
                line, _ = date_rows[value]
                candidates.append(FieldCandidate(
                    field_path=path, value=value, normalized_value=value,
                    source_document=source, source_method="document_parser",
                    confidence=min(0.82, line.confidence * 0.88),
                    evidence_text=line.text, bounding_box=line.bounding_box,
                    validation_passed=True,
                    warnings=["IDP_DATE_SEQUENCE_LAYOUT_FALLBACK"],
                ))
    elif len(ordered_dates) == 2:
        issue_value, expiry_value = ordered_dates
        # When the holder DOB row is unreadable, a modern issue/valid-until
        # pair remains safe to distinguish by chronology. Avoid treating a
        # young holder's older birth year as the issue date.
        if int(issue_value[:4]) >= date.today().year - 20 and issue_value <= expiry_value:
            for path, value in (
                ("international_driving_permit.issue_date", issue_value),
                ("international_driving_permit.expiry_date", expiry_value),
            ):
                line, _ = date_rows[value]
                candidates.append(FieldCandidate(
                    field_path=path, value=value, normalized_value=value,
                    source_document=source, source_method="document_parser",
                    confidence=min(0.80, line.confidence * 0.86),
                    evidence_text=line.text, bounding_box=line.bounding_box,
                    validation_passed=True,
                    warnings=["IDP_TWO_DATE_SEQUENCE_LAYOUT_FALLBACK"],
                ))
    elif len(ordered_dates) == 1 and _idp_delivery_row_present(lines):
        # The 1968 booklet's cover carries "Délivré à <place> le <date>" and no
        # other date at all: the holder's dates and the permit's validity are
        # printed inside. One date beside that wording is the issue date, and
        # the alternative -- reading it as nothing -- left the field empty on a
        # page that states it plainly. It is refused if it has not happened
        # yet, or is older than any permit still in use.
        value = ordered_dates[0]
        line, _ = date_rows[value]
        today = date.today()
        if str(today.year - 15) <= value[:4] and value <= today.isoformat():
            candidates.append(FieldCandidate(
                field_path="international_driving_permit.issue_date",
                value=value, normalized_value=value, source_document=source,
                source_method="document_parser",
                confidence=min(0.80, line.confidence * 0.86),
                evidence_text=line.text, bounding_box=line.bounding_box,
                validation_passed=True,
                warnings=["IDP_COVER_DELIVERY_ROW_ISSUE_DATE"],
            ))
    candidates.extend(idp_number_candidates(lines, source, red_boxes=red_boxes))
    return candidates


# What the cover says when it records where and when the booklet was handed
# over: the French wording of the 1968 model and the Arabic beside it.
_IDP_DELIVERY_WORDING = (
    "DELIVRE A", "DELIVRE LE", "DELIVREE LE", "ISSUED AT",
    # The Arabic beside it, and the two spellings a recogniser returns for the
    # first of them once the dal is read as a waw.
    "صدرت في", "صورت في", "بتاريخ", "بتارخ", "سلمت في",
)


def _idp_delivery_row_present(lines: list[OCRLine]) -> bool:
    return any(
        wording in fold_for_match(line.text)
        for line in lines for wording in _IDP_DELIVERY_WORDING
    )


def private_international_driver_licence_candidates(
    lines: list[OCRLine], source: str,
) -> list[FieldCandidate]:
    """Read an accepted commercial International Driver's License.

    This is intentionally separate from ``idp_layout_candidates``: the card is
    accepted by this rental's business policy, but it is not a
    1949/1968 treaty IDP and therefore cannot borrow the treaty-booklet
    assumptions about red numbers, convention text or issuing state.
    """
    candidates: list[FieldCandidate] = []
    number_rows: dict[str, OCRLine] = {}
    date_rows: dict[str, OCRLine] = {}
    labelled_dates: dict[str, tuple[str, OCRLine]] = {}
    for line in lines:
        if re.search(r"\bORIGINAL\s*(?:DL|DRIVER'?S?\s*LICEN[CS]E)\b", line.text, re.I):
            # This is the domestic licence number that the translation cites;
            # it is never the association card's own international number.
            continue
        formatted_matches = list(re.finditer(
            r"(?<![A-Z0-9])(\d{2}\s*[A-Z]{2}\s*\d{6})(?![A-Z0-9])",
            line.text.upper(),
        ))
        for match in formatted_matches:
            value = re.sub(r"\s+", " ", match.group(1)).strip()
            current = number_rows.get(value)
            if current is None or line.confidence > current.confidence:
                number_rows[value] = line
        # Do not also capture the digits at the end of a grouped association
        # card number such as ``01 EA 056795``. That would create a second,
        # artificial candidate (``056795``) and make the number look ambiguous.
        for match in re.finditer(r"(?<![A-Z0-9])\d{6,15}(?![A-Z0-9])", line.text.upper()):
            if any(start <= match.start() < end for start, end in (
                formatted.span(1) for formatted in formatted_matches
            )):
                continue
            value = match.group(0)
            current = number_rows.get(value)
            if current is None or line.confidence > current.confidence:
                number_rows[value] = line
        date_text = split_run_together_dates(line.text)
        for match in re.finditer(rf"(?=({DATE_PATTERN}))", date_text, re.I):
            normalized = normalize_date(match.group(1), day_first_hint=True)
            if normalized.value is None:
                continue
            current = date_rows.get(normalized.value)
            if current is None or line.confidence > current.confidence:
                date_rows[normalized.value] = line
            folded = fold_for_match(line.text)
            if re.search(r"\b(?:ISSUED|ISSUE\s+DATE)\b", folded):
                labelled_dates["international_driving_permit.issue_date"] = (
                    normalized.value, line,
                )
            elif re.search(r"\b(?:EXPIRES|EXPIRY\s+DATE|EXPIRATION\s+DATE)\b", folded):
                labelled_dates["international_driving_permit.expiry_date"] = (
                    normalized.value, line,
                )
    # The translation card has one long card identifier. Multiple long runs
    # would be ambiguous (for example, a separate domestic licence number), so
    # the reader leaves that field empty rather than choosing one.
    if len(number_rows) == 1:
        value, line = next(iter(number_rows.items()))
        candidates.append(FieldCandidate(
            field_path="international_driving_permit.number",
            value=value, normalized_value=value, source_document=source,
            source_method="document_parser",
            confidence=min(0.80, line.confidence * 0.84),
            evidence_text=line.text, bounding_box=line.bounding_box,
            validation_passed=True,
            warnings=["PRIVATE_DRIVER_LICENCE_NUMBER_ACCEPTED_BY_POLICY"],
        ))
    if set(labelled_dates) == {
        "international_driving_permit.issue_date",
        "international_driving_permit.expiry_date",
    }:
        selected_dates = [
            (path, *labelled_dates[path])
            for path in (
                "international_driving_permit.issue_date",
                "international_driving_permit.expiry_date",
            )
        ]
    else:
        ordered_dates = sorted(date_rows)
        selected_dates = (
            [
                ("international_driving_permit.issue_date", ordered_dates[0], date_rows[ordered_dates[0]]),
                ("international_driving_permit.expiry_date", ordered_dates[1], date_rows[ordered_dates[1]]),
            ]
            if len(ordered_dates) == 2 else []
        )
    for path, value, line in selected_dates:
        candidates.append(FieldCandidate(
            field_path=path, value=value, normalized_value=value,
            source_document=source, source_method="document_parser",
            confidence=min(0.80, line.confidence * 0.84),
            evidence_text=line.text, bounding_box=line.bounding_box,
            validation_passed=True,
            warnings=["PRIVATE_DRIVER_LICENCE_DATES_ACCEPTED_BY_POLICY"],
        ))
    return candidates


# The sign that stands beside the permit number on every 1949 and 1968
# convention booklet. NFKD decomposes "№" itself to "No", so folding the row
# first makes one spelling cover the sign, the "No" an English-set booklet
# prints, the "N°" a French one does, and the "N0" a recogniser returns when it
# reads the ordinal as a zero.
# The sign must start the row or follow a space. Folding Cyrillic to its Latin
# shapes turns "НОмер" and "натиОНальный" into text containing a literal "NO",
# and a marker found in the middle of a word is not a marker.
_IDP_NUMERO_SIGN = re.compile(r"(?:(?<=^)|(?<=\s))N[O0]\s*[°º.:]?\s*(?=[0-9A-Z])")

# The row carrying the *national* licence number, printed across the foot of
# the booklet's left column in the issuing state's languages.
#
# This is the number the reader kept returning as the permit number, and the
# reason is structural rather than accidental: it is printed as a bare digit
# run on a line of its own, while the permit number above it is grouped
# ("01 EA 044761") and shares its line with a marker. A search that ranks by
# "is this whole line one number" prefers the wrong one every time. The two are
# different identifiers -- one issued by the association, one by the holder's
# own state -- so the row is refused outright rather than ranked down.
_IDP_NATIONAL_LICENCE_ROW = re.compile(
    r"НАЦИОНАЛЬН|НАЦІОНАЛЬН|"
    r"NATIONAL\s*(?:DRIVING\s*)?(?:LICENCE|LICENSE|PERMIT|DRIVER)|"
    # An Armenian booklet captions the row "Number of domestic driving permit",
    # and both halves of that wording fell outside the guard: "domestic" was
    # only admitted before "licence", and "number of" only before "national".
    # The row was then read as the permit's own number, so the booklet reported
    # the holder's national licence number as its permit number and no national
    # licence number at all.
    r"NUMBER\s*OF\s*(?:THE\s*)?(?:NATIONAL|DOMESTIC)|"
    r"DOMESTIC\s*(?:DRIVING\s*)?(?:LICENCE|LICENSE|PERMIT|DRIVER)|"
    r"PERMIS\s*NATIONAL|PERMIS\s*DE\s*CONDUIRE\s*NATIONAL|"
    r"NACIONAL|NAZIONALE|NACIONAIS|NATIONALEN\s*FUHRERSCHEIN|"
    # What the Cyrillic spellings above look like once every confusable letter
    # has been folded to its Latin shape, which is the form the row scan sees.
    r"HAЦNOHAЛЬH|HAЦIOHAЛЬH",
    re.I,
)


def _is_national_licence_row(raw: str, folded: str) -> bool:
    """True where this row is the booklet's national-licence line.

    Both spellings are tested. Folding a row to Latin shapes is what lets one
    marker pattern cover every script, but it also rewrites the Cyrillic label
    this guard is looking for -- "НАЦИОНАЛЬНОГО" comes out as "HAЦNOHAЛЬHOГO" --
    so a guard written in Cyrillic alone silently stops firing on exactly the
    booklets that print the row in Cyrillic.
    """
    return bool(
        _IDP_NATIONAL_LICENCE_ROW.search(raw)
        or _IDP_NATIONAL_LICENCE_ROW.search(folded)
    )

# A run that could spell a permit number: groups of letters and digits joined by
# a single space or hyphen. Issuers group it differently and the grouping is
# printed, so it is captured as printed and judged on what it reduces to. A dot
# is not a separator here: it is what a date is written with, and "01.08.2026"
# reduces to nine characters of which eight are digits.
_IDP_NUMBER_RUN = re.compile(
    r"(?<![0-9A-Z])[0-9A-Z]{1,15}(?:[ \-][0-9A-Z]{1,15}){0,4}(?![0-9A-Z])",
)

# The shape an association-issued booklet prints: two digits, a two-letter
# series, then six digits. Recognising it lets the number be found on a page
# whose numero sign was lost to the guilloche behind it.
_IDP_SERIES_NUMBER = re.compile(r"^\d{2}[A-Z]{2}\d{6}$")


def _fold_row(text: str) -> str:
    """The row upper-cased, deaccented, and with every script folded to Latin.

    ``fold_for_match`` upper-cases before it decomposes, so the "No" that "№"
    decomposes into keeps a lower-case o. Upper-casing again afterwards is what
    lets one marker pattern match both spellings.
    """
    return fold_for_match(text).upper()


def _idp_number_shape(run: str) -> str | None:
    """The permit number `run` spells, as printed, or None if it cannot be one.

    Six to fifteen characters with at least four digits covers every issuer in
    this corpus -- an IDA booklet's "01 EA 044761", a Russian booklet's nine
    bare digits -- while refusing a page's short codes, its country letters and
    the vehicle-class cells beside the photograph.

    Letters are capped at a short series because the row is read whole, and a
    row of prose with numbers in it is otherwise a run of the same shape: the
    convention line the booklet prints across its own cover reduces to
    "19SEPTEMBER1949", which is fifteen characters with eight digits and is not
    anybody's permit number.
    """
    compact = re.sub(r"[ \-]", "", run)
    if not 6 <= len(compact) <= 15 or not compact.isalnum():
        return None
    if sum(char.isdigit() for char in compact) < 4 or len(set(compact)) <= 2:
        return None
    letter_groups = re.findall(r"[A-Z]+", compact)
    if sum(len(group) for group in letter_groups) > 4 or any(
        len(group) > 3 for group in letter_groups
    ):
        return None
    return " ".join(run.split())


def _idp_rows(lines: list[OCRLine]) -> list[list[OCRLine]]:
    """Boxes grouped into the printed rows they belong to, left to right.

    Nothing guarantees a recogniser returns "№ 01 EA 044761" as one box: a
    marker set that small is routinely split from the value beside it, and the
    national-licence row at the foot of the page is printed as a long label on
    the left with its value on the right. Both have to be read as rows for a
    marker to be found and for a label to be able to disqualify a number.

    A gap wider than a twelfth of the page ends the row, which is what keeps
    the booklet's two columns -- the country and permit block on the left, the
    holder's numbered rows on the right -- from being joined into one.
    """
    ordered = sorted(lines, key=lambda line: (_line_rect(line)[1], _line_rect(line)[0]))
    gap_limit = _page_width(lines) * 0.08
    rows: list[list[OCRLine]] = []
    for line in ordered:
        x1, y1, _, y2 = _line_rect(line)
        for row in rows:
            _, ry1, rx2, ry2 = (
                min(_line_rect(member)[0] for member in row),
                min(_line_rect(member)[1] for member in row),
                max(_line_rect(member)[2] for member in row),
                max(_line_rect(member)[3] for member in row),
            )
            overlap = max(0.0, min(y2, ry2) - max(y1, ry1))
            if (
                overlap >= 0.5 * max(1.0, min(y2 - y1, ry2 - ry1))
                and -gap_limit <= x1 - rx2 <= gap_limit
            ):
                row.append(line)
                break
        else:
            rows.append([line])
    return [sorted(row, key=lambda line: _line_rect(line)[0]) for row in rows]


def _overlaps_red(line: OCRLine, red_boxes: list[tuple[float, float, float, float]]) -> bool:
    x1, y1, x2, y2 = _line_rect(line)
    area = max(1.0, (x2 - x1) * (y2 - y1))
    for rx1, ry1, rx2, ry2 in red_boxes:
        overlap = max(0.0, min(x2, rx2) - max(x1, rx1)) * max(0.0, min(y2, ry2) - max(y1, ry1))
        if overlap / area >= 0.10:
            return True
    return False


def idp_number_candidates(
    lines: list[OCRLine], source: str,
    red_boxes: list[tuple[float, float, float, float]] | None = None,
) -> list[FieldCandidate]:
    """The permit's own number, told apart from the other numbers beside it.

    A permit booklet carries at least two long numbers on one page, and only one
    of them keys the rental: the permit number, printed in red beside the numero
    sign under the title, and the holder's national licence number, printed in
    black at the foot of the same column. Choosing the longest bare digit run
    returned the national one on every association-issued booklet, because that
    is the one printed alone on its line while the permit number sits beside a
    marker and is grouped by spaces.

    Three cues decide it, strongest first, and each is a thing the booklet
    itself prints rather than a property of the capture:

    * red ink, which only the permit number is set in;
    * the numero sign that stands immediately before it;
    * the two-digit/two-letter/six-digit series an association issues.

    The national-licence row is refused before any of them are weighed, so the
    fallback that runs when all three are missing cannot reach it either.
    """
    red_boxes = red_boxes or []
    treaty_years = {
        year
        for line in lines
        if re.search(r"CONVENTION|ROAD\s+TRAFFIC|КОНВЕНЦ|KOHBEHЦ", line.text, re.I)
        for year in re.findall(r"(?:19|20)\d{2}", line.text)
    }
    options: list[tuple[tuple[Any, ...], OCRLine, str]] = []
    page_height = max((_line_rect(line)[3] for line in lines), default=1.0) or 1.0
    for row in _idp_rows(lines):
        spans: list[tuple[OCRLine, int, int]] = []
        folded_lines: list[str] = []
        for line in row:
            start = sum(len(text) + 1 for text in folded_lines)
            folded_lines.append(_fold_row(line.text))
            spans.append((line, start, start + len(folded_lines[-1])))
        row_text = " ".join(folded_lines)
        if (
            _is_national_licence_row(" ".join(line.text for line in row), row_text)
            or re.search(r"CONVENTION|ROAD\s+TRAFFIC|КОНВЕНЦ|KOHBEHЦ", row_text, re.I)
            or re.search(DATE_PATTERN, row_text, re.I)
            # The date the cover was filled in by hand is a date whatever its
            # strokes were read as. Left unrecognised, "26/1112023" was mined
            # for the digit run inside it and a permit was filed under the day
            # it was issued.
            or handwritten_date_values(row_text)
            or "<" in row_text
        ):
            continue
        marker = _IDP_NUMERO_SIGN.search(row_text)
        marker_at = marker.end() if marker else None
        # The marker is blanked before the row is scanned. Left in, its "N" and
        # "O" are read as part of the value beside it and the booklet is filed
        # under "NO 01 EA 044761". The row is scanned whole rather than box by
        # box because a recogniser splits "01 EA 044761" into three boxes as
        # readily as it returns one.
        scan = _IDP_NUMERO_SIGN.sub(lambda hit: " " * len(hit.group()), row_text)
        for match in _IDP_NUMBER_RUN.finditer(scan):
            number = _idp_number_shape(match.group())
            if number is None:
                continue
            covered = [
                line for line, start, end in spans
                if start < match.end() and match.start() < end
            ]
            if not covered:
                continue
            line = covered[0]
            compact = number.replace(" ", "")
            red_confirmed = any(_overlaps_red(one, red_boxes) for one in covered)
            marker_confirmed = marker_at is not None and match.start() >= marker_at
            series_confirmed = bool(_IDP_SERIES_NUMBER.match(compact))
            # A convention date can be returned twice: once as its complete
            # sentence and once as a digits-only OCR fragment (``196808`` for
            # the 8 November 1968 wording).  The sentence is already refused
            # above; refuse its orphan fragment too unless the booklet itself
            # anchors it as a number by colour, numero sign, or series shape.
            if (
                treaty_years
                and not (red_confirmed or marker_confirmed or series_confirmed)
                and any(year in compact for year in treaty_years)
            ):
                continue
            centers = [sum(_line_rect(one)[1::2]) * 0.5 for one in covered]
            options.append(((
                red_confirmed,
                marker_confirmed,
                series_confirmed,
                scan.strip() == match.group().strip(),
                # Recogniser confidence, and deliberately no preference for the
                # East-Slavic recogniser over the English one. The rank this
                # replaces carried such a preference, and it is a bias rather
                # than evidence: every country outside a short table is read
                # with ("en", "ru"), so both recognisers read a Latin-script
                # Argentine booklet, and preferring the Slavic one handed the
                # field to whichever of them was worse at Latin letters. A
                # number has no script of its own; the letters beside it belong
                # to whichever the booklet was printed in.
                round(min(one.confidence for one in covered), 2),
                # Between two rows with nothing else to separate them, the
                # permit number is the higher on the page: it is printed under
                # the title, and every other number on the booklet is printed
                # below it.
                -(sum(centers) / len(centers)) / page_height,
                len(compact),
            ), line, number))
    if not options:
        return []
    key, line, number = max(options, key=lambda option: option[0])
    warnings = ["IDP_VISIBLE_NUMBER_LAYOUT_FALLBACK"]
    confidence = min(0.84, line.confidence * 0.90)
    if key[0]:
        warnings.append("IDP_NUMBER_CONFIRMED_BY_RED_INK")
        confidence = min(0.93, line.confidence * 0.97)
    if key[1]:
        warnings.append("IDP_NUMBER_ANCHORED_ON_NUMERO_SIGN")
        confidence = max(confidence, min(0.90, line.confidence * 0.95))
    rivals = {value for _, _, value in options} - {number}
    if not any(key[:3]) and rivals:
        # Nothing on the page says this is the permit number rather than one of
        # the others beside it -- no red, no marker, no series shape -- and the
        # page holds more than one number it could be. That is the case this
        # whole parser exists because of, so it is handed to an operator rather
        # than guessed at. A single unrivalled number needs no such caution:
        # there is nothing for it to be confused with.
        warnings.append("IDP_NUMBER_UNANCHORED_REQUIRES_REVIEW")
        confidence = min(0.62, line.confidence * 0.66)
    return [FieldCandidate(
        field_path="international_driving_permit.number",
        value=number, normalized_value=number, source_document=source,
        source_method="document_parser", confidence=confidence,
        evidence_text=line.text, bounding_box=line.bounding_box,
        validation_passed=True, warnings=warnings,
    )]


_NORMALIZED_UNSET = object()


def _candidate(
    path: str, value: str, line: OCRLine, source: str,
    validation: bool | None, normalized: str | None | object = _NORMALIZED_UNSET,
    warnings: list[str] | None = None, proximity: float = 1.0,
) -> FieldCandidate:
    confidence = max(0.0, min(1.0, line.confidence * proximity))
    source_method = (
        "document_parser"
        if line.model_name.startswith("PaddleOCR-VL")
        else "labelled_ocr"
    )
    return FieldCandidate(
        field_path=path, value=value,
        normalized_value=value if normalized is _NORMALIZED_UNSET else normalized,
        source_document=source, source_method=source_method, confidence=confidence,
        evidence_text=line.text, bounding_box=line.bounding_box, validation_passed=validation,
        warnings=warnings or [],
    )


# Wording that marks a numbered row as a control or reference number rather than
# the licence number, in the languages that print it beside designator 5.
#
# "DD" and a bare "REF" are the AAMVA spelling of the same thing. The standard
# calls it the document discriminator: a serial identifying this piece of
# plastic rather than the entitlement, reissued with every replacement card.
# The Ontario licence prints it as "5 DD/RÉF  IS0831624" directly under the
# licence number itself, so the two rows are one line apart and the wrong one
# is the one a hungry parser reaches first.
_REFERENCE_ROW = re.compile(
    r"\b(?:R[EÉ]F[EÉ]RENCE|REFERENCE|R[EÉ]F\.?|REF\.?\s*N|DD|"
    r"DOCUMENT\s+DISCRIMINATOR|CONTROL|CONTRÔLE)\b",
    re.I,
)

# The same wordings as labels, for the lookups that must not treat such a row
# as naming the licence number.
_DISCRIMINATOR_LABELS = (
    "DD/RÉF", "DD/REF", "DD / RÉF", "DD / REF", "DD", "RÉF", "REF",
    "RÉFÉRENCE", "REFERENCE", "DOCUMENT DISCRIMINATOR",
    # A passport that replaces an earlier one prints the earlier one's number
    # on the back, under a caption that says so. An Indian booklet's
    # "Old Passport No. with Date and Place of Issue" row put V1424269 beside
    # the C2859311 printed on the data page, and the two competed as equals:
    # the field came back as a conflict for a person to settle between a
    # passport's number and the number of a passport that no longer exists.
    "OLD PASSPORT NO", "OLD PASSPORT NUMBER", "PREVIOUS PASSPORT NO",
    "PREVIOUS PASSPORT NUMBER", "FILE NO",
)


def _looks_like_licence_number(raw: str) -> bool:
    match = re.search(r"(?=[A-Z0-9\-/]{4,25}\b)(?=[A-Z0-9\-/]*\d)[A-Z0-9][A-Z0-9\-/]*", raw, re.I)
    return match is not None


_ARABIC_RUN = re.compile(r"[؀-ۿݐ-ݿﭐ-﷿ﹰ-﻿]+")


def _strip_rtl_label_residue(value: str, is_gcc_document: bool) -> str:
    """Drop an Arabic label that a right-to-left row left beside its value.

    An Algerian licence prints each row as "1. BENALI اللقب" -- designator,
    Latin value, then the Arabic label on the right-hand side. OCR returns the
    lot as one line, so the label came through as part of the value and the
    reader stored الاسم, the word "name", as somebody's middle name.

    Only applied where the value is already Latin, and never on a GCC card,
    whose Arabic side is a real name kept in its own field.
    """
    if is_gcc_document or not value:
        return value
    if not _ARABIC_RUN.search(value) or not re.search(r"[A-Za-z0-9]", value):
        return value
    return " ".join(_ARABIC_RUN.sub(" ", value).split())


# The field names a card may print beside a numbered designator instead of a
# value. Matching one means the row names its field and carries nothing else.
_NUMBERED_ROW_LABELS = frozenset({
    "FAMILY NAME", "SURNAME", "GIVEN NAME", "GIVEN NAMES", "OTHER NAMES",
    "DATE OF BIRTH", "PLACE OF BIRTH", "DATE OF ISSUE", "DATE OF EXPIRY",
    "ISSUING AUTHORITY", "LICENCE NUMBER", "LICENSE NUMBER", "SIGNATURE",
    "CATEGORIES OF VEHICLES", "PERMANENT PLACE OF RESIDENCE", "ADDRESS",
    # Senegal prints the same key in French, values on the row beneath.
    "NOM", "PRENOM", "PRENOMS", "DATE ET LIEU DE NAISSANCE",
    "DATE D'EMISSION", "DATE D'EXPIRATION", "DELIVRE PAR", "N PERMIS",
    "NUMERO DU PERMIS", "CATEGORIES", "ADRESSE", "SIGNATURE DU TITULAIRE",
    # The Latin-American bilingual layout, which prints each key twice -- once
    # in Spanish and once in English, separated by a slash -- and puts the
    # value on the row beneath. Both halves are listed because either can be
    # the one OCR returns cleanly.
    "N LICENCIA", "LICENSE N", "LICENCE N", "APELLIDO", "NOMBRE", "NOMBRES",
    "FIRST NAME", "LAST NAME", "FECHA DE NAC", "FECHA DE NACIMIENTO",
    "OTORGAMIENTO", "VENCIMIENTO", "EXPIRES", "DOMICILIO", "CLASES", "CLASS",
    "CLASSES", "FIRMA DEL TITULAR", "LICENCIA",
})


def _strip_leading_row_label(raw: str) -> str:
    """Drop the field name a card prints between the designator and the value.

    Senegal writes "4c. Délivré par MITTD" on one line, so the authority came
    through as "Délivré par MITTD" with its own label attached.
    """
    folded = fold_for_match(re.sub(r"[ªº°]", "", raw))
    for label in sorted(_NUMBERED_ROW_LABELS, key=len, reverse=True):
        prefix = fold_for_match(label)
        if folded.startswith(prefix) and len(raw) > len(prefix):
            return raw[len(prefix):].strip(" :#-.")
    return raw


def _is_label_only_row(raw: str) -> bool:
    """True where the row names its field and carries no value of its own.

    Accents and the ordinal in "N° Permis" are folded away so one entry covers
    every way a card spells the same label.

    A bilingual card states the key twice on one row, its own language and
    English either side of a slash, and prints the value on the row beneath:
    the Argentine licence writes "1. Apellido / Last name" above "DORAO". Read
    as a value, that row made the holder's surname the words "Apellido / Last
    name" and their given name "bre", the tail of "Nombre" -- a card whose every
    printed field was legible produced nothing usable but its dates. Either half
    naming a field is enough, because either can be the half OCR returns
    cleanly.
    """
    known = {
        " ".join(fold_for_match(label).split()) for label in _NUMBERED_ROW_LABELS
    }

    def folded(text: str) -> str:
        return " ".join(
            fold_for_match(re.sub(r"[ªº°]", "", text.strip(" :#-."))).split()
        )

    stripped = folded(raw)
    if stripped in known:
        return True
    if "/" not in stripped:
        return False
    halves = [folded(half) for half in stripped.split("/")]
    return any(half in known for half in halves if half)


# Values that sit on a licence and are not its number. A wrong number is worse
# than none -- it is the field the rental is keyed on -- so anything resembling
# one of these is refused outright.
_FOLDED_LICENCE_TITLES = tuple(
    dict.fromkeys(fold_for_match(title) for title in LICENCE_TITLES)
)

_NOT_A_LICENCE_NUMBER = re.compile(
    r"\b(?:BLOOD|GROUP|HEIGHT|WEIGHT|CLASS|CATEG|RESTRICT|ENDORS|PHONE|TEL|"
    r"POSTAL|ZIP|SERIAL|VERSION|REF|CONTROL|SELLO|CNIC|NIN|RUT|CPF|CURP)\b",
    re.I,
)


# The machine-readable line across the foot of an EU card: "D1" for the document
# type, the issuing state, then the card's own data including the licence number.
# A hyphen inside the zone is part of it. A Slovak card prints
# "DLSVKM0730824-6060522140022427", and read as an unbroken run the zone
# ended at the hyphen after eight characters, matched nothing, and the
# licence number the card states twice was reported as absent.
_CARD_ZONE = re.compile(
    r"(?<![0-9A-Z])D[1IL][A-Z]{3}[0-9A-Z][0-9A-Z-]{10,}[0-9A-Z]"
)

# The shortest token worth testing against the zone. Below this a number could
# sit inside a thirty-character zone by chance; at eight it does not.
_MIN_CORROBORATED_NUMBER = 8

# The letter/digit pairs a recognizer trades for one another, folded to one side
# so a row read with a letter O still meets the zone's 0.
_CONFUSABLE = str.maketrans(CONFUSIONS)


def licence_number_corroborated_by_card_zone(
    lines: list[OCRLine], source: str,
) -> list[FieldCandidate]:
    """Bind the number the card prints twice: in a row, and in its own zone.

    Designator 5 is the licence number, and a hologram sits over exactly that
    part of a Dutch card. When the "5" is lost or run into the digits behind it,
    the anchored read finds nothing and the rental is left with no licence
    number at all -- which is what happened on a real upload, twice.

    The card also prints the number inside the machine-readable line along its
    foot, and that line is set by the issuing authority rather than read off a
    label. A row that the zone repeats is therefore not a guess: it is the same
    number, stated twice, and the two confirm each other. The holder's national
    number on the reverse -- BSN 220126689 on that card -- appears nowhere in
    the zone, which is what keeps this from binding the wrong number.

    The zone's spelling wins, so a row whose 0 came back as a letter O is
    corrected rather than merely rejected.
    """
    zones: list[str] = []
    zone_lines: set[int] = set()
    for line in lines:
        for match in _CARD_ZONE.finditer(
            "".join(ascii_numerals(line.text).upper().split())
        ):
            zones.append(match.group(0))
            zone_lines.add(id(line))
    if not zones:
        return []
    candidates: list[FieldCandidate] = []
    seen: set[str] = set()
    for zone in zones:
        folded_zone = zone.translate(_CONFUSABLE)
        for line in lines:
            # The zone corroborates the rows around it; it cannot corroborate
            # itself. Reading tokens out of the zone line put a slice of the
            # zone -- "D1NLD250579642709B" -- into the licence number field.
            if id(line) in zone_lines:
                continue
            for token in re.findall(r"[0-9A-Z]{%d,18}" % _MIN_CORROBORATED_NUMBER,
                                    ascii_numerals(line.text).upper()):
                position = folded_zone.find(token.translate(_CONFUSABLE))
                if position < 0:
                    continue
                # Report what the zone prints, not what the row's OCR made of it.
                value = zone[position:position + len(token)]
                if value in seen or not any(char.isdigit() for char in value):
                    continue
                seen.add(value)
                candidates.append(FieldCandidate(
                    field_path="national_driving_licence.number",
                    value=value, normalized_value=value,
                    source_document=source, source_method="document_parser",
                    confidence=min(0.95, line.confidence),
                    evidence_text=f"{line.text} = {zone}",
                    bounding_box=line.bounding_box, validation_passed=True,
                    warnings=["LICENCE_NUMBER_CONFIRMED_BY_CARD_ZONE"],
                ))
    return candidates


def canadian_licence_candidates(
    lines: list[OCRLine], source: str, known_birth_date: str | None = None,
    known_surname: str | None = None,
) -> list[FieldCandidate]:
    """Read a Canadian licence by the shape its own province gives its number.

    A Canadian card need not name the field at all. The Quebec licence heads
    its number with the designator "4d" set in grey four-point type and nothing
    else, and on a photographed card the recogniser returned that designator as
    the letters "BL". The number itself came back as G3006-140404-00 at 0.9997
    -- the clearest thing on the page -- and there was no label within reach to
    bind it to, so the licence was reported as having no number.

    What replaces the missing label is the province, which the card does state,
    and the number's own arithmetic. Quebec builds the holder's birth date into
    positions six to eleven and Ontario into its last six digits, so a run of
    characters shaped like that province's number can be checked against the
    date of birth read off the same card. Agreement is not a coincidence worth
    entertaining: it is the same person, stated twice, and the number is bound
    at full confidence. Disagreement rejects the token outright rather than
    demoting it, because a wrong licence number is worse than none.

    Where the province builds in no birth date the shape still has to be
    particular enough to identify itself -- Alberta's 6-3 digits, Manitoba's
    letter block, Nova Scotia's five letters -- and the value is marked for a
    person to confirm. The four provinces whose number is a bare run of digits
    are not read this way at all; nothing distinguishes their number from a
    postal code, and they are read from their printed labels instead.
    """
    province = province_from_text(" ".join(line.text for line in lines))
    if province is None:
        return []
    candidates: list[FieldCandidate] = [FieldCandidate(
        field_path="national_driving_licence.issued_by_name",
        value=province.name, normalized_value=province.name,
        source_document=source, source_method="document_parser",
        confidence=0.92, evidence_text=province.name, bounding_box=None,
        validation_passed=True,
        warnings=[f"CANADIAN_ISSUER_FROM_LICENCE_TEXT:{province.code}"],
    )]
    if province.number_pattern is None or not province.bindable_by_shape:
        return candidates
    surname_initial_match = re.search(
        r"[A-Z]", fold_for_match(known_surname or ""),
    )
    surname_initial = surname_initial_match.group(0) if surname_initial_match else None
    # Ontario begins its number with the surname initial.  These are the only
    # prefix substitutions allowed, and only when the surname and the encoded
    # birth date independently confirm the repaired number.
    prefix_confusions = (
        frozenset({"I", "L", "1", "|"}), frozenset({"O", "Q", "0"}),
        frozenset({"S", "5"}), frozenset({"B", "8"}),
        frozenset({"Z", "2"}), frozenset({"G", "6"}),
    )

    def compatible_initial(read: str, expected: str) -> bool:
        return read == expected or any(
            read in group and expected in group for group in prefix_confusions
        )

    found: list[tuple[str, OCRLine, bool]] = []
    for line in lines:
        text = ascii_numerals(line.text)
        # A row the card labels as something else cannot be donating the
        # licence number, whatever its shape. "N° de référence : R4MSA2R21" on
        # the Quebec card and "5 DD/RÉF IS0831624" on the Ontario one are the
        # rows this protects against -- both sit directly beside the number
        # they are not.
        if _REFERENCE_ROW.search(text) or _NOT_A_LICENCE_NUMBER.search(text):
            continue
        # The number as printed, with the air the card sets around its hyphens
        # closed up: Ontario prints "A1059 - 35419 - 80608".  When the card's
        # first I was read as the digit 1, a strict letter-first expression
        # discarded an otherwise perfect number.  The surname initial and the
        # birth date encoded in the last six digits make that repair exact.
        compact = compact_identifiers(text.upper())
        if province.code == "ON" and surname_initial:
            matches = re.finditer(
                r"(?<![A-Z0-9])([A-Z0-9|])\d{4}-?\d{5}-?\d{5}(?![A-Z0-9])",
                compact,
            )
        else:
            matches = province.number_pattern.finditer(compact)
        for match in matches:
            token = match.group(0)
            repaired_prefix = False
            if province.code == "ON" and surname_initial:
                if not compatible_initial(token[0], surname_initial):
                    continue
                if token[0] != surname_initial:
                    token = surname_initial + token[1:]
                    repaired_prefix = True
                if province.number_pattern.fullmatch(token) is None:
                    continue
            stated = birth_date_in_number(province, token)
            if province.encodes_birth_date:
                if stated is None:
                    continue          # shaped like the number but spells no date
                if known_birth_date and stated != known_birth_date:
                    continue          # a different person's number, or not one
                if repaired_prefix and not known_birth_date:
                    continue          # repair needs both independent checks
            found.append((token, line, repaired_prefix))
    unique = {token for token, _, _ in found}
    if len(unique) != 1:
        # Two readings of the same card disagreeing about its number is not
        # something to settle by picking one.
        return candidates
    token, line, repaired_prefix = found[0]
    confirmed = (
        province.encodes_birth_date
        and known_birth_date is not None
        and birth_date_in_number(province, token) == known_birth_date
    )
    candidates.append(FieldCandidate(
        field_path="national_driving_licence.number",
        value=token, normalized_value=token.upper(),
        source_document=source, source_method="document_parser",
        confidence=min(0.97, line.confidence) if confirmed
        else min(0.70, line.confidence * 0.72),
        evidence_text=line.text, bounding_box=line.bounding_box,
        validation_passed=True,
        warnings=[f"CANADIAN_LICENCE_NUMBER_FORMAT:{province.code}"] + (
            ["CANADIAN_LICENCE_NUMBER_OCR_PREFIX_REPAIRED"]
            if repaired_prefix else []
        ) + (
            ["LICENCE_NUMBER_CONFIRMED_BY_ENCODED_BIRTH_DATE"] if confirmed
            else ["LICENCE_NUMBER_FROM_PROVINCIAL_FORMAT_REQUIRES_REVIEW"]
        ),
    ))
    return candidates


_NUMBERED_FIELD_ROW = re.compile(
    r"^\s*(4\s*[ABCD]|[1235])\s*(?:[.):\-]+\s*|\s+)\S", re.I,
)
_UNLABELLED_IDENTIFIER = re.compile(r"(?<![A-Z0-9])[A-Z0-9][A-Z0-9\-/]{4,24}(?![A-Z0-9])")
_UNREAD_DESIGNATOR_MINIMUM_CONFIDENCE = 0.9
_UNREAD_DESIGNATOR_MINIMUM_DIGITS = 4


# A mailing address is not an identifier, and its house number is not a field
# designator. An American licence prints the holder's address under the name --
# "2 WILMINGTON DR" over "MELVILLE, NY 11747" -- and both halves were read as
# something else: the house number matched the numbered-card designator shape,
# which sent the page down the recovery meant for an EU card whose "5" was
# lost, and the ZIP code was the one identifier-shaped token that recovery
# found. A New York licence was reported as licence number 11747.
_MAILING_ADDRESS_ROW = re.compile(
    # A two-letter state and a ZIP closing the row: no licence number is
    # printed in that form.
    r",\s*[A-Z]{2}\.?\s+\d{5}(?:-\d{4})?\s*$"
    # A house number opening a row that a street type or unit closes.
    r"|^\s*\d{1,6}\s+[A-Z0-9 .'#-]*?\b(?:ST|STREET|AVE|AVENUE|RD|ROAD|DR"
    r"|DRIVE|LN|LANE|BLVD|BOULEVARD|CT|COURT|PL|PLACE|TER|TERRACE|WAY|HWY"
    r"|PKWY|CIR|CIRCLE|APT|UNIT|SUITE|STE)\b",
    re.I,
)

# What an American card captions its number with when it prints no designator:
# New York sets "ID 730 096 135" beside the title and Michigan "DL A 434 366
# 067 242", and the recogniser returns the groups with the print spacing kept.
# The caption is what separates those figures from the other runs on the card,
# so they are joined back only where it stands in front of them and what comes
# out is a shape a state issues.
# ``DLN`` is admitted without a boundary in front of it: the designator that
# precedes it is set in the smallest type on the card, and a recogniser runs
# the two together. Pennsylvania's row came back as "KODLN 28 954 357", where
# the caption is unmistakable and the eight figures behind it are the number.
_AMERICAN_ID_ROW = re.compile(
    r"(?:(?<![A-Z0-9])(?:ID|DL)|DLN)\s*([A-Z]?[\d\s.\-]{4,18}\d)(?![\d])", re.I,
)
# The document discriminator is the one long number on an AAMVA card that is
# not the licence number, and Florida prints it as "5DD G742403010344" -- a
# letter and twelve figures, the same shape several states issue their licence
# numbers in. The designator runs into the caption, so a word-boundary test in
# front of "DD" does not see it.
_DISCRIMINATOR_NUMBER_ROW = re.compile(r"(?<![A-Z])DD(?![A-Z])", re.I)
# The states do not agree on a format -- eight figures in Pennsylvania, nine in
# New York, twelve in North Carolina, a letter and twelve in Michigan -- so the
# caption in front is what identifies the number and this only keeps a stray
# run of figures from passing for one.
_AMERICAN_LICENCE_NUMBER = re.compile(r"\d{6,15}|[A-Z]\d{6,14}")
# Where the caption itself was lost. A Michigan row came back as "g. S 300 772
# 609025", its "4d" reduced to a "g", and the shape that survives is specific
# enough to stand on its own: a single letter followed by twelve figures, on a
# page already established as an American licence, and only where the page
# carries exactly one such run.
_AMERICAN_UNCAPTIONED_NUMBER = re.compile(
    r"(?<![A-Z0-9])([A-Z][\d\s]{10,16}\d)(?![A-Z0-9])", re.I,
)


def _prints_us_state_heading(lines: list[OCRLine]) -> bool:
    """Whether a driving licence page heads itself with one U.S. state.

    Most American cards name no country at all. Maryland heads its licence
    "MARYLAND" over "Driver's License", captions its rows "Date of birth",
    "Date of exp" and "Date of issue", and prints neither an AAMVA designator
    nor the letters USA anywhere -- so nothing said the card was American, its
    dates were read day first, and "01/16/2025" named a sixteenth month and
    was reported as no issue date at all.
    """
    if not page_licence_title_present(lines):
        return False
    return len(us_states_named(" ".join(line.text for line in lines))) == 1


def prints_american_licence_layout(lines: list[OCRLine]) -> bool:
    """Whether this page is an American card, by its designators, captions or state."""
    return (
        _prints_aamva_field_codes(lines)
        or _prints_american_licence_captions(lines)
        or _prints_us_state_heading(lines)
    )


# The DVLA writes a licence number to a fixed sixteen-character formula: five
# characters of surname (padded with 9s), six figures encoding the birth date,
# two of the holder's initials (9 where there is none), one arbitrary figure
# and two computer-check letters. Nothing else printed on either side of the
# card takes that shape, which is what lets the number be read from a card
# whose "5" designator was returned in a box of its own -- as this photograph
# of a card lying on its side returned it.
_UK_LICENCE_NUMBER = re.compile(
    r"(?<![A-Z0-9])[A-Z9]{5}\d{6}[A-Z9]{2}\d[A-Z]{2}(?![A-Z0-9])",
)
# The card prints the issue number in the same cell, two figures after a gap,
# and the reader keeps it: that is the value field 5 states.
_UK_LICENCE_NUMBER_ROW = re.compile(
    _UK_LICENCE_NUMBER.pattern + r"(?:\s+\d{2}(?!\d))?",
)
_UK_LICENCE_AUTHORITY = re.compile(r"(?<![A-Z])(?:DVLA|DVA)(?![A-Z])", re.I)


def uk_licence_number_candidates(
    lines: list[OCRLine], source: str,
) -> list[FieldCandidate]:
    """Read a DVLA licence number by the formula the agency issues it to."""
    latin = [latinize_lookalikes(line.text) for line in lines]
    if not any(_UK_LICENCE_AUTHORITY.search(text) for text in latin):
        return []
    found: dict[str, OCRLine] = {}
    for line, text in zip(lines, latin):
        for match in _UK_LICENCE_NUMBER_ROW.finditer(
            ascii_numerals(text).upper(),
        ):
            token = match.group(0)
            current = found.get(token)
            if current is None or line.confidence > current.confidence:
                found[token] = line
    if len(found) != 1:
        return []
    token, line = next(iter(found.items()))
    return [FieldCandidate(
        field_path="national_driving_licence.number",
        value=token, normalized_value=token,
        source_document=source, source_method="document_parser",
        confidence=min(0.80, line.confidence),
        evidence_text=line.text, bounding_box=line.bounding_box,
        validation_passed=True,
        warnings=["UK_LICENCE_NUMBER_FORMAT_REQUIRES_REVIEW"],
    )]


def _american_id_row_licence_number(
    lines: list[OCRLine], source: str,
) -> list[FieldCandidate]:
    """The number an American card prints against a bare ``ID``/``DL`` caption."""
    if not prints_american_licence_layout(lines):
        return []
    found: dict[str, OCRLine] = {}
    uncaptioned: dict[str, OCRLine] = {}
    for line in lines:
        # A date printed on the number's own row is not part of the number, and
        # removing it is what leaves the figures the caption introduced.
        row = re.sub(
            DATE_PATTERN, " ",
            split_run_together_dates(ascii_numerals(line.text)), flags=re.I,
        )
        if _DISCRIMINATOR_NUMBER_ROW.search(row):
            continue
        match = _AMERICAN_ID_ROW.search(row)
        if match is not None:
            token = re.sub(r"[\s.\-]+", "", match.group(1)).upper()
            if _AMERICAN_LICENCE_NUMBER.fullmatch(token) is not None:
                current = found.get(token)
                if current is None or line.confidence > current.confidence:
                    found[token] = line
                continue
        for loose in _AMERICAN_UNCAPTIONED_NUMBER.finditer(row):
            token = re.sub(r"[\s.\-]+", "", loose.group(1)).upper()
            if re.fullmatch(r"[A-Z]\d{12}", token) is None:
                continue
            current = uncaptioned.get(token)
            if current is None or line.confidence > current.confidence:
                uncaptioned[token] = line
    if not found and len(uncaptioned) == 1:
        found = uncaptioned
    if len(found) != 1:
        return []
    token, line = next(iter(found.items()))
    return [FieldCandidate(
        field_path="national_driving_licence.number", value=token,
        normalized_value=token, source_document=source,
        source_method="document_parser",
        confidence=min(0.78, line.confidence),
        evidence_text=line.text, bounding_box=line.bounding_box,
        validation_passed=True,
        warnings=["AMERICAN_ID_ROW_LICENCE_NUMBER_REQUIRES_REVIEW"],
    )]


def _identifier_designator_was_read(lines: list[OCRLine]) -> bool:
    for line in lines:
        text = _normalize_numbered_designator_text(line.text)
        bare = _BARE_DESIGNATOR.match(text)
        if bare is not None and designator_name(bare.group(1)) == "5":
            return True
        row = _NUMBERED_FIELD_ROW.match(text)
        if row is not None and designator_name(row.group(1)) in {"5", "4D"}:
            return True
    return False


def _unread_designator_five_candidates(
    lines: list[OCRLine], source: str,
) -> list[FieldCandidate]:
    """Recover field 5 on a numbered card whose 5 marker was never returned.

    A cropped or small-print marker is lost often enough that a card can prove
    it numbers its fields, print its number at full confidence, and still
    report none. The recovery is refused wherever 5 or a filled 4d was read --
    those own the field -- and wherever the page offers more than one token of
    the right shape, so it never chooses between two readings. A card carrying
    a machine-readable zone states its number there and is left to it.
    """
    if _identifier_designator_was_read(lines):
        return []
    if any(_CARD_ZONE.search(ascii_numerals(line.text).upper()) for line in lines):
        return []
    options: list[tuple[str, OCRLine]] = []
    for line in lines:
        if (
            line.confidence < _UNREAD_DESIGNATOR_MINIMUM_CONFIDENCE
            or _NUMBERED_FIELD_ROW.match(line.text)
        ):
            continue
        text = ascii_numerals(line.text)
        if (
            re.search(DATE_PATTERN, text, re.I)
            or _NOT_A_LICENCE_NUMBER.search(text)
            or _MAILING_ADDRESS_ROW.search(text)
        ):
            continue
        for token in _UNLABELLED_IDENTIFIER.findall(text.upper()):
            digits = sum(character.isdigit() for character in token)
            if digits < _UNREAD_DESIGNATOR_MINIMUM_DIGITS:
                continue
            if len(set(token)) <= 2:
                continue
            if not _plausible_raw_value("national_driving_licence.number", token):
                continue
            options.append((token, line))
    if len(options) != 1:
        return []
    token, line = options[0]
    return [FieldCandidate(
        field_path="national_driving_licence.number", value=token,
        normalized_value=token, source_document=source,
        source_method="document_parser",
        confidence=min(0.62, line.confidence * 0.66),
        evidence_text=line.text, bounding_box=line.bounding_box,
        validation_passed=True,
        warnings=["LICENCE_NUMBER_POSITION_FALLBACK_REQUIRES_REVIEW"],
    )]


# The New Zealand card captions its number "Licence number", and the recogniser
# frequently returns only the first word of that caption -- the reported card
# came back as a bare "Licence" beside "DH936529 Version 178", so no label
# matched and the field the rental is keyed on was reported as absent. The
# card's own format answers it instead: a New Zealand driver licence number is
# two letters and six figures, and nothing else printed on either side of the
# card takes that shape. The version and card numbers beside it are longer and
# carry a separator, so the boundaries below exclude them; where more than one
# distinct token still matches, none is taken.
_NEW_ZEALAND_LICENCE_NUMBER = re.compile(
    r"(?<![A-Z0-9])[A-Z]{2}[0-9]{6}(?![A-Z0-9])",
)


def _new_zealand_licence_number_candidates(
    lines: list[OCRLine], source: str,
) -> list[FieldCandidate]:
    """Read a New Zealand licence number from the shape the card fixes."""
    # Gated on what this page itself prints, not on the bundle's country: a
    # tourist bundle is read a page at a time, and on the page that proves the
    # country the reconciled value is not available yet, so the licence front
    # was reaching this rule with no country at all.
    if not any(
        marker in fold_for_match(line.text)
        for line in lines for marker in ("NEW ZEALAND", "AOTEAROA")
    ):
        return []
    found: list[tuple[str, OCRLine]] = []
    for line in lines:
        text = ascii_numerals(line.text).upper()
        for token in _NEW_ZEALAND_LICENCE_NUMBER.findall(text):
            found.append((token, line))
    if not found or len({token for token, _ in found}) != 1:
        return []
    token, line = found[0]
    return [FieldCandidate(
        field_path="national_driving_licence.number", value=token,
        normalized_value=token, source_document=source,
        source_method="document_parser",
        confidence=min(0.9, line.confidence),
        evidence_text=line.text, bounding_box=line.bounding_box,
        validation_passed=True,
        warnings=["LICENCE_NUMBER_FROM_NEW_ZEALAND_FORMAT"],
    )]


# Armenia numbers its licence with two Latin letters and six figures, printed
# in field 5 at the top right of the card -- away from the 1/2/3 column that
# the recovery for a lone "5" marker anchors on, and the only bare designator
# the recogniser returned, so the column rule had nothing to pair it with. The
# card's own format answers it instead: nothing else on either side takes that
# shape, and where more than one token does, none is taken.
_ARMENIAN_LICENCE_NUMBER = re.compile(r"(?<![A-Z0-9])[A-Z]{2}[0-9]{6}(?![A-Z0-9])")
_ARMENIAN_PAGE = re.compile(r"(?<![A-Z])ARMENIA(?![A-Z])|ՀԱՅԱՍՏԱՆ", re.I)


def _armenian_licence_number_candidates(
    lines: list[OCRLine], source: str,
) -> list[FieldCandidate]:
    """Read an Armenian licence number by the format the republic issues."""
    if not any(_ARMENIAN_PAGE.search(line.text) for line in lines):
        return []
    found: dict[str, OCRLine] = {}
    for line in lines:
        for match in _ARMENIAN_LICENCE_NUMBER.finditer(
            latinize_lookalikes(ascii_numerals(line.text)).upper(),
        ):
            token = match.group(0)
            current = found.get(token)
            if current is None or line.confidence > current.confidence:
                found[token] = line
    if len(found) != 1:
        return []
    token, line = next(iter(found.items()))
    return [FieldCandidate(
        field_path="national_driving_licence.number",
        value=token, normalized_value=token,
        source_document=source, source_method="document_parser",
        confidence=min(0.80, line.confidence),
        evidence_text=line.text, bounding_box=line.bounding_box,
        validation_passed=True,
        warnings=["ARMENIAN_LICENCE_NUMBER_FORMAT_REQUIRES_REVIEW"],
    )]


def national_licence_number_fallback(
    lines: list[OCRLine], source: str,
) -> list[FieldCandidate]:
    """Take the licence number from its position when its label is unknown.

    Every licence prints its number high on the card, near the title, and it is
    the one long mixed token there that is not a date. That is weak evidence, so
    it is used only when no label matched, is refused wherever a competing
    reading is plausible, and is always marked for a person to confirm.
    """
    # Anchored on the verified titles rather than on guessed word roots: a
    # Brazilian CNH says "HABILITAÇÃO" and matches none of "licence", "permis"
    # or "driving".
    # A numbered card states where its number is, so a token picked from
    # elsewhere must not stand against a designator the page actually printed:
    # on an Albanian licence that guess landed on the holder's national ID from
    # row 4d, and on a noisy capture it landed on the OCR fragment "2Rbew".
    british = uk_licence_number_candidates(lines, source)
    if british:
        return british
    american = _american_id_row_licence_number(lines, source)
    if american:
        return american
    if any(
        _NUMBERED_FIELD_ROW.match(line.text)
        and not _MAILING_ADDRESS_ROW.search(line.text)
        for line in lines
    ):
        return _unread_designator_five_candidates(lines, source)
    title_row = next((
        index for index, line in enumerate(lines)
        if any(title in fold_for_match(line.text) for title in _FOLDED_LICENCE_TITLES)
    ), None)
    if title_row is None:
        return []
    options: list[tuple[int, str, OCRLine]] = []
    for offset, line in enumerate(lines[title_row: title_row + 6]):
        text = ascii_numerals(line.text)
        if (
            re.search(DATE_PATTERN, text, re.I)
            or _NOT_A_LICENCE_NUMBER.search(text)
            or _MAILING_ADDRESS_ROW.search(text)
        ):
            continue
        for token in re.findall(r"(?<![A-Z0-9])[A-Z0-9][A-Z0-9\-/]{4,24}(?![A-Z0-9])", text.upper()):
            if not any(character.isdigit() for character in token):
                continue
            if len(set(token)) <= 2:            # 00000000 and the like
                continue
            # The same shape test the labelled path applies, so a value this
            # pass would never have accepted from a label cannot enter through
            # the weaker door instead. A street name reaches here on a licence
            # that prints the holder's address under its title.
            if not _plausible_raw_value("national_driving_licence.number", token):
                continue
            options.append((offset, token, line))
    if len(options) != 1:
        # Nothing to choose between two candidates without knowing the country's
        # format, and guessing here is exactly the failure this avoids.
        return []
    _, token, line = options[0]
    return [FieldCandidate(
        field_path="national_driving_licence.number", value=token,
        normalized_value=token, source_document=source,
        source_method="document_parser",
        confidence=min(0.62, line.confidence * 0.66),
        evidence_text=line.text, bounding_box=line.bounding_box,
        validation_passed=True,
        warnings=["LICENCE_NUMBER_POSITION_FALLBACK_REQUIRES_REVIEW"],
    )]


_AAMVA_FRONT_DATE_ANCHOR = re.compile(
    r"^\s*(?:3\s*(?:[.):\-]+\s*|\s+)?DOB\b|"
    r"4A\s*(?:[.):\-]+\s*|\s+)?ISS\b|"
    r"4B\s*(?:[.):\-]+\s*|\s+)?EXP\b)",
    re.I,
)
_AAMVA_UNLABELLED_NUMBER = re.compile(
    r"[A-Z0-9](?:[A-Z0-9]*[A-Z0-9])?(?:-[A-Z0-9]+)*",
    re.I,
)


def american_unlabelled_licence_number_candidates(
    lines: list[OCRLine], source: str,
) -> list[FieldCandidate]:
    """Recover a visible U.S. licence number whose tiny ``4d`` was lost.

    This pass is called only after the issuing country has independently been
    established as the United States and the ordinary AAMVA designator parser
    found no number.  It still requires at least two front-side AAMVA date
    anchors, a single strong identifier-shaped row in the upper part of the
    card, and refuses every labelled discriminator/reference row.  The result
    is intentionally below the automatic-acceptance threshold: the operator
    sees the number instead of an empty field, but must confirm it.
    """
    if sum(
        1 for line in lines if _AAMVA_FRONT_DATE_ANCHOR.search(line.text)
    ) < 2:
        return []
    if not lines:
        return []
    page_top = min(_line_rect(line)[1] for line in lines)
    page_bottom = max(_line_rect(line)[3] for line in lines)
    upper_limit = page_top + max(1.0, page_bottom - page_top) * 0.55
    found: dict[str, OCRLine] = {}
    for line in lines:
        if line.confidence < 0.80:
            continue
        _, top, _, bottom = _line_rect(line)
        if (top + bottom) * 0.5 > upper_limit:
            continue
        raw = ascii_numerals(line.text).upper().strip()
        if (
            not raw
            or _REFERENCE_ROW.search(raw)
            or _NOT_A_LICENCE_NUMBER.search(raw)
            or re.search(DATE_PATTERN, raw, re.I)
        ):
            continue
        # Preserve meaningful hyphens while tolerating whitespace OCR inserts
        # immediately around them. Arbitrary internal spaces remain a refusal:
        # they are how a labelled DD/reference row differs from the value row.
        token = re.sub(r"\s*-\s*", "-", raw)
        if re.search(r"\s", token) or _AAMVA_UNLABELLED_NUMBER.fullmatch(token) is None:
            continue
        compact = token.replace("-", "")
        digits = sum(character.isdigit() for character in compact)
        if not 6 <= len(compact) <= 20 or digits < 5 or len(set(compact)) <= 2:
            continue
        # Purely numeric identifiers exist, but a short run is more likely a
        # ZIP code, class, height or control fragment than a licence number.
        if compact.isdigit() and not 8 <= len(compact) <= 16:
            continue
        if not _plausible_raw_value("national_driving_licence.number", token):
            continue
        current = found.get(token)
        if current is None or line.confidence > current.confidence:
            found[token] = line
    if len(found) != 1:
        return []
    token, line = next(iter(found.items()))
    return [FieldCandidate(
        field_path="national_driving_licence.number",
        value=token, normalized_value=token,
        source_document=source, source_method="document_parser",
        confidence=min(0.65, line.confidence * 0.68),
        evidence_text=line.text, bounding_box=line.bounding_box,
        validation_passed=True,
        warnings=["AAMVA_UNLABELLED_TOP_NUMBER_REQUIRES_REVIEW"],
    )]


_CALIFORNIA_DLN = re.compile(r"^[A-Z]\d{7}$", re.I)
_CALIFORNIA_ABBREVIATED_DATE = re.compile(
    r"(?<![A-Z])(?P<label>ISS(?:UE(?:D)?)?|EXP(?:IR(?:Y|ATION|ES)?)?)"
    r"(?![A-Z])\s*[:.\-]?\s*(?P<value>" + DATE_PATTERN + r")",
    re.I,
)
_CALIFORNIA_SHORT_DATE_LABEL = re.compile(
    r"^\s*(?P<label>ISS(?:UE(?:D)?)?|EXP(?:IR(?:Y|ATION|ES)?)?)"
    r"\s*[:.\-]?\s*$",
    re.I,
)
_CALIFORNIA_SEX = re.compile(
    r"(?<![A-Z])(?:\d{1,2}\s*)?SEX\s*[:.\-]?\s*(?P<value>[MFX])\b",
    re.I,
)


def is_california_driver_licence(lines: list[OCRLine]) -> bool:
    """Whether visible page wording identifies California's English DL form."""
    page = " ".join(fold_for_match(line.text) for line in lines)
    return "CALIFORNIA" in page and "DRIVER" in page and "LICEN" in page


def california_licence_layout_candidates(
    lines: list[OCRLine], source: str,
) -> list[FieldCandidate]:
    """Recover the compact California DL layout after AAMVA codes are lost.

    California prints the customer number as one letter followed by seven
    digits, while its ``ISS``/``EXP``/``SEX`` captions do not carry the AAMVA
    4a/4b/8 designators.  On a phone capture the tiny designators are often
    the first ink lost to glare.  The generic U.S. fallback correctly refuses
    to choose among several ordinary-looking identifiers in that situation;
    California's state-specific number format and short, explicit captions
    provide the missing distinction without guessing.  Every recovered value
    remains review-level evidence.
    """
    if not is_california_driver_licence(lines):
        return []

    candidates: list[FieldCandidate] = []
    numbers: dict[str, OCRLine] = {}
    for line in lines:
        raw = ascii_numerals(line.text).upper().strip()
        if line.confidence < 0.80 or _CALIFORNIA_DLN.fullmatch(raw) is None:
            continue
        existing = numbers.get(raw)
        if existing is None or line.confidence > existing.confidence:
            numbers[raw] = line
    # More than one valid California-shaped reading means that OCR disagreed
    # about the customer number.  Keep the field empty instead of choosing a
    # sequence simply because it looks plausible.
    if len(numbers) == 1:
        number, line = next(iter(numbers.items()))
        candidates.append(FieldCandidate(
            field_path="national_driving_licence.number",
            value=number, normalized_value=number,
            source_document=source, source_method="document_parser",
            confidence=min(0.65, line.confidence * 0.68),
            evidence_text=line.text, bounding_box=line.bounding_box,
            validation_passed=True,
            warnings=["CALIFORNIA_DLN_FORMAT_REQUIRES_REVIEW"],
        ))

    dated: dict[tuple[str, str], OCRLine] = {}
    for line in lines:
        raw = ascii_numerals(line.text)
        match = _CALIFORNIA_ABBREVIATED_DATE.search(raw)
        if match is None:
            continue
        path = (
            "national_driving_licence.issue_date"
            if match.group("label").upper().startswith("ISS")
            else "national_driving_licence.expiry_date"
        )
        normalized = normalize_date(match.group("value"), day_first_hint=False)
        if normalized.value is None:
            continue
        key = (path, normalized.value)
        existing = dated.get(key)
        if existing is None or line.confidence > existing.confidence:
            dated[key] = line
    # The California card puts its short caption directly above or beside the
    # date on some editions.  OCR then yields ``ISS`` and ``04/16/2025`` as
    # separate boxes, despite both being clear.  Pair only a complete date in
    # the label's immediate lower/right neighbourhood; this is layout evidence
    # for the same printed row, not a date chosen from elsewhere on the card.
    for label_line in lines:
        label_match = _CALIFORNIA_SHORT_DATE_LABEL.fullmatch(
            ascii_numerals(label_line.text),
        )
        if label_match is None or not label_line.bounding_box:
            continue
        path = (
            "national_driving_licence.issue_date"
            if label_match.group("label").upper().startswith("ISS")
            else "national_driving_licence.expiry_date"
        )
        left, top, right, bottom = _line_rect(label_line)
        height = max(1.0, bottom - top)
        closest: tuple[float, OCRLine, str] | None = None
        for value_line in lines:
            if value_line is label_line or not value_line.bounding_box:
                continue
            raw = ascii_numerals(value_line.text).strip()
            if re.fullmatch(DATE_PATTERN, raw, re.I) is None:
                continue
            normalized = normalize_date(raw, day_first_hint=False)
            if normalized.value is None:
                continue
            value_left, value_top, _value_right, value_bottom = _line_rect(value_line)
            overlap = min(bottom, value_bottom) - max(top, value_top)
            is_same_row = overlap >= 0.35 * min(
                height, max(1.0, value_bottom - value_top),
            )
            is_row_below = (
                value_top >= top - 0.20 * height
                and value_top <= bottom + 2.2 * height
            )
            if not (is_same_row or is_row_below):
                continue
            if value_left < left - 0.5 * height or value_left > right + 8.0 * height:
                continue
            distance = abs(value_left - right) + abs(value_top - top)
            if closest is None or distance < closest[0]:
                closest = (distance, value_line, normalized.value)
        if closest is not None:
            _, value_line, value = closest
            key = (path, value)
            existing = dated.get(key)
            if existing is None or value_line.confidence > existing.confidence:
                dated[key] = value_line
    for (path, value), line in dated.items():
        candidates.append(FieldCandidate(
            field_path=path, value=value, normalized_value=value,
            source_document=source, source_method="document_parser",
            confidence=min(0.68, line.confidence * 0.72),
            evidence_text=line.text, bounding_box=line.bounding_box,
            validation_passed=True,
            warnings=["CALIFORNIA_ABBREVIATED_DATE_LABEL_REQUIRES_REVIEW"],
        ))

    genders: dict[str, OCRLine] = {}
    for line in lines:
        match = _CALIFORNIA_SEX.search(ascii_numerals(line.text))
        if match is None:
            continue
        value = match.group("value").upper()
        existing = genders.get(value)
        if existing is None or line.confidence > existing.confidence:
            genders[value] = line
    if len(genders) == 1:
        value, line = next(iter(genders.items()))
        candidates.append(FieldCandidate(
            field_path="personal_info.gender",
            value=value, normalized_value=value,
            source_document=source, source_method="document_parser",
            confidence=min(0.68, line.confidence * 0.72),
            evidence_text=line.text, bounding_box=line.bounding_box,
            validation_passed=True,
            warnings=["CALIFORNIA_SEX_LAYOUT_REQUIRES_REVIEW"],
        ))
    return candidates


_CATEGORY_TABLE_COLUMN = re.compile(r"^\s*(9|10|11|12)\s*[.):]?\s*$", re.I)


def licence_category_table_dates(lines: list[OCRLine]) -> frozenset[str]:
    """Dates printed under an EU/Vienna licence's category table columns.

    The reverse has headings 9, 10, 11 and 12 printed across one row. Dates
    below them describe individual vehicle entitlements; they are not the
    document's 4a issue date or 4b expiry date. Combined front/back images put
    both kinds of dates in one OCR page, so their geometry is the only generic
    way to keep the table out of document-level fallbacks and VLM grounding.

    Three distinct headings and a meaningful horizontal spread are required.
    That rejects the explanatory legend below the table, where the same
    numbers are stacked vertically, and stray digits elsewhere on a card.
    """
    markers: list[tuple[str, OCRLine, float, float]] = []
    page_width = 0.0
    for line in lines:
        if not line.bounding_box:
            continue
        left, top, right, bottom = _line_rect(line)
        page_width = max(page_width, right)
        match = _CATEGORY_TABLE_COLUMN.match(line.text)
        if match is not None:
            markers.append((
                match.group(1), line, (top + bottom) * 0.5,
                (left + right) * 0.5,
            ))
    if len(markers) < 3 or page_width <= 0:
        return frozenset()

    vertical_tolerance = max(45.0, page_width * 0.065)
    best: tuple[tuple[int, float], list[tuple[str, OCRLine, float, float]]] | None = None
    for _, _, anchor_y, _ in markers:
        nearby = [
            marker for marker in markers
            if abs(marker[2] - anchor_y) <= vertical_tolerance
        ]
        # Multiple OCR variants can return the same heading. Keep the reading
        # closest to this candidate row so duplicates do not prove a table.
        unique: dict[str, tuple[str, OCRLine, float, float]] = {}
        for marker in nearby:
            current = unique.get(marker[0])
            if current is None or abs(marker[2] - anchor_y) < abs(current[2] - anchor_y):
                unique[marker[0]] = marker
        cluster = list(unique.values())
        if len(cluster) < 3:
            continue
        horizontal_span = max(item[3] for item in cluster) - min(item[3] for item in cluster)
        if horizontal_span < page_width * 0.18:
            continue
        score = (len(cluster), horizontal_span)
        if best is None or score > best[0]:
            best = (score, cluster)
    if best is None:
        return frozenset()

    headers = best[1]
    header_bottom = max(_line_rect(item[1])[3] for item in headers)
    left_limit = min(item[3] for item in headers) - page_width * 0.05
    right_limit = max(item[3] for item in headers) + page_width * 0.05
    values: set[str] = set()
    elsewhere: set[str] = set()
    for line in lines:
        if not line.bounding_box:
            continue
        left, top, right, bottom = _line_rect(line)
        center_x = (left + right) * 0.5
        center_y = (top + bottom) * 0.5
        inside = center_y > header_bottom and left_limit <= center_x <= right_limit
        target = values if inside else elsewhere
        for match in re.finditer(DATE_PATTERN, ascii_numerals(line.text), re.I):
            parsed = normalize_date(match.group(0), day_first_hint=True)
            if parsed.value is not None:
                target.add(parsed.value)
    # A category is very often granted on the day the licence itself was
    # issued, so the table repeats 4a's date. Only a date printed nowhere but
    # under the table belongs to the table alone; one that also appears in the
    # identity block is a document date and must stay readable there.
    return frozenset(values - elsewhere)


VALIDITY_CAPTIONS: tuple[str, ...] = (
    "VALID", "VALIDITY", "VALID FROM", "GELDIG", "VALIDO", "VÁLIDO",
)
_VALIDITY_WORDS = frozenset(VALIDITY_CAPTIONS)


def _is_validity_caption(text: str) -> bool:
    """A "Valid" caption, alone or beside its translation.

    A South African card heads the row "Valid/Geldig" in Afrikaans and
    "Valid/Kamogelesego" in Setswana; the second half is whatever language the
    card was printed in, so only the first is worth listing. All three name
    the same cell.
    """
    first = re.sub(r"[\s.:]+", " ", text.split("/")[0]).strip().upper()
    return first in _VALIDITY_WORDS
_RANGE_SEPARATORS = frozenset({"-", "\u2013", "\u2014", "TO", "/"})


def licence_validity_range_dates(
    lines: list[OCRLine], source: str,
) -> list[FieldCandidate]:
    """A licence's "Valid <from>-<to>" cell states its issue and expiry.

    A South African card prints no issue or expiry caption at all. It prints
    "Valid" beside one cell holding both dates, and separately "First issue"
    beside the day the holder first qualified -- years earlier, and not the day
    this licence was issued. With only the captioned read, the first-issue date
    was reported as the issue date and the expiry, which the card states, as
    absent.

    The caption is required and has to share the row: two dates joined by a
    dash appear in category tables too, where they describe one entitlement
    rather than the document.
    """
    captions = [
        line for line in lines
        if line.bounding_box and _is_validity_caption(line.text)
    ]
    month_first = prints_american_licence_layout(lines)
    for line in lines:
        if not line.bounding_box:
            continue
        text = split_run_together_dates(ascii_numerals(line.text)).strip()
        found = list(re.finditer(DATE_PATTERN, text, re.I))
        if len(found) != 2:
            continue
        if found[1].end() != len(text):
            continue
        if text[found[0].end():found[1].start()].strip().upper() not in _RANGE_SEPARATORS:
            continue
        # The caption can be a box of its own beside the cell, or the head of
        # the same box: one card sets "Valid/Kamogelesego: 22/11/2022 -
        # 21/11/2027" as a single printed line.
        prefix = text[:found[0].start()]
        if prefix.strip():
            if not _is_validity_caption(prefix):
                continue
        else:
            _, top, _, bottom = _line_rect(line)
            if not any(
                min(bottom, _line_rect(caption)[3]) > max(top, _line_rect(caption)[1])
                for caption in captions
            ):
                continue
        issued = normalize_date(found[0].group(0), day_first_hint=not month_first)
        expires = normalize_date(found[1].group(0), day_first_hint=not month_first)
        if issued.value is None or expires.value is None:
            continue
        if issued.value >= expires.value:
            continue
        return [
            FieldCandidate(
                field_path=path, value=value, normalized_value=value,
                source_document=source, source_method="document_parser",
                confidence=min(0.86, line.confidence * 0.9),
                evidence_text=line.text, bounding_box=line.bounding_box,
                validation_passed=True,
                warnings=["LICENCE_DATES_FROM_VALIDITY_RANGE_CELL"],
            )
            for path, value in (
                ("national_driving_licence.issue_date", issued.value),
                ("national_driving_licence.expiry_date", expires.value),
            )
        ]
    return []


_UNCAPTIONED_GENDER = re.compile(r"^\s*(MALE|FEMALE)\s*$", re.I)


def uncaptioned_gender_word(
    lines: list[OCRLine], source: str,
) -> list[FieldCandidate]:
    """A card that spells the holder's sex out and captions it with nothing.

    The South African licence prints "MALE" in its own cell with no "Sex"
    beside it, so the labelled read had nothing to bind and the field was
    reported as absent from a card that states it in full. Only one such word
    may appear: two disagreeing readings decide nothing.
    """
    words = {
        _UNCAPTIONED_GENDER.match(line.text).group(1).upper()[0]: line
        for line in lines
        if _UNCAPTIONED_GENDER.match(line.text)
    }
    if len(words) != 1:
        return []
    value, line = next(iter(words.items()))
    return [FieldCandidate(
        field_path="personal_info.gender",
        value=value, normalized_value=value,
        source_document=source, source_method="document_parser",
        confidence=min(0.8, line.confidence * 0.85),
        evidence_text=line.text, bounding_box=line.bounding_box,
        validation_passed=True,
        warnings=["GENDER_FROM_UNCAPTIONED_WORD"],
    )]


def national_licence_date_sequence(
    lines: list[OCRLine], source: str, known_birth_date: str | None = None,
) -> list[FieldCandidate]:
    """Order a licence's dates when no label in its language was recognised.

    Every licence carries the same three dates in the same relation: the holder
    was born, then years later the licence was issued, then it expires. Where
    the labels are in a language whose wording is not known, that ordering is
    still readable, and it is the only thing left to read.

    This is the same reasoning the permit booklet already relies on. It fires
    only after the labelled pass found nothing, and every value it produces is
    marked so it reaches a person rather than being presented as read.
    """
    category_dates = licence_category_table_dates(lines)
    # The same smeared box that the labelled read refuses states nothing here
    # either: ordering three dates says which is which only where all three
    # were printed on a row of their own.
    outsized = _outsized_rows(tuple(
        _line_rect(line) for line in lines if line.bounding_box
    ))
    placed = [line for line in lines if line.bounding_box]
    smeared = {id(placed[position]) for position in outsized}
    # The order of three dates says which is which; the order of the figures
    # inside one of them is a separate question, and an American card answers
    # it differently. Read day first, a Michigan licence's "Exp 03/26/2030"
    # named a twenty-sixth month, every date on the card was unreadable, and
    # the ordering had nothing to order.
    month_first = prints_american_licence_layout(lines)
    seen: dict[str, OCRLine] = {}
    for line in lines:
        if id(line) in smeared:
            continue
        row = split_run_together_dates(ascii_numerals(line.text))
        for match in re.finditer(DATE_PATTERN, row, re.I):
            parsed = normalize_date(match.group(0), day_first_hint=not month_first)
            if parsed.value is None:
                continue
            if parsed.value in category_dates:
                continue
            year = int(parsed.value[:4])
            if year < 1900 or year > date.today().year + 30:
                continue          # another calendar, as on an Iranian card
            seen.setdefault(parsed.value, line)
    ordered = sorted(seen)
    birth = known_birth_date if known_birth_date in seen else None
    if birth is None and ordered and len(ordered) >= 3:
        birth = ordered[0]
    remaining = [value for value in ordered if value != birth]
    if len(remaining) < 2:
        return []
    issue, expiry = remaining[0], remaining[-1]
    # An issue date must follow the holder's birth by at least a driving age,
    # and precede the expiry. Anything else is not this pattern.
    if birth is not None and int(issue[:4]) - int(birth[:4]) < MINIMUM_DRIVING_AGE_YEARS:
        return []
    if issue >= expiry:
        return []
    candidates: list[FieldCandidate] = []
    for path, value in (
        ("national_driving_licence.issue_date", issue),
        ("national_driving_licence.expiry_date", expiry),
    ):
        line = seen[value]
        candidates.append(FieldCandidate(
            field_path=path, value=value, normalized_value=value,
            source_document=source, source_method="document_parser",
            # Below the review threshold on purpose, and by enough that a clean
            # capture cannot lift it over. At the old cap of 0.70 the score
            # landed on 0.78 exactly -- the boundary -- so a date this pass had
            # inferred from nothing but the order of three numbers was presented
            # as HIGH_CONFIDENCE, which is the one thing its own warning says it
            # must never be.
            confidence=min(0.65, line.confidence * 0.75),
            evidence_text=line.text, bounding_box=line.bounding_box,
            validation_passed=True,
            warnings=["LICENCE_DATE_ORDER_FALLBACK_REQUIRES_REVIEW"],
        ))
    return candidates


_ALGERIAN_LICENCE_MRZ_ROW = re.compile(r"^DLDZA[0-9A-Z<]{10,}$")


def algerian_national_licence_front_dates(
    lines: list[OCRLine], source: str,
) -> list[FieldCandidate]:
    """Read the three front-side dates without borrowing category validity.

    Algerian licences commonly arrive as one photograph containing the front
    above the back.  The back's category table has older dates in larger,
    cleaner print than the front's tiny ``4a`` label, so an ordering fallback
    selected a category entitlement date as the document issue date.  The same
    back carries a TD1-style ``DLDZA`` zone.  Its position separates the two
    sides without a new OCR pass: on a vertical capture the front is the block
    above the zone; on a side-by-side capture it is the opposite half.

    Only the standard three-date front layout is accepted (birth, issue,
    expiry with a legal driving-age gap).  Anything less exact is left empty
    for the ordinary labelled path or review rather than guessed.
    """
    zone = next((
        line for line in lines
        if _ALGERIAN_LICENCE_MRZ_ROW.fullmatch(
            "".join(ascii_numerals(line.text).upper().split())
        )
        and line.bounding_box
    ), None)
    if zone is None:
        return []
    page_width = _page_width(lines)
    page_height = max((_line_rect(line)[3] for line in lines if line.bounding_box), default=0.0)
    if page_width <= 0 or page_height <= 0:
        return []
    zx1, zy1, zx2, _ = _line_rect(zone)
    side_by_side = (zx2 - zx1) < page_width * 0.65
    zone_center_x = (zx1 + zx2) * 0.5

    dated: dict[str, OCRLine] = {}
    for line in lines:
        if not line.bounding_box:
            continue
        x1, y1, x2, y2 = _line_rect(line)
        center_x, center_y = (x1 + x2) * 0.5, (y1 + y2) * 0.5
        if side_by_side:
            on_front = (
                center_x < page_width * 0.50
                if zone_center_x >= page_width * 0.50
                else center_x > page_width * 0.50
            )
        else:
            # In the common stacked capture the reverse begins well before the
            # MRZ.  The 56% cut includes all three front rows in the reported
            # specimen while excluding every date under category columns 10/11.
            on_front = center_y < zy1 * 0.56
        if not on_front:
            continue
        for match in re.finditer(DATE_PATTERN, ascii_numerals(line.text), re.I):
            parsed = normalize_date(match.group(0), day_first_hint=True)
            if parsed.value is None:
                continue
            current = dated.get(parsed.value)
            if current is None or line.confidence > current.confidence:
                dated[parsed.value] = line
    ordered = sorted(dated)
    if len(ordered) != 3:
        return []
    birth, issue, expiry = ordered
    if int(issue[:4]) - int(birth[:4]) < MINIMUM_DRIVING_AGE_YEARS or issue >= expiry:
        return []
    candidates: list[FieldCandidate] = []
    for path, value in (
        ("national_driving_licence.issue_date", issue),
        ("national_driving_licence.expiry_date", expiry),
    ):
        line = dated[value]
        candidates.append(FieldCandidate(
            field_path=path, value=value, normalized_value=value,
            source_document=source, source_method="document_parser",
            confidence=min(0.94, line.confidence * 0.95),
            evidence_text=line.text, bounding_box=line.bounding_box,
            validation_passed=True,
            warnings=["ALGERIAN_LICENCE_FRONT_DATES_SEPARATED_FROM_CATEGORY_TABLE"],
        ))
    return candidates


def _clean_designator_value(raw: str) -> str:
    """Strip the Arabic label an RTL numbered row prints beside its value.

    The Algerian licence is the EU numbered card laid out right to left: each
    row reads "1. BENALI اللقب", so the label sits after the value on the same
    OCR line. Without this the surname was stored as "BENALI اللقب" and the word
    الاسم -- "name" -- became a middle name.
    """
    return _strip_rtl_label_residue(raw.strip(" :#-"), is_gcc_document=False)


# Tunisia numbers a licence to its own scheme and prints the key on the card:
# "1. Numéro du permis de conduire  2. Date de délivrance  3. Nom  4. Prénom
#  5. Date et lieu de naissance". Read against the EU meaning, designator 5 put
# the holder's date of birth into the licence-number field and designator 1 put
# the licence number into the surname. The card states its own key, so that is
# what selects the mapping.
_TUNISIAN_LEGEND = re.compile(
    r"NUM[EÉ]RO\s+DU\s+PERMIS\s+DE\s+CONDUIRE|R[EÉ]PUBLIQUE\s+TUNISIENNE", re.I,
)

_TUNISIAN_MAPPING = {
    "1": "national_driving_licence.number",
    "2": "national_driving_licence.issue_date",
    "3": "personal_info.last_name",
    "4": "personal_info.first_name",
    "5": "personal_info.date_of_birth",
}


# A designator and the value it names are one printed row, but OCR returns one
# box per cluster of ink and the gap between "5" and the number beside it is
# wide enough to split them. On the Albanian licence only rows 4b and 4d
# survived as single lines: the surname, the given name, the birth date, the
# issue date, the authority and the licence number itself each arrived as a
# bare designator plus a separate value, matched nothing, and were left to the
# model to guess. The row is still visible in the layout -- the designator sits
# immediately left of its value, on the same line of the card -- so rejoin them
# there.
# The trailing comma is there because that is what a recogniser returns for the
# full stop after a designator: the Bulgarian licence's "5." came back as "5,"
# from one pass and "5" from another, and only the second was recognised.
def designator_name(text: str) -> str:
    """The designator a row carries, however the print spaced it.

    "4b" is one designator, but it is set as a figure and a letter, and a
    recogniser reports what it sees: a French licence in this project's bug
    report returned "4 b.18.01.2034". Matched as the two characters "4B", that
    row was not a designator at all, and the expiry date -- the one date a
    rental is refused on -- was reported missing from a card that prints it
    plainly.
    """
    return text.upper().replace(" ", "")


_BARE_DESIGNATOR = re.compile(r"^\s*(4\s*[ABCD]|[1235])\s*[.,):\-]*\s*$", re.I)

# A worn or hologram-covered 4a marker can lose only its suffix and arrive as
# ``4.``.  It is not enough to see a date somewhere on the card: that date may
# belong to a vehicle-category table on the reverse.  The marker must be on
# the same printed row, immediately to the left of the date it is allowed to
# recover.
_ORPHANED_4A_DESIGNATOR = re.compile(r"^\s*4\s*[.,):\-]*\s*$", re.I)


_RIGHT_TO_LEFT = re.compile(r"[\u0590-\u05FF\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]")
_BIDI_SPLIT_FOUR_SERIES = re.compile(
    rf"^\s*4\s*[.,):\-]?\s*(?P<between>.*?)"
    rf"(?<![0-9A-Za-z])(?P<letter>[ABCD])\s*[.,):\-]\s*"
    rf"(?P<value>{DATE_PATTERN})",
    re.I | re.S,
)


def _normalize_numbered_designator_text(text: str) -> str:
    """Repair script-confusable letters only in an EU field designator.

    Greek OCR commonly returns the Latin ``b`` in printed ``4b`` as Greek
    beta (``β``/``Β``).  Replacing it only when it immediately follows 4 keeps
    ordinary Greek content untouched while preserving the standard's meaning.
    """
    normalized = re.sub(r"(?<=4)[βΒ]", "B", text)
    # The sub-letter of a 4-series designator is set smaller than the figure,
    # and a recogniser reading that size difference as punctuation returns it
    # bracketed: this Hungarian licence came back as "4(a). 2019.11.21" and
    # "4(b). 2029.03.25.". Neither was a designator to this reader, so both
    # rows fell through to the ordering fallback, which took the row above --
    # the holder's birth date -- for the issue date and the issue date for the
    # expiry. Every field on the card was legible and every one of them was
    # reported one row out. The brackets are dropped only from around a single
    # a-d immediately after the 4, so nothing else on the page is touched.
    normalized = re.sub(
        r"(?<=4)\s*[(\[{]\s*([A-Da-d])\s*[)\]}]", r"\1", normalized,
    )
    # On a worn card ``5. 230300356`` can be read as ``s.230300356``.  This
    # function is used only by the numbered-national-licence parser, and the
    # repair requires a complete identifier-shaped value with a digit, so an
    # ordinary word beginning with S cannot become the licence-number field.
    confusable_five = re.match(
        r"^(\s*)[Ss](\s*[.,):\-]+\s*)([A-Z0-9][A-Z0-9\-/\s]{3,24})\s*$",
        normalized, re.I,
    )
    if confusable_five is not None and any(
        character.isdigit() for character in confusable_five.group(3)
    ):
        return f"{confusable_five.group(1)}5{confusable_five.group(2)}{confusable_five.group(3)}"
    # A card that prints Hebrew or Arabic beside its Latin fields is a
    # bidirectional line, and the recogniser returns such a line in logical
    # order, not in the order it is printed. That splits ``4a.`` in two: the
    # Israeli front below came back as ``4. 4 <hebrew> a.18.12.2022``, with the
    # figure emitted at the head of the row and its sub-letter carried across
    # the Hebrew run to sit against the Gregorian date. Neither half is a
    # designator on its own, so the issue date -- printed, legible, and read at
    # 0.96 -- was reported as absent while 4b one row below came through
    # whole. The two halves are rejoined only where right-to-left script
    # actually stands between them, which is the condition that causes the
    # split, and only where a date follows the sub-letter.
    rejoined = _BIDI_SPLIT_FOUR_SERIES.match(normalized)
    if rejoined is not None and _RIGHT_TO_LEFT.search(rejoined.group("between")):
        return f"4{rejoined.group('letter').upper()}. {rejoined.group('value')}"
    # The "a" and "b" of 4a and 4b are single-storey letters on a card's small
    # print, and a recogniser that cannot read one returns a digit in its
    # place: "48.16.06.2025" on one blurred French licence, "42.06.12.2023" on
    # another. Any digit standing where the letter belongs is the same damage,
    # so it is repaired once here rather than one spelling at a time -- but
    # which of the two dated rows it is cannot be told from the row alone, so
    # the marker is left undecided and the page settles it below. The repair
    # applies only where a date follows immediately, so an ordinary figure at
    # the head of a row -- a street number, a height, a class -- cannot become
    # a date field.
    return re.sub(
        rf"^(\s*)4\d(?=\s*[.,):\-]\s*{DATE_PATTERN})",
        r"\g<1>4X", normalized, flags=re.I,
    )


def _page_width(lines: list[OCRLine]) -> float:
    return max((_line_rect(line)[2] for line in lines if line.bounding_box), default=0.0)


_DESIGNATOR_SEQUENCE = ("1", "2", "3", "4A", "4B", "4C", "4D", "5")


def _designator_column(lines: list[OCRLine]) -> list[tuple[str, OCRLine]]:
    """The bare designators that form a printed column, in card order.

    A lone digit proves nothing: a category table, a page number and the
    micro-printing along the edge of a card all produce one. What identifies
    the real thing is that a numbered card prints its designators down a
    single column, in the order the standard fixes them. Both were needed --
    the reverse of this Albanian licence, photographed together with the
    front, carries a stray "5" above a stray "2" at the right-hand edge, and
    they share a column closely enough to pass on alignment alone.
    """
    bare = [
        (designator_name(match.group(1)), line)
        for line, match in (
            (line, _BARE_DESIGNATOR.match(_normalize_numbered_designator_text(line.text)))
            for line in lines
        )
        if match is not None and line.bounding_box
    ]
    if len(bare) < 2:
        return []
    tolerance = max(8.0, 0.02 * _page_width(lines))
    columns: list[list[tuple[str, OCRLine]]] = []
    for entry in sorted(bare, key=lambda item: _line_rect(item[1])[0]):
        left = _line_rect(entry[1])[0]
        for column in columns:
            if abs(left - _line_rect(column[0][1])[0]) <= tolerance:
                column.append(entry)
                break
        else:
            columns.append([entry])
    ordered: list[tuple[str, OCRLine]] = []
    singletons: list[list[tuple[str, OCRLine]]] = []
    for column in columns:
        if len(column) < 2:
            singletons.append(column)
            continue
        column.sort(key=lambda item: _line_rect(item[1])[1])
        positions = [_DESIGNATOR_SEQUENCE.index(designator) for designator, _ in column]
        # Equal positions are the same row read twice by two image variants.
        if all(before <= after for before, after in zip(positions, positions[1:])):
            ordered.extend(column)
    # A designator standing on its own, once the card has already proved it
    # numbers its fields.
    #
    # The column rule is what tells a real designator from a stray digit, and
    # it holds -- but it assumes every designator joins the same stack. The
    # Bulgarian licence does not stack them: 1, 2, 3 and the 4a-4f block run
    # down the left, and 5, the licence number, is printed by itself under the
    # photograph at the other side of the card. Alone in its column it was
    # discarded, and a number the recogniser had read at full confidence was
    # reported as absent.
    #
    # Requiring a proven column elsewhere on the page is what keeps this from
    # reopening the door the rule was built to shut: the stray "5" over a
    # stray "2" on the edge of an Albanian card has no such column to appeal
    # to, and a page carrying a real one is a numbered card by then anyway.
    if ordered:
        known = {designator for designator, _ in ordered}
        for column in singletons:
            designator, line = column[0]
            if designator not in known:
                ordered.append((designator, line))
                known.add(designator)
    return ordered


# ``image_processing._estimate_skew`` levels a page that is up to 15 degrees
# off and refuses anything past that on purpose: beyond that angle its
# whole-page ``minAreaRect`` measures the frame rather than the print. A card
# photographed at 60 degrees therefore reaches extraction exactly as shot, and
# every layout rule below that asks "beside" or "on the same row" -- all of
# them axis-aligned tests -- silently stops matching. An Italian front turned
# that far had ``5.`` and ``VA5656782B`` read at full confidence and reported
# the licence number as absent. The rows themselves record which way the card
# was turned, so the page is replotted upright from its own baselines before
# those rules run. Below 20 degrees the deskew step has already done this;
# past 80 the page is a quarter turn, which the orientation classifier owns
# and where "column" and "row" trade places.
_TEXT_FRAME_MINIMUM_ANGLE = 20.0
_TEXT_FRAME_MAXIMUM_ANGLE = 80.0
_TEXT_FRAME_AGREEMENT_DEGREES = 7.0
_TEXT_FRAME_MINIMUM_CONFIDENCE = 0.90
_TEXT_FRAME_MAJORITY = 0.60


def _reading_direction(line: OCRLine) -> tuple[float, float] | None:
    """The angle this OCR row reads at, and how long the row is.

    A row's quadrilateral starts at its first character, so the first edge is
    the direction the text runs in. That edge is trusted only where it is
    clearly the longer one: a two-character box like ``5.`` is square enough
    that either edge could be taken for the baseline.
    """
    box = line.bounding_box
    if not box or len(box) < 3:
        return None
    (x0, y0), (x1, y1), (x2, y2) = box[0], box[1], box[2]
    along = math.hypot(x1 - x0, y1 - y0)
    across = math.hypot(x2 - x1, y2 - y1)
    if along < 1.0 or along < 1.5 * across:
        return None
    angle = math.degrees(math.atan2(y1 - y0, x1 - x0))
    if angle > 90.0:
        angle -= 180.0
    elif angle <= -90.0:
        angle += 180.0
    return angle, along


def _page_text_angle(lines: list[OCRLine]) -> float:
    """How far off level this page's printing runs, or 0.0 when it is level.

    A row votes with its length times how sure the recogniser is of it, and
    only rows read above ``_TEXT_FRAME_MINIMUM_CONFIDENCE`` vote at all. Both
    halves matter. A long row's baseline is measured over many characters
    while a short one is mostly box padding -- but length alone is not enough,
    because the guilloche under an Italian licence returns dozens of long
    garbled reads of the words milled into the background, and those boxes
    point wherever the pattern happens to run. What the recogniser could
    actually read is what was actually printed as a line of text.

    The winning direction must also carry most of that weight: rows pointing
    several ways are a card photographed beside its own reverse, or a legend
    fanned around a title, neither of which is one page to re-plot.
    """
    readings: list[tuple[float, float]] = []
    for line in lines:
        if line.confidence <= _TEXT_FRAME_MINIMUM_CONFIDENCE:
            continue
        measured = _reading_direction(line)
        if measured is None:
            continue
        angle, length = measured
        readings.append((
            angle, length * (line.confidence - _TEXT_FRAME_MINIMUM_CONFIDENCE),
        ))
    if len(readings) < 3:
        return 0.0
    total = sum(weight for _, weight in readings)
    if total <= 0.0:
        return 0.0
    best_angle, best_weight, best_count = 0.0, 0.0, 0
    for angle, _ in readings:
        near = [
            item for item in readings
            if abs(item[0] - angle) <= _TEXT_FRAME_AGREEMENT_DEGREES
        ]
        weight = sum(item[1] for item in near)
        if weight > best_weight:
            centre = sum(item[0] * item[1] for item in near) / weight
            best_angle, best_weight, best_count = centre, weight, len(near)
    if best_count < 3 or best_weight < _TEXT_FRAME_MAJORITY * total:
        return 0.0
    if not _TEXT_FRAME_MINIMUM_ANGLE <= abs(best_angle) <= _TEXT_FRAME_MAXIMUM_ANGLE:
        return 0.0
    return best_angle


def _upright_view(
    lines: list[OCRLine],
) -> tuple[list[OCRLine], dict[int, OCRLine]] | None:
    """This page's rows re-plotted level, with each copy's original beside it.

    Only the coordinates are turned. Every reading, confidence and evidence
    box the report goes on to quote stays the one the recogniser produced, so
    a value recovered here is reported at the place it was actually printed.
    """
    angle = _page_text_angle(lines)
    if not angle:
        return None
    radians = math.radians(-angle)
    cos_a, sin_a = math.cos(radians), math.sin(radians)
    turned: list[tuple[OCRLine, list[list[float]]]] = []
    for line in lines:
        if not line.bounding_box:
            continue
        turned.append((line, [
            [x * cos_a - y * sin_a, x * sin_a + y * cos_a]
            for x, y in line.bounding_box
        ]))
    if not turned:
        return None
    left = min(point[0] for _, box in turned for point in box)
    top = min(point[1] for _, box in turned for point in box)
    replotted: list[OCRLine] = []
    origin: dict[int, OCRLine] = {}
    for line, box in turned:
        copy = replace(
            line,
            bounding_box=[[x - left, y - top] for x, y in box],
        )
        replotted.append(copy)
        origin[id(copy)] = line
    return replotted, origin


def _rows_from_upright_view(
    lines: list[OCRLine],
    read: Callable[[list[OCRLine]], list[tuple[str, str, OCRLine]]],
) -> list[tuple[str, str, OCRLine]]:
    """Read a rotated page's numbered layout, and answer in its own frame."""
    view = _upright_view(lines)
    if view is None:
        return []
    replotted, origin = view
    return [
        (designator, value, origin[id(line)])
        for designator, value, line in read(replotted)
        if id(line) in origin
    ]


def _paired_designator_rows(lines: list[OCRLine]) -> list[tuple[str, str, OCRLine]]:
    """Rejoin a designator OCR split away from the value printed beside it."""
    rows: list[tuple[str, str, OCRLine]] = []
    for designator, marker in _designator_column(lines):
        left, top, right, bottom = _line_rect(marker)
        height = max(1.0, bottom - top)
        best: tuple[float, OCRLine] | None = None
        for line in lines:
            if line is marker or not line.bounding_box:
                continue
            if _BARE_DESIGNATOR.match(line.text) or len(line.text.strip()) < 2:
                continue
            if line.confidence < 0.5:
                # Binding by position is weaker than reading a printed row, so
                # it is not spent on text the recognizer itself doubts.
                continue
            value_left, value_top, _, value_bottom = _line_rect(line)
            # To the right of the designator, allowing the slight box overlap
            # OCR produces when the two are close.
            if value_left <= left or value_left < right - 0.5 * (right - left):
                continue
            overlap = min(bottom, value_bottom) - max(top, value_top)
            if overlap < 0.35 * min(height, max(1.0, value_bottom - value_top)):
                continue                      # a different printed row
            # Designator 5 is the licence number.  Several EU cards put that
            # number beneath the photo, farther right than the date rows: the
            # Slovak front placed ``M0730824`` 160px from its 5 marker while
            # the marker itself was only 32px high.  The ordinary four-row
            # window rejected an otherwise exact, high-confidence reading and
            # sent the whole card through later OCR/VLM fallbacks.  Keep the
            # narrower limit for every other designator; 5 is a number-only
            # field and its independently proven designator column makes the
            # slightly wider same-row window safe.
            maximum_gap = (6 if designator == "5" else 4) * height
            if value_left - right > maximum_gap:
                continue                      # across the card, not beside it
            # Nearest first, and where two renderings return the same box, the
            # one the recogniser is surer of. Position alone let a 0.86 reading
            # spelled "2 8 5 1 1 0 62 1" stand in front of the 1.00 reading of
            # the same row, and the licence number was then thrown away for
            # having spaces in it.
            ranking = (value_left, -line.confidence)
            if best is None or ranking < best[0]:
                best = (ranking, line)
        if best is not None:
            rows.append((designator, _clean_designator_value(best[1].text), best[1]))
    return rows


def _inline_supported_number_rows(
    lines: list[OCRLine], inline_rows: list[tuple[str, str, OCRLine]],
) -> list[tuple[str, str, OCRLine]]:
    """Bind a lone split 5 only below a proven inline 1/2/3 field column.

    Google read the Russian front's names and birth date as complete rows,
    but returned ``5.`` and its number separately. The bare-column rule quite
    deliberately rejects a lone digit. Three valid inline fields, aligned in
    order above it, provide the missing evidence without weakening that rule.
    """
    already_bound = {id(line) for _, line in _designator_column(lines)}
    recovered: list[tuple[str, str, OCRLine]] = []
    for marker in lines:
        match = _BARE_DESIGNATOR.fullmatch(marker.text)
        if (
            match is None or designator_name(match.group(1)) != "5"
            or id(marker) in already_bound or not marker.bounding_box
            or marker.confidence < 0.75
        ):
            continue
        left, top, right, bottom = _line_rect(marker)
        height = max(1.0, bottom - top)
        anchors: dict[str, list[OCRLine]] = {"1": [], "2": [], "3": []}
        for designator, raw, line in inline_rows:
            if (
                designator not in anchors or not line.bounding_box
                or line.confidence < 0.75 or line.variant != marker.variant
                or not _same_language_pass(marker, line)
                or _is_label_only_row(raw)
            ):
                continue
            x1, _, _, y2 = _line_rect(line)
            if abs(x1 - left) > max(8.0, 0.75 * height) or y2 >= top:
                continue
            if designator in {"1", "2"}:
                valid = _plausible_person_name(raw)
            else:
                date_match = re.search(DATE_PATTERN, ascii_numerals(raw), re.I)
                valid = date_match is not None and normalize_date(
                    date_match.group(0), day_first_hint=True,
                ).value is not None
            if valid:
                anchors[designator].append(line)
        if not any(
            _line_rect(surname)[3] < _line_rect(given)[1]
            and _line_rect(given)[3] < _line_rect(birth)[1]
            # A front card above a separate reverse in one photograph must
            # not authorize a stray digit far below on that second card.
            and top - _line_rect(birth)[3] <= 12 * height
            for surname in anchors["1"]
            for given in anchors["2"]
            for birth in anchors["3"]
        ):
            continue

        choices: dict[str, OCRLine] = {}
        for line in lines:
            if (
                line is marker or not line.bounding_box or line.confidence < 0.75
                or line.variant != marker.variant or not _same_language_pass(marker, line)
            ):
                continue
            x1, y1, _, y2 = _line_rect(line)
            overlap = min(bottom, y2) - max(top, y1)
            if not (
                right <= x1 <= right + 6 * height
                and overlap >= 0.5 * min(height, max(1.0, y2 - y1))
            ):
                continue
            raw = ascii_numerals(line.text).strip().upper()
            # Require the entire box to be one identifier, not a date, field
            # caption, control-number label, or number buried in other text.
            if _REFERENCE_ROW.search(raw) or re.fullmatch(DATE_PATTERN, raw, re.I):
                continue
            # A printed serial comes back with the gaps the card sets in it,
            # and they do not fall only between figures: the UK photocard
            # prints field 5 as "RANA9061023EM9ND 93", breaking between a
            # letter and a digit. Joining only digit to digit left that box
            # looking like two tokens rather than one identifier, and a
            # number read at 0.98 was reported as absent.
            #
            # What separates that from a caption standing beside a number is
            # not which characters the gap falls between but what is on each
            # side of it: every part of a printed serial carries a figure. A
            # run of letters alone is a word -- an authority's abbreviation,
            # a field name -- and a gap next to one is never closed, so
            # "GIBDD 1234" stays two tokens and binds to nothing.
            parts = raw.split()
            joined = (
                "".join(parts)
                if all(any(character.isdigit() for character in part) for part in parts)
                else raw
            )
            if not re.fullmatch(r"(?=[A-Z0-9/-]*\d)[A-Z0-9][A-Z0-9/-]{3,24}", joined):
                continue
            if joined not in choices or line.confidence > choices[joined].confidence:
                choices[joined] = line
        # Two different plausible numbers on the same row are not permission
        # to choose whichever one OCR rated higher. Leave that case unbound.
        if len(choices) == 1:
            line = next(iter(choices.values()))
            recovered.append(("5", _clean_designator_value(line.text), line))
    return recovered


_CONFUSABLE_4A_DESIGNATOR = re.compile(
    r"^\s*[IL|]\s*A\s*[.,):\-]*\s*$", re.I,
)


def _confusable_4a_date_rows(
    lines: list[OCRLine],
    marker_pattern: re.Pattern[str] | None = None,
    designator: str = "4A",
) -> list[tuple[str, str, OCRLine]]:
    """Bind a date beside an OCR-damaged ``4a`` field designator.

    On the reported Andorran licence, the Latin recognizer read the small
    printed ``4a`` marker as ``la`` while it read ``16/11/2021`` at 0.9999.
    The deferred Cyrillic recognizer happened to recognize that tiny marker
    correctly, so an otherwise complete Latin card incurred a second full-page
    OCR pass.  This is not a general letter substitution: the damaged marker
    must be a standalone token and must sit immediately left of one valid date
    on the same visual row.  Those two layout facts preserve the meaning of
    standard field 4a without allowing prose such as Catalan ``la`` to become
    an issue-date label.
    """
    recovered: list[tuple[str, str, OCRLine]] = []
    for marker in lines:
        if (
            (marker_pattern or _CONFUSABLE_4A_DESIGNATOR).fullmatch(marker.text) is None
            or not marker.bounding_box
        ):
            continue
        left, top, right, bottom = _line_rect(marker)
        height = max(1.0, bottom - top)
        best: tuple[tuple[float, float], OCRLine, str] | None = None
        for line in lines:
            if line is marker or not line.bounding_box:
                continue
            date_match = re.search(DATE_PATTERN, ascii_numerals(line.text), re.I)
            if date_match is None:
                continue
            if normalize_date(date_match.group(0), day_first_hint=True).value is None:
                continue
            value_left, value_top, _, value_bottom = _line_rect(line)
            overlap = min(bottom, value_bottom) - max(top, value_top)
            if overlap < 0.35 * min(height, max(1.0, value_bottom - value_top)):
                continue
            if value_left < right - 0.5 * (right - left):
                continue
            horizontal_gap = value_left - right
            if horizontal_gap > 4 * height:
                continue
            ranking = (horizontal_gap, -line.confidence)
            if best is None or ranking < best[0]:
                best = (ranking, line, date_match.group(0))
        if best is not None:
            recovered.append((designator, best[2], best[1]))
    return recovered


def _orphaned_marker_date_rows(lines: list[OCRLine]) -> list[tuple[str, str, OCRLine]]:
    """Bind the date beside a 4-series marker that lost its own sub-letter.

    An Israeli licence prints "4a." over its issue date and returned the
    marker as a bare "4." in its own box, with the date immediately to its
    right. Only 4a and 4b are dated, and which of the two this is cannot be
    told from a marker whose letter is gone -- so the row is left undecided
    and the page settles it, exactly as it does for a letter read as a digit.
    """
    return _confusable_4a_date_rows(lines, _ORPHANED_4A_DESIGNATOR, "4X")


def _has_orphaned_4a_date_row(lines: list[OCRLine]) -> bool:
    """Whether a date is visibly paired with an OCR-damaged ``4a`` marker."""
    for marker in lines:
        if (
            _ORPHANED_4A_DESIGNATOR.fullmatch(marker.text) is None
            or not marker.bounding_box
        ):
            continue
        left, top, right, bottom = _line_rect(marker)
        height = max(1.0, bottom - top)
        for line in lines:
            if line is marker or not line.bounding_box:
                continue
            if re.search(DATE_PATTERN, ascii_numerals(line.text), re.I) is None:
                continue
            value_left, value_top, _, value_bottom = _line_rect(line)
            overlap = min(bottom, value_bottom) - max(top, value_top)
            if overlap < 0.35 * min(height, max(1.0, value_bottom - value_top)):
                continue
            if value_left < right - 0.5 * (right - left):
                continue
            if value_left - right <= 4 * height:
                return True
    return False


def _aamva_date_rows(lines: list[OCRLine]) -> list[tuple[str, str, OCRLine]]:
    """Recover AAMVA dates split into designator, label and value boxes.

    The Illinois production image is read both as one imperfect contrast row
    (``4b ExP: 08122/2026``) and as three accurate normal-image boxes
    (``4b``, ``EXP:``, ``08/22/2026``).  The generic nearest-neighbour join
    stops at ``EXP:`` because it is the closest box.  For AAMVA 4a/4b we know
    the value is a U.S. date, so bind the closest valid date on the same visual
    row instead.
    """
    recovered: list[tuple[str, str, OCRLine]] = []
    for marker in lines:
        match = re.fullmatch(
            r"\s*(4\s*[AB])\s*(?:(?:ISS|EXP)(?:UE|IRY|IRES|DATE)?\s*)?[.):#\-]*\s*",
            marker.text, re.I,
        )
        if match is None or not marker.bounding_box:
            continue
        marker_left, marker_top, marker_right, marker_bottom = _line_rect(marker)
        marker_height = max(1.0, marker_bottom - marker_top)
        best: tuple[tuple[float, float, float], OCRLine, str] | None = None
        for line in lines:
            if line is marker or not line.bounding_box:
                continue
            date_match = re.search(DATE_PATTERN, ascii_numerals(line.text), re.I)
            if date_match is None:
                continue
            normalized = normalize_date(date_match.group(0), day_first_hint=False)
            if normalized.value is None:
                continue
            value_left, value_top, _, value_bottom = _line_rect(line)
            overlap = min(marker_bottom, value_bottom) - max(marker_top, value_top)
            same_row = overlap >= 0.30 * min(
                marker_height, max(1.0, value_bottom - value_top),
            )
            same_row = (
                same_row
                and value_left >= marker_left - marker_height
                and max(0.0, value_left - marker_right) <= 5 * marker_height
            )
            # Massachusetts stacks ``4b EXP`` over ``04/27/2027``.  The boxes
            # overlap by only a few pixels, too little to count as one line,
            # but their left edges and vertical gap still identify one field.
            stacked = (
                value_top >= marker_top - 0.25 * marker_height
                and value_top <= marker_bottom + 2.0 * marker_height
                and abs(value_left - marker_left) <= 1.5 * marker_height
            )
            if not same_row and not stacked:
                continue
            vertical_gap = max(0.0, value_top - marker_bottom)
            horizontal_gap = (
                abs(value_left - marker_left)
                if stacked else max(0.0, value_left - marker_right)
            )
            ranking = (vertical_gap, horizontal_gap, -line.confidence)
            if best is None or ranking < best[0]:
                best = (ranking, line, date_match.group(0))
        if best is not None:
            recovered.append((designator_name(match.group(1)), best[2], best[1]))
    return recovered


# The "d" of 4d is small print, and a recogniser that cannot read it returns a
# digit: this Massachusetts card's caption came back as "40 NUMBER". Accepting
# any digit there is safe because the rest of the caption still has to be one
# of AAMVA's number words, and because this pattern is used only on a licence
# already established as American, where 4d is the number the rental is keyed
# on. It is the same damage the dated designators take, repaired the same way.
_AAMVA_4D_NUMBER_LABEL = re.compile(
    r"^\s*4\s*[D0-9]\s*(?:LIC(?:ENSE)?\s*(?:NO|NUMBER)?|DL\s*(?:NO|NUMBER)?|NUMBER)"
    r"\s*[.:#-]*\s*$",
    re.I,
)


def _aamva_number_rows(lines: list[OCRLine]) -> list[tuple[str, str, OCRLine]]:
    """Recover an AAMVA 4d value printed directly below its label.

    Massachusetts stacks ``4d NUMBER`` over the customer identifier.  On the
    production capture, the normal rendering read the label while the
    illumination-flattened rendering read ``SA0111809`` below it.  Neither
    rendering therefore contained an inline ``4d <value>`` row, even though
    the merged OCR had both pieces at essentially perfect confidence.

    This is intentionally AAMVA-only: on the EU/Vienna card model 4d is a
    holder identifier, not the driving-licence number.
    """
    recovered: list[tuple[str, str, OCRLine]] = []
    for marker in lines:
        if (
            _AAMVA_4D_NUMBER_LABEL.fullmatch(marker.text) is None
            or not marker.bounding_box
        ):
            continue
        marker_left, marker_top, marker_right, marker_bottom = _line_rect(marker)
        marker_height = max(1.0, marker_bottom - marker_top)
        best: tuple[tuple[float, float, float], OCRLine, str] | None = None
        for line in lines:
            if line is marker or not line.bounding_box or line.confidence < 0.5:
                continue
            raw = " ".join(line.text.strip().split())
            if (
                not raw
                or _AAMVA_4D_NUMBER_LABEL.fullmatch(raw)
                or _is_label_only_row(raw)
                or _NOT_A_LICENCE_NUMBER.search(raw)
            ):
                continue
            if re.search(DATE_PATTERN, ascii_numerals(raw), re.I):
                # The card sets 4d, 4b and 3 side by side, and the recogniser
                # returned all three columns as one box:
                # "SA0111809 04/27/2027 04/27/2006". A licence number is never
                # a date, so the dates on that row are the neighbouring
                # columns' values and what is left over is this field's. Only
                # where exactly one token is left, so a row of unread
                # fragments is never guessed at.
                remainder = re.sub(
                    DATE_PATTERN, " ", ascii_numerals(raw), flags=re.I,
                ).split()
                if len(remainder) != 1:
                    continue
                raw = remainder[0]
            joined = re.sub(
                r"(?<=[A-Z0-9])\s+(?=[A-Z0-9])", "", ascii_numerals(raw).upper(),
            )
            number_match = re.fullmatch(r"[A-Z0-9][A-Z0-9\-/]{3,24}", joined)
            if number_match is None or not _looks_like_licence_number(joined):
                continue
            value_left, value_top, value_right, value_bottom = _line_rect(line)
            # The value may overlap the label box slightly (OCR boxes are not
            # typographic baselines), but it must begin at or immediately
            # below the label and share its left-hand column.
            if value_top < marker_top - 0.25 * marker_height:
                continue
            if value_top > marker_bottom + 2.0 * marker_height:
                continue
            if abs(value_left - marker_left) > 1.5 * marker_height:
                continue
            if value_right < marker_left or value_left > marker_right + 4.0 * marker_height:
                continue
            vertical_gap = max(0.0, value_top - marker_bottom)
            ranking = (vertical_gap, abs(value_left - marker_left), -line.confidence)
            if best is None or ranking < best[0]:
                best = (ranking, line, number_match.group(0))
        if best is not None:
            recovered.append(("4D", best[2], best[1]))
    return recovered


# The surname is designator 1 and the given name designator 2, printed one
# above the other in the same column on every card built to the EU and Vienna
# models. Where OCR loses the "1." itself -- it is two glyphs of grey ink, and
# on the Albanian card it came back as a dash -- the row is still identifiable
# by that layout, and losing it costs the one name the licence can be checked
# against the passport with. Inferred, never read, so it is marked and carries
# a confidence that keeps it in front of a person.
def _surname_above_given_name(
    rows: list[tuple[str, str, OCRLine]], lines: list[OCRLine],
) -> tuple[str, str, OCRLine] | None:
    if any(designator == "1" for designator, _, _ in rows):
        return None
    given = next((line for designator, _, line in rows if designator == "2"), None)
    if given is None or not given.bounding_box:
        return None
    bound = {id(line) for _, _, line in rows}
    left, top, _, bottom = _line_rect(given)
    tolerance = max(6.0, 0.02 * _page_width(lines))
    height = max(1.0, bottom - top)
    best: tuple[float, OCRLine] | None = None
    for line in lines:
        if id(line) in bound or not line.bounding_box:
            continue
        value = _clean_person_name(line.text)
        if not _plausible_person_name(value):
            continue
        folded = fold_for_match(value)
        if _is_label_only_row(value) or any(
            title in folded for title in _FOLDED_LICENCE_TITLES
        ):
            continue
        candidate_left, _, _, candidate_bottom = _line_rect(line)
        gap = top - candidate_bottom
        if abs(candidate_left - left) > tolerance or not -0.25 * height <= gap <= 1.2 * height:
            continue
        if best is None or gap < best[0]:
            best = (gap, line)
    if best is None:
        return None
    return ("1", _clean_designator_value(best[1].text), best[1])


def _given_name_below_surname(
    rows: list[tuple[str, str, OCRLine]], lines: list[OCRLine],
) -> list[tuple[str, str, OCRLine]]:
    """Recover the missing field-2 row from the standard numbered layout.

    The front of a European licence puts surname (1), given names (2), then
    the dated 4a/4b rows in one left-aligned stack.  Small 1/2 markers are
    often the first glyphs a photographed card loses.  In the reported
    Andorran image, both text rows were perfect -- ``TUILIER CURCO`` and
    ``KENNETH YANNICK`` -- while only marker 1 survived a second recognizer.
    The first-name field must retain the full given-name row rather than
    silently shortening it to the passport's first token.

    This remains layout evidence, not an unconstrained name guess: there must
    be a proven 4a or 4b row below, both name rows must be directly consecutive
    and aligned, and no explicit field-2 value may be present.
    """
    if any(designator == "2" for designator, _, _ in rows):
        return []
    dated = [
        line for designator, _, line in rows
        if designator in {"4A", "4B"} and line.bounding_box
    ]
    if not dated:
        return []
    first_date_top = min(_line_rect(line)[1] for line in dated)
    date_height = max(
        _line_rect(line)[3] - _line_rect(line)[1] for line in dated
    )
    surname = next((line for designator, _, line in rows if designator == "1"), None)
    bound = {id(line) for _, _, line in rows}

    def plausible(line: OCRLine) -> bool:
        if not line.bounding_box or id(line) in bound:
            return False
        value = _clean_person_name(line.text)
        return (
            bool(value)
            and _plausible_person_name(value)
            and not _is_label_only_row(value)
            and not any(title in fold_for_match(value) for title in _FOLDED_LICENCE_TITLES)
        )

    if surname is not None and surname.bounding_box:
        surname_left, _surname_top, _surname_right, surname_bottom = _line_rect(surname)
        surname_height = max(1.0, _line_rect(surname)[3] - _line_rect(surname)[1])
        options: list[tuple[float, OCRLine]] = []
        for line in lines:
            if not plausible(line):
                continue
            left, top, _right, _bottom = _line_rect(line)
            if top < surname_bottom - 0.25 * surname_height or top >= first_date_top:
                continue
            if top - surname_bottom > 1.5 * surname_height:
                continue
            if abs(left - surname_left) > max(12.0, 0.03 * _page_width(lines)):
                continue
            options.append((top - surname_bottom, line))
        if not options:
            return []
        line = min(options, key=lambda item: item[0])[1]
        return [("2", _clean_designator_value(line.text), line)]

    # Neither marker survived.  The two adjacent, aligned names immediately
    # above the dated fields are still the numbered 1/2 stack when the geometry
    # proves that relationship.
    options = [
        line for line in lines
        if plausible(line)
        and _line_rect(line)[3] <= first_date_top
        and first_date_top - _line_rect(line)[3] <= 5.0 * max(1.0, date_height)
    ]
    for upper, lower in zip(sorted(options, key=lambda line: _line_rect(line)[1]),
                            sorted(options, key=lambda line: _line_rect(line)[1])[1:]):
        upper_left, _upper_top, _upper_right, upper_bottom = _line_rect(upper)
        lower_left, lower_top, _lower_right, _lower_bottom = _line_rect(lower)
        upper_height = max(1.0, _line_rect(upper)[3] - _line_rect(upper)[1])
        if lower_top - upper_bottom > 1.5 * upper_height:
            continue
        if abs(lower_left - upper_left) > max(12.0, 0.03 * _page_width(lines)):
            continue
        return [
            ("1", _clean_designator_value(upper.text), upper),
            ("2", _clean_designator_value(lower.text), lower),
        ]
    return []


# A designator anywhere on the line, not only at its start. The lookbehind is
# what keeps it out of the middle of a number: the "5" of 1995 is preceded by a
# digit, so a birth row cannot be split at it.
_EMBEDDED_DESIGNATOR = re.compile(
    r"(?<![0-9A-Za-z])(4[ABCD]|[1235])\s*(?:[.):\-]+\s*|\s+)", re.I,
)


def _split_designator_row(text: str) -> list[tuple[str, str]]:
    """Separate the designators a card prints side by side on one line.

    Returns nothing for the ordinary one-field row, so the anchored read below
    still handles it -- this only speaks up where a second designator is
    genuinely there.
    """
    text = _normalize_numbered_designator_text(text)
    matches = list(_EMBEDDED_DESIGNATOR.finditer(text))
    if len(matches) < 2:
        return []
    pairs: list[tuple[str, str]] = []
    for position, match in enumerate(matches):
        end = (
            matches[position + 1].start()
            if position + 1 < len(matches) else len(text)
        )
        value = text[match.end():end].strip()
        if value:
            pairs.append((designator_name(match.group(1)), value))
    # Two designators that yielded one value between them is a misread, not a
    # shared row; the anchored read is the safer answer for it.
    return pairs if len(pairs) >= 2 else []


def _split_caption_row(line: OCRLine) -> list[OCRLine]:
    if not line.bounding_box:
        return []
    text = _normalize_numbered_designator_text(line.text)
    segments = _split_designator_row(text)
    if len(segments) < 2:
        return []
    if not all(_is_label_only_row(value) for _, value in segments):
        return []
    order = [
        _DESIGNATOR_SEQUENCE.index(designator)
        for designator, _ in segments
        if designator in _DESIGNATOR_SEQUENCE
    ]
    if len(order) != len(segments) or any(
        before >= after for before, after in zip(order, order[1:])
    ):
        return []
    matches = list(_EMBEDDED_DESIGNATOR.finditer(text))
    if len(matches) != len(segments) or not text:
        return []
    left, top, right, bottom = _line_rect(line)
    width = right - left
    if width <= 0:
        return []
    parts: list[OCRLine] = []
    for position, match in enumerate(matches):
        start = match.start()
        end = (
            matches[position + 1].start()
            if position + 1 < len(matches) else len(text)
        )
        x1 = left + width * start / len(text)
        x2 = left + width * end / len(text)
        parts.append(OCRLine(
            text[start:end].strip(), line.confidence,
            [[x1, top], [x2, top], [x2, bottom], [x1, bottom]],
            line.language, line.variant, line.model_name,
        ))
    return parts


# A designator prints as a number, optionally a letter, then a stop: "3.",
# "4a.", "12.". A caption carries exactly one -- its own. What follows the stop
# must not be another digit, or every dotted date on the card counts as two.
_FIELD_DESIGNATOR = re.compile(r"(?<![\w])\d{1,2}[a-e]?\s*\.(?!\s*\d)", re.I)

# How many different fields a row must name before it is a key to the document
# rather than the caption of one. A caption may name its field several times
# over, once per language, and a birth row may name the date and the place
# together, so two is not enough.
_LEGEND_DISTINCT_FIELDS = 3


@lru_cache(maxsize=1)
def _label_fields() -> tuple[tuple[str, str, str], ...]:
    """Every known label paired with the field it names.

    The four name paths count as one field: a row reading
    "Nome e Sobrenome / Name and Surname" names the holder once, in two
    languages, and must not look like a row naming two different things.
    """
    pairs: list[tuple[str, str, str]] = []
    for table in (
        FIELD_LABELS, PASSPORT_VIZ_LABELS,
        COMMON_NATIONAL_LABELS, LANGUAGE_FIELD_LABELS,
    ):
        for path, labels in table.items():
            if path.startswith("_"):
                continue
            field = "name" if path in _NAME_PATHS else path.rpartition(".")[2]
            for label in labels:
                pairs.append((compact_label(label), label, field))
    return tuple(dict.fromkeys(pairs))


def _is_field_legend_row(text: str) -> bool:
    """True where a row is the card's printed key rather than a caption.

    A licence prints, usually along its foot in small type, a list explaining
    what each numbered field on it means, in every language the document is
    issued in. Those rows are made entirely of captions and nothing else, so
    every rule that looks for a caption finds several, none of which names the
    value printed anywhere near it.

    Two signals say a row is enumerating fields rather than naming one: it
    carries more than one field designator, or it names three different
    fields. Either is enough, because OCR routinely loses the stops that make
    a designator recognisable.
    """
    # A key names fields; it never states one. A Swiss licence sets its issue
    # date, its masked expiry and its authority on one row -- "4a. 07.05.2021
    # 4b. ********** 4c. BE-CH" -- and the masked field left two designators
    # countable, so the row was taken for a key and the issue date with it.
    if any(
        normalize_date(match.group(0), day_first_hint=True).value
        for match in re.finditer(DATE_PATTERN, text, re.I)
    ):
        return False
    upper = text.upper().strip()
    if len(_FIELD_DESIGNATOR.findall(text)) >= 2:
        return True
    compact = compact_label(upper)
    fields = {
        field for folded, label, field in _label_fields()
        if folded in compact and label_pattern(label).search(upper)
    }
    return len(fields) >= _LEGEND_DISTINCT_FIELDS


def _drop_field_legend_rows(
    lines: list[OCRLine], doc_type: DocumentType | None = None,
) -> list[OCRLine]:
    """Remove the card's printed key, including the rows it wraps onto.

    Cards only. A passport's data page prints no key: its captions are the
    field captions, and two of them share a row wherever two columns meet --
    "6. Sugu/Sex/Sexe 7. Sünnikoht/Place of birth" on the Estonian page in this
    project's bug report. Read as a key, that row seeded a block that then grew
    down the whole left-aligned column, because a data page's captions and the
    values under them are exactly what this looks for: same left edge, same
    height, one line-space apart. The page lost its name, its sex, its birth
    date and both of its date rows -- captions and values together -- and the
    passport came back with a document number and nothing else.

    A Brazilian licence sets its key as one small paragraph, and only some of
    its lines are recognisable on their own -- the line that happens to hold
    two designators, or three field names. The rest are the same paragraph:
    left-aligned with a recognised line, the same height, one line-space
    apart. Taking the block whole is what stops "5. Número de registro da
    CNH/ Driver License Number/Número de Permiso de Conducir-9" from being
    read as the licence number, and stops "4a. Data de Emissão" from
    persuading the reader that the card's real, misread issue caption was
    already found somewhere.
    """
    if doc_type == DocumentType.PASSPORT_BIODATA:
        return lines
    boxed = [line for line in lines if line.bounding_box]
    if not boxed:
        return lines
    rects = {id(line): _line_rect(line) for line in boxed}
    legend = {id(line) for line in boxed if _is_field_legend_row(line.text)}
    if not legend:
        return lines
    for _ in range(len(boxed)):
        grown = set(legend)
        for line in boxed:
            if id(line) in legend:
                continue
            left, top, right, bottom = rects[id(line)]
            height = max(bottom - top, 1.0)
            for other in boxed:
                if id(other) not in legend:
                    continue
                oleft, otop, oright, obottom = rects[id(other)]
                other_height = max(obottom - otop, 1.0)
                if abs(left - oleft) > max(height, other_height):
                    continue
                if abs(height - other_height) > max(height, other_height) * 0.5:
                    continue
                gap = max(top - obottom, otop - bottom)
                if gap <= max(height, other_height):
                    grown.add(id(line))
                    break
        if grown == legend:
            break
        legend = grown
    return [line for line in lines if id(line) not in legend]


def _split_caption_rows(lines: list[OCRLine]) -> list[OCRLine]:
    expanded: list[OCRLine] = []
    for line in lines:
        parts = _split_caption_row(line)
        expanded.extend(parts or [line])
    return expanded


_GLUED_NUMERIC_DATE = r"\d{1,2}[-/.]\d{1,2}[-/.]\d{4}"
_GLUED_4A_4B_DATE_ROW = re.compile(
    r"^\s*4\s*A?\s*[.,):\-]*\s*(" + _GLUED_NUMERIC_DATE + r")"
    r"\s*4\s*B\s*[.,):\-]*\s*(" + _GLUED_NUMERIC_DATE + r")\s*$",
    re.I,
)


def _numbered_national_licence_candidates(
    lines: list[OCRLine], source: str, licence_country: str | None = None,
) -> list[FieldCandidate]:
    """Parse numbered licence fields, including the U.S. AAMVA variant.

    AAMVA and the EU/Vienna layouts share 1, 2, 3, 4a and 4b, but assign the
    two identifier rows differently.  On a U.S. AAMVA card 4d is the customer
    identifier (the licence number) and 5 is the document discriminator.  On
    the EU/Vienna model 5 is the licence number and 4d is an administrative
    holder identifier.  The issuing country must therefore select the mapping;
    guessing from a missing OCR row confuses two real, incompatible standards.
    """
    # The operator's country selection is one way to know the card model; the
    # card itself is the other, and the surer one. A Colombian passport beside
    # a Florida licence left the country resolved as Colombia, so the American
    # reading never engaged and the card's dates were read day first: an
    # expiry of 07/03/2024 came back as March, four months early.
    is_us_aamva = (
        licence_country == "United States" or _prints_aamva_field_codes(lines)
    )
    mapping = {
        "1": "personal_info.last_name",
        "2": "personal_info.first_name",
        "3": "personal_info.date_of_birth",
        "4A": "national_driving_licence.issue_date",
        "4B": "national_driving_licence.expiry_date",
        "4C": "national_driving_licence.issued_by_name",
        "5": "national_driving_licence.number",
        "2,1": "personal_info.full_name",
    }
    if _TUNISIAN_LEGEND.search(" ".join(line.text for line in lines)):
        mapping = dict(_TUNISIAN_MAPPING)
    # Designator 5 is the licence number on the EU and Vienna models, and 4d is
    # an administrative number -- Turkey puts its 11-digit national ID there.
    # Quebec reverses the two: 4d is the licence number and 5 is the "N° de
    # référence", a control number. Deciding per row would break one or the
    # other, so the page is inspected first: 4d only becomes the licence number
    # where 5 is absent or is visibly a reference row.
    rows: list[tuple[str, str, OCRLine]] = []
    for line in lines:
        # An EU card prints its field names once, set sideways along the edge:
        # "4c. Afgegeven door 5.Rijbewijsnummer" down the side of a Belgian
        # licence, "6. Fuhrerscheinnummer 10. Guitig ab" down an Austrian one.
        # Split as a numbered row it yields the field's own name as its value,
        # and "Rijbewijsnummer" then competes with the number the card states
        # plainly. A printed field row is wider than it is tall; that strip is
        # taller than the card is wide at that point.
        if line.bounding_box:
            left, top, right, bottom = _line_rect(line)
            if bottom - top > right - left:
                continue
        numbered_text = _normalize_numbered_designator_text(line.text)
        # A date whose leading zero was lost can look exactly like a numbered
        # row: ``05.06.2035`` becomes ``5.06.2035``, then the first ``5.`` is
        # mistaken for the EU licence-number designator and ``2035`` survives
        # as a competing serial. A standalone date is not a numbered field
        # row; reject it before splitting off any apparent designator. Keep the
        # match anchored to the whole OCR line so a genuine row such as
        # ``3. 02.11.2003`` or a side-by-side numbered row is unaffected.
        if re.fullmatch(
            DATE_PATTERN, ascii_numerals(numbered_text).strip(), re.I,
        ):
            continue
        # "2,1." is one row carrying both given names and surname, as on the Sri
        # Lankan licence. Reading it as designator 2 left the comma and the "1."
        # inside the value, and the guard below then discarded the whole row, so
        # the holder's name was lost outright. It is a combined row, so it is
        # treated as the full name and split like one.
        combined = re.match(r"^\s*2\s*,\s*1\s*[.)]?\s*[:\-]?\s*(.+?)\s*$", numbered_text)
        if combined is not None:
            rows.append(("2,1", _clean_designator_value(combined.group(1)), line))
            continue
        # Cards print two designators on one line wherever the layout is
        # tighter than one field per row: the Dutch licence sets 4a and 4b side
        # by side, the Italian 4a and 4c. Anchoring at the start of the line
        # read the first and swallowed the second into its value -- a Dutch
        # expiry date vanished into the issue row and was then supplied by the
        # ordering fallback instead of being read off the card.
        side_by_side = _split_designator_row(numbered_text)
        if side_by_side:
            rows.extend(
                (designator, _clean_designator_value(value), line)
                for designator, value in side_by_side
            )
            continue
        # The same side-by-side row with both separators lost to wear: an
        # Armenian card returned "429-12-2020 4b29-12-2030" for a row printed
        # "4a. 29-12-2020 4b. 29-12-2030". A designator glued to a fragment is
        # OCR noise and stays rejected; glued to a complete numeric date, and
        # paired with the 4b that follows it, the row can only be the issue and
        # expiry pair the standard fixes in that order.
        glued_pair = _GLUED_4A_4B_DATE_ROW.match(numbered_text)
        if glued_pair is not None:
            rows.append(("4A", _clean_designator_value(glued_pair.group(1)), line))
            rows.append(("4B", _clean_designator_value(glued_pair.group(2)), line))
            continue
        # A separator between the designator and its value is required. Cards
        # print "1." or "1 ", never "1SE" -- that shape is OCR running a stray
        # mark into a fragment, and accepting it made "SE" a surname and "Rbew"
        # a given name on an Albanian licence whose real name rows were lost.
        separator = (
            # U.S. cards commonly print a tiny designator directly against its
            # abbreviation ("4bEXP", "4dLIC NO").  OCR then preserves no
            # whitespace between them, so allow that exact AAMVA shape only on
            # a licence already established as American.
            r"(?:[.):\-]+\s*|\s+|(?=(?:DOB|ISS|EXP|LIC(?:ENSE)?|DLN|DL)\b))"
            if is_us_aamva else r"(?:[.):\-]+\s*|\s+)"
        )
        match = re.match(
            rf"^\s*(4\s*[ABCDX]|[1235])\s*{separator}(.+?)\s*$",
            numbered_text, re.I,
        )
        if match is not None:
            rows.append((designator_name(match.group(1)), _clean_designator_value(match.group(2)), line))
    # A date is never 4d. That cell holds an administrative identifier on the
    # EU model and the customer number on an AAMVA card, so a row read as
    # "4D 16.05.2039" -- as this French licence's was, its b read as a d -- is
    # the 4b row, which is the expiry the rental is refused on. Only where no
    # 4b row was read at all: a card that printed both is telling the truth.
    if not any(designator == "4B" for designator, _, _ in rows):
        rows = [
            ("4B", value, line)
            if designator == "4D"
            and normalize_date(value, day_first_hint=True).value
            else (designator, value, line)
            for designator, value, line in rows
        ]
    # A marker whose letter the recogniser replaced with a digit is one of the
    # two dated rows, and the page says which: a row read as 4a or 4b already
    # owns its field, so the undecided row takes whichever of the pair is
    # still unclaimed, and is discarded where both were read. Where neither
    # was read it is 4a -- the field a licence prints first, and the one this
    # damage has been reported on. Deciding it here, from the whole page,
    # rather than from the damaged row is what keeps an unreadable 4b from
    # handing an expiry date to the issue-date field.
    # A marker read without its sub-letter is recovered before the pair is
    # settled below, so the page can decide which of the two dated rows it is.
    read_lines = {id(line) for _, _, line in rows}
    for row in _orphaned_marker_date_rows(lines):
        if id(row[2]) not in read_lines:
            rows.append(row)
    if any(designator == "4X" for designator, _, _ in rows):
        claimed = {designator for designator, _, _ in rows}
        resolved: list[tuple[str, str, OCRLine]] = []
        for designator, value, line in rows:
            if designator != "4X":
                resolved.append((designator, value, line))
                continue
            free = [name for name in ("4A", "4B") if name not in claimed]
            if not free:
                continue
            # Where the page has not already settled it, the date itself does:
            # no document was issued after today, so a row dated in the future
            # is the expiry. A Greek card returned "46. 04.04.2033" for the row
            # printed "4β. 04.04.2033" and was reported as issued in 2033, with
            # no expiry at all -- a licence eight years from expiring, offered
            # as one issued eight years from now.
            issued = normalize_date(value, day_first_hint=True).value
            preferred = "4B" if issued and issued > date.today().isoformat() else "4A"
            name = preferred if preferred in free else free[0]
            claimed.add(name)
            resolved.append((name, value, line))
            other = "4B" if name == "4A" else "4A"
            if other in claimed or issued is None:
                continue
            # OCR returned one box for two printed rows and gave two readings
            # of it: the same rectangle came back as "46. 04.04.2033" and as
            # "REPUBLICO. 04.04.2018", the second being the 4a row with its
            # designator lost in the guilloche printed across it. That box
            # holds the pair, and an issue date precedes an expiry, so the
            # sibling row is read off the reading that is not this one.
            bound_lines = {id(one) for _, _, one in rows}
            left, top, right, bottom = _line_rect(line)
            area = max(1.0, (right - left) * (bottom - top))
            for other_line in lines:
                if other_line is line or not other_line.bounding_box:
                    continue
                if id(other_line) in bound_lines:
                    continue        # already read as a field of its own
                # The same rectangle, not merely a neighbouring one: this is
                # two readings of one box, and a row printed near it is a
                # different row with a field of its own.
                other_left, other_top, other_right, other_bottom = _line_rect(other_line)
                overlap = (
                    max(0.0, min(right, other_right) - max(left, other_left))
                    * max(0.0, min(bottom, other_bottom) - max(top, other_top))
                )
                union = (
                    area
                    + max(1.0, (other_right - other_left) * (other_bottom - other_top))
                    - overlap
                )
                if overlap / max(1.0, union) < 0.9:
                    continue
                found = re.search(
                    DATE_PATTERN, close_split_year(ascii_numerals(other_line.text)), re.I,
                )
                if found is None:
                    continue
                sibling = normalize_date(found.group(0), day_first_hint=True).value
                if sibling is None or sibling == issued:
                    continue
                if (sibling < issued) != (other == "4A"):
                    continue
                claimed.add(other)
                resolved.append((other, found.group(0), other_line))
                break
        rows = resolved
    # Rows OCR split into a designator and a value are recovered from layout;
    # a whole-line read suppresses only the split reading of that same row.
    layout_bound: dict[tuple[str, int], str] = {}
    inline: dict[str, list[OCRLine]] = {}
    for designator, _, line in rows:
        inline.setdefault(designator, []).append(line)
    paired_rows = (
        _paired_designator_rows(lines)
        or _rows_from_upright_view(lines, _paired_designator_rows)
    )
    # The new singleton recovery uses the EU/Vienna meaning of 5. Do not
    # apply it to Tunisia's alternate map or North American control numbers.
    inline_supported = (
        _inline_supported_number_rows(lines, rows)
        if mapping.get("5") == "national_driving_licence.number"
        and licence_country not in {"United States", "Canada"} else []
    )
    for row in [*paired_rows, *inline_supported]:
        # OCR variants often return one printed row both whole (``5. ABC``)
        # and split (``5.``, ``ABC``). Suppress that duplicate only when the
        # two readings occupy the same physical row. A reverse-side legend can
        # also say ``5. Licence number`` while the front-side value is split;
        # suppressing by designator alone discarded the real number from every
        # combined front/back image with that layout.
        if any(
            _same_text_region(row[2], line)
            for line in inline.get(row[0], ())
        ):
            continue
        layout_bound[(row[0], id(row[2]))] = (
            "NUMBER_DESIGNATOR_SUPPORTED_BY_INLINE_COLUMN"
            if any(row[2] is recovered[2] for recovered in inline_supported)
            else "DESIGNATOR_PAIRED_BY_LAYOUT"
        )
        rows.append(row)
    for row in _confusable_4a_date_rows(lines):
        # The marker itself is not a valid 4a reading; retain the geometry in
        # the evidence record so the audit trail says exactly why the date was
        # accepted.  A direct 4a row remains independent evidence and is not
        # removed by this recovery.
        layout_bound[(row[0], id(row[2]))] = (
            "CONFUSABLE_4A_DESIGNATOR_PAIRED_BY_LAYOUT"
        )
        rows.append(row)
    if is_us_aamva:
        for row in _aamva_date_rows(lines):
            layout_bound[(row[0], id(row[2]))] = "AAMVA_DATE_PAIRED_BY_LAYOUT"
            rows.append(row)
        for row in _aamva_number_rows(lines):
            layout_bound[(row[0], id(row[2]))] = "AAMVA_4D_NUMBER_PAIRED_BY_LAYOUT"
            rows.append(row)
    inferred = _surname_above_given_name(rows, lines)
    if inferred is not None:
        layout_bound[(inferred[0], id(inferred[2]))] = (
            "SURNAME_INFERRED_FROM_ROW_ABOVE_GIVEN_NAME"
        )
        rows.append(inferred)
    for row in _given_name_below_surname(rows, lines):
        layout_bound[(row[0], id(row[2]))] = (
            "STANDARD_NAME_ROWS_INFERRED_BY_LAYOUT"
        )
        rows.append(row)
    usable_five = not is_us_aamva and any(
        designator == "5" and not _REFERENCE_ROW.search(line.text)
        for designator, _, line in rows
    )
    # 4d only replaces 5 where the card says 5 is a reference number, as Quebec
    # does. A missing 5 means OCR lost the row, not that the card numbers its
    # fields differently: on every EU and Vienna card 4d is an administrative
    # number. Treating absence as permission bound an Albanian licence to the
    # holder's national ID -- K00721078T, which is an Albanian NID, not a
    # licence number -- because row 5 had simply not been read.
    #
    # Reading it off row 5 works only while row 5 is legible. On the Quebec
    # card that row is "N° de référence : R4MSA2R21" set in six-point type,
    # and its designator is small enough that a photographed card frequently
    # returns neither. The province itself settles the question -- it is the
    # province, not the row, that decides where the number is printed -- so a
    # Canadian card no longer depends on the reference row surviving OCR. The
    # wording test stays as the answer for a card that names no province.
    #
    # Gated on the bundle's country and not on the page text alone. There is a
    # town called Alberta in Virginia, and a page naming one in an address
    # would otherwise have moved that card's licence number onto 4d -- which on
    # a European card is the holder's national number, and is exactly the wrong
    # value in exactly the field the rental is keyed on.
    province = (
        province_from_text(" ".join(line.text for line in lines))
        if licence_country == "Canada" else None
    )
    quebec_style = (
        province is not None and province.licence_number_designator == "4D"
    ) or any(
        designator == "5" and _REFERENCE_ROW.search(line.text)
        for designator, _, line in rows
    )

    candidates: list[FieldCandidate] = []
    for designator, raw, line in rows:
        # Tanzania spells the field names out beside the designators and prints
        # each value on the line below: "1. Family name" then "MWANGI". Binding
        # the row's own text stored the words "Family name" as the surname and
        # "Issuing authority" as the authority. A row that is only a label has
        # no value on it; the labelled path picks the value up from below.
        if _is_label_only_row(raw):
            continue
        raw = _strip_leading_row_label(raw)
        if is_us_aamva:
            # The compact English abbreviations are part of the field label,
            # not the value.  Keep this local to AAMVA so "DD" in another
            # country's genuine identifier is never stripped.
            raw = re.sub(
                # DLN before DL: North Carolina captions the row "4d-DLN",
                # and "DL" followed by a letter is not the caption "DL", so
                # nothing was stripped and the abbreviation for the field's own
                # name was stored as the first three characters of the licence
                # number.
                r"^(?:DOB|ISS|EXP|LIC(?:ENSE)?\s*(?:NO|NUMBER)?"
                r"|DLN|DL\s*(?:NO|NUMBER)?)\b\s*[.:#-]*\s*",
                "", raw, flags=re.I,
            )
        if is_us_aamva and designator == "5":
            continue                      # AAMVA document discriminator, not DL number
        if designator == "5" and not usable_five:
            continue                      # a reference row, not the licence number
        if designator == "4D":
            if is_us_aamva and _looks_like_licence_number(raw):
                path = "national_driving_licence.number"
            elif not quebec_style or not _looks_like_licence_number(raw):
                # An administrative number, as on the EU model -- the holder's
                # national number, which is not the licence number but is the
                # identifier the passport also carries.
                path = "national_driving_licence.holder_id"
            else:
                path = "national_driving_licence.number"
        else:
            path = mapping[designator]
        value, normalized = raw, raw
        if path.endswith(("date_of_birth", "issue_date", "expiry_date")):
            date_match = re.search(DATE_PATTERN, close_split_year(raw), re.I)
            if date_match is None:
                continue
            value = date_match.group(0)
            normalized_date = normalize_date(
                value, day_first_hint=False if is_us_aamva else True,
            )
            if normalized_date.value is None:
                continue
            normalized = normalized_date.value
        elif path.endswith((".number", ".holder_id")):
            # A printed serial comes back with spaces dropped into it, and the
            # value is already isolated from its designator, so joining the
            # alphanumeric fragments is safe. The labelled path has done this
            # since it was written; the designator path had not, and a
            # Bulgarian licence whose number was returned as "2 8 5 1 1 0 62 1"
            # matched nothing and was reported as having no number at all.
            printed = ascii_numerals(raw).upper()
            if path.endswith(".number"):
                printed = _latin_identifier_glyphs(printed)
            joined = re.sub(r"(?<=[A-Z0-9])\s+(?=[A-Z0-9])", "", printed)
            number_match = re.search(
                r"(?=[A-Z0-9\-/]{4,25}\b)(?=[A-Z0-9\-/]*\d)[A-Z0-9][A-Z0-9\-/]*",
                joined, re.I,
            )
            if number_match is None:
                continue
            value = number_match.group(0)
            # A gap the card itself prints is part of how the number reads,
            # and the UK photocard sets one: "RANA9061023EM9ND 93". Scatter is
            # what joining was written for -- the Bulgarian row returned as
            # "2 8 5 1 1 0 62 1" -- and the two are told apart by shape, not
            # by country: one gap, between two blocks that each carry figures
            # and neither of which is a single character, is a printed gap and
            # is kept. Anything more broken than that is put back together.
            parts = printed.split()
            if (
                len(parts) == 2
                and "".join(parts) == value
                and all(
                    len(part) > 1 and any(character.isdigit() for character in part)
                    for part in parts
                )
            ):
                value = " ".join(parts)
            normalized = value.upper()
        elif path == "personal_info.full_name":
            raw = _clean_person_name(raw)
            if not _plausible_person_name(raw):
                continue
            value = normalized = raw
        elif path == "personal_info.first_name":
            raw = _clean_person_name(raw)
            if not raw or not _plausible_person_name(raw):
                continue
            # ``first_name`` is the CRM's given-names field, not just the
            # first token.  The EU model's field 2 can legally contain several
            # given names (``KENNETH YANNICK`` on the Andorran licence), and
            # taking ``split()[0]`` silently discarded every later one.
            value = normalized = raw
        elif path == "personal_info.last_name":
            raw = _clean_person_name(raw)
            if not _plausible_person_name(raw):
                continue
            value = normalized = raw
        elif path.endswith("issued_by_name"):
            if not any(character.isalpha() for character in raw):
                continue
            normalized = " ".join(raw.split())
        layout_warning = layout_bound.get((designator, id(line)))
        candidate = _candidate(
            path, value, line, source, True, normalized,
            [f"STANDARD_FIELD_DESIGNATOR:{designator}"]
            + ([layout_warning] if layout_warning else []),
            0.72 if layout_warning == "SURNAME_INFERRED_FROM_ROW_ABOVE_GIVEN_NAME"
            else 0.98 if layout_warning == "AAMVA_DATE_PAIRED_BY_LAYOUT"
            else 0.96 if layout_warning else 0.98,
        )
        candidate.source_method = "document_parser"
        candidates.append(candidate)
        if designator == "2" and len(raw.split()) > 1:
            middle = " ".join(raw.split()[1:])
            derived = _candidate(
                "personal_info.middle_name", middle, line, source, True, middle,
                ["DERIVED_FROM_NUMBERED_GIVEN_NAMES"], 0.98,
            )
            derived.source_method = "document_parser"
            candidates.append(derived)
    best = {}
    for path in ("personal_info.first_name", "personal_info.middle_name", "personal_info.last_name"):
        options = [candidate for candidate in candidates if candidate.field_path == path]
        if options:
            best[path] = max(options, key=lambda candidate: candidate.confidence)
    if "personal_info.first_name" in best and "personal_info.last_name" in best:
        name_parts = [best["personal_info.first_name"].normalized_value]
        if "personal_info.middle_name" in best:
            middle_name = best["personal_info.middle_name"].normalized_value
            # Field 2 is the complete given-names row, while middle_name is a
            # convenience projection of its later tokens.  Do not repeat that
            # projection in the derived full name.
            if not name_parts[0].endswith(f" {middle_name}"):
                name_parts.append(middle_name)
        name_parts.append(best["personal_info.last_name"].normalized_value)
        full_name = " ".join(part for part in name_parts if part)
        evidence = best["personal_info.first_name"]
        candidates.append(FieldCandidate(
            field_path="personal_info.full_name", value=full_name, normalized_value=full_name,
            source_document=source, source_method="document_parser",
            confidence=min(candidate.confidence for candidate in best.values()),
            evidence_text=evidence.evidence_text, bounding_box=evidence.bounding_box,
            validation_passed=True, warnings=["DERIVED_FROM_STANDARD_NUMBERED_FIELDS"],
        ))
    return candidates


# The vehicle categories of the EU and Vienna models, in the order the card
# prints them. Longest first when matching, so "C1E" is never read as "C1".
_LICENCE_CATEGORIES = (
    "AM", "A1", "A2", "A", "B1", "B", "BE", "C1", "C1E", "C", "CE",
    "D1", "D1E", "D", "DE",
    # National additions several states print in the same table.
    "F", "G", "T", "K", "L", "M",
)
_CATEGORY_ORDER = {code: position for position, code in enumerate(_LICENCE_CATEGORIES)}
_CATEGORY_ROW = re.compile(
    r"^\s*(" + "|".join(sorted(_LICENCE_CATEGORIES, key=len, reverse=True)) + r")\s*[.)]?\s*$",
)
# Designator 9 on the front summarises the same entitlement in one row.
_CATEGORY_SUMMARY_ROW = re.compile(r"^\s*9\s*[.)]?\s*(.+?)\s*$")


def _category_summary_rows(lines: list[OCRLine]) -> list[tuple[OCRLine, str]]:
    """Designator 9 on the front, whether or not OCR kept it in one box.

    The Italian licence prints "9. AM B" and OCR returned the "9." and the
    "AM B" as two boxes, so the row matched nothing and the entitlement was
    lost -- on a card whose reverse was too small to read the same codes from.
    """
    rows: list[tuple[OCRLine, str]] = []
    for line in lines:
        match = _CATEGORY_SUMMARY_ROW.match(line.text)
        if match is None:
            continue
        if match.group(1).strip(" .)"):
            rows.append((line, match.group(1)))
            continue
        if not line.bounding_box:
            continue
        left, top, right, bottom = _line_rect(line)
        height = max(1.0, bottom - top)
        beside = [
            other for other in lines
            if other is not line and other.bounding_box
            and _line_rect(other)[0] > left
            and _line_rect(other)[0] - right <= 4 * height
            and min(bottom, _line_rect(other)[3]) - max(top, _line_rect(other)[1])
            >= 0.35 * min(height, max(1.0, _line_rect(other)[3] - _line_rect(other)[1]))
        ]
        if beside:
            nearest = min(beside, key=lambda other: _line_rect(other)[0])
            rows.append((nearest, nearest.text))
    return rows


# A licence that grants each category separately dates each one separately,
# and prints those dates as a column under a single caption rather than beside
# it. The Moroccan reverse is laid out that way: "Date de délivrance" heads a
# column whose rows are the categories, and every category the holder does not
# hold is filled with asterisks. The caption is therefore two hundred pixels
# and six rows above the one date it names, far outside the row-below window
# every other label is bound by -- and rightly so, since reaching that far on
# an ordinary card swaps one field for its neighbour.
#
# What makes this case answerable is that the column holds exactly one date.
# A European reverse captions "10. Valid from" over a column of entitlement
# dates, one per category, and no single value there is the licence's own; a
# column with more than one date is left alone for that reason.
_COLUMN_HEADER_MAXIMUM_ROWS = 12


def _column_header_date_candidates(
    lines: list[OCRLine], source: str,
) -> list[FieldCandidate]:
    """Bind a date caption to the single date in the column beneath it."""
    found: list[FieldCandidate] = []
    for path in (
        "national_driving_licence.issue_date",
        "national_driving_licence.expiry_date",
    ):
        labels = COMMON_NATIONAL_LABELS.get(path, ())
        for label in lines:
            if not label.bounding_box:
                continue
            upper = label.text.upper().strip()
            compact = compact_label(upper)
            if not any(
                compact_label(wording) in compact
                and label_pattern(wording).search(upper)
                for wording in labels
            ):
                continue
            left, _top, right, bottom = _line_rect(label)
            height = max(1.0, bottom - _top)
            column: list[tuple[OCRLine, str]] = []
            for line in lines:
                if line is label or not line.bounding_box:
                    continue
                value_left, value_top, value_right, _bottom = _line_rect(line)
                if value_top < bottom:
                    continue                  # beside the caption or above it
                if value_top - bottom > height * _COLUMN_HEADER_MAXIMUM_ROWS:
                    continue                  # another block of the card
                overlap = min(right, value_right) - max(left, value_left)
                if overlap < 0.5 * min(right - left, max(1.0, value_right - value_left)):
                    continue                  # a different column
                match = re.search(DATE_PATTERN, close_split_year(line.text), re.I)
                if match is None:
                    continue
                normalized = normalize_date(match.group(0), day_first_hint=True).value
                if normalized is not None:
                    column.append((line, normalized))
            if len({value for _, value in column}) != 1:
                continue
            line, normalized = column[0]
            candidate = _candidate(
                path, match_text := line.text.strip(), line, source, True,
                normalized, ["LICENCE_DATE_UNDER_COLUMN_HEADING"], 0.9,
            )
            candidate.source_method = "document_parser"
            found.append(candidate)
            break
    return found


def national_licence_category_candidates(
    lines: list[OCRLine], source: str,
) -> list[FieldCandidate]:
    """Read which vehicles the holder may drive.

    The reverse of an EU or Vienna licence is a table: a column of category
    codes, and beside each one the dates it is valid between. A code with no
    dates against it is a category the holder does not hold -- the row is
    printed on every card whether or not it was granted -- so the dates are
    what distinguishes the entitlement from the form.

    Nothing here was being read at all, which on a car rental is the one fact
    the counter cannot do without: a licence carrying only AM or A is a moped
    or motorcycle licence and does not permit a car.
    """
    table: list[tuple[str, OCRLine]] = []
    for line in lines:
        match = _CATEGORY_ROW.match(line.text)
        if match is not None and line.bounding_box:
            table.append((match.group(1), line))
    granted: dict[str, OCRLine] = {}
    # Three rows is a table. One or two are a stray letter -- a "B" alone is
    # also a blood group, a class on an American licence and an OCR fragment.
    if len(table) >= 3:
        left_edge = min(_line_rect(line)[0] for _, line in table)
        for code, line in table:
            left, top, _, bottom = _line_rect(line)
            if left > left_edge + max(40.0, 0.08 * _page_width(lines)):
                continue              # a legend elsewhere on the card, not the column
            height = max(1.0, bottom - top)
            for other in lines:
                if other is line or not other.bounding_box:
                    continue
                other_left, other_top, _, other_bottom = _line_rect(other)
                if other_left <= left:
                    continue
                overlap = min(bottom, other_bottom) - max(top, other_top)
                if overlap < 0.35 * min(height, max(1.0, other_bottom - other_top)):
                    continue
                if re.search(DATE_PATTERN, ascii_numerals(other.text), re.I):
                    granted.setdefault(code, line)
                    break
    from_table = bool(granted)
    if not granted:
        for line, listed_text in _category_summary_rows(lines):
            listed = [
                token.strip(" .)")
                for token in re.split(r"[,;/+\s]+", listed_text.upper())
                if token.strip(" .)")
            ]
            if listed and all(token in _CATEGORY_ORDER for token in listed):
                for code in listed:
                    granted.setdefault(code, line)
                break
    if not granted:
        return []
    ordered = sorted(granted, key=lambda code: _CATEGORY_ORDER[code])
    evidence = granted[ordered[0]]
    return [FieldCandidate(
        field_path="national_driving_licence.categories",
        value=", ".join(ordered), normalized_value=", ".join(ordered),
        source_document=source, source_method="document_parser",
        confidence=min(0.90, evidence.confidence * 0.92),
        evidence_text=evidence.text, bounding_box=evidence.bounding_box,
        validation_passed=True,
        # The table is printed on the reverse and the summary row on the front,
        # so which one was read also says which side of the card was captured.
        warnings=[
            "LICENCE_CATEGORY_TABLE" if from_table else "LICENCE_CATEGORY_SUMMARY_ROW"
        ],
    )]


def _saudi_licence_layout_candidates(
    lines: list[OCRLine], source: str,
) -> list[FieldCandidate]:
    """Recover the Saudi DOB/issue/expiry rows when tiny labels are unreadable."""
    dated: dict[str, OCRLine] = {}
    for line in lines:
        text = ascii_numerals(line.text)
        for match in re.finditer(DATE_PATTERN, text, re.I):
            normalized = normalize_date(match.group(0), day_first_hint=True)
            if normalized.value is None:
                continue
            year = int(normalized.value[:4])
            if not 1900 <= year <= date.today().year + 30:
                continue
            current = dated.get(normalized.value)
            if current is None or line.confidence > current.confidence:
                dated[normalized.value] = line
    ordered = sorted(dated)
    if len(ordered) != 3:
        return []
    birth, issue, expiry = ordered
    if int(issue[:4]) - int(birth[:4]) < 15 or issue > expiry:
        return []
    candidates: list[FieldCandidate] = []
    for path, value in (
        ("personal_info.date_of_birth", birth),
        ("gcc_driving_licence.issue_date", issue),
        ("gcc_driving_licence.expiry_date", expiry),
    ):
        line = dated[value]
        candidates.append(FieldCandidate(
            field_path=path, value=value, normalized_value=value,
            source_document=source, source_method="document_parser",
            confidence=min(0.88, line.confidence * 0.92),
            evidence_text=line.text, bounding_box=line.bounding_box,
            validation_passed=True,
            warnings=["SAUDI_LICENCE_THREE_DATE_LAYOUT"],
        ))
    return candidates


def _saudi_tourist_licence_number_candidate(
    lines: list[OCRLine], source: str,
) -> list[FieldCandidate]:
    """Read the front identifier on the Saudi bilingual driving-licence layout.

    Tourist uploads do not know their issuing country until the bundle is
    reconciled, so they cannot rely on the GCC-only Saudi profile.  This card
    proves its own layout in English: the ten-digit identifier beside ``الرقم``
    is printed on the front, while the longer ``SN`` on the reverse is only the
    card serial and must never replace it.
    """
    page_text = " ".join(ascii_numerals(line.text).upper() for line in lines)
    if not (
        "KINGDOM OF SAUDI ARABIA" in page_text
        and "DRIVING LICENSE" in page_text
    ):
        return []
    found: dict[str, OCRLine] = {}
    for line in lines:
        text = ascii_numerals(line.text).upper()
        if re.search(r"\bS\s*N\b", text):
            continue                         # back-of-card serial number
        for match in re.finditer(r"(?<!\d)([12]\d{9})(?!\d)", text):
            value = match.group(1)
            current = found.get(value)
            if current is None or line.confidence > current.confidence:
                found[value] = line
    # A Saudi card can also print a traffic-file or other administrative
    # number.  Do not choose between two same-shaped values without a visible
    # label; the dedicated GCC path remains available when the customer chose
    # the GCC workflow.
    if len(found) != 1:
        return []
    value, line = next(iter(found.items()))
    return [FieldCandidate(
        field_path="national_driving_licence.number",
        value=value, normalized_value=value,
        source_document=source, source_method="document_parser",
        confidence=min(0.97, line.confidence), evidence_text=line.text,
        bounding_box=line.bounding_box, validation_passed=True,
        warnings=["SAUDI_LICENCE_FRONT_IDENTIFIER_LAYOUT"],
    )]


def _russian_licence_front_issue_date_candidate(
    lines: list[OCRLine], source: str,
) -> list[FieldCandidate]:
    """Recover 4a from the date cell immediately left of a Russian 4b row.

    On this layout 4a and 4b share one horizontal row.  The tiny 4a caption
    is commonly lost while the two dates and the 4b caption remain clear.  A
    date from the entitlement table is much lower on the reverse and cannot
    satisfy this same-row geometry, so this recovery never guesses from it.
    """
    page_text = " ".join(line.text.upper() for line in lines)
    # Latin OCR commonly reads the date row and ``GIBDD`` clearly while it
    # cannot read the licence title in Cyrillic.  GIBDD is the Russian traffic
    # authority, so it is sufficient layout evidence on its own; requiring the
    # Cyrillic title forced a second complete recognizer pass merely to enable
    # a same-row date rule that the first pass had all the data to prove.
    has_russian_title = "УДОСТОВЕРЕНИЕ" in page_text and "ВОДИ" in page_text
    has_russian_authority = "ГИБДД" in page_text or "GIBDD" in page_text
    if not (has_russian_title or has_russian_authority):
        return []
    expiry_rows: list[tuple[OCRLine, str]] = []
    for line in lines:
        text = ascii_numerals(line.text)
        if not re.search(r"\b4\s*[BВ]\s*[).:]", text, re.I):
            continue
        date_match = re.search(DATE_PATTERN, text, re.I)
        if date_match is None:
            continue
        expiry = normalize_date(date_match.group(0), day_first_hint=True).value
        if expiry is not None:
            expiry_rows.append((line, expiry))
    for expiry_line, expiry in expiry_rows:
        expiry_x1, expiry_y1, _expiry_x2, expiry_y2 = _line_rect(expiry_line)
        expiry_height = max(expiry_y2 - expiry_y1, 1.0)
        choices: list[tuple[float, OCRLine, str]] = []
        for line in lines:
            if line is expiry_line:
                continue
            x1, y1, x2, y2 = _line_rect(line)
            if x2 > expiry_x1 or abs((y1 + y2) / 2 - (expiry_y1 + expiry_y2) / 2) > max(42.0, expiry_height * 1.5):
                continue
            for date_match in re.finditer(DATE_PATTERN, ascii_numerals(line.text), re.I):
                issue = normalize_date(date_match.group(0), day_first_hint=True).value
                if issue is None or issue >= expiry:
                    continue
                # The nearest date to the left of 4b is its 4a partner.
                choices.append((expiry_x1 - x2, line, issue))
        if choices:
            _distance, line, issue = min(choices, key=lambda item: item[0])
            return [FieldCandidate(
                field_path="national_driving_licence.issue_date",
                value=issue, normalized_value=issue,
                source_document=source, source_method="document_parser",
                confidence=min(0.94, line.confidence), evidence_text=line.text,
                bounding_box=line.bounding_box, validation_passed=True,
                warnings=["RUSSIAN_LICENCE_4A_LEFT_OF_4B_LAYOUT"],
            )]
    return []


_GCC_DOCUMENT_TYPES = frozenset({
    DocumentType.GCC_IDENTITY_FRONT,
    DocumentType.GCC_IDENTITY_BACK,
    DocumentType.GCC_DRIVING_LICENCE_FRONT,
    DocumentType.GCC_DRIVING_LICENCE_BACK,
})


def _gcc_fragment_key(value: str) -> str:
    """Comparable GCC label text with harmless OCR punctuation removed."""
    return re.sub(r"[\s.:#\-/]+", "", value).upper()


def _gcc_name_script(value: str) -> str:
    """Return the visible script, independent of Paddle's language tag."""
    if re.search(r"[\u0600-\u06ff]", value):
        return "arabic"
    if re.search(r"[A-Za-z]", value):
        return "latin"
    return "other"


def _gcc_same_printed_row(first: OCRLine, second: OCRLine) -> bool:
    """Whether two OCR boxes are adjacent pieces of one printed GCC row."""
    first_left, first_top, first_right, first_bottom = _line_rect(first)
    second_left, second_top, second_right, second_bottom = _line_rect(second)
    first_height = max(1.0, first_bottom - first_top)
    second_height = max(1.0, second_bottom - second_top)
    overlap = min(first_bottom, second_bottom) - max(first_top, second_top)
    if overlap < 0.25 * min(first_height, second_height):
        return False
    horizontal_gap = max(
        first_left - second_right, second_left - first_right, 0.0,
    )
    return horizontal_gap <= max(300.0, 4.0 * max(first_height, second_height))


def _gcc_union_box(rows: list[OCRLine]) -> list[list[float]]:
    rectangles = [_line_rect(row) for row in rows]
    left = min(rectangle[0] for rectangle in rectangles)
    top = min(rectangle[1] for rectangle in rectangles)
    right = max(rectangle[2] for rectangle in rectangles)
    bottom = max(rectangle[3] for rectangle in rectangles)
    return [[left, top], [right, top], [right, bottom], [left, bottom]]


def augment_gcc_ocr_lines(
    lines: list[OCRLine], licence_country: str | None,
) -> list[OCRLine]:
    """Rejoin GCC labels and names that OCR split into adjacent boxes.

    Laminated GCC cards use large bilingual captions. Paddle commonly returns
    ``الرقم الشخصي`` as two boxes and a long Latin holder name as two or three
    boxes on the same row. The ordinary label binder intentionally consumes
    one neighbouring box only, so without a bounded layout repair the first
    fragment wins: Qatar's holder number becomes an identity number on a
    driving licence, and a one-token name cannot yield first/last name fields.

    This repair is restricted to the selected GCC profile, exact known labels,
    one OCR variant/language and one visual row. It never joins arbitrary page
    prose and does not change any non-GCC workflow.
    """
    profile = profile_for_gcc_country(licence_country)
    if profile is None or any(
        "+gcc-fragment-join" in line.model_name for line in lines
    ):
        return lines

    gcc_paths = (
        "personal_info.full_name", "personal_info.date_of_birth",
        "personal_info.nationality_name", "personal_info.gender",
        "gcc_identity.number", "gcc_identity.issue_date",
        "gcc_identity.expiry_date", "gcc_driving_licence.number",
        "gcc_driving_licence.issued_by_name",
        "gcc_driving_licence.issue_date",
        "gcc_driving_licence.expiry_date",
    )
    known_labels = tuple(dict.fromkeys([
        *profile.identity_titles,
        *profile.licence_titles,
        *(
            label
            for path in gcc_paths
            for label in (
                *FIELD_LABELS.get(path, ()),
                *gcc_labels_for_path(path, licence_country),
            )
        ),
    ]))
    targets = {
        _gcc_fragment_key(label): label
        for label in known_labels
        if len(_gcc_fragment_key(label)) >= 4
    }
    additions: list[OCRLine] = []
    seen = {
        (line.variant, line.language, _gcc_fragment_key(line.text))
        for line in lines
    }
    grouped: dict[tuple[str, str], list[OCRLine]] = {}
    for line in lines:
        grouped.setdefault((line.variant, line.language), []).append(line)

    # Exact two-piece labels only. Requiring a known target and shared visual
    # row is what makes it safe to recreate ``الرقم`` + ``الشخصي`` without
    # teaching the licence-number field to accept the dangerously broad word
    # ``الرقم`` on its own.
    for (variant, language), group in grouped.items():
        fragments = [
            line for line in group
            if not any(character.isdigit() for character in line.text)
            and 1 <= len(_gcc_fragment_key(line.text)) <= 24
            and any(
                _gcc_fragment_key(line.text) in target
                for target in targets
            )
        ]
        for first_index, first in enumerate(fragments):
            for second in fragments[first_index + 1:]:
                if not _gcc_same_printed_row(first, second):
                    continue
                for ordered in ((first, second), (second, first)):
                    joined_key = "".join(_gcc_fragment_key(line.text) for line in ordered)
                    canonical = targets.get(joined_key)
                    marker = (variant, language, joined_key)
                    if canonical is None or marker in seen:
                        continue
                    additions.append(OCRLine(
                        canonical,
                        min(line.confidence for line in ordered),
                        _gcc_union_box(list(ordered)),
                        language,
                        variant,
                        f"{first.model_name}+gcc-fragment-join",
                    ))
                    seen.add(marker)

    augmented = [*lines, *additions]
    index = build_line_index(augmented, known_labels)
    full_name_labels = tuple(dict.fromkeys((
        *FIELD_LABELS["personal_info.full_name"],
        *gcc_labels_for_path("personal_info.full_name", licence_country),
    )))

    # Recreate only a name row that has an explicit Name/الاسم caption and at
    # least two adjacent value boxes. A single box retains the established
    # behaviour, while unrelated words elsewhere on the card are unreachable.
    for position, label_line in enumerate(index.lines):
        matched_label = next((
            label for label in full_name_labels
            if label_pattern(label).search(index.uppers[position])
        ), None)
        if matched_label is None:
            continue
        label_left, label_top, label_right, label_bottom = index.rects[position]
        label_height = max(1.0, label_bottom - label_top)
        label_script = _gcc_name_script(label_line.text)
        rtl = index.is_rtl[position]
        values: list[OCRLine] = []
        for candidate_position, candidate in enumerate(index.lines):
            if candidate is label_line or index.is_label[candidate_position]:
                continue
            if (
                candidate.variant != label_line.variant
                # Paddle assigned the Latin Omani name to its Arabic stream in
                # the reported capture. The visible script is reliable here;
                # the recognizer's language metadata is not.
                or _gcc_name_script(candidate.text) != label_script
                or any(character.isdigit() for character in candidate.text)
                or not _plausible_person_name(candidate.text)
            ):
                continue
            candidate_left, candidate_top, candidate_right, candidate_bottom = (
                index.rects[candidate_position]
            )
            candidate_height = max(1.0, candidate_bottom - candidate_top)
            vertical_overlap = min(label_bottom, candidate_bottom) - max(
                label_top, candidate_top,
            )
            same_row = vertical_overlap >= 0.25 * min(
                label_height, candidate_height,
            )
            wrapped_row = (
                0.0 <= candidate_top - label_bottom
                <= max(80.0, 1.75 * max(label_height, candidate_height))
            )
            if not same_row and not wrapped_row:
                continue
            if rtl:
                if candidate_left > label_right + label_height:
                    continue
            elif candidate_right < label_left - label_height:
                continue
            values.append(candidate)
        values.sort(
            key=lambda line: (
                _line_rect(line)[1],
                -_line_rect(line)[0] if rtl else _line_rect(line)[0],
            ),
        )
        chain: list[OCRLine] = []
        cursor = label_left if rtl else label_right
        for candidate in values:
            candidate_left, _, candidate_right, _ = _line_rect(candidate)
            gap = (
                max(0.0, cursor - candidate_right)
                if rtl else max(0.0, candidate_left - cursor)
            )
            if gap > max(320.0, 4.0 * label_height):
                if chain:
                    break
                continue
            chain.append(candidate)
            cursor = candidate_left if rtl else candidate_right
            if len(chain) == 4:
                break
        if len(chain) < 2:
            continue
        joined_value = " ".join(candidate.text.strip() for candidate in chain)
        if not _plausible_person_name(joined_value):
            continue
        joined_text = f"{matched_label}: {joined_value}"
        joined_key = _gcc_fragment_key(joined_text)
        marker = (label_line.variant, label_line.language, joined_key)
        if marker in seen:
            continue
        additions.append(OCRLine(
            joined_text,
            min([label_line.confidence, *(line.confidence for line in chain)]),
            _gcc_union_box([label_line, *chain]),
            label_line.language,
            label_line.variant,
            f"{label_line.model_name}+gcc-fragment-join",
        ))
        seen.add(marker)

    return [*lines, *additions]


_GIVEN_NAME_CAPTIONS: tuple[str, ...] = tuple(dict.fromkeys((
    *PASSPORT_VIZ_LABELS.get("personal_info.first_name", ()),
    *PLURAL_GIVEN_NAME_LABELS,
    "GIVEN NAMES", "GIVEN NAME", "FORENAME", "FORENAMES", "FIRST NAME",
)))


def _rect(box: list[list[float]] | None) -> tuple[float, float, float, float]:
    xs = [point[0] for point in box or []]
    ys = [point[1] for point in box or []]
    return (min(xs), min(ys), max(xs), max(ys)) if xs else (0.0, 0.0, 0.0, 0.0)


def _given_names_under_a_shared_name_caption(
    lines: list[OCRLine], candidates: list[FieldCandidate], source: str,
) -> list[FieldCandidate]:
    """The given-names row of a passport that captions both name rows once.

    An Australian passport prints one "Name / Nom" over two rows: the surname
    on the first and the given names on the second, with no caption of its own.
    The label binds the row beneath it, the surname is read, and the row after
    that has nothing to bind to -- so the holder's given names, printed at full
    confidence, were reported absent whenever the machine-readable zone failed.
    Two of them failed in one week, for two unrelated reasons, and the zone is
    the only other place this passport states the name.

    The rule is confined to that layout. A page that captions its given names
    is excluded outright, which is most of the world's passports and every one
    the reader already handles. What remains must be the row immediately under
    the surname, in its column, shaped like a name and bound to nothing else.
    It is offered below the zone's confidence, so a readable zone still wins
    and a contradiction between the two is visible rather than silent.
    """
    if any(
        candidate.field_path == "personal_info.first_name"
        and candidate.normalized_value
        for candidate in candidates
    ):
        return []
    surname = next(
        (
            candidate for candidate in candidates
            if candidate.field_path == "personal_info.last_name"
            and candidate.normalized_value and candidate.bounding_box
        ),
        None,
    )
    if surname is None:
        return []
    page_text = " ".join(line.text for line in lines)
    if any(
        label_pattern(label).search(page_text)
        for label in _GIVEN_NAME_CAPTIONS
    ):
        return []
    left, _top, right, bottom = _rect(surname.bounding_box)
    height = max(1.0, bottom - _rect(surname.bounding_box)[1])
    spoken_for = {
        _rect(candidate.bounding_box) for candidate in candidates
        if candidate.bounding_box
    }
    options: list[tuple[float, OCRLine]] = []
    for line in lines:
        if not line.bounding_box or line.confidence < 0.6:
            continue
        rect = _line_rect(line)
        if rect in spoken_for:
            continue
        value = _clean_person_name(line.text)
        if (
            not value or not _plausible_person_name(value)
            or _is_label_only_row(value) or _non_primary_name_field_label(line.text)
        ):
            continue
        value_left, value_top, _value_right, _value_bottom = rect
        if value_top < bottom - 0.25 * height:
            continue
        if value_top - bottom > 1.6 * height:
            continue
        if abs(value_left - left) > max(12.0, 0.03 * _page_width(lines)):
            continue
        options.append((value_top - bottom, line))
    if not options:
        return []
    line = min(options, key=lambda item: item[0])[1]
    value = _clean_person_name(line.text)
    return [FieldCandidate(
        field_path="personal_info.first_name", value=value,
        normalized_value=value, source_document=source,
        source_method="document_parser",
        confidence=min(0.78, line.confidence * 0.8),
        evidence_text=line.text, bounding_box=line.bounding_box,
        validation_passed=True,
        warnings=["GIVEN_NAMES_ROW_UNDER_SHARED_NAME_CAPTION"],
    )]


def labelled_ocr_candidates(
    lines: list[OCRLine], doc_type: DocumentType, source: str,
    licence_country: str | None = None,
    allowed_paths: frozenset[str] | None = None,
) -> list[FieldCandidate]:
    """Bind visible labels to values for one document.

    ``allowed_paths`` restricts the lookup to the fields the calling workflow
    actually needs. The GCC route passes its thirteen customer fields, so no
    time is spent scanning for a blood group or licence class it will never
    show, and no such value can reach the result.
    """
    candidates: list[FieldCandidate] = []
    # Tourist pages are processed one at a time.  On the first uploaded page
    # the bundle-level country may not have been reconciled yet even though the
    # card's own TD1 zone already proves it.  Use that page-local evidence now
    # so Algerian layout rules do not depend on upload order.
    if (
        licence_country is None
        and doc_type in {
            DocumentType.NATIONAL_DRIVING_LICENCE_FRONT,
            DocumentType.NATIONAL_DRIVING_LICENCE_BACK,
        }
        and any(
            _ALGERIAN_LICENCE_MRZ_ROW.fullmatch(
                "".join(ascii_numerals(line.text).upper().split())
            )
            for line in lines
        )
    ):
        licence_country = "Algeria"
    if doc_type == DocumentType.PASSPORT_BIODATA:
        lines = _join_split_passport_dates(lines)
    if doc_type in _GCC_DOCUMENT_TYPES:
        lines = augment_gcc_ocr_lines(lines, licence_country)
    lines = _drop_field_legend_rows(_split_caption_rows(lines), doc_type)
    local_labels = country_labels(licence_country)
    # Whether this page carries a row labelled as the surname on its own. When
    # it does, a "name" row beside it holds given names only and must not be
    # split for a surname.
    # Boundary-aware, not a substring test: "NOM" sits inside the Portuguese
    # "NOME", which is a full-name label, and a substring match made a Brazilian
    # CNH look like a card that prints its surname separately.
    page_text = " ".join(line.text for line in lines)
    # "Vornamen" says the row above it is a surname just as plainly as the word
    # "Surname" does, and on a German passport it is the only thing that says so.
    page_prints_plural_given_names = (
        doc_type == DocumentType.PASSPORT_BIODATA
        and any(
            label_pattern(label).search(page_text)
            for label in PLURAL_GIVEN_NAME_LABELS
        )
    )
    # ...and a row that is the bare word and nothing else. Labels are matched as
    # substrings, and "NAME" sits inside "VOORNAMEN", "GIVEN NAMES" and
    # "SURNAME": a Belgian passport prints "1. Naam / Nom / Name / Surname" over
    # "2. Voornamen / Prenoms / Vornamen / Given names", and treating the bare
    # word as a surname label there captured the given-name row as well. A row
    # that carries only the bare word is the German layout, where the English
    # half that would have said "Surname" is missing or was lost to glare.
    page_prints_bare_name_row = page_prints_plural_given_names and any(
        _bare_label_row(text) in BARE_NAME_AS_SURNAME_LABELS
        for text in (line.text for line in lines)
    )
    page_prints_its_own_surname = page_prints_plural_given_names or any(
        label_pattern(label).search(page_text) for label in DEDICATED_SURNAME_LABELS
    )
    multilingual_types = {
        DocumentType.PASSPORT_BIODATA,
        DocumentType.INTERNATIONAL_DRIVING_PERMIT,
        DocumentType.NATIONAL_DRIVING_LICENCE_FRONT,
        DocumentType.NATIONAL_DRIVING_LICENCE_BACK,
        DocumentType.GCC_IDENTITY_FRONT,
        DocumentType.GCC_IDENTITY_BACK,
        DocumentType.GCC_DRIVING_LICENCE_FRONT,
        DocumentType.GCC_DRIVING_LICENCE_BACK,
    }
    if doc_type in multilingual_types:
        english_lines = [line for line in lines if line.language == "en"]
        # The Russian recognizer also attempts Latin rows, sometimes dropping
        # spaces (AMINA NOOR -> AMINANOOR). Prefer the dedicated English model
        # at overlapping Latin-only boxes and reserve Russian OCR for Cyrillic.
        lines = [
            line for line in lines
            if not (
                line.language == "ru"
                and not re.search(r"[А-ЯЁа-яё]", line.text)
                and any(
                    _duplicate_reading_of(line, english)
                    for english in english_lines
                )
            )
        ]
    if doc_type in {
        DocumentType.EMIRATES_ID_FRONT,
        DocumentType.EMIRATES_ID_BACK,
        DocumentType.UAE_DRIVING_LICENCE_FRONT,
        DocumentType.UAE_DRIVING_LICENCE_BACK,
        DocumentType.GCC_IDENTITY_FRONT,
        DocumentType.GCC_IDENTITY_BACK,
        DocumentType.GCC_DRIVING_LICENCE_FRONT,
        DocumentType.GCC_DRIVING_LICENCE_BACK,
    }:
        english_lines = [line for line in lines if line.language == "en"]
        # The Arabic recognizer is valuable for Arabic-only labels, but when it
        # re-reads a Latin row it may insert/drop a space inside its digits.
        # Do not let that second interpretation conflict with the dedicated
        # English recognizer at the same box -- unless it is the reading that
        # carries the field label.
        lines = [
            line for line in lines
            if not (
                line.language == "ar"
                and not re.search(r"[\u0600-\u06ff]", line.text)
                and _redundant_reading(line, english_lines, local_labels)
            )
        ]
    if doc_type == DocumentType.PASSPORT_BIODATA:
        # Russian passports print Cyrillic and Latin transliterations close
        # together. The English recognizer can turn the Cyrillic row into a
        # plausible-looking but wrong Latin name. When the Russian recognizer
        # confirms Cyrillic at the same box, suppress that duplicate English row.
        cyrillic_lines = [
            line for line in lines
            if line.language == "ru" and re.search(r"[А-ЯЁа-яё]", line.text)
        ]
        known_labels = tuple(label for labels in FIELD_LABELS.values() for label in labels)
        lines = [
            line for line in lines
            if not (
                line.language == "en"
                and not any(label in line.text.upper() for label in known_labels)
                and any(
                    _duplicate_reading_of(line, cyrillic)
                    for cyrillic in cyrillic_lines
                )
            )
        ]
    allowed_prefix = {
        DocumentType.EMIRATES_ID_FRONT: ("personal_info.", "emirates_id."),
        DocumentType.EMIRATES_ID_BACK: ("emirates_id.",),
        DocumentType.PASSPORT_BIODATA: ("personal_info.", "passport."),
        DocumentType.UAE_DRIVING_LICENCE_FRONT: ("personal_info.", "uae_driving_licence."),
        DocumentType.UAE_DRIVING_LICENCE_BACK: ("uae_driving_licence.",),
        DocumentType.GCC_IDENTITY_FRONT: ("personal_info.", "gcc_identity."),
        # The Bahraini and Kuwaiti identity cards print the holder's birth date
        # and sex on the reverse only, so the back must be allowed to supply
        # personal fields rather than card fields alone.
        DocumentType.GCC_IDENTITY_BACK: ("personal_info.", "gcc_identity."),
        DocumentType.GCC_DRIVING_LICENCE_FRONT: ("personal_info.", "gcc_driving_licence."),
        # Likewise the Omani licence prints birth date and nationality on its
        # reverse, beside the permitted classes.
        DocumentType.GCC_DRIVING_LICENCE_BACK: ("personal_info.", "gcc_driving_licence."),
        DocumentType.INTERNATIONAL_DRIVING_PERMIT: ("personal_info.", "international_driving_permit."),
        DocumentType.NATIONAL_DRIVING_LICENCE_FRONT: ("personal_info.", "national_driving_licence."),
        DocumentType.NATIONAL_DRIVING_LICENCE_BACK: ("national_driving_licence.",),
    }.get(doc_type, ())
    is_gcc_document = doc_type in {
        DocumentType.GCC_IDENTITY_FRONT, DocumentType.GCC_IDENTITY_BACK,
        DocumentType.GCC_DRIVING_LICENCE_FRONT, DocumentType.GCC_DRIVING_LICENCE_BACK,
    }
    # On a bilingual GCC card the Arabic label is recognized by the Arabic
    # engine while its Latin/numeric value comes from the English one, so these
    # shared personal fields must be allowed to bind across recognizers.
    gcc_bilingual_personal_paths = {
        "personal_info.date_of_birth", "personal_info.gender",
        "personal_info.nationality_name",
    } if is_gcc_document else set()
    # An American card writes its dates month first. Read day first, a Florida
    # licence expiring on 07/03/2024 was reported as expiring in March, four
    # months before it actually does, and a birth date of 12/31/1979 could not
    # be read at all. The card says which order it uses: the AAMVA designators
    # are printed on it and no other model uses them.
    # And where the designators are absent the bundle can still have proved
    # the country: New York captions its rows in words -- "DOB", "Issued",
    # "Expires" -- so no designator is printed, while the card's own AAMVA
    # barcode states DCG=USA. Read day first, that licence reported an issue
    # date of 05/31/2022 as no date at all (there is no thirty-first month)
    # and gave a birth date of 04/06/1998 as June, against the passport's
    # April, so the holder's own date of birth came back CONFLICTING. The
    # country speaks only for the licence: an Uzbek passport in the same
    # bundle still writes its dates day first.
    month_first = _prints_aamva_field_codes(lines) or (
        doc_type in {
            DocumentType.NATIONAL_DRIVING_LICENCE_FRONT,
            DocumentType.NATIONAL_DRIVING_LICENCE_BACK,
        }
        and (
            licence_country == "United States"
            or _prints_american_licence_captions(lines)
            or _prints_us_state_heading(lines)
        )
    )
    index = build_line_index(
        [*lines, *_rejoined_date_rows(lines)], local_labels,
    )
    if (
        is_gcc_document
        and not identity_issue_date_printed(profile_for_gcc_country(licence_country))
    ):
        # Of the five supported states only Qatar prints an issue date on the
        # identity card. Elsewhere any value reaching this field came from a
        # birth date, an expiry, a Hijri row or a card version, so the field
        # stays null rather than carrying a guess.
        allowed_paths = (
            allowed_paths if allowed_paths is not None
            else frozenset(FIELD_LABELS)
        ) - {"gcc_identity.issue_date"}
    for path, base_labels in FIELD_LABELS.items():
        if not any(path.startswith(prefix) for prefix in allowed_prefix): continue
        if allowed_paths is not None and path not in allowed_paths: continue
        labels = base_labels
        if doc_type in {DocumentType.NATIONAL_DRIVING_LICENCE_FRONT, DocumentType.NATIONAL_DRIVING_LICENCE_BACK}:
            labels = tuple(dict.fromkeys((
                *labels,
                *COMMON_NATIONAL_LABELS.get(path, ()),
                # The hand-built table above covers the languages whose
                # specimens were inspected. This one covers the official
                # language of every tourist country, so a licence printed in
                # Swahili, Amharic, Khmer or Georgian still finds its rows.
                *LANGUAGE_FIELD_LABELS.get(path, ()),
            )))
        if doc_type == DocumentType.PASSPORT_BIODATA:
            # A passport's printed rows are labelled in the issuing state's own
            # language, and they are the only evidence left when glare takes the
            # machine-readable zone -- which on a phone photo held under a
            # ceiling light is the usual way a page fails. English and Russian
            # were the only wordings listed, so a French, Italian or Latvian
            # page whose zone was lost gave up its name and birth date entirely.
            labels = tuple(dict.fromkeys((
                *labels,
                *PASSPORT_VIZ_LABELS.get(path, ()),
                *(
                    BARE_NAME_AS_SURNAME_LABELS
                    if path == "personal_info.last_name" and page_prints_bare_name_row
                    else ()
                ),
                *LANGUAGE_FIELD_LABELS.get(path, ()),
            )))
        if doc_type in {
            DocumentType.GCC_IDENTITY_FRONT, DocumentType.GCC_IDENTITY_BACK,
            DocumentType.GCC_DRIVING_LICENCE_FRONT, DocumentType.GCC_DRIVING_LICENCE_BACK,
        }:
            labels = tuple(dict.fromkeys((*labels, *gcc_labels_for_path(path, licence_country))))
        for line, raw, proximity in _label_value(
            index, labels, lambda value, field_path=path: _plausible_raw_value(
                field_path, value, licence_country,
            ),
            strict_neighbor=(
                (
                    path in _STRICT_NEIGHBOR_PATHS
                    # The strict rule keeps a value on its label's own row, which
                    # is how the tightly stacked GCC cards print. A passport
                    # biodata page prints the birthplace on the line *under* its
                    # label with a full line of air between them, so the same
                    # rule refused the city that was plainly there.
                    and not (
                        path == "personal_info.place_of_birth"
                        and doc_type == DocumentType.PASSPORT_BIODATA
                    )
                )
                or (is_gcc_document and path in _GCC_STRICT_PATHS)
            ),
            strict_next_row=(
                doc_type in EMIRATES_ID_TYPES
                and path in {"emirates_id.issue_date", "emirates_id.expiry_date"}
            ),
            cross_language_values=(
                path in _CROSS_LANGUAGE_PATHS or path in gcc_bilingual_personal_paths
            ),
            cross_variant_values=path in _CROSS_VARIANT_PATHS,
            avoid_labels=(
                (
                    *_NON_HOLDER_NAME_LABELS,
                    *DEDICATED_SURNAME_LABELS,
                    *FIELD_LABELS["personal_info.last_name"],
                    *FIELD_LABELS["personal_info.first_name"],
                    *LANGUAGE_FIELD_LABELS.get("personal_info.first_name", ()),
                    # On a passport laid out "Name" above "Vornamen", the bare
                    # row is the surname and has already been read as one.
                    # Leaving it available here bound it a second time as a
                    # full name, and splitting that gave the surname back as a
                    # given name.
                    *(
                        BARE_NAME_AS_SURNAME_LABELS
                        if page_prints_bare_name_row else ()
                    ),
                )
                if path == "personal_info.full_name"
                else _NON_HOLDER_NAME_LABELS
                if path in _NAME_PATHS
                # The row naming the document discriminator names the one
                # number on an AAMVA card that is not the licence number, and
                # it is printed one line from the row that is. Left available,
                # Ontario's "5 DD/RÉF" claimed the field and reported "6DD/R".
                else _DISCRIMINATOR_LABELS
                if path.endswith(".number") else ()
            ),
            # Two rows of reach: a bilingual passport prints the name in its
            # own script and again transliterated underneath, so the Latin
            # reading a rental needs sits one row further down than the label.
            #
            # Every other field gets a reach of its own for the same reason.
            # A caption names what is printed against it, and a card's own
            # spacing says how far that is -- a few line heights, never half
            # the page. Without a limit the window fell back to an absolute
            # 500 pixels, and the "LICENCE NO." column header at the foot of
            # this Ghanaian card's category table reached six rows down to a
            # legacy number, which then tied on confidence with the real one
            # printed under "Licence #" beside the holder's name. The field
            # came back CONFLICTING and the rental was sent to manual review
            # for a number the card states plainly.
            max_rows_below=2.5 if path in _NAME_PATHS else 3.5,
        ):
            if path in _NAME_PATHS and _non_primary_name_field_label(line.text):
                # A multilingual descriptor for a religious/stage/pseudonymous
                # name is another field's label. High OCR confidence proves the
                # letters were seen clearly; it does not turn that label into
                # the holder's primary name.
                continue
            value = _strip_rtl_label_residue(raw.strip(" :#-"), is_gcc_document)
            if (
                doc_type == DocumentType.PASSPORT_BIODATA
                and path == "personal_info.last_name"
            ):
                value = _PRIMARY_PASSPORT_SURNAME_DESIGNATOR.sub("", value)
            normalized: str | None = value
            validation: bool | None = None
            warnings: list[str] = []
            if path == "emirates_id.number":
                match = re.search(r"784(?:[-\s]?\d){12}", value)
                if not match: continue
                value = match.group(0)
                norm = normalize_emirates_id(value)
                normalized, warnings, validation = norm.value, list(norm.warnings), norm.value is not None
            elif path.endswith(("date_of_birth", "issue_date", "expiry_date")):
                value = close_split_year(
                    split_run_together_dates(ascii_numerals(value)),
                )
                # A validity range printed as one cell is one OCR box holding
                # two dates. Taking the first of them gave the issue date to
                # both fields, and the expiry -- being then a duplicate of a
                # value already bound -- was dropped, so a Queensland class row
                # reading "31.05.25 31.05.27" produced an issue date and no
                # expiry at all. Which of the two belongs to which field is not
                # a matter of position on the card: an issue date precedes an
                # expiry date, on every document that prints both.
                found = [
                    match.group(0)
                    for match in re.finditer(DATE_PATTERN, value, re.I)
                ]
                usable = [
                    (normalize_date(token, day_first_hint=not month_first), token)
                    for token in found
                ]
                if not usable: continue
                # A row printed "00/00/0000" is a placeholder the card itself
                # carries. It stays a candidate with nothing normalized, so the
                # operator is told the row exists and states nothing rather
                # than left with an empty field to go hunting the card for.
                readable = [(norm, token) for norm, token in usable if norm.value]
                recovered_from_row = False
                if not readable:
                    # The value the caption bound states no readable date, and
                    # the row it sits on still does. A bilingual Saudi licence
                    # prints "ISS 02/12/2021 <arabic caption> <hijri date>":
                    # the Arabic caption stands between the Gregorian date and
                    # the Hijri one, so the value taken after it was the Hijri
                    # -- and the recogniser had dropped a figure from its year,
                    # leaving "143/04/27", which is not a date at all. The row
                    # was reported as having no issue date while the same
                    # card's expiry, captioned only in English, came through.
                    row = close_split_year(
                        split_run_together_dates(ascii_numerals(line.text)),
                    )
                    readable = [
                        (norm, token)
                        for norm, token in (
                            (normalize_date(
                                match.group(0), day_first_hint=not month_first,
                            ), match.group(0))
                            for match in re.finditer(DATE_PATTERN, row, re.I)
                        )
                        if norm.value
                    ]
                    recovered_from_row = bool(readable)
                if len(readable) > 1 and path.endswith(("issue_date", "expiry_date")):
                    readable.sort(key=lambda item: item[0].value or "")
                    norm, value = readable[-1] if path.endswith("expiry_date") else readable[0]
                else:
                    norm, value = (readable or usable)[0]
                normalized, warnings, validation = norm.value, list(norm.warnings), norm.value is not None
                if recovered_from_row:
                    warnings.append("DATE_RECOVERED_FROM_BILINGUAL_ROW")
                # An out-of-range year means the row was printed in another
                # calendar. The GCC cards print Hijri; Iran prints Solar Hijri,
                # so "تاریخ صدور 1398/03/01" parsed cleanly and would have been
                # stored as the year 1398 AD. Thailand's Buddhist rows (25xx)
                # are caught the same way, leaving its Gregorian row to win.
                if path.startswith((
                    "gcc_identity.", "gcc_driving_licence.",
                    "national_driving_licence.", "international_driving_permit.",
                )) and normalized:
                    year = int(normalized[:4])
                    if year < 1900 or year > date.today().year + 30:
                        normalized, validation = None, False
                        warnings.append("HIJRI_OR_IMPLAUSIBLE_GREGORIAN_YEAR_REJECTED")
                if path == "personal_info.date_of_birth" and implausible_birth_date(
                    normalized,
                    minimum_age=(
                        MINIMUM_DRIVING_AGE_YEARS
                        if doc_type in DRIVING_DOCUMENTS_REQUIRING_ADULT_HOLDER
                        else None
                    ),
                ):
                    normalized, validation = None, False
                    warnings.append("IMPLAUSIBLE_BIRTH_DATE_REJECTED")
            elif path == "personal_info.gender":
                arabic_gender = next((
                    normalized_gender for marker, normalized_gender in (
                        ("أنثى", "F"), ("انثى", "F"), ("ذكر", "M"),
                    ) if marker in value
                ), None)
                match = re.search(r"\b(MALE|FEMALE|M|F|X)\b", value, re.I)
                if arabic_gender is None and match is None:
                    continue
                value = arabic_gender or match.group(1).upper()
                normalized = {"MALE": "M", "FEMALE": "F"}.get(value, value)
                validation = True
            elif path == "personal_info.nationality_name":
                code, name = nationality_country(value)
                if code is None:
                    # A nationality the tables cannot place is still what the
                    # card states, and dropping it reported a legible row as
                    # having no evidence at all. It is carried as printed and
                    # left for the operator rather than mapped to a guess.
                    cleaned = " ".join(value.split())
                    folded_value = compact_label(cleaned.upper())
                    if (
                        not _BARE_NATIONALITY_WORD.fullmatch(cleaned)
                        or _is_label_only_row(cleaned)
                        or any(
                            compact_label(wording) in folded_value
                            for wording in labels
                        )
                    ):
                        # A row that still carries its own caption is the
                        # caption, not a nationality, and must never stand in
                        # front of a value the passport's zone resolved.
                        continue
                    normalized, validation = cleaned, False
                    warnings = ["UNKNOWN_COUNTRY"]
                else:
                    normalized, validation, warnings = name, True, []
            elif path == "passport.issued_by_code":
                code, _, warnings = normalize_country(value)
                if code is None:
                    continue
                value, normalized, validation = code, code, True
            elif path == "uae_driving_licence.number":
                if re.search(DATE_PATTERN, value, re.I):
                    continue
                compact = re.sub(r"[\s-]", "", value)
                if re.fullmatch(r"\d{4,15}", compact) is None:
                    continue
                value, normalized, validation = compact, compact, True
            elif path == "gcc_identity.number":
                normalized = normalize_gcc_number(value, licence_country, identity=True)
                if normalized is None:
                    continue
                value, validation = normalized, True
            elif path == "gcc_driving_licence.number":
                normalized = normalize_gcc_number(value, licence_country, identity=False)
                if normalized is None:
                    continue
                value, validation = normalized, True
            elif path.endswith(".number"):
                # OCR frequently inserts a space inside a printed serial.  The
                # value is already isolated from its label, so joining only
                # alphanumeric fragments is safe and prevents a false conflict
                # between two OCR languages reading the same physical row.
                # Lebanon, Iran, Iraq, Syria and Sudan print their numbers in
                # Arabic-Indic digits (١٨٤٦٩٩٤). Dates were already converted
                # here; numbers were not, so those licences yielded no number at
                # all. Converting first also lets the pattern below apply.
                # The ordinal indicators are dropped before the NFKC pass, which
                # would otherwise fold "º" into the letter "o" and turn the
                # Argentine "Licencia Nº 32419010" into "O32419010".
                number_value = ascii_numerals(re.sub(r"[ªº°]", " ", value))
                number_value = re.sub(r"(?<=[A-Z0-9])\s+(?=[A-Z0-9])", "", number_value.upper())
                match = re.search(r"\b(?=[A-Z0-9\-/]{4,25}\b)(?=[A-Z0-9\-/]*\d)[A-Z0-9][A-Z0-9\-/]*", number_value, re.I)
                if not match: continue
                value, normalized, validation = match.group(0), match.group(0).upper(), True
            elif path.endswith("full_name"):
                value = _clean_person_name(value)
                if not _plausible_person_name(value): continue
                if doc_type == DocumentType.PASSPORT_BIODATA and re.search(r"[А-ЯЁа-яё]", value): continue
                normalized, validation = value, True
            elif path.endswith(("first_name", "middle_name", "last_name")):
                # The comma a North American card prints after the surname is
                # punctuation marking the layout -- "ABOUCHERE," above
                # "IBRAHIM,BECHIR" on the Ontario licence, "MALHOTRA," above
                # "PUNEET" on the British Columbia one. Kept, it reaches the
                # rental as part of the holder's name and drags the check
                # against their passport down with it.
                value = _clean_person_name(value)
                if not _plausible_person_name(value): continue
                if doc_type == DocumentType.PASSPORT_BIODATA and re.search(r"[А-ЯЁа-яё]", value): continue
                normalized, validation = value, True
            elif path.endswith("issued_by_name"):
                if len(value) < 2 or not any(char.isalpha() for char in value): continue
                normalized, validation = " ".join(value.split()), True
            elif path == "personal_info.place_of_birth":
                if len(value) < 2 or not any(char.isalpha() for char in value): continue
                normalized, validation = " ".join(value.split()), True
            # A bilingual GCC card often prints the same holder name in Arabic
            # and Latin.  Arabic OCR is useful evidence, but it is not a safe
            # source for the Latin CRM first/last-name fields: comparing the
            # scripts creates a false conflict and transliteration would be a
            # guess.  Preserve it in the dedicated Arabic-name field instead.
            candidate_path = path
            if (
                is_gcc_document
                and path == "personal_info.full_name"
                and normalized
                and re.search(r"[\u0600-\u06ff]", normalized)
            ):
                candidate_path = "personal_info.full_name_arabic"
                warnings.append("ARABIC_NAME_PRESERVED_SEPARATELY")
            candidate = _candidate(
                candidate_path, value, line, source, validation, normalized,
                warnings, proximity,
            )
            candidates.append(candidate)
            is_gcc = is_gcc_document
            if path == "personal_info.first_name" and len(raw.strip(" :#-").split()) > 1 and not is_gcc:
                middle = " ".join(raw.strip(" :#-").split()[1:])
                candidates.append(_candidate(
                    "personal_info.middle_name", middle, line, source, True,
                    middle, ["DERIVED_FROM_GIVEN_NAMES"], proximity,
                ))
            if candidate_path == "personal_info.full_name" and normalized:
                if is_gcc and licence_country == "Saudi Arabia" and "," in normalized:
                    surname, given_names = (part.strip(" ,") for part in normalized.split(",", 1))
                    derived = {
                        "personal_info.first_name": given_names,
                        "personal_info.last_name": surname,
                    }
                else:
                    parts = normalized.split()
                    surname, spans = _family_name(parts)
                    derived = {
                        "personal_info.first_name": parts[0] if parts else "",
                        # An empty surname (truncated line) fails the all() guard
                        # below, so the cut page contributes no name at all.
                        "personal_info.last_name": surname,
                    }
                    if spans and len(parts) > 1 + spans and not is_gcc:
                        derived["personal_info.middle_name"] = " ".join(
                            parts[1:len(parts) - spans],
                        )
                if page_prints_its_own_surname and not is_gcc:
                    # The page labels the surname itself, so this row is given
                    # names and nothing else. Argentina, Chile and Peru print
                    # "Apellido" above "Nombre"; splitting the "Nombre" row made
                    # "NOMBRE JUAN CARLOS" yield the surname CARLOS, which then
                    # tied with the real surname from the Apellido row at equal
                    # confidence and left the field CONFLICTING and empty.
                    derived.pop("personal_info.last_name", None)
                    derived.pop("personal_info.middle_name", None)
                if all(derived.values()):
                    for derived_path, derived_value in derived.items():
                        candidates.append(_candidate(
                            derived_path, derived_value, line, source, True,
                            derived_value, ["DERIVED_FROM_VISIBLE_FULL_NAME"], proximity,
                        ))
    if doc_type == DocumentType.PASSPORT_BIODATA:
        moldovan_fields = _moldovan_passport_layout_candidates(lines, source)
        moldovan_values = {
            candidate.field_path: str(candidate.normalized_value)
            for candidate in moldovan_fields if candidate.normalized_value
        }
        # Once a country-scoped visible row supplies one of these fields, do
        # not let the generic matcher retain the damaged caption it replaced.
        # If both name rows were recovered, the derived full name must follow
        # those two values as well.
        if moldovan_values:
            replace_paths = set(moldovan_values)
            first = moldovan_values.get("personal_info.first_name")
            last = moldovan_values.get("personal_info.last_name")
            if first and last:
                replace_paths.add("personal_info.full_name")
            candidates = [
                candidate for candidate in candidates
                if candidate.field_path not in replace_paths
            ]
            if first and last:
                evidence = next(
                    candidate for candidate in moldovan_fields
                    if candidate.field_path == "personal_info.first_name"
                )
                candidates.append(FieldCandidate(
                    field_path="personal_info.full_name", value=f"{first} {last}",
                    normalized_value=f"{first} {last}", source_document=source,
                    source_method="document_parser",
                    confidence=min(candidate.confidence for candidate in moldovan_fields
                                   if candidate.field_path in {
                                       "personal_info.first_name", "personal_info.last_name",
                                   }),
                    evidence_text=evidence.evidence_text,
                    bounding_box=evidence.bounding_box, validation_passed=True,
                    warnings=["MOLDOVAN_PASSPORT_VISIBLE_LAYOUT"],
                ))
            candidates.extend(moldovan_fields)
        # Haiti's printed bilingual captions are small enough that OCR often
        # damages the label while preserving both names. This layout recovery
        # is country- and geometry-bound, and supplies the independent visible
        # evidence needed when the MRZ name row itself was not captured.
        haitian_names = _haitian_passport_name_candidates(lines, source)
        haitian_values = {
            candidate.field_path: str(candidate.normalized_value)
            for candidate in haitian_names if candidate.normalized_value
        }
        first = haitian_values.get("personal_info.first_name")
        last = haitian_values.get("personal_info.last_name")
        if first and last:
            # A single upload can show a residence card above the passport.
            # Its generic ``Name/Nom`` row is not a passport name field; on the
            # reported page it bound BOURSIQUOT as a one-token full name and
            # then derived the same surname as the first name. Once the two
            # explicitly scoped passport rows are present, keep only name
            # interpretations compatible with them.
            expected = {
                "personal_info.first_name": {first},
                "personal_info.last_name": {last},
                "personal_info.full_name": {f"{first} {last}", f"{last} {first}"},
            }
            candidates = [
                candidate for candidate in candidates
                if candidate.field_path not in expected
                or str(candidate.normalized_value) in expected[candidate.field_path]
            ]
            evidence = next(
                candidate for candidate in haitian_names
                if candidate.field_path == "personal_info.first_name"
            )
            candidates.append(FieldCandidate(
                field_path="personal_info.full_name",
                value=f"{first} {last}", normalized_value=f"{first} {last}",
                source_document=source, source_method="document_parser",
                confidence=min(candidate.confidence for candidate in haitian_names),
                evidence_text=evidence.evidence_text,
                bounding_box=evidence.bounding_box, validation_passed=True,
                warnings=["HAITIAN_PASSPORT_NAME_LAYOUT"],
            ))
        candidates.extend(haitian_names)
    if (
        licence_country == "Saudi Arabia"
        and doc_type in {
            DocumentType.GCC_IDENTITY_FRONT, DocumentType.GCC_DRIVING_LICENCE_FRONT,
        }
        and not any(candidate.field_path == "personal_info.full_name" for candidate in candidates)
    ):
        # Both verified Saudi fronts print the Latin holder name as
        # SURNAME, GIVEN NAMES without a separate English Name label.
        # Bind only a clearly comma-separated all-letter line so headers,
        # numbers and redaction artifacts cannot become a personal name.
        for line in sorted(lines, key=lambda item: (_line_rect(item)[1], _line_rect(item)[0])):
            text = " ".join(line.text.strip().split())
            match = re.fullmatch(
                r"([A-Za-z][A-Za-z' -]{1,50}),\s*([A-Za-z][A-Za-z'. -]{1,100})",
                text,
            )
            if match is None:
                continue
            surname = " ".join(match.group(1).split()).upper()
            given_names = " ".join(match.group(2).split()).upper()
            first_name = given_names
            full_name = f"{surname}, {given_names}"
            layout_warning = ["SAUDI_SURNAME_FIRST_LAYOUT"]
            candidates.extend([
                _candidate(
                    "personal_info.full_name", full_name, line, source, True,
                    full_name, layout_warning, 1.0,
                ),
                _candidate(
                    "personal_info.first_name", first_name, line, source, True,
                    first_name, layout_warning, 1.0,
                ),
                _candidate(
                    "personal_info.last_name", surname, line, source, True,
                    surname, layout_warning, 1.0,
                ),
            ])
            break
    if doc_type == DocumentType.NATIONAL_DRIVING_LICENCE_FRONT:
        # Before the last-resort passes below, not after them. The designators
        # are printed evidence and the ordering fallback is a guess, and running
        # the guess first hid what the card had actually stated: a Dutch licence
        # whose 4a row was covered by the hologram had 4b read correctly from
        # its designator, while the guess -- which had already run, seeing
        # neither date bound -- supplied an issue date of 10.06.13 taken off the
        # category table on the reverse. The wrong date reached the operator at
        # HIGH_CONFIDENCE; the empty field it should have been would have been
        # typed in and forgotten.
        saudi_number = _saudi_tourist_licence_number_candidate(lines, source)
        if saudi_number:
            candidates = [
                candidate for candidate in candidates
                if candidate.field_path != "national_driving_licence.number"
            ]
            candidates.extend(saudi_number)
        numbered = _numbered_national_licence_candidates(
            lines, source, licence_country=licence_country,
        )
        numbered_paths = {candidate.field_path for candidate in numbered}
        if numbered_paths:
            candidates = [candidate for candidate in candidates if candidate.field_path not in numbered_paths]
            candidates.extend(numbered)
        russian_issue = _russian_licence_front_issue_date_candidate(lines, source)
        if russian_issue:
            candidates = [
                candidate for candidate in candidates
                if candidate.field_path != "national_driving_licence.issue_date"
            ]
            candidates.extend(russian_issue)
        if licence_country == "Algeria":
            algerian_dates = algerian_national_licence_front_dates(lines, source)
            if algerian_dates:
                date_paths = {candidate.field_path for candidate in algerian_dates}
                candidates = [
                    candidate for candidate in candidates
                    if candidate.field_path not in date_paths
                ]
                candidates.extend(algerian_dates)
    if doc_type in {
        DocumentType.NATIONAL_DRIVING_LICENCE_FRONT,
        DocumentType.NATIONAL_DRIVING_LICENCE_BACK,
    }:
        # A cell captioned "Valid" holding both dates outranks whatever the
        # captioned read bound: on the card that prints it, the only other
        # dated caption is "First issue", which names a different day.
        validity_range = licence_validity_range_dates(lines, source)
        if validity_range:
            ranged = {candidate.field_path for candidate in validity_range}
            candidates = [
                candidate for candidate in candidates
                if candidate.field_path not in ranged
            ]
            candidates.extend(validity_range)
        if not any(
            candidate.field_path == "personal_info.gender"
            and candidate.normalized_value
            for candidate in candidates
        ):
            candidates.extend(uncaptioned_gender_word(lines, source))
        # Last resort for each missing date. A licence printed in a language
        # whose wording is not in the tables would otherwise carry no dates at
        # all, but a correctly read expiry does not prove that its issue row was
        # also read. Where birth, issue and expiry are all visibly present, the
        # chronological sequence can safely recover only the missing member.
        # A validity range printed as one cell states both of its dates, even
        # where only one of the two column headings above it survived the
        # capture. On the Queensland card the recogniser returned "Effective"
        # and lost "Expiry", so the cell bound an issue date and the expiry it
        # also names was reported as absent -- from a row an operator can read
        # off the card in front of them. The later date in the cell the issue
        # date came from is that expiry.
        bound = {candidate.field_path for candidate in candidates}
        if (
            "national_driving_licence.issue_date" in bound
            and "national_driving_licence.expiry_date" not in bound
        ):
            issued = next(
                candidate for candidate in candidates
                if candidate.field_path == "national_driving_licence.issue_date"
            )
            dates = sorted({
                normalize_date(match.group(0), day_first_hint=True).value
                for match in re.finditer(
                    DATE_PATTERN,
                    split_run_together_dates(ascii_numerals(issued.evidence_text or "")),
                    re.I,
                )
            } - {None})
            if len(dates) >= 2 and dates[-1] != issued.normalized_value:
                candidates.append(FieldCandidate(
                    field_path="national_driving_licence.expiry_date",
                    value=dates[-1], normalized_value=dates[-1],
                    source_document=source, source_method=issued.source_method,
                    confidence=min(issued.confidence, 0.84),
                    evidence_text=issued.evidence_text,
                    bounding_box=issued.bounding_box, validation_passed=True,
                    warnings=["EXPIRY_FROM_VALIDITY_RANGE_CELL"],
                ))
        bound = {
            candidate.field_path for candidate in candidates
            if candidate.normalized_value
        }
        missing_date_paths = {
            "national_driving_licence.issue_date",
            "national_driving_licence.expiry_date",
        } - bound
        if missing_date_paths:
            birth = next((
                candidate.normalized_value for candidate in candidates
                if candidate.field_path == "personal_info.date_of_birth"
            ), None)
            # Filling only one of the pair needs a printed DOB to establish
            # which of the two remaining dates is the issue date.  With no DOB
            # the two dates can be an issue/expiry pair, or a birth/issue pair;
            # guessing an expiry from the latter is exactly how a Swiss
            # category-table capture acquired a false validity date.  Preserve
            # the older fully-unlabelled fallback, which can establish birth
            # itself only when all three dates are visible.
            both_dates_missing = not bound & {
                "national_driving_licence.issue_date",
                "national_driving_licence.expiry_date",
            }
            recover_damaged_4a = (
                "national_driving_licence.issue_date" in missing_date_paths
                and birth is not None
                and _has_orphaned_4a_date_row(lines)
            )
            if both_dates_missing or recover_damaged_4a:
                candidates.extend(
                    candidate
                    for candidate in national_licence_date_sequence(lines, source, birth)
                    if candidate.field_path in missing_date_paths
                )
        # The zone confirms a printed row, so it is consulted whatever the
        # anchored read produced -- including where that read bound a number
        # whose zeroes came back as letters.
        corroborated = licence_number_corroborated_by_card_zone(lines, source)
        if corroborated:
            # Two readings of one number, differing only where a recognizer
            # trades a letter for a digit, are not competing values -- and left
            # as competing they cancelled each other and emptied the field. The
            # zone is machine-readable, so its spelling stands and the row's is
            # dropped rather than set against it.
            confirmed = {
                candidate.normalized_value.translate(_CONFUSABLE)
                for candidate in corroborated if candidate.normalized_value
            }
            candidates = [
                candidate for candidate in candidates
                if candidate.field_path != "national_driving_licence.number"
                or not candidate.normalized_value
                or candidate.normalized_value.translate(_CONFUSABLE) not in confirmed
            ]
            candidates.extend(corroborated)
        # The agency's own formula outranks a designator reading, because a
        # card photographed on its side returns its "5" in a box of its own and
        # the designator parser then reads the row it can pair -- the printed
        # legend along the foot of the card, which explains what field 5 is
        # rather than stating it. A number already in the DVLA's format was
        # read correctly and is left alone.
        british = uk_licence_number_candidates(lines, source)
        if british and not any(
            candidate.field_path == "national_driving_licence.number"
            and candidate.normalized_value
            and _UK_LICENCE_NUMBER.search(str(candidate.normalized_value).upper())
            for candidate in candidates
        ):
            candidates = [
                candidate for candidate in candidates
                if candidate.field_path != "national_driving_licence.number"
            ]
            candidates.extend(british)
        bound = {candidate.field_path for candidate in candidates}
        if "national_driving_licence.number" not in bound:
            candidates.extend(
                _new_zealand_licence_number_candidates(lines, source)
                if licence_country in (None, "New Zealand") else []
            )
        bound = {candidate.field_path for candidate in candidates}
        if "national_driving_licence.number" not in bound:
            candidates.extend(
                _armenian_licence_number_candidates(lines, source)
                if licence_country in (None, "Armenia") else []
            )
        bound = {candidate.field_path for candidate in candidates}
        if "national_driving_licence.number" not in bound:
            candidates.extend(national_licence_number_fallback(lines, source))
        candidates.extend(national_licence_category_candidates(lines, source))
        bound = {candidate.field_path for candidate in candidates}
        candidates.extend(
            row for row in _column_header_date_candidates(lines, source)
            if row.field_path not in bound
        )
    if doc_type == DocumentType.PASSPORT_BIODATA and not any(
        candidate.field_path == "personal_info.full_name" for candidate in candidates
    ):
        best = {}
        for path in ("personal_info.first_name", "personal_info.middle_name", "personal_info.last_name"):
            options = [candidate for candidate in candidates if candidate.field_path == path]
            if options:
                best[path] = max(options, key=lambda candidate: candidate.confidence)
        if "personal_info.first_name" in best and "personal_info.last_name" in best:
            first_name = best["personal_info.first_name"].normalized_value
            values = [first_name]
            if "personal_info.middle_name" in best:
                middle_name = best["personal_info.middle_name"].normalized_value
                # first_name now preserves the passport's complete GIVEN NAMES
                # row, so its secondary tokens may also appear in the internal
                # middle_name field. Do not repeat them in the derived full name.
                if not first_name.endswith(f" {middle_name}"):
                    values.append(middle_name)
            values.append(best["personal_info.last_name"].normalized_value)
            full_name = " ".join(value for value in values if value)
            evidence = best["personal_info.first_name"]
            candidates.append(FieldCandidate(
                field_path="personal_info.full_name", value=full_name, normalized_value=full_name,
                source_document=source, source_method="labelled_ocr",
                confidence=min(candidate.confidence for candidate in best.values()),
                evidence_text=evidence.evidence_text, bounding_box=evidence.bounding_box,
                validation_passed=True, warnings=["DERIVED_FROM_GIVEN_AND_SURNAME_FIELDS"],
            ))
    if doc_type == DocumentType.PASSPORT_BIODATA:
        layout_candidates = _passport_layout_candidates(lines, source)
        # The layout pass exists for pages whose labels could not be read. Its
        # standardized two-column date fallback cannot improve a date already
        # bound to its own explicit caption: Syrian passports print expiry on
        # the left and issue on the right, the inverse of that fallback's
        # assumption. In that layout the direct bindings were correct, then
        # replaced by the fallback and discarded by chronology validation.
        # Preserve explicit date evidence and use the layout only to fill a
        # date the label pass genuinely could not read.
        directly_bound_passport_dates = {
            candidate.field_path for candidate in candidates
            if (
                candidate.field_path in {"passport.issue_date", "passport.expiry_date"}
                and candidate.normalized_value
            )
        }
        if directly_bound_passport_dates:
            layout_candidates = [
                candidate for candidate in layout_candidates
                if not (
                    candidate.field_path in directly_bound_passport_dates
                    and "PASSPORT_STANDARD_LAYOUT_FALLBACK" in candidate.warnings
                )
            ]
        # A visible "passport no." beside a value beats a token chosen for
        # having the most digits, which on the Albanian page was the personal
        # number.
        if any(
            candidate.field_path == "passport.number"
            and "VISIBLE_PASSPORT_NUMBER_PATTERN" not in candidate.warnings
            for candidate in candidates
        ):
            layout_candidates = [
                candidate for candidate in layout_candidates
                if candidate.field_path != "passport.number"
            ]
        layout_paths = {candidate.field_path for candidate in layout_candidates}
        if layout_paths:
            candidates = [candidate for candidate in candidates if candidate.field_path not in layout_paths]
            candidates.extend(layout_candidates)
        tajik_names = _tajik_passport_name_candidates(lines, source)
        tajik_name_paths = {candidate.field_path for candidate in tajik_names}
        if tajik_name_paths:
            candidates = [
                candidate for candidate in candidates
                if candidate.field_path not in tajik_name_paths
            ]
            candidates.extend(tajik_names)
        # Last, and only for what is still empty. The sex and nationality rows
        # are the two a worn booklet gives up last -- three characters of label
        # for one, an adjective the country tables do not carry for the other --
        # and a page can otherwise read perfectly and hand the rental two blank
        # boxes it has no way to fill but by asking for the passport again.
        bound = {
            candidate.field_path for candidate in candidates
            if candidate.normalized_value
        }
        issuing = next(
            (
                candidate.normalized_value for candidate in candidates
                if candidate.field_path
                in {"passport.issued_by_name", "passport.issued_by_code"}
                and candidate.normalized_value
            ),
            None,
        )
        candidates.extend(
            candidate
            for candidate in passport_sex_and_nationality(
                lines, source, issuing_country=issuing,
            )
            if candidate.field_path not in bound
        )
        candidates.extend(
            _given_names_under_a_shared_name_caption(lines, candidates, source)
        )
    if doc_type == DocumentType.INTERNATIONAL_DRIVING_PERMIT:
        candidates = [
            candidate for candidate in candidates
            if not (
                candidate.field_path == "international_driving_permit.expiry_date"
                and candidate.evidence_text
                and re.search(r"CONVENTION\s+ON\s+ROAD\s+TRAFFIC", candidate.evidence_text, re.I)
            )
        ]
        layout_candidates = idp_layout_candidates(lines, source)
        layout_paths = {candidate.field_path for candidate in layout_candidates}
        if layout_paths:
            candidates = [candidate for candidate in candidates if candidate.field_path not in layout_paths]
            candidates.extend(layout_candidates)
    if doc_type in {DocumentType.EMIRATES_ID_FRONT, DocumentType.EMIRATES_ID_BACK}:
        # One OCR box cannot prove two different semantic date fields. If the
        # Issue Date lookup borrowed the exact Expiry Date box, discard the
        # issue candidate so contrast OCR/Qwen can try to recover independent
        # visible evidence. Expiry is retained because it was the established
        # field before Issue Date was added to the schema.
        expiry_evidence = {
            (
                candidate.source_document,
                tuple(tuple(point) for point in (candidate.bounding_box or [])),
                candidate.normalized_value,
            )
            for candidate in candidates
            if candidate.field_path == "emirates_id.expiry_date"
        }
        candidates = [
            candidate for candidate in candidates
            if not (
                candidate.field_path == "emirates_id.issue_date"
                and (
                    candidate.source_document,
                    tuple(tuple(point) for point in (candidate.bounding_box or [])),
                    candidate.normalized_value,
                ) in expiry_evidence
            )
        ]
    if doc_type in {
        DocumentType.GCC_IDENTITY_FRONT, DocumentType.GCC_IDENTITY_BACK,
        DocumentType.GCC_DRIVING_LICENCE_FRONT, DocumentType.GCC_DRIVING_LICENCE_BACK,
    }:
        section = (
            "gcc_identity"
            if doc_type in {DocumentType.GCC_IDENTITY_FRONT, DocumentType.GCC_IDENTITY_BACK}
            else "gcc_driving_licence"
        )
        expiry_evidence = {
            (
                candidate.source_document,
                tuple(tuple(point) for point in (candidate.bounding_box or [])),
                candidate.normalized_value,
            )
            for candidate in candidates
            if candidate.field_path == f"{section}.expiry_date"
        }
        candidates = [
            candidate for candidate in candidates
            if not (
                candidate.field_path == f"{section}.issue_date"
                and (
                    candidate.source_document,
                    tuple(tuple(point) for point in (candidate.bounding_box or [])),
                    candidate.normalized_value,
                ) in expiry_evidence
            )
        ]
        if (
            licence_country == "Saudi Arabia"
            and doc_type == DocumentType.GCC_DRIVING_LICENCE_FRONT
        ):
            present_paths = {
                candidate.field_path for candidate in candidates
                if candidate.normalized_value and candidate.validation_passed is not False
            }
            candidates.extend(
                candidate for candidate in _saudi_licence_layout_candidates(lines, source)
                if candidate.field_path not in present_paths
            )
    # Preserve only explicitly labelled Arabic names. Treating every Arabic
    # line as a name creates hundreds of false candidates from labels/headers.
    for line, raw, proximity in _label_value(
        index,
        ("الاسم", "الإسم", "اسم حامل البطاقة"),
        lambda value: re.search(r"[\u0600-\u06ff]", value) is not None and not any(char.isdigit() for char in value),
    ):
        value = raw.strip(" :#-")
        if len(value) >= 3 and re.search(r"[\u0600-\u06ff]", value) and not any(char.isdigit() for char in value):
            normalized_value = " ".join(value.split())
            if not any(
                candidate.field_path == "personal_info.full_name_arabic"
                and candidate.normalized_value == normalized_value
                and candidate.source_document == source
                for candidate in candidates
            ):
                candidates.append(_candidate(
                    "personal_info.full_name_arabic", value, line, source, None,
                    normalized_value, ["ARABIC_NAME_REQUIRES_LABEL_REVIEW"], proximity,
                ))
    if not any(candidate.field_path == "personal_info.full_name_arabic" for candidate in candidates):
        latin_name_labels = [
            line for line in lines
            if any(label in line.text.upper() for label in FIELD_LABELS["personal_info.full_name"])
        ]
        for label_line in latin_name_labels:
            lx1, ly1, lx2, ly2 = _line_rect(label_line)
            height = max(ly2 - ly1, 1)
            nearby = []
            for line in lines:
                if not re.search(r"[\u0600-\u06ff]", line.text) or any(char.isdigit() for char in line.text):
                    continue
                x1, y1, x2, _ = _line_rect(line)
                if ly1 - height <= y1 <= ly2 + height * 2.5 and not (x2 < lx1 - height or x1 > lx2 + 500):
                    nearby.append((abs(y1 - ly1) + abs(x1 - lx1) * 0.25, line))
            if nearby:
                _, line = min(nearby, key=lambda item: item[0])
                value = " ".join(line.text.strip().split())
                candidates.append(_candidate(
                    "personal_info.full_name_arabic", value, line, source, None,
                    value, ["ARABIC_NAME_REQUIRES_LABEL_REVIEW"], 0.82,
                ))
                break
    return candidates


# MRZ keys that describe the holder rather than the passport booklet. An
# Emirates ID card carries a TD1 MRZ whose document number, issuing country and
# expiry belong to the card, so only these keys may be read from a non-passport
# document; the rest would otherwise be written into the passport.* fields.
MRZ_HOLDER_KEYS = frozenset({"gender"})


# Emirates IDs encode the holder's family and given names in their TD1 MRZ.
# A generic printed ``Name`` row is not a reliable split into those components,
# especially for compound Indian, Arabic and Hispanic names, so the valid zone
# may supply the two CRM fields as a fallback. This is deliberately scoped to
# Emirates IDs; other card MRZs keep their established country-specific rules.
EMIRATES_ID_TYPES = frozenset({
    DocumentType.EMIRATES_ID_FRONT,
    DocumentType.EMIRATES_ID_BACK,
})

EMIRATES_ID_MRZ_NAME_KEYS = frozenset({"first_name", "last_name"})


# The tail of TD1 row two is often cropped by phone captures.  The birth date,
# sex and expiry bytes precede that tail, and the two dates carry their own
# check digits.  Do not accept a bare date-like row: at least two trailing MRZ
# fillers plus the UAE card's nationality byte are required as its layout proof.
_EMIRATES_ID_PARTIAL_TD1_GENDER = re.compile(
    r"(?P<birth>\d{6})(?P<birth_check>\d)(?P<gender>[MF])"
    r"(?P<expiry>\d{6})(?P<expiry_check>\d)(?P<nationality>[A-Z]{3})"
    r"(?P<filler><{2,})$",
)


def emirates_id_partial_mrz_gender_candidates(
    lines: list[OCRLine], source: str, doc_type: DocumentType,
) -> list[FieldCandidate]:
    """Recover sex from a clipped Emirates-ID TD1 middle row safely.

    A full zone remains the preferred source. This applies only when its tail
    was clipped but the row still proves its fixed offsets through independently
    checksummed birth and expiry dates. It is local string validation, so it
    adds no OCR or model pass to the UAE fast path.
    """
    if doc_type not in EMIRATES_ID_TYPES:
        return []
    candidates: list[FieldCandidate] = []
    for line in lines:
        row = normalize_mrz_line(line.text)
        matched = _EMIRATES_ID_PARTIAL_TD1_GENDER.fullmatch(row)
        if matched is None:
            continue
        birth = matched.group("birth")
        expiry = matched.group("expiry")
        if not (
            validate_check(birth, matched.group("birth_check"))
            and validate_check(expiry, matched.group("expiry_check"))
        ):
            continue
        candidates.append(FieldCandidate(
            field_path="personal_info.gender",
            value=matched.group("gender"),
            normalized_value=matched.group("gender"),
            source_document=source,
            source_method="mrz",
            confidence=0.99,
            evidence_text=line.text,
            bounding_box=line.bounding_box,
            validation_passed=True,
            warnings=[
                "EMIRATES_ID_PARTIAL_TD1_GENDER",
                "MRZ_BIRTH_AND_EXPIRY_CHECKSUMMED",
            ],
        ))
    return candidates


# Unlike the document number and dates, the name row carries no check digit.
# A fully valid zone therefore proves its layout but not whether OCR dropped a
# letter from the holder's name. Printed-name agreement can still promote the
# field to VERIFIED during reconciliation; the MRZ alone may not do so.
MRZ_UNCHECKSUMMED_NAME_KEYS = frozenset({
    "first_name", "middle_name", "last_name", "full_name",
})


# The reverse of the Bahraini, Kuwaiti, Omani and current Saudi identity cards
# carries a TD1 machine-readable zone. Its birth date and expiry are protected
# by their own check digits, which makes them the most reliable evidence on the
# whole document — and free, because the zone is already parsed for routing.
# The document-number field is claimed as the identity number only where the
# state is documented as printing the national number there; elsewhere it can
# be a card serial, so the visible labelled row remains the only source.
GCC_IDENTITY_MRZ_KEYS: dict[str, str] = {
    "date_of_birth": "personal_info.date_of_birth",
    "expiry_date": "gcc_identity.expiry_date",
    "gender": "personal_info.gender",
    "nationality_code": "personal_info.nationality_code",
}


GCC_IDENTITY_TYPES = frozenset({
    DocumentType.GCC_IDENTITY_FRONT, DocumentType.GCC_IDENTITY_BACK,
})


def mrz_candidates(
    parsed: ParsedMRZ, source: str, doc_type: DocumentType | None = None,
    licence_country: str | None = None,
) -> list[FieldCandidate]:
    if not parsed.mrz_type: return []
    holder_only = doc_type is not None and doc_type != DocumentType.PASSPORT_BIODATA
    profile = profile_for_gcc_country(licence_country)
    gcc_identity = holder_only and doc_type in GCC_IDENTITY_TYPES and profile is not None
    emirates_id = holder_only and doc_type in EMIRATES_ID_TYPES
    algerian_licence = (
        holder_only
        and doc_type in {
            DocumentType.NATIONAL_DRIVING_LICENCE_FRONT,
            DocumentType.NATIONAL_DRIVING_LICENCE_BACK,
        }
        and parsed.mrz_type == "TD1"
        and parsed.fields.get("document_code") == "DL"
        and parsed.fields.get("issuing_country_code") == "DZA"
        # The zone is page-local, checksum-protected country evidence.  It must
        # work even when this is the first page and the bundle country has not
        # yet been resolved.  A contradictory established country still wins
        # and prevents cross-country field mapping.
        and licence_country in {None, "Algeria"}
    )
    # The birth date and expiry each carry their own check digit. A GCC card
    # whose document-number field fails -- routine, because several of these
    # states use a number longer than the nine characters TD1 reserves -- still
    # proves those two values, and two passing check digits on the middle row
    # prove the row's alignment, which is what makes the sex character at a
    # fixed offset trustworthy. Demanding whole-zone validity threw all of that
    # away and left the operator typing a birth date the card had encoded.
    aligned = (
        parsed.checks.get("date_of_birth") is True
        and parsed.checks.get("expiry_date") is True
    )
    if holder_only and not parsed.valid and not (gcc_identity and aligned):
        return []
    mapping = {
        "document_number": "passport.number", "issuing_country_code": "passport.issued_by_code",
        "expiry_date": "passport.expiry_date", "first_name": "personal_info.first_name",
        "middle_name": "personal_info.middle_name", "last_name": "personal_info.last_name",
        "full_name": "personal_info.full_name", "date_of_birth": "personal_info.date_of_birth",
        "gender": "personal_info.gender", "nationality_code": "personal_info.nationality_code",
    }
    if gcc_identity:
        mapping = dict(GCC_IDENTITY_MRZ_KEYS)
        # The zone prints the holder's name in a fixed machine-readable form,
        # which settles the spelling when the two OCR engines disagree over a
        # letter in the printed Latin row. A country's profile can disable the
        # unchecked name row where its layout does not use ICAO surname-first
        # order; the checksum-protected holder fields are unaffected.
        if profile.identity_mrz_names:
            mapping.update({
                "first_name": "personal_info.first_name",
                "last_name": "personal_info.last_name",
            })
        if profile.identity_mrz_id_source == "document_number":
            mapping["document_number"] = "gcc_identity.number"
    elif algerian_licence:
        # Algeria prints a complete three-line DL zone on the reverse.  The
        # number, birth date and expiry each carry their own check digit; using
        # the passport-only mapping threw all three away even after the zone
        # had parsed as fully valid.
        mapping = {
            "document_number": "national_driving_licence.number",
            "issuing_country_code": "national_driving_licence.issued_by_code",
            "expiry_date": "national_driving_licence.expiry_date",
            "first_name": "personal_info.first_name",
            "last_name": "personal_info.last_name",
            "full_name": "personal_info.full_name",
            "date_of_birth": "personal_info.date_of_birth",
            "gender": "personal_info.gender",
            "nationality_code": "personal_info.nationality_code",
        }
    elif emirates_id:
        # TD1 row three is ``SURNAME<<GIVEN<NAMES``. The zone's numeric rows
        # have already proved its alignment and check digits, while the name
        # itself remains marked as unchecked evidence below.
        mapping = {
            "first_name": "personal_info.first_name",
            "last_name": "personal_info.last_name",
        }
        if aligned:
            # Row two carries the card's expiry and the holder's birth date,
            # each under its own check digit, with the sex byte between them
            # and the nationality immediately after. Taking only the two name
            # keys meant the sole evidence kept from the zone was the part it
            # cannot prove, while a card whose expiry had just passed its own
            # check digit still reported no expiry at all. Both date checks
            # passing is what proves the row's alignment, and so the fixed
            # offsets of the sex and nationality characters as well.
            mapping.update({
                "expiry_date": "emirates_id.expiry_date",
                "date_of_birth": "personal_info.date_of_birth",
                "gender": "personal_info.gender",
                "nationality_code": "personal_info.nationality_code",
            })
    elif holder_only:
        mapping = {key: path for key, path in mapping.items() if key in MRZ_HOLDER_KEYS}
    checksum_map = {"document_number": "document_number", "expiry_date": "expiry_date", "date_of_birth": "date_of_birth"}
    candidates: list[FieldCandidate] = []
    for key, path in mapping.items():
        value = parsed.fields.get(key)
        if not value: continue
        normalized = value
        if path == "gcc_identity.number":
            # A machine-readable number still has to look like that country's
            # personal number before it is allowed into the field.
            normalized = normalize_gcc_number(value, licence_country, identity=True)
            if normalized is None:
                continue
        check = parsed.checks.get(checksum_map[key]) if key in checksum_map else parsed.valid
        if key == "gender" and aligned:
            # Sex occupies the byte between the independently checksummed birth
            # and expiry dates. When both date checks pass, their alignment
            # proves that fixed position even if a crop removed the document or
            # composite check digit elsewhere in the same TD2/TD3 row.
            check = True
        if key in {"nationality_code", "issuing_country_code"} and not re.fullmatch(
            r"[A-Z]{3}", value,
        ):
            # No check digit covers these three characters, so a passing zone
            # says nothing about them. Anything that is not a country code is
            # a misreading and must not be presented as proven.
            check = None
        if key in MRZ_UNCHECKSUMMED_NAME_KEYS and not gcc_identity:
            check = None
        if gcc_identity and key not in checksum_map:
            # Backed by the row alignment the two passing check digits prove.
            check = aligned or parsed.valid
        confidence = 0.99 if check is True else 0.88 if parsed.valid else 0.65
        candidates.append(FieldCandidate(
            field_path=path, value=value, normalized_value=normalized, source_document=source,
            source_method="mrz", confidence=confidence, evidence_text=None, bounding_box=None,
            validation_passed=check,
            warnings=list(dict.fromkeys([
                *parsed.warnings,
                *(
                    ["EMIRATES_ID_MRZ_NAME_FALLBACK"]
                    if emirates_id and key in EMIRATES_ID_MRZ_NAME_KEYS else []
                ),
                *([] if parsed.valid else ["MRZ_NOT_FULLY_VALID"]),
            ])),
        ))
    return candidates


def barcode_candidates(
    barcodes: Iterable[BarcodeCandidate], source: str, known_fields: set[str],
    doc_type: DocumentType | None = None,
    licence_country: str | None = None,
) -> list[FieldCandidate]:
    if licence_country == "Saudi Arabia" and doc_type in {
        DocumentType.GCC_IDENTITY_FRONT, DocumentType.GCC_IDENTITY_BACK,
        DocumentType.GCC_DRIVING_LICENCE_FRONT, DocumentType.GCC_DRIVING_LICENCE_BACK,
    }:
        # The verified Saudi samples expose a QR/barcode while the human-readable
        # ID is independently redacted. Do not bypass that redaction or replace
        # the visible labelled value with machine-readable payload content.
        return []
    aliases: dict[str, str] = {
        "id_number": "emirates_id.number", "emirates_id": "emirates_id.number",
        "license_number": "uae_driving_licence.number", "licence_number": "uae_driving_licence.number",
        "passport_number": "passport.number", "permit_number": "international_driving_permit.number",
        "expiry_date": "passport.expiry_date", "date_of_birth": "personal_info.date_of_birth",
        "name": "personal_info.full_name", "first_name": "personal_info.first_name",
        "middle_name": "personal_info.middle_name", "last_name": "personal_info.last_name",
    }
    if source.startswith("national_licence") or doc_type in {
        DocumentType.NATIONAL_DRIVING_LICENCE_FRONT,
        DocumentType.NATIONAL_DRIVING_LICENCE_BACK,
    }:
        aliases.update({
            "license_number": "national_driving_licence.number",
            "licence_number": "national_driving_licence.number",
            "issue_date": "national_driving_licence.issue_date",
            "expiry_date": "national_driving_licence.expiry_date",
            "issuing_country": "national_driving_licence.issued_by_code",
            # The AAMVA zone names the province or state that issued the card.
            # On a Canadian licence that is the whole of the issuer -- there is
            # no federal one -- and it is stated by the issuing authority
            # rather than read off the front, so it outranks anything OCR makes
            # of the wording printed there.
            "jurisdiction": "national_driving_licence.issued_by_name",
        })
    elif doc_type in {
        DocumentType.GCC_IDENTITY_FRONT, DocumentType.GCC_IDENTITY_BACK,
    }:
        aliases.update({
            "id_number": "gcc_identity.number",
            "emirates_id": "gcc_identity.number",
            "issue_date": "gcc_identity.issue_date",
            "expiry_date": "gcc_identity.expiry_date",
            "issuing_country": "gcc_identity.issued_by_code",
        })
    elif doc_type in {
        DocumentType.GCC_DRIVING_LICENCE_FRONT, DocumentType.GCC_DRIVING_LICENCE_BACK,
    }:
        aliases.update({
            "license_number": "gcc_driving_licence.number",
            "licence_number": "gcc_driving_licence.number",
            "issue_date": "gcc_driving_licence.issue_date",
            "expiry_date": "gcc_driving_licence.expiry_date",
            "issuing_country": "gcc_driving_licence.issued_by_code",
        })
    elif source.startswith("idp_pages") or doc_type == DocumentType.INTERNATIONAL_DRIVING_PERMIT:
        aliases.update({
            "license_number": "international_driving_permit.number",
            "issue_date": "international_driving_permit.issue_date",
            "expiry_date": "international_driving_permit.expiry_date",
            "issuing_country": "international_driving_permit.issued_by_code",
        })
    elif source.startswith("uae_licence") or doc_type in {
        DocumentType.UAE_DRIVING_LICENCE_FRONT,
        DocumentType.UAE_DRIVING_LICENCE_BACK,
    }:
        aliases.update({
            "issue_date": "uae_driving_licence.issue_date",
            "expiry_date": "uae_driving_licence.expiry_date",
            "issuing_country": "uae_driving_licence.issued_by_code",
        })
    candidates: list[FieldCandidate] = []
    for barcode in barcodes:
        for key, value in barcode.structured_candidate.items():
            path = aliases.get(key)
            if path not in known_fields: continue
            normalized = value
            if path == "gcc_identity.number":
                parsed_number = normalize_gcc_number(str(value), licence_country, identity=True)
                if parsed_number is None:
                    continue
                normalized = parsed_number
            elif path == "gcc_driving_licence.number":
                parsed_number = normalize_gcc_number(str(value), licence_country, identity=False)
                if parsed_number is None:
                    continue
                normalized = parsed_number
            elif path == "uae_driving_licence.number":
                compact = re.sub(r"[\s-]", "", str(value))
                if re.fullmatch(r"\d{4,15}", compact) is None:
                    continue
                normalized = compact
            elif path == "national_driving_licence.number":
                # AAMVA stores the identifier without the visual grouping the
                # Canadian card prints.  Restore that grouping when DAJ names
                # a province whose format is known, so barcode and visible OCR
                # reconcile to one value instead of conflicting solely over
                # hyphens.
                compact = re.sub(r"[^A-Z0-9]", "", str(value).upper())
                province = province_for(
                    barcode.structured_candidate.get("jurisdiction"),
                )
                if province and province.code == "ON" and re.fullmatch(
                    r"[A-Z]\d{14}", compact,
                ):
                    normalized = f"{compact[:5]}-{compact[5:10]}-{compact[10:]}"
                elif province and province.code == "QC" and re.fullmatch(
                    r"[A-Z]\d{12}", compact,
                ):
                    normalized = f"{compact[:5]}-{compact[5:11]}-{compact[11:]}"
                else:
                    normalized = str(value).strip()
            elif path.endswith(("date_of_birth", "issue_date", "expiry_date")):
                parsed_date = normalize_date(value, day_first_hint=False)
                if parsed_date.value is None:
                    continue
                normalized = parsed_date.value
            elif path.endswith("issued_by_code"):
                normalized = normalize_country(value)[0] or value.upper()
            elif key == "jurisdiction":
                # DAJ carries the two-letter code, "QC" rather than "Quebec".
                # An operator reading the field should see the province.
                province = province_for(value)
                if province is None:
                    continue
                normalized = province.name
            candidates.append(FieldCandidate(
                field_path=path, value=value, normalized_value=normalized, source_document=source,
                source_method=f"barcode:{barcode.barcode_type}", confidence=0.86,
                evidence_text=None, bounding_box=None, validation_passed=None,
                warnings=[*barcode.warnings, "BARCODE_REQUIRES_VISIBLE_OCR_AGREEMENT"],
            ))
        person = barcode.structured_candidate
        if (
            source.startswith("national_licence")
            or doc_type in {
                DocumentType.NATIONAL_DRIVING_LICENCE_FRONT,
                DocumentType.NATIONAL_DRIVING_LICENCE_BACK,
                DocumentType.GCC_DRIVING_LICENCE_FRONT,
                DocumentType.GCC_DRIVING_LICENCE_BACK,
            }
        ) and person.get("first_name") and person.get("last_name"):
            parts = [person["first_name"], person.get("middle_name"), person["last_name"]]
            full_name = " ".join(part.strip() for part in parts if part and part.strip())
            candidates.append(FieldCandidate(
                field_path="personal_info.full_name", value=full_name,
                normalized_value=full_name, source_document=source,
                source_method=f"barcode:{barcode.barcode_type}", confidence=0.86,
                evidence_text=None, bounding_box=None, validation_passed=None,
                warnings=[*barcode.warnings, "BARCODE_REQUIRES_VISIBLE_OCR_AGREEMENT"],
            ))
    return candidates
