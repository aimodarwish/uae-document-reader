from __future__ import annotations

import re

from .country_documents import LICENCE_TITLES, PASSPORT_TITLES
from .normalize import fold_for_match
from .ocr import OCRLine
from .schemas import DocumentType


KEYWORDS: dict[DocumentType, tuple[str, ...]] = {
    DocumentType.EMIRATES_ID_FRONT: ("EMIRATES ID", "IDENTITY CARD", "UNITED ARAB EMIRATES", "ID NUMBER"),
    DocumentType.EMIRATES_ID_BACK: ("CARD HOLDER'S SIGNATURE", "IF YOU FIND THIS CARD", "ISSUING DATE"),
    DocumentType.UAE_DRIVING_LICENCE_FRONT: (
        "DRIVING LICENCE", "DRIVING LICENSE", "TRAFFIC NO", "TRAFFIC NUMBER",
        "LICENCE NO", "LICENSE NO", "LICENCE NUMBER", "LICENSE NUMBER",
        "PLACE OF ISSUE",
    ),
    DocumentType.UAE_DRIVING_LICENCE_BACK: ("VEHICLES PERMITTED", "CATEGORIES", "DRIVER SIGNATURE"),
    DocumentType.GCC_IDENTITY_FRONT: (
        "NATIONAL IDENTITY CARD", "IDENTITY CARD", "CIVIL ID", "PERSONAL NUMBER",
        "NATIONAL ID", "البطاقة المدنية", "البطاقة الشخصية", "بطاقة الهوية",
    ),
    DocumentType.GCC_IDENTITY_BACK: (
        "CARD SERIAL", "ADDRESS", "EXPIRY DATE", "الرقم المسلسل", "العنوان",
    ),
    DocumentType.PASSPORT_BIODATA: ("PASSPORT", "SURNAME", "GIVEN NAMES", "DATE OF EXPIRY"),
    DocumentType.INTERNATIONAL_DRIVING_PERMIT: (
        "INTERNATIONAL DRIVING PERMIT", "CONVENTION ON ROAD TRAFFIC",
        "PERMIS DE CONDUIRE INTERNATIONAL", "INTERNATIONAL CONVENTION OF",
        "ISSUE OF PERMIT",
    ),
    DocumentType.NATIONAL_DRIVING_LICENCE_FRONT: ("DRIVER LICENSE", "DRIVING LICENCE", "DRIVER LICENCE"),
    DocumentType.NATIONAL_DRIVING_LICENCE_BACK: ("RESTRICTIONS", "CLASS", "CATEGORIES"),
}


STRONG_MULTILINGUAL_KEYWORDS: dict[DocumentType, tuple[str, ...]] = {
    DocumentType.PASSPORT_BIODATA: (
        # "РОССИЙСКАЯ ФЕДЕРАЦИЯ" was here and is deliberately gone: it names the
        # issuing state, not the document. A Russian driving licence prints it
        # just as prominently as a Russian passport does, so it classified that
        # licence as a passport.
        "ПАСПОРТ", "REISEPASS", "PASAPORTE", "PASSEPORT", "PASSAPORTO",
        "PASSAPORTE", "PASZPORT", "PASAPOARTE", "ÚTLEVÉL", "CESTOVNÍ PAS",
        "ΔΙΑΒΑΤΗΡΙΟ", "جواز سفر", "护照", "旅券", "パスポート", "여권", "דרכון",
        "HỘ CHIẾU", "หนังสือเดินทาง", "PASPOR", "PASAPORT",
    ),
    DocumentType.INTERNATIONAL_DRIVING_PERMIT: (
        # Ghana's inner issue page identifies the booklet by its convention
        # rather than repeating the cover title.
        "INTERNATIONAL CONVENTION OF 1968", "ISSUE OF PERMIT",
        "МЕЖДУНАРОДНОЕ ВОДИТЕЛЬСКОЕ УДОСТОВЕРЕНИЕ",
        "МЕЖДУНАРОДНЫЕ ВОДИТЕЛЬСКИЕ ПРАВА",
        "КОНВЕНЦИЯ О ДОРОЖНОМ ДВИЖЕНИИ",
        "PERMISO INTERNACIONAL DE CONDUCIR",
        "INTERNATIONALER FÜHRERSCHEIN",
        "PERMESSO INTERNAZIONALE DI GUIDA",
        "رخصة قيادة دولية", "国际驾驶许可证", "国際運転免許証", "국제운전면허증",
        # The booklet's inner page does not repeat the cover title: it opens
        # with the convention's own sentence about the contracting states, in
        # French and in Arabic. Without it, an Algerian permit's issue page was
        # filed as a national licence back -- for the word "catégories" in that
        # same sentence -- and the two dates the rental turns on, "Fait le" and
        # "Jusqu'au", were discarded as back-side values.
        "LE PRESENT PERMIS EST VALABLE",
        "THE PRESENT PERMIT IS VALID",
        "الدول المتعاقدة", "الدولة المتعاقدة",
    ),
    DocumentType.NATIONAL_DRIVING_LICENCE_FRONT: (
        "ВОДИТЕЛЬСКОЕ УДОСТОВЕРЕНИЕ", "FÜHRERSCHEIN", "PERMISO DE CONDUCIR",
        "PERMIS DE CONDUIRE", "PATENTE DI GUIDA", "PRAWO JAZDY",
        # Brazil issues the CNH, whose title shares no wording with Portugal's
        # "carta de condução"; without its own entry the card matched nothing
        # and fell through to the permit scorer.
        "CARTEIRA NACIONAL DE HABILITAÇÃO", "CARTEIRA NACIONAL DE HABILITACAO",
        "CARTA DE CONDUÇÃO", "RIJBEWIJS", "KÖRKORT", "AJOKORTTI", "KØREKORT", "KOREKORT",
        "PERMIS DE CONDUCERE", "ŘIDIČSKÝ PRŮKAZ", "VODIČSKÝ PREUKAZ",
        "VEZETŐI ENGEDÉLY", "JUHILUBA", "VADĪTĀJA APLIECĪBA",
        "VAIRUOTOJO PAŽYMĖJIMAS", "СВИДЕТЕЛСТВО ЗА УПРАВЛЕНИЕ",
        "ВОЗАЧКА ДОЗВОЛА", "LEJE DREJTIMI", "ПОСВІДЧЕННЯ ВОДІЯ",
        "ΑΔΕΙΑ ΟΔΗΓΗΣΗΣ",
        # Alberta issues an "Operator's Licence", not a driver's licence, and
        # the title tables carry every wording but that one. A card heading
        # itself with a phrase the reader does not know scores nothing as a
        # front, while the word "Class" printed on it scores a third as a back
        # -- so an Alberta front was filed as a back, which is a page the
        # reader takes no name and no date of birth from.
        "OPERATOR'S LICENCE", "OPERATORS LICENCE", "OPERATOR'S LICENSE",
        "OPERATORS LICENSE",
        "رخصة قيادة", "رخصة السياقة", "机动车驾驶证", "駕駛執照", "運転免許証",
        "운전면허증", "רישיון נהיגה", "SÜRÜCÜ BELGESİ",
    ),
}


# Both sides of every comparison below are folded once, at import. Folding only
# the page text would silently break the Arabic markers: NFKD decomposes أ, إ and
# آ into a bare alif plus a combining hamza, so a literal "الإمارات" would stop
# matching its own folded page.
_FOLDED_KEYWORDS = {
    doc_type: tuple(fold_for_match(keyword) for keyword in keywords)
    for doc_type, keywords in KEYWORDS.items()
}
# A document title is a word, and it has to be matched as one.
#
# The tables below carry every state's title in its own language, and several
# are short: Germany, Sweden, Norway and Denmark all call a passport a "Pass".
# Tested as a bare substring, "PASS" is inside the French "dépassant" -- which
# is printed on the back of an Ontario driving licence, in the sentence
# limiting a trailer to one "ne dépassant pas 4600 kg". That licence scored
# 0.80 as a passport on the strength of it, tied with its own real title, and
# the tie went to whichever document type had been typed into the table first.
# The licence was filed as a passport: every licence field came back MISSING
# and the card's own rows were read into the passport's.
#
# The boundary is the one the country markers in tourist_detect already use,
# and it is applied after folding, so "DEPASSANT" keeps "PASS" out. Scripts
# that do not separate words -- Chinese, Japanese, Korean -- are unaffected,
# because none of their characters is in the boundary class.
def _bounded(folded: str) -> re.Pattern[str]:
    return re.compile(rf"(?<![0-9A-Z]){re.escape(folded)}(?![0-9A-Z])")


_FOLDED_STRONG = {
    doc_type: tuple(dict.fromkeys(
        fold_for_match(keyword)
        for keyword in (
            *keywords,
            # The hand-written lists above covered roughly a third of the world.
            # The generated tables carry the licence and passport title in every
            # tourist country's own language, so a Thai, Vietnamese, Indonesian
            # or Latin American card is recognised as the document it is instead
            # of falling through to whatever else happened to score.
            *(LICENCE_TITLES if doc_type == DocumentType.NATIONAL_DRIVING_LICENCE_FRONT else ()),
            *(PASSPORT_TITLES if doc_type == DocumentType.PASSPORT_BIODATA else ()),
        )
    ))
    for doc_type, keywords in STRONG_MULTILINGUAL_KEYWORDS.items()
}
_STRONG_PATTERNS = {
    doc_type: tuple(_bounded(folded) for folded in folded_keywords)
    for doc_type, folded_keywords in _FOLDED_STRONG.items()
}
_UAE_COUNTRY_MARKERS = tuple(fold_for_match(marker) for marker in (
    "UNITED ARAB EMIRATES", "الإمارات العربية المتحدة",
))
_UAE_LICENCE_TITLES = tuple(fold_for_match(marker) for marker in (
    "DRIVING LICENCE", "DRIVING LICENSE", "DRIVER LICENCE", "DRIVER LICENSE",
    "رخصة القيادة", "رخصة سياقة", "رخصة السواقة",
))
_UAE_COMPACT_LICENCE_TITLES = tuple(
    re.sub(r"[\s.]", "", marker) for marker in _UAE_LICENCE_TITLES
)
_UAE_NAME = fold_for_match("UNITED ARAB EMIRATES")


def has_uae_driving_licence_title(text: str) -> bool:
    """Match the UAE licence title despite harmless OCR spacing damage.

    A frequent PP-OCR output is ``Driving L icense``.  It is visually the
    actual title but cannot match a normal phrase search, which used to route
    a genuine licence as an Emirates ID because both cards say United Arab
    Emirates.  Only spacing and full stops are ignored here: altered letters
    still require the card's regular classification evidence.
    """
    folded = fold_for_match(text)
    compact = re.sub(r"[\s.]", "", folded)
    return (
        any(marker in folded for marker in _UAE_LICENCE_TITLES)
        or any(marker in compact for marker in _UAE_COMPACT_LICENCE_TITLES)
    )


# The Overseas Citizen of India certificate is a lifelong residence document.
# It is not a passport and not a licence, but it is shaped like both: a booklet
# page carrying surname, given names, sex, date and place of birth, nationality,
# a photograph, a "Date of Issue", a "Place of Issue", a referenced passport
# number and a machine-readable zone. Uploaded beside the real documents it was
# routed as a licence front on one page and a passport on the other, and the
# dates it carries -- issued 2013 in London, referencing a 2012 passport --
# were offered against the licence's own 4a row and the passport's own issue
# date as competing well-supported values, leaving three fields for an operator
# to settle and the licence country unresolved between India and the United
# Kingdom. The certificate names itself on both of its pages.
_OCI_CERTIFICATE_MARKERS = tuple(fold_for_match(marker) for marker in (
    "OVERSEAS CITIZEN OF INDIA", "OCI CERTIFICATE",
))


def is_overseas_citizen_of_india_certificate(lines: list[OCRLine]) -> bool:
    """Whether this page is an OCI certificate rather than a travel document."""
    text = fold_for_match(" ".join(line.text for line in lines))
    return any(marker in text for marker in _OCI_CERTIFICATE_MARKERS)


def _rect(line: OCRLine) -> tuple[float, float, float, float]:
    xs = [point[0] for point in line.bounding_box]
    ys = [point[1] for point in line.bounding_box]
    return min(xs), min(ys), max(xs), max(ys)


# A card's title is set in display type across the head of the page, and the
# state name beside it larger still. A recogniser returns them as separate
# boxes and orders them by position, so a Michigan licence came back as
# "DRIVER'S", "Michigan", "LICENSE" -- with the state standing between the two
# halves of the title. Joined end to end the page named no licence at all, so a
# photograph holding both sides of the card was filed by the generic words on
# its reverse, and the tourist route then discarded every value on it.
_STACKED_TITLE_MAXIMUM_GAP = 1.0
_STACKED_TITLE_ALIGNMENT = 0.06
_STACKED_TITLE_MAXIMUM_LENGTH = 40


def stacked_heading_texts(lines: list[OCRLine]) -> list[str]:
    """Each pair of rows that is one heading broken across two boxes."""
    boxed = [
        line for line in lines
        if getattr(line, "bounding_box", None)
        and len(line.text) <= _STACKED_TITLE_MAXIMUM_LENGTH
    ]
    if not boxed:
        return []
    page_width = max(_rect(line)[2] for line in boxed)
    tolerance = max(12.0, _STACKED_TITLE_ALIGNMENT * page_width)
    joined: list[str] = []
    for first in boxed:
        left, _top, _right, bottom = _rect(first)
        height = max(1.0, bottom - _top)
        for second in boxed:
            if second is first:
                continue
            other_left, other_top, _other_right, other_bottom = _rect(second)
            other_height = max(1.0, other_bottom - other_top)
            gap = other_top - bottom
            if gap < 0 or gap > _STACKED_TITLE_MAXIMUM_GAP * min(height, other_height):
                continue
            if abs(other_left - left) > tolerance:
                continue
            joined.append(f"{first.text} {second.text}")
    return joined


def page_licence_title_present(lines: list[OCRLine]) -> bool:
    """Whether the page heads itself with a driving-licence title."""
    text = fold_for_match(
        " ".join([*(line.text for line in lines), *stacked_heading_texts(lines)])
    )
    return any(fold_for_match(title) in text for title in LICENCE_TITLES)


def classify_document(lines: list[OCRLine], expected: DocumentType, has_mrz: bool = False, barcode_types: list[str] | None = None) -> tuple[DocumentType, dict[str, float]]:
    text = fold_for_match(
        " ".join([*(line.text for line in lines), *stacked_heading_texts(lines)])
    )
    # Only a structurally parsed MRZ is sufficient to override the upload slot.
    # A loose "P<" substring appears in ordinary multilingual booklet text and
    # previously caused a Russian IDP page to be rejected as a passport.
    if has_mrz:
        return DocumentType.PASSPORT_BIODATA, {DocumentType.PASSPORT_BIODATA.value: 1.0}
    # UAE cards share the country heading and several generic labels (name,
    # nationality, DOB, expiry).  The previous keyword ratio treated the
    # country heading alone as Emirates-ID evidence and therefore routed UAE
    # licences printed with the American spelling "Driving License" into the
    # ID extractor.  Bind a visible licence title to UAE country evidence
    # before applying generic card scores.  Both English spellings and the
    # Arabic title occur on genuine cards issued by different emirates.
    uae_country = any(marker in text for marker in _UAE_COUNTRY_MARKERS)
    uae_licence_title = has_uae_driving_licence_title(text)
    if uae_country and uae_licence_title:
        return DocumentType.UAE_DRIVING_LICENCE_FRONT, {
            DocumentType.UAE_DRIVING_LICENCE_FRONT.value: 1.0,
        }
    scores: dict[DocumentType, float] = {}
    titled: dict[DocumentType, bool] = {}
    for doc_type, keywords in _FOLDED_KEYWORDS.items():
        matched = sum(1 for keyword in keywords if keyword in text)
        scores[doc_type] = matched / len(keywords)
        titled[doc_type] = any(
            pattern.search(text) for pattern in _STRONG_PATTERNS.get(doc_type, ())
        )
        if titled[doc_type]:
            scores[doc_type] = max(scores[doc_type], 0.80)
    if scores.get(expected, 0) > 0:
        scores[expected] += 0.12  # slot is a weak prior, never sufficient alone
    # Ranked, and the ranking is written down. ``max`` alone breaks a tie by
    # whichever key was typed into the table first, which is not a fact about
    # the document: an Ontario licence tied with a passport at 0.80 and was
    # filed as a passport because that entry is nine lines higher up. A page
    # that prints one document's title and not the other's is that document,
    # whatever generic keywords it shares.
    best = max(
        scores,
        # A document title is direct evidence of what the page is; generic
        # back-side words are only layout evidence. This matters when a single
        # upload contains both sides of a card: Ghana's front says DRIVER
        # LICENCE while the lower half contains RESTRICTIONS, CLASS and
        # CATEGORIES. Counting those three generic words first filed the whole
        # image as a back and discarded its clearly labelled front fields.
        key=lambda doc_type: (titled[doc_type], scores[doc_type]),
    )
    # Resolve the shared driving-licence phrase using the upload slot and country evidence.
    if best in {DocumentType.UAE_DRIVING_LICENCE_FRONT, DocumentType.NATIONAL_DRIVING_LICENCE_FRONT}:
        if _UAE_NAME in text or expected == DocumentType.UAE_DRIVING_LICENCE_FRONT:
            best = DocumentType.UAE_DRIVING_LICENCE_FRONT
        elif expected == DocumentType.NATIONAL_DRIVING_LICENCE_FRONT:
            best = DocumentType.NATIONAL_DRIVING_LICENCE_FRONT
    if scores[best] < 0.22:
        return DocumentType.UNKNOWN, {key.value: round(value, 3) for key, value in scores.items()}
    return best, {key.value: round(value, 3) for key, value in scores.items()}
