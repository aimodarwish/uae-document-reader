"""Decide, from the documents alone, which licence policy a tourist falls under.

The tourist workflow used to depend on an operator picking the licence country
from a dropdown. Everything the dropdown was there to answer is printed on the
documents themselves:

* whether the customer presented an international permit or a national licence
  is settled by the convention wording every conforming permit carries
  ("Convention on Road Traffic of 19 September 1949" / "of 8 November 1968"),
  not by the country;
* which country issued a national licence is settled by the issuer wording, the
  AAMVA barcode's country element, or the official document title.

The passport is deliberately *not* the primary source. The NATIONAL_ONLY rule
exists to check the holder's nationality against the licence's country, so
deriving the licence country from the passport would compare a value with
itself and pass every time. The passport is therefore a flagged fallback, used
only when the licence carries no country evidence of its own.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from .classify import page_licence_title_present
from .country_documents import COUNTRY_ISSUER_MARKERS, LICENCE_TITLES
from .licence_profiles import COUNTRY_NAMES, GCC_COUNTRIES, policy_for_country
from .normalize import fold_for_match, normalize_country


class DetectionSource:
    LICENCE_BARCODE = "licence_barcode"
    LICENCE_TEXT = "licence_text"
    VLM_VISUAL = "vlm_visual"
    # A private translation card sometimes states its underlying permit state
    # as ``PERMIT: MEXICO``.  The row is useful document evidence, but it is not
    # equivalent to an issuing authority or a government barcode and must stay
    # behind operator review.
    IDP_PERMIT_LABEL = "idp_permit_label"
    PASSPORT_TEXT = "passport_text"
    PASSPORT_NATIONALITY = "passport_nationality"
    OPERATOR = "operator_selection"


@dataclass(frozen=True)
class CountryEvidence:
    """One country reading, with where it came from and what proved it."""

    country: str
    source: str
    confidence: float
    evidence_text: str


# ---------------------------------------------------------------------------
# International driving permit
# ---------------------------------------------------------------------------
# Both UN models print their convention and its date on the front cover, in
# English, on every conforming booklet. That string is the only field on an IDP
# that is typeset rather than filled in by hand, which makes it the one piece of
# an IDP a reader can rely on.
CONVENTION_1949 = (
    "19 SEPTEMBER 1949", "19TH SEPTEMBER 1949", "SEPTEMBER 19, 1949",
    "19 SEPTEMBRE 1949", "1949 CONVENTION", "CONVENTION OF 1949",
)
CONVENTION_1968 = (
    "8 NOVEMBER 1968", "8TH NOVEMBER 1968", "NOVEMBER 8, 1968",
    "8 NOVEMBRE 1968", "1968 CONVENTION", "CONVENTION OF 1968",
    # Czech IDP cover: ``Úmluva o silničním provozu z 8. listopadu
    # 1968``. The final Y is a recurring OCR substitution on the textured
    # booklet paper, so both the printed form and that one-letter read are
    # accepted.
    "8. LISTOPADU 1968", "8 LISTOPADU 1968", "8. LISTOPADY 1968",
)

# Titles that appear on the cover of a permit and on no national licence. The
# Cyrillic and CJK forms matter because the cover of a Russian, Japanese, Korean
# or Chinese permit prints its own language above the English.
IDP_TITLE_MARKERS = (
    "INTERNATIONAL DRIVING PERMIT", "INTERNATIONAL DRIVER'S PERMIT",
    "INTERNATIONAL DRIVING LICENCE", "INTERNATIONAL DRIVING LICENSE",
    "INTERNATIONAL MOTOR TRAFFIC", "CONVENTION ON ROAD TRAFFIC",
    "PERMIS DE CONDUIRE INTERNATIONAL", "CIRCULATION ROUTIERE INTERNATIONALE",
    "CIRCULATION ROUTIÈRE INTERNATIONALE",
    # Czech booklets put their own title above the French translation. The
    # base phrase ``ŘIDIČSKÝ PRŮKAZ`` is also the title of a Czech national
    # licence, so the complete ``MEZINÁRODNÍ`` wording must be recognised
    # before routing is allowed to fall back to that embedded national title.
    "MEZINÁRODNÍ ŘIDIČSKÝ PRŮKAZ", "MEZINARODNI RIDICSKY PRUKAZ",
    # The official cover uses this word order, while the existing marker only
    # covered ``PERMIS DE CONDUIRE INTERNATIONAL``.
    "PERMIS INTERNATIONAL DE CONDUIRE",
    "PERMISO INTERNACIONAL DE CONDUCIR", "INTERNATIONALER FÜHRERSCHEIN",
    "PERMESSO INTERNAZIONALE DI GUIDA", "AUTORIZAÇÃO INTERNACIONAL",
    "МЕЖДУНАРОДНОЕ ВОДИТЕЛЬСКОЕ УДОСТОВЕРЕНИЕ",
    "МЕЖДУНАРОДНЫЕ ВОДИТЕЛЬСКИЕ ПРАВА",
    "КОНВЕНЦИЯ О ДОРОЖНОМ ДВИЖЕНИИ",
    "رخصة قيادة دولية", "إجازة سوق دولية",
    "国际驾驶许可证", "國際駕駛許可證", "国際運転免許証", "국제운전면허증",
)

IDP_1968_VIENNA = "1968_VIENNA"
IDP_1949_GENEVA = "1949_GENEVA"
IDP_MODEL_UNSPECIFIED = "UNSPECIFIED"


# Fragments that hint at a permit without proving one. "INTERNATIONAL" alone
# also appears on national cards, so these are only consulted where the country
# is already known to require a permit and the question is merely which page of
# the bundle it is.
IDP_WEAK_MARKERS = (
    "INTERNATIONAL", "IDP NO", "PERMIT NO", "NO. DU PERMIS", "МЕЖДУНАРОД", "МВУ",
)


def idp_convention(text: str) -> str | None:
    """Return which IDP model this page is, or None if it is not a permit.

    A page that names a convention is a permit even when the title line was
    lost to glare, and a page carrying a permit title is a permit even when the
    convention line fell outside the crop.
    """
    # OCR returns a cover title a line at a time.  A Tunisian booklet's
    # ``Permis international de conduire`` consequently reached this function
    # as ``Permis international\nde conduire`` and failed an exact-space match,
    # despite being an unambiguous official permit title.  Fold accents and
    # collapse every whitespace run on both sides; this preserves the specific
    # multi-word markers while making their line wrapping irrelevant.
    upper = " ".join(fold_for_match(text).split())

    def contains(marker: str) -> bool:
        return " ".join(fold_for_match(marker).split()) in upper

    if any(contains(marker) for marker in CONVENTION_1968):
        return IDP_1968_VIENNA
    if any(contains(marker) for marker in CONVENTION_1949):
        return IDP_1949_GENEVA
    # Some official French covers state both treaty years in one sentence
    # (for example ``... du 16 septembre 1949 et 7-8 octobre 1968``), whose
    # local date order differs from the model's English specimen.  The
    # convention and road-traffic wording together make the page a permit;
    # prefer the newer 1968 model when both appear, as above.
    is_road_traffic_convention = (
        "CONVENTION" in upper
        and ("CIRCULATION ROUTIERE" in upper or "ROAD TRAFFIC" in upper)
    )
    if is_road_traffic_convention and "1968" in upper:
        return IDP_1968_VIENNA
    if is_road_traffic_convention and "1949" in upper:
        return IDP_1949_GENEVA
    if any(contains(marker) for marker in IDP_TITLE_MARKERS):
        return IDP_MODEL_UNSPECIFIED
    return None


def looks_like_idp(text: str) -> bool:
    return idp_convention(text) is not None


_IDP_PERMIT_COUNTRY_ROW = re.compile(
    r"^\s*PERMIT\s*[:\-]\s*(.+?)\s*$", re.I | re.M,
)


def country_from_idp_permit_label(text: str) -> CountryEvidence | None:
    """Read ``PERMIT: <country>`` without confusing the IDP title or birthplace.

    Private translation cards use this non-standard row to state the country
    of the national licence they translate.  It is intentionally parsed only
    as a complete labelled row: ``INTERNATIONAL DRIVING PERMIT`` and ``COUNTRY
    OF BIRTH: COLOMBIA`` therefore cannot donate a country to this field.
    """
    found: dict[str, tuple[str, str]] = {}
    for match in _IDP_PERMIT_COUNTRY_ROW.finditer(text):
        raw = match.group(1).strip(" .")
        code, country, _ = normalize_country(raw)
        if code and country:
            found[code] = (country, match.group(0).strip())
    if len(found) != 1:
        # Conflicting labelled states are not settled by choosing one.
        return None
    country, evidence = next(iter(found.values()))
    return CountryEvidence(
        country=country, source=DetectionSource.IDP_PERMIT_LABEL,
        confidence=0.65, evidence_text=evidence,
    )


def idp_is_non_government_translation(text: str) -> bool:
    """True when the card itself disclaims government-IDP status."""
    folded = " ".join(fold_for_match(text).split())
    return (
        "NON GOVERNMENT IDENTIFICATION CARD" in folded
        or "TRANSLATION BASED ON CURRENT DRIVER" in folded
    )


def idp_is_private_translation_document(text: str) -> bool:
    """Identify a private driver-licence translation, not a treaty IDP.

    An official document uses the term *International Driving Permit* and
    cites one of the 1949 or 1968 conventions.  The reported cards instead
    call themselves an ``International Driver's License``, state that they
    translate a foreign licence, and cite a non-existent 1964 convention.
    None of these claims can become a licence number, country, or permit.
    """
    folded = " ".join(fold_for_match(text).split())
    private_translation = (
        "INTERNATIONAL DRIVER'S LICENSE" in folded
        and "TRANSLATION" in folded
        and "FOREIGN DRIVER" in folded
    )
    # International Drivers Association cards use this alternate title. It is
    # still a commercial translation for this workflow, not a treaty booklet;
    # recognise its own words rather than mistaking its original-DL field for a
    # national licence.
    # On the compact IAA card OCR often emits ``ASSOCIATION`` as eleven
    # separate one-letter rows.  Keep the title requirement strict, but allow
    # that one harmless typography variation in the provider name.
    ida_association = (
        "INTERNATIONAL DRIVERS ASSOCIATION" in folded
        or bool(re.search(
            r"\bINTERNATIONAL\s+DRIVERS\s+A\s+S\s+S\s+O\s+C\s+I\s+A\s+T\s+I\s+O\s+N\b",
            folded,
        ))
    )
    ida_translation = (
        "INTERNATIONAL TRANSLATION OF DRIVER'S LICENSE" in folded
        and ida_association
    )
    false_convention = (
        "THE UN CONVENTIONS ON ROAD TRAFFIC" in folded
        and "1949" in folded
        and "1964" in folded
        and "INTERNATIONAL DRIVING DOCUMENT" in folded
    )
    return private_translation or ida_translation or false_convention


# ---------------------------------------------------------------------------
# Which country issued a national licence
# ---------------------------------------------------------------------------
# Wording that identifies exactly one issuing state. Anything ambiguous between
# two countries is left out entirely: a wrong country silently selects the wrong
# acceptance rule, which is worse than detecting nothing and asking.
#
# Document titles are absent from this table on purpose, and that is a rule with
# teeth. "FÜHRERSCHEIN" is printed by Germany, Austria, Switzerland and
# Liechtenstein alike. The same trap caught this table once already: Turkish
# "SÜRÜCÜ BELGESİ" was listed under Turkey, so a Cypriot licence printed in
# Turkish named Turkey as its issuer, and Hungarian "VEZETŐI ENGEDÉLY" listed
# under Hungary did the same to a Slovenian card. A title says which language the
# card is printed in; only the state's own name says who issued it. Every title
# now lives in ``country_documents.LICENCE_TITLES``.
COUNTRY_TEXT_MARKERS: dict[str, tuple[str, ...]] = {
    # A UAE resident may present this bilingual card through the Tourist route.
    # Its national heading is the issuing-country evidence; the holder's
    # separately labelled nationality can of course be any country.
    "United Arab Emirates": ("UNITED ARAB EMIRATES",),
    # The other five Gulf states had no wording here at all, so a card issued
    # by any of them could not name its own issuer on the Tourist route. A
    # Palestinian working in Saudi Arabia presented a licence headed "KINGDOM
    # OF SAUDI ARABIA" and the reader, finding no country on the page, took
    # the issuer from his passport and reported a Saudi licence as issued by
    # Palestine. The holder's own nationality row is kept out of issuer
    # evidence by the caption test in the pipeline, in Arabic as in English.
    "Saudi Arabia": (
        "KINGDOM OF SAUDI ARABIA", "المملكة العربية السعودية", "SAUDI ARABIA",
    ),
    "Qatar": ("STATE OF QATAR", "دولة قطر", "QATAR"),
    "Kuwait": ("STATE OF KUWAIT", "دولة الكويت", "KUWAIT"),
    "Oman": ("SULTANATE OF OMAN", "سلطنة عمان", "OMAN"),
    "Bahrain": ("KINGDOM OF BAHRAIN", "مملكة البحرين", "BAHRAIN"),
    "United Kingdom": (
        "DRIVER AND VEHICLE LICENSING AGENCY", "DVLA", "UNITED KINGDOM OF GREAT BRITAIN",
        "DRIVER & VEHICLE AGENCY", "GREAT BRITAIN",
    ),
    "Ireland": ("NATIONAL DRIVER LICENCE SERVICE", "EIRE", "ÉIRE", "IRELAND"),
    "Germany": ("BUNDESREPUBLIK DEUTSCHLAND", "DEUTSCHLAND", "FEDERAL REPUBLIC OF GERMANY"),
    # Austria names its issuing authority where it does not name itself: the
    # card front carries FÜHRERSCHEIN, which Germany, Switzerland and
    # Liechtenstein print too, and its own state name nowhere. These stand
    # here for the reason DVLA does. The abbreviation is needed as well as
    # the full form: the reported Vienna card prints "4c. LPD Wien VA".
    "Austria": (
        "REPUBLIK ÖSTERREICH", "OSTERREICH", "ÖSTERREICH", "REPUBLIC OF AUSTRIA",
        "LANDESPOLIZEIDIREKTION", "BEZIRKSHAUPTMANNSCHAFT", "LPD",
    ),
    "Switzerland": ("SCHWEIZERISCHE EIDGENOSSENSCHAFT", "CONFEDERATION SUISSE", "CONFÉDÉRATION SUISSE", "SUISSE", "SVIZZERA", "SWISS CONFEDERATION", "SWITZERLAND", "MANISCHAR"),
    "France": ("REPUBLIQUE FRANCAISE", "RÉPUBLIQUE FRANÇAISE", "FRANCE", "FRENCH REPUBLIC"),
    "Belgium": ("ROYAUME DE BELGIQUE", "KONINKRIJK BELGIE", "KONINKRIJK BELGIË", "BELGIQUE", "BELGIE", "BELGIË", "KINGDOM OF BELGIUM"),
    # RDW is the Dutch vehicle authority and the only issuer named anywhere on
    # the licence: the card prints three titles across the top -- RIJBEWIJS,
    # PERMIS DE CONDUIRE, FÜHRERSCHEIN -- and its country nowhere. It stands
    # here for the same reason DVLA does.
    "Netherlands": ("KONINKRIJK DER NEDERLANDEN", "NEDERLAND", "KINGDOM OF THE NETHERLANDS", "NETHERLANDS", "RDW"),
    "Luxembourg": ("GRAND-DUCHE DE LUXEMBOURG", "GRAND-DUCHÉ DE LUXEMBOURG", "LUXEMBOURG"),
    "Spain": ("REINO DE ESPANA", "REINO DE ESPAÑA", "ESPANA", "ESPAÑA", "KINGDOM OF SPAIN"),
    "Portugal": ("REPUBLICA PORTUGUESA", "REPÚBLICA PORTUGUESA", "PORTUGAL", "PORTUGUESE REPUBLIC"),
    # Brazil requires a permit, so naming the issuer of a Brazilian CNH is what
    # makes the reader ask for one instead of accepting the card on its own.
    # DENATRAN/CONTRAN are the federal traffic bodies and appear on every CNH.
    "Brazil": (
        "REPUBLICA FEDERATIVA DO BRASIL", "REPÚBLICA FEDERATIVA DO BRASIL",
        "DEPARTAMENTO NACIONAL DE TRANSITO", "DEPARTAMENTO NACIONAL DE TRÂNSITO",
        "DENATRAN", "CONTRAN", "BRASIL",
    ),
    "Italy": ("REPUBBLICA ITALIANA", "ITALIA", "ITALIAN REPUBLIC"),
    "Greece": ("ΕΛΛΗΝΙΚΗ ΔΗΜΟΚΡΑΤΙΑ", "ΕΛΛΑΣ", "ELLAS", "HELLENIC REPUBLIC", "GREECE"),
    "Denmark": ("DANMARK", "KINGDOM OF DENMARK", "DENMARK"),
    "Sweden": ("SVERIGE", "KINGDOM OF SWEDEN", "SWEDEN"),
    "Norway": ("NORGE", "NOREG", "KINGDOM OF NORWAY", "NORWAY"),
    "Finland": ("SUOMI", "FINLAND", "REPUBLIC OF FINLAND"),
    "Iceland": ("ISLAND", "ÍSLAND", "ICELAND"),
    "Poland": ("RZECZPOSPOLITA POLSKA", "POLSKA", "REPUBLIC OF POLAND", "POLAND"),
    "Romania": ("ROMANIA", "ROMÂNIA"),
    "Bulgaria": ("РЕПУБЛИКА БЪЛГАРИЯ", "БЪЛГАРИЯ", "BULGARIA"),
    "Hungary": ("MAGYARORSZAG", "MAGYARORSZÁG", "HUNGARY"),
    "Czechia (Czech Republic)": ("CESKA REPUBLIKA", "ČESKÁ REPUBLIKA", "CZECH REPUBLIC", "CZECHIA"),
    "Slovakia": ("SLOVENSKA REPUBLIKA", "SLOVENSKÁ REPUBLIKA", "SLOVAK REPUBLIC", "SLOVAKIA"),
    "Slovenia": ("REPUBLIKA SLOVENIJA", "SLOVENIJA", "REPUBLIC OF SLOVENIA", "SLOVENIA"),
    "Croatia": ("REPUBLIKA HRVATSKA", "HRVATSKA", "REPUBLIC OF CROATIA", "CROATIA"),
    "Estonia": ("EESTI VABARIIK", "EESTI", "REPUBLIC OF ESTONIA", "ESTONIA"),
    "Latvia": ("LATVIJAS REPUBLIKA", "LATVIJA", "REPUBLIC OF LATVIA", "LATVIA"),
    "Lithuania": ("LIETUVOS RESPUBLIKA", "LIETUVA", "REPUBLIC OF LITHUANIA", "LITHUANIA"),
    "Malta": ("MALTA",),
    "Cyprus": ("ΚΥΠΡΙΑΚΗ ΔΗΜΟΚΡΑΤΙΑ", "KYPROS", "CYPRUS"),
    "Albania": ("REPUBLIKA E SHQIPERISE", "REPUBLIKA E SHQIPËRISË", "SHQIPERIA", "SHQIPËRIA"),
    "Serbia": ("РЕПУБЛИКА СРБИЈА", "REPUBLIKA SRBIJA", "СРБИЈА"),
    "Montenegro": ("CRNA GORA", "ЦРНА ГОРА"),
    "North Macedonia": ("РЕПУБЛИКА СЕВЕРНА МАКЕДОНИЈА", "SEVERNA MAKEDONIJA", "NORTH MACEDONIA"),
    "Ukraine": ("УКРАЇНА", "UKRAINE"),
    "Azerbaijan": ("AZƏRBAYCAN", "AZERBAYCAN", "AZERBAIJAN"),
    "Turkey": ("TURKIYE CUMHURIYETI", "TÜRKİYE CUMHURİYETİ", "TURKIYE", "TÜRKİYE"),
    "Israel": ("מדינת ישראל", "STATE OF ISRAEL"),
    "Japan": ("日本国",),
    "South Korea": ("대한민국", "REPUBLIC OF KOREA"),
    "China": ("中华人民共和国",),
    "Hong Kong": ("HONG KONG", "香港", "香港特別行政區"),
    "Singapore": ("SINGAPORE", "REPUBLIC OF SINGAPORE"),
    "India": ("UNION OF INDIA", "INDIAN UNION", "भारत", "THE UNION OF INDIA"),
    "Tajikistan": (
        "REPUBLIC OF TAJIKISTAN", "TAJIKISTAN", "TOJIKISTON", "ТОЧИКИСТОН",
    ),
    # Tunisian permits carry the state name in French/Arabic rather than the
    # English wording found in the generic issuer table.  It is the issuing
    # state printed across the official booklet cover, not the holder's
    # nationality, so it is safe country evidence for an IDP.
    "Tunisia": (
        "REPUBLIC OF TUNISIA", "TUNISIA", "TUNISIE",
        "REPUBLIQUE TUNISIENNE", "RÉPUBLIQUE TUNISIENNE",
        "الجمهورية التونسية", "تونس",
    ),
    "South Africa": ("REPUBLIC OF SOUTH AFRICA", "SUID-AFRIKA", "RSA"),
    "Australia": ("AUSTRALIA", "NEW SOUTH WALES", "VICTORIA", "QUEENSLAND",
                  "WESTERN AUSTRALIA", "SOUTH AUSTRALIA", "TASMANIA",
                  "AUSTRALIAN CAPITAL TERRITORY", "NORTHERN TERRITORY"),
    "New Zealand": ("NEW ZEALAND", "AOTEAROA", "NZ TRANSPORT AGENCY", "WAKA KOTAHI"),
    # A Canadian licence is issued by a province and never prints the word
    # "Canada": a British Columbia card heads itself "DRIVER'S LICENCE /
    # British Columbia CAN", so the only marker listed here matched nothing and
    # the country fell through to the model's visual guess -- correct, but at
    # 0.72 and marked for review, from a card that states its issuer twice in
    # plain text. Australia's states have been listed here since its licences
    # were worked on; Canada's provinces never were.
    #
    # "Ontario" is left out on purpose, and it is the one that costs most:
    # there is an Ontario in California, and an address line naming it would
    # outrank "USA" on length and hand a Californian licence to Canada. A wrong
    # country silently selects the wrong acceptance rule, which is worse than
    # detecting nothing and asking. The same reasoning keeps out "New
    # Brunswick", which is also a city in New Jersey.
    "Canada": (
        "CANADA", "BRITISH COLUMBIA", "ALBERTA", "SASKATCHEWAN", "MANITOBA",
        "NOVA SCOTIA", "NEWFOUNDLAND", "PRINCE EDWARD ISLAND", "QUEBEC",
        "QUÉBEC", "YUKON", "NORTHWEST TERRITORIES", "NUNAVUT",
    ),
    "United States": (
        "UNITED STATES OF AMERICA", "USA",
        # Massachusetts licences name the issuing state, not the country.  The
        # AAMVA PDF417 normally supplies DCG=USA, but photographed cards often
        # leave the barcode too small to decode.  The state heading is still
        # explicit issuer evidence and must win over a visual-model guess.
        "MASSACHUSETTS", "MASS.GOV/RMV",
    ),
}


# ISO3 codes carried by the AAMVA barcode's DCG element.
BARCODE_COUNTRY_CODES = {"USA": "United States", "CAN": "Canada"}


# The EU card model prints a machine-readable line across the foot of the
# licence: "D1NLD25057964270..." on the Dutch card, where D1 is the document
# code and the three letters after it are the issuing state.
#
# This matters far beyond one country. The wording table can only name a state
# that the card actually prints, and several EU cards never print theirs: the
# Dutch licence says RIJBEWIJS, PERMIS DE CONDUIRE and FÜHRERSCHEIN across the
# top -- three titles, no country -- and names only its authority, RDW. The card
# zone is printed by that authority, in a fixed position, in Latin characters,
# and it says NLD outright.
#
# OCR reads the digit 1 as a capital I often enough to be worth allowing.
_CARD_ZONE_STATE = re.compile(r"(?<![0-9A-Z])D[1IL]([A-Z]{3})[0-9A-Z]")


def country_from_card_zone(text: str) -> CountryEvidence | None:
    """Read the issuing state from an EU licence's own machine-readable line."""
    for match in _CARD_ZONE_STATE.finditer(text.upper()):
        policy = policy_for_country(match.group(1))
        if policy is not None:
            return CountryEvidence(
                country=policy.country, source=DetectionSource.LICENCE_TEXT,
                confidence=0.95, evidence_text=match.group(0),
            )
    return None


# The curated markers above are the authority names and document wording that no
# reference dataset carries -- DVLA, DENATRAN, Waka Kotahi. The generated table
# adds every state's own official name, in its own language, for all 190
# countries. Curated entries come first so a hand-checked marker wins a tie.
ALL_COUNTRY_MARKERS: dict[str, tuple[str, ...]] = {
    country: tuple(dict.fromkeys((
        *COUNTRY_TEXT_MARKERS.get(country, ()),
        *COUNTRY_ISSUER_MARKERS.get(country, ()),
    )))
    for country in set(COUNTRY_TEXT_MARKERS) | set(COUNTRY_ISSUER_MARKERS)
}

_FOLDED_MARKERS: tuple[tuple[str, str, str], ...] = tuple(
    (fold_for_match(marker), country, marker)
    for country, markers in ALL_COUNTRY_MARKERS.items()
    for marker in markers
)

# Word-boundary patterns, compiled once at import rather than rebuilt per page.
# Running the regex for all 657 markers on every driving page cost ~12 ms, which
# is two orders of magnitude more than classification and grows with every
# country added. Nearly all markers can be dismissed by a plain substring test
# first -- that runs in C, and only the handful that survive it need the
# boundary check that distinguishes MALI from SMALTATURA.
_MARKER_PATTERNS: dict[str, re.Pattern[str]] = {
    folded: re.compile(rf"(?<![0-9A-Z]){re.escape(folded)}(?![0-9A-Z])")
    for folded, _, _ in _FOLDED_MARKERS
}

# The shortest marker is the cheapest possible reject: a page shorter than it
# cannot contain any marker at all.
_MIN_MARKER_LENGTH = min((len(f) for f, _, _ in _FOLDED_MARKERS), default=0)


def _tokens(text: str) -> str:
    """Collapse whitespace so a marker split across OCR boxes still matches."""
    return " ".join(fold_for_match(text).split())


# The country the EU model reserves a field for. Every licence built to it
# carries its state's distinguishing sign inside the ring of stars at the top
# left, and a card can print that and name itself nowhere else: an Austrian
# licence issued to a Bosnian national gave "A" in the oval, FÜHRERSCHEIN --
# which Germany and Switzerland print too -- and "4c. BH Schärding" for the
# Bezirkshauptmannschaft. Nothing said Austria, so the reader fell back on the
# holder's passport, called the licence Bosnian, and demanded an international
# permit that an Austrian licence does not need.
#
# "BH" is deliberately absent from Austria's markers and always will be: it is
# the everyday abbreviation for Bosnia and Herzegovina, printed on the very
# passport in that bundle.
_EU_DISTINGUISHING_SIGNS: dict[str, str] = {
    "A": "Austria", "B": "Belgium", "BG": "Bulgaria", "CY": "Cyprus",
    "CZ": "Czech Republic", "D": "Germany", "DK": "Denmark", "E": "Spain",
    "EST": "Estonia", "FIN": "Finland", "F": "France", "GR": "Greece",
    "H": "Hungary", "HR": "Croatia", "I": "Italy", "IRL": "Ireland",
    "IS": "Iceland", "L": "Luxembourg", "LT": "Lithuania", "LV": "Latvia",
    "M": "Malta", "N": "Norway", "NL": "Netherlands", "P": "Portugal",
    "PL": "Poland", "RO": "Romania", "S": "Sweden", "SK": "Slovakia",
    "SLO": "Slovenia", "FL": "Liechtenstein", "CH": "Switzerland",
}

# The oval sits beside the card's title, not at some fraction of the page: a
# cropped capture moves every absolute position but not that relationship.
#
# "Beside" has to be measured against the sign, though, not against whatever
# matched a title. The multilingual legend down the edge of an EU card carries
# the word Führerschein inside "Fuhrerscheinnummer" and OCR returns it as one
# box forty pixels wide and six hundred tall; every letter in the category
# table then sits to its left and inside its vertical span, and A, B and D
# were read as Austria, Belgium and Germany at once. So the title must be a
# printed line rather than a strip on its side, its type must be of a size
# with the sign's, and the gap is counted in the sign's own height.
_OVAL_TITLE_GAP = 5.0
_OVAL_MIN_HEIGHT_RATIO = 0.5
_OVAL_MAX_HEIGHT_RATIO = 2.0
# Row-sharing is measured between the two centres, not by how far the boxes
# overlap. A Belgian card sets its sign a little below the baseline of a title
# spanning most of the card, and the boxes met over four pixels of thirty-seven
# -- plainly the same row to any eye, and rejected by an overlap rule.
_OVAL_MAX_ROW_OFFSET = 1.0


def _rect(line: Any) -> tuple[float, float, float, float]:
    xs = [point[0] for point in line.bounding_box]
    ys = [point[1] for point in line.bounding_box]
    return min(xs), min(ys), max(xs), max(ys)


_FOLDED_LICENCE_TITLES: tuple[str, ...] = tuple(dict.fromkeys(
    fold_for_match(title) for title in LICENCE_TITLES
))


def country_from_eu_distinguishing_sign(lines: list[Any]) -> CountryEvidence | None:
    """Read the issuing state from the blue oval of an EU-model licence.

    This is the standard's own country field, not an inference from wording, so
    it is read the way designator 4c or the card zone is. It is offered below a
    state that names itself in words, and only where the page proves it is a
    licence built to that model and exactly one sign stands beside its title.
    """
    titles = [
        line for line in lines
        if getattr(line, "bounding_box", None)
        and any(title in _tokens(line.text) for title in _FOLDED_LICENCE_TITLES)
    ]
    if not titles:
        return None
    found: list[tuple[str, str]] = []
    for line in lines:
        if not getattr(line, "bounding_box", None):
            continue
        token = line.text.strip().upper()
        country = _EU_DISTINGUISHING_SIGNS.get(token)
        if country is None or policy_for_country(country) is None:
            continue
        _left, top, right, bottom = _rect(line)
        height = max(1.0, bottom - top)
        for title in titles:
            title_left, title_top, title_right, title_bottom = _rect(title)
            title_height = max(1.0, title_bottom - title_top)
            if title_right - title_left <= title_height:
                continue                      # a legend on its side, not a line
            ratio = height / title_height
            if not _OVAL_MIN_HEIGHT_RATIO <= ratio <= _OVAL_MAX_HEIGHT_RATIO:
                continue                      # not set in the title's type
            if right > title_left:
                continue
            if title_left - right > _OVAL_TITLE_GAP * height:
                continue
            offset = abs((top + bottom) - (title_top + title_bottom)) * 0.5
            if offset > _OVAL_MAX_ROW_OFFSET * max(height, title_height):
                continue                      # not on the title's own row
            found.append((token, country))
            break
    countries = {country for _, country in found}
    if len(countries) != 1:
        return None
    token, country = found[0]
    return CountryEvidence(
        country=country, source=DetectionSource.LICENCE_TEXT,
        confidence=0.75, evidence_text=f"EU_DISTINGUISHING_SIGN:{token}",
    )


# The United States prints no country name on most of its licences: the card
# heads itself with the state that issued it. Maryland's says "MARYLAND" over
# "Driver's License" and names the country nowhere on either side, so a bundle
# whose passport happened to be American resolved the licence's country from
# that passport -- an inference, which the rule below then refuses to write
# into the issuing-country field, leaving it empty on a card that says plainly
# who issued it. A state named on a driving licence is that evidence.
#
# Georgia is deliberately absent: it is a state and a country, and a page
# naming it says nothing on its own. The two-word names are matched whole, so
# "NEW MEXICO" is never read as Mexico.
US_STATE_NAMES: tuple[str, ...] = (
    "ALABAMA", "ALASKA", "ARIZONA", "ARKANSAS", "CALIFORNIA", "COLORADO",
    "CONNECTICUT", "DELAWARE", "DISTRICT OF COLUMBIA", "FLORIDA", "HAWAII",
    "IDAHO", "ILLINOIS", "INDIANA", "IOWA", "KANSAS", "KENTUCKY", "LOUISIANA",
    "MAINE", "MARYLAND", "MASSACHUSETTS", "MICHIGAN", "MINNESOTA",
    "MISSISSIPPI", "MISSOURI", "MONTANA", "NEBRASKA", "NEVADA",
    "NEW HAMPSHIRE", "NEW JERSEY", "NEW MEXICO", "NEW YORK",
    "NORTH CAROLINA", "NORTH DAKOTA", "OHIO", "OKLAHOMA", "OREGON",
    "PENNSYLVANIA", "PUERTO RICO", "RHODE ISLAND", "SOUTH CAROLINA",
    "SOUTH DAKOTA", "TENNESSEE", "TEXAS", "UTAH", "VERMONT", "VIRGINIA",
    "WASHINGTON", "WEST VIRGINIA", "WISCONSIN", "WYOMING",
)
_FOLDED_US_STATES = tuple(
    (fold_for_match(name), name)
    # Longest first: "WEST VIRGINIA" must not be read as "VIRGINIA", and a page
    # naming both is naming one state twice.
    for name in sorted(US_STATE_NAMES, key=len, reverse=True)
)
_US_STATE_PATTERNS = {
    folded: re.compile(rf"(?<![A-Z0-9]){re.escape(folded)}(?![A-Z0-9])")
    for folded, _ in _FOLDED_US_STATES
}


def us_states_named(text: str) -> set[str]:
    """Which U.S. states this text names, longer names taken over shorter."""
    compact = _tokens(text)
    found: set[str] = set()
    matched: list[str] = []
    for folded, name in _FOLDED_US_STATES:
        if folded not in compact:
            continue
        if not _US_STATE_PATTERNS[folded].search(compact):
            continue
        if any(folded in longer for longer in matched):
            continue
        matched.append(folded)
        found.add(name)
    return found


def country_from_us_state(lines: list[Any]) -> CountryEvidence | None:
    """Read the United States from the state a licence names as its issuer.

    One state, and it must be the card's own heading rather than a line of the
    holder's address: a Maryland licence prints "ROCKVILLE MD 20850" under
    "Address" and "MARYLAND" across the top, and only the second says who
    issued the card. Two states named would be exactly that ambiguity, so
    nothing is claimed.
    """
    text = " ".join(getattr(line, "text", "") for line in lines)
    states = us_states_named(text)
    if len(states) != 1:
        return None
    if not page_licence_title_present(lines):
        return None
    state = next(iter(states))
    return CountryEvidence(
        country="United States", source=DetectionSource.LICENCE_TEXT,
        confidence=0.90, evidence_text=f"US_STATE:{state}",
    )


def country_from_barcode(structured: dict[str, str] | None) -> CountryEvidence | None:
    """Read the issuing country from an AAMVA PDF417 payload.

    The barcode is printed by the issuing authority and is not an OCR reading,
    so nothing on the page outranks it.
    """
    if not structured:
        return None
    raw = (structured.get("issuing_country") or "").strip().upper()
    country = BARCODE_COUNTRY_CODES.get(raw)
    if country is None:
        return None
    return CountryEvidence(
        country=country, source=DetectionSource.LICENCE_BARCODE,
        confidence=0.99, evidence_text=f"DCG={raw}",
    )


def country_from_text(text: str) -> CountryEvidence | None:
    """Read the issuing country from the wording printed on a licence page.

    Longer markers are tried first so "UNITED STATES OF AMERICA" is not beaten
    by a bare country name appearing in a translation table elsewhere on the
    page. A match must sit on a word boundary, so "MALI" cannot be found inside
    "NORMALISED".
    """
    compact = _tokens(text)
    if len(compact) < _MIN_MARKER_LENGTH:
        return None
    matches: list[tuple[int, str, str, str]] = []
    for folded, country, marker in _FOLDED_MARKERS:
        # Cheap substring reject first; the boundary regex only runs for the
        # few markers that are actually present in the page.
        if folded not in compact:
            continue
        if _MARKER_PATTERNS[folded].search(compact):
            matches.append((len(folded), country, marker, folded))
    if not matches:
        return None
    # "GUINEA" sits inside "GUINEA-BISSAU" and "SUDAN" inside "SOUTH SUDAN", so a
    # card naming the longer state also matches the shorter one. A match wholly
    # contained in a longer match is that longer match, not a second country.
    survivors = [
        item for item in matches
        if not any(
            item[3] != other[3] and item[3] in other[3]
            for other in matches
        )
    ]
    matches = survivors or matches
    matched_countries = {country for _, country, _, _ in matches}
    length, country, marker, _ = max(matches, key=lambda item: item[0])
    # Two genuinely different states named on one page means a translation panel,
    # a convention list or a foreign address, not an issuer. Detecting nothing is
    # the safe outcome; the operator is asked instead of being given a guess.
    if len(matched_countries) > 1:
        return None
    return CountryEvidence(
        country=country, source=DetectionSource.LICENCE_TEXT,
        confidence=0.90 if length >= 12 else 0.80, evidence_text=marker,
    )


_PASSPORT_NON_ISSUER_LABELS = (
    "NATIONALITY", "COUNTRY OF BIRTH", "PLACE OF BIRTH", "RESIDENCE",
    "ADDRESS", "NATIONALITE", "NATIONALITAT", "STAATSANGEHORIGKEIT",
)


def country_from_passport_lines(lines: list[str]) -> CountryEvidence | None:
    """Recover a passport issuer when its complete MRZ was not readable.

    A passport states the issuing state twice independently: in the fixed
    three-character issuer slot immediately after ``P<`` and as the official
    state heading on the biodata page.  A cropped/glared page can leave either
    one readable while preventing a checksummed MRZ parse.

    Heading matches are accepted only when the state name dominates its own
    OCR row.  This deliberately rejects ``Nationality: Colombia``, ``Place of
    birth: Germany`` and addresses: nationality and birthplace do not prove
    who issued a travel document. Conflicting structural and heading evidence
    yields no answer rather than a guessed country.
    """
    heading: dict[str, tuple[str, str]] = {}
    for text in lines:
        folded_line = fold_for_match(text)
        if any(label in folded_line for label in _PASSPORT_NON_ISSUER_LABELS):
            continue
        found = country_from_text(text)
        if found is None:
            continue
        marker = fold_for_match(found.evidence_text)
        line_size = len(re.sub(r"[^0-9A-Z]", "", folded_line))
        marker_size = len(re.sub(r"[^0-9A-Z]", "", marker))
        if line_size == 0 or marker_size / line_size < 0.60:
            continue
        policy = policy_for_country(found.country)
        if policy is not None:
            heading[policy.iso3] = (policy.country, text.strip())

    structural: dict[str, tuple[str, str]] = {}
    for text in lines:
        compact = "".join(text.upper().split())
        if not compact.startswith("P<") or len(compact) < 5:
            continue
        raw = compact[2:5].replace("<", "")
        # Germany is the ICAO exception printed as one letter plus two filler
        # characters (``P<D<<``); the workflow stores ISO alpha-3.
        raw = "DEU" if raw == "D" else raw
        policy = policy_for_country(raw)
        if policy is not None:
            structural[policy.iso3] = (policy.country, text.strip())

    # A country value can appear on its own after OCR separates it from the
    # ``Nationality`` or ``Place of birth`` caption.  That bare row is not an
    # issuer heading, and must not cancel the state that both the formal page
    # heading and the MRZ issuer slot independently establish.  A real
    # heading/MRZ disagreement still has no matching code here and remains a
    # refusal below.
    if len(structural) == 1:
        code = next(iter(structural))
        if code in heading:
            country, evidence = heading[code]
            return CountryEvidence(
                country=country,
                source=DetectionSource.PASSPORT_TEXT,
                confidence=0.90,
                evidence_text=evidence,
            )

    countries = set(heading) | set(structural)
    if len(countries) != 1:
        return None
    code = next(iter(countries))
    country, evidence = heading.get(code) or structural[code]
    confidence = 0.90 if code in heading and code in structural else (
        0.84 if code in heading else 0.70
    )
    return CountryEvidence(
        country=country, source=DetectionSource.PASSPORT_TEXT,
        confidence=confidence, evidence_text=evidence,
    )


def country_from_nationality(iso3: str | None) -> CountryEvidence | None:
    """Fall back to the passport's nationality, clearly marked as a fallback."""
    if not iso3:
        return None
    policy = policy_for_country(iso3)
    if policy is None or policy.country in GCC_COUNTRIES:
        return None
    return CountryEvidence(
        country=policy.country, source=DetectionSource.PASSPORT_NATIONALITY,
        confidence=0.60, evidence_text=iso3,
    )


def resolve_licence_country(
    operator_choice: str | None,
    licence_evidence: list[CountryEvidence],
    passport_nationality_iso3: str | None,
) -> tuple[CountryEvidence | None, list[str]]:
    """Settle which country's acceptance rule applies, and say how it was known.

    Returns the winning evidence and the warnings the result should carry, so
    that a country the reader inferred is never presented as one the documents
    proved.
    """
    warnings: list[str] = []
    if operator_choice:
        policy = policy_for_country(operator_choice)
        if policy is not None:
            return CountryEvidence(
                country=policy.country, source=DetectionSource.OPERATOR,
                confidence=1.0, evidence_text=operator_choice,
            ), warnings

    ranked = sorted(licence_evidence, key=lambda item: item.confidence, reverse=True)
    distinct = {item.country for item in ranked}
    if len(distinct) > 1:
        # Text printed on the card and an authority barcode are document
        # evidence. The visual router is a fallback used when a page has no
        # readable issuer; it must not manufacture a bundle conflict against a
        # state the licence front explicitly names. A Danish front stating
        # DANMARK was previously cancelled by the model guessing ARE from its
        # titleless category-table reverse.
        explicit = [
            item for item in ranked
            if item.source in {
                DetectionSource.LICENCE_BARCODE,
                DetectionSource.LICENCE_TEXT,
            }
        ]
        explicit_countries = {item.country for item in explicit}
        if len(explicit_countries) == 1:
            explicit_country = next(iter(explicit_countries))
            contradicting = [
                item for item in ranked if item.country != explicit_country
            ]
            if contradicting and all(
                item.source == DetectionSource.VLM_VISUAL
                for item in contradicting
            ):
                warnings.append(
                    "LICENCE_COUNTRY_WEAK_VISUAL_INFERENCE_IGNORED:"
                    + ":".join(sorted({item.country for item in contradicting}))
                )
                ranked = [
                    item for item in ranked if item.country == explicit_country
                ]
                distinct = {explicit_country}
        if len(distinct) == 1:
            pass
        else:
            # Two licence pages naming two states is not something to average
            # over when both readings came from the documents themselves.
            warnings.append(
                "LICENCE_COUNTRY_CONFLICT:" + ":".join(sorted(distinct))
            )
            return None, warnings

    if ranked:
        winner = ranked[0]
        fallback = country_from_nationality(passport_nationality_iso3)
        if fallback and fallback.country != winner.country:
            # Legitimate and common -- a resident abroad drives on the licence of
            # the country they live in. The licence still decides, but the pair
            # is surfaced because NATIONAL_ONLY turns on exactly this gap.
            warnings.append(
                f"LICENCE_COUNTRY_DIFFERS_FROM_NATIONALITY:{winner.country}:{fallback.country}"
            )
        return winner, warnings

    fallback = country_from_nationality(passport_nationality_iso3)
    if fallback is not None:
        warnings.append("LICENCE_COUNTRY_INFERRED_FROM_PASSPORT")
        return fallback, warnings

    warnings.append("LICENCE_COUNTRY_UNDETERMINED")
    return None, warnings


# Every country the detector can name, for the UI's optional override list.
DETECTABLE_COUNTRIES = tuple(
    country for country in COUNTRY_NAMES if country in COUNTRY_TEXT_MARKERS
)
