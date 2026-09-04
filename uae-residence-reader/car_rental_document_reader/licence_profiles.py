from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

import pycountry


class LicenceRequirement(str, Enum):
    NEED_IDL = "NEED_IDL"
    ALL_EXCEPTION = "ALL_EXCEPTION"
    NATIONAL_ONLY = "NATIONAL_ONLY"


class LicenceTemplateFamily(str, Enum):
    INTERNATIONAL_BOOKLET_1949_1968 = "INTERNATIONAL_BOOKLET_1949_1968"
    EU_NUMBERED = "EU_NUMBERED"
    VIENNA_NUMBERED = "VIENNA_NUMBERED"
    AAMVA = "AAMVA"
    AUSTRALIA_NZ = "AUSTRALIA_NZ"
    GCC_BILINGUAL = "GCC_BILINGUAL"
    EAST_ASIA = "EAST_ASIA"
    INDIA = "INDIA"
    SOUTH_AFRICA = "SOUTH_AFRICA"
    TURKEY = "TURKEY"


@dataclass(frozen=True)
class CountryLicencePolicy:
    country: str
    iso3: str
    requirement: LicenceRequirement
    template_family: LicenceTemplateFamily
    ocr_languages: tuple[str, ...]


# Exact country order and spelling transcribed from the supplied 196-row policy
# image.  This source of truth belongs to the rental business; the external
# standards below describe document layout, not the acceptance decision.
COUNTRY_NAMES = (
    "Afghanistan", "Albania", "Algeria", "Andorra", "Angola", "Antigua and Barbuda",
    "Argentina", "Armenia", "Australia", "Austria", "Azerbaijan", "Bahamas", "Bahrain",
    "Bangladesh", "Barbados", "Belarus", "Belgium", "Belize", "Benin", "Bhutan", "Bolivia",
    "Bosnia and Herzegovina", "Botswana", "Brazil", "Brunei", "Bulgaria", "Burkina Faso",
    "Burundi", "Cabo Verde", "Cambodia", "Cameroon", "Canada", "Central African Republic",
    "Chad", "Chile", "China", "Colombia", "Comoros", "Congo (Congo-Brazzaville)",
    "Costa Rica", "Côte d'Ivoire", "Croatia", "Cuba", "Cyprus", "Czechia (Czech Republic)",
    "Democratic Republic of the Congo", "Denmark", "Djibouti", "Dominica", "Dominican Republic",
    "Ecuador", "Egypt", "El Salvador", "Equatorial Guinea", "Eritrea", "Estonia", "Eswatini",
    "Ethiopia", "Fiji", "Finland", "France", "Gabon", "Gambia", "Georgia", "Germany", "Ghana",
    "Greece", "Grenada", "Guatemala", "Guinea", "Guinea-Bissau", "Guyana", "Haiti", "Holy See",
    "Honduras", "Hong Kong", "Hungary", "Iceland", "India", "Indonesia", "Iran", "Iraq",
    "Ireland", "Israel", "Italy", "Jamaica", "Japan", "Jordan", "Kazakhstan", "Kenya", "Kiribati",
    "Kuwait", "Kyrgyzstan", "Laos", "Latvia", "Lebanon", "Lesotho", "Liberia", "Libya",
    "Liechtenstein", "Lithuania", "Luxembourg", "Madagascar", "Malawi", "Malaysia", "Maldives",
    "Mali", "Malta", "Marshall Islands", "Mauritania", "Mauritius", "Mexico", "Micronesia",
    "Moldova", "Monaco", "Mongolia", "Montenegro", "Morocco", "Mozambique", "Myanmar", "Namibia",
    "Nauru", "Nepal", "Netherlands", "New Zealand", "Nicaragua", "Niger", "Nigeria", "North Korea",
    "North Macedonia", "Norway", "Oman", "Pakistan", "Palau", "Palestine State", "Panama",
    "Papua New Guinea", "Paraguay", "Peru", "Philippines", "Poland", "Portugal", "Qatar", "Romania",
    "Russia", "Rwanda", "Saint Kitts and Nevis", "Saint Lucia", "Saint Vincent and the Grenadines",
    "Samoa", "San Marino", "Sao Tome and Principe", "Saudi Arabia", "Senegal", "Serbia", "Seychelles",
    "Sierra Leone", "Singapore", "Slovakia", "Slovenia", "Solomon Islands", "Somalia", "South Africa",
    "South Korea", "South Sudan", "Spain", "Sri Lanka", "Sudan", "Suriname", "Sweden", "Switzerland",
    "Syria", "Tajikistan", "Tanzania", "Thailand", "Timor-Leste", "Togo", "Tonga",
    "Trinidad and Tobago", "Tunisia", "Turkey", "Turkmenistan", "Tuvalu", "Uganda", "Ukraine",
    "United Arab Emirates", "United Kingdom", "United States", "Uruguay", "Uzbekistan", "Vanuatu",
    "Venezuela", "Vietnam", "Yemen", "Zambia", "Zimbabwe",
)


# India is deliberately absent. The business's own acceptance list records it as
# "Need Idl", and it was the single entry where this set disagreed with that
# list: 33 countries here against 32 marked as accepted there. Listing it meant
# an Indian licence was accepted on its own, without the international permit the
# policy requires -- and because the Indian card reads cleanly, nothing else in
# the flow would have stopped it. One line to restore if the list is the stale
# one rather than this set.
ALL_EXCEPTION_COUNTRIES = frozenset({
    "Australia", "Austria", "Bahrain", "Belgium", "Canada", "Denmark", "Finland", "France",
    "Germany", "Greece", "Hong Kong", "Ireland", "Italy", "Japan", "Kuwait",
    "Netherlands", "New Zealand", "Norway", "Oman", "Poland", "Qatar", "Romania", "Saudi Arabia",
    "South Africa", "South Korea", "Spain", "Sweden", "Switzerland", "Turkey",
    "United Arab Emirates", "United Kingdom", "United States",
})


NATIONAL_ONLY_COUNTRIES = frozenset({
    "Albania", "Azerbaijan", "Bulgaria", "China", "Cyprus", "Estonia", "Hungary", "Iceland",
    "Israel", "Latvia", "Lithuania", "Luxembourg", "Malta", "Montenegro", "Portugal", "Serbia",
    "Singapore", "Slovakia", "Slovenia", "Ukraine",
})


EU_NUMBERED_COUNTRIES = frozenset({
    "Austria", "Belgium", "Bulgaria", "Cyprus", "Denmark", "Estonia", "Finland", "France",
    "Germany", "Greece", "Hungary", "Iceland", "Ireland", "Italy", "Latvia", "Lithuania",
    "Luxembourg", "Malta", "Netherlands", "Norway", "Poland", "Portugal", "Romania", "Slovakia",
    "Slovenia", "Spain", "Sweden", "Switzerland", "United Kingdom",
})


VIENNA_NUMBERED_COUNTRIES = frozenset({
    "Albania", "Azerbaijan", "Montenegro", "Serbia", "Ukraine",
    # Verified against PRADO specimens: the current Israeli, Turkish and Russian
    # licences all print the Annex 6 designators, so the numbered parser reads
    # them without a word of Hebrew, Turkish or Russian. Russia writes 4a) 4b)
    # 4c) with a closing bracket, which the designator pattern already accepts.
    "Israel", "Turkey", "Russia",
})
# Countries whose licence prints the holder's national number in designator 4d
# -- the same number their passport prints as the personal number, so the two
# documents can be checked against each other.
#
# The EU model defines 4d only as "a number for administrative purposes other
# than the licence number", and several states fill it with a card or holder
# reference that has nothing to do with the passport. Two different numbers on
# such a pair are normal, not a sign of two different people, so a country
# earns a place here only on evidence that the two are the same identifier.
# Albania is listed from a bundle in which licence 4d and the passport's
# personal number were both read as K00721078T; add others the same way rather
# than by assuming the model is used consistently.
NATIONAL_ID_IN_LICENCE_4D = frozenset({"Albania"})


GCC_COUNTRIES = frozenset({"Bahrain", "Kuwait", "Oman", "Qatar", "Saudi Arabia", "United Arab Emirates"})
AAMVA_COUNTRIES = frozenset({"Canada", "United States"})
AUSTRALIA_NZ_COUNTRIES = frozenset({"Australia", "New Zealand"})
# Israel is deliberately absent: its current licence prints the same numbered
# field designators as the EU model (1, 2, 3, 4, 4a, 4b, 4d ID, 5, 8, 9), as the
# PRADO specimen ISR-FO-01001 shows. It was grouped here on the assumption that
# a Hebrew card must need bespoke handling; it does not, and Hebrew is not an
# East Asian script either.
EAST_ASIA_COUNTRIES = frozenset({"China", "Hong Kong", "Japan", "South Korea", "Singapore"})


ISO3_OVERRIDES = {
    "Bolivia": "BOL", "Brunei": "BRN", "Cabo Verde": "CPV", "Congo (Congo-Brazzaville)": "COG",
    "Côte d'Ivoire": "CIV", "Czechia (Czech Republic)": "CZE", "Democratic Republic of the Congo": "COD",
    "Eswatini": "SWZ", "Holy See": "VAT", "Hong Kong": "HKG", "Iran": "IRN", "Laos": "LAO",
    "Micronesia": "FSM", "Moldova": "MDA", "Myanmar": "MMR", "North Korea": "PRK",
    "Palestine State": "PSE", "Russia": "RUS", "South Korea": "KOR", "Syria": "SYR",
    "Tanzania": "TZA", "Turkey": "TUR", "Venezuela": "VEN", "Vietnam": "VNM",
}


def _iso3(country: str) -> str:
    if country in ISO3_OVERRIDES:
        return ISO3_OVERRIDES[country]
    match = pycountry.countries.lookup(country)
    return match.alpha_3


def _family(country: str, requirement: LicenceRequirement) -> LicenceTemplateFamily:
    if requirement == LicenceRequirement.NEED_IDL:
        return LicenceTemplateFamily.INTERNATIONAL_BOOKLET_1949_1968
    if country in EU_NUMBERED_COUNTRIES:
        return LicenceTemplateFamily.EU_NUMBERED
    if country in VIENNA_NUMBERED_COUNTRIES:
        return LicenceTemplateFamily.VIENNA_NUMBERED
    if country in GCC_COUNTRIES:
        return LicenceTemplateFamily.GCC_BILINGUAL
    if country in AAMVA_COUNTRIES:
        return LicenceTemplateFamily.AAMVA
    if country in AUSTRALIA_NZ_COUNTRIES:
        return LicenceTemplateFamily.AUSTRALIA_NZ
    if country in EAST_ASIA_COUNTRIES:
        return LicenceTemplateFamily.EAST_ASIA
    if country == "India":
        return LicenceTemplateFamily.INDIA
    if country == "South Africa":
        return LicenceTemplateFamily.SOUTH_AFRICA
    if country == "Turkey":
        return LicenceTemplateFamily.TURKEY
    return LicenceTemplateFamily.VIENNA_NUMBERED


COUNTRY_OCR_LANGUAGES: dict[str, tuple[str, ...]] = {
    "China": ("en", "ch"), "Hong Kong": ("en", "ch"), "Japan": ("en", "japan"),
    "South Korea": ("en", "korean"), "Israel": ("en", "he"), "Turkey": ("en", "tr"),
    "Ukraine": ("en", "ru"), "Azerbaijan": ("en", "ru"),
    **{country: ("en", "ar") for country in GCC_COUNTRIES},
}


def _build_policies() -> dict[str, CountryLicencePolicy]:
    policies: dict[str, CountryLicencePolicy] = {}
    for country in COUNTRY_NAMES:
        if country in ALL_EXCEPTION_COUNTRIES:
            requirement = LicenceRequirement.ALL_EXCEPTION
        elif country in NATIONAL_ONLY_COUNTRIES:
            requirement = LicenceRequirement.NATIONAL_ONLY
        else:
            requirement = LicenceRequirement.NEED_IDL
        policies[country] = CountryLicencePolicy(
            country=country,
            iso3=_iso3(country),
            requirement=requirement,
            template_family=_family(country, requirement),
            ocr_languages=COUNTRY_OCR_LANGUAGES.get(country, ("en", "ru")),
        )
    return policies


COUNTRY_POLICIES = _build_policies()


COUNTRY_ALIASES = {
    "UAE": "United Arab Emirates", "UNITED STATES OF AMERICA": "United States", "USA": "United States",
    "UK": "United Kingdom", "RUSSIAN FEDERATION": "Russia", "REPUBLIC OF KOREA": "South Korea",
    "KOREA": "South Korea", "PRC": "China", "PEOPLE'S REPUBLIC OF CHINA": "China",
    "CZECH REPUBLIC": "Czechia (Czech Republic)", "VATICAN CITY": "Holy See",
}


def policy_for_country(value: str | None) -> CountryLicencePolicy | None:
    if not value:
        return None
    cleaned = " ".join(value.strip().upper().split())
    canonical = COUNTRY_ALIASES.get(cleaned)
    if canonical:
        return COUNTRY_POLICIES[canonical]
    for country, policy in COUNTRY_POLICIES.items():
        if country.upper() == cleaned or policy.iso3 == cleaned:
            return policy
    return None


# Auditable primary/authoritative layout sources used to define the families.
LAYOUT_SOURCES: dict[LicenceTemplateFamily, tuple[str, ...]] = {
    LicenceTemplateFamily.INTERNATIONAL_BOOKLET_1949_1968: (
        "https://unece.org/DAM/trans/conventn/Conv_road_traffic_EN.pdf",
        "https://unece.org/DAM/trans/conventn/Convention_on_Road_Traffic_of_1949.pdf",
    ),
    LicenceTemplateFamily.EU_NUMBERED: (
        "https://eur-lex.europa.eu/eli/dir/2025/2205",
        "https://www.consilium.europa.eu/prado/en/search-by-document-country.html",
    ),
    LicenceTemplateFamily.VIENNA_NUMBERED: (
        "https://unece.org/DAM/trans/conventn/Conv_road_traffic_EN.pdf",
        "https://www.consilium.europa.eu/prado/en/search-by-document-country.html",
    ),
    LicenceTemplateFamily.AAMVA: (
        "https://www.aamva.org/assets/best-practices%2C-guides%2C-standards%2C-manuals%2C-whitepapers/aamva-dl-id-card-design-standard-%282020%29",
    ),
    LicenceTemplateFamily.AUSTRALIA_NZ: (
        "https://www.service.nsw.gov.au/verifying-your-identity/document-examples",
    ),
    LicenceTemplateFamily.GCC_BILINGUAL: ("https://www.rta.ae/",),
    LicenceTemplateFamily.EAST_ASIA: (
        "https://www.td.gov.hk/en/about_us/history_of_transport_department/licensing_services/development_and_changes_of_driving_licences_/hong_kong_driving_licence/",
    ),
    LicenceTemplateFamily.INDIA: (
        "https://parivahan.gov.in/parivahan/sites/default/files/DownloadForm/form7.pdf",
    ),
    LicenceTemplateFamily.SOUTH_AFRICA: ("https://www.gov.za/",),
    LicenceTemplateFamily.TURKEY: ("https://www.nvi.gov.tr/",),
}


COMMON_NATIONAL_LABELS: dict[str, tuple[str, ...]] = {
    # A licence prints the holder's nationality as often as a passport does,
    # and in the same words, but only the English and Russian captions were
    # listed for a licence: the Mexico City card's "Nacionalidad / MEXICANA"
    # named nothing and the field was reported as having no evidence at all.
    "personal_info.nationality_name": (
        "NACIONALIDAD", "NACIONALIDADE", "NATIONALITÉ", "NATIONALITE",
        "CITTADINANZA", "STAATSANGEHÖRIGKEIT", "NATIONALITEIT",
        "OBYWATELSTWO", "DRŽAVLJANSTVO", "UYRUĞU", "الجنسية",
    ),
    "personal_info.full_name": (
        # "NOM" is deliberately absent: in French it labels the surname, never
        # the whole name. A Moroccan licence prints "Nom" above "Prénom", and
        # treating "Nom" as a full name split the surname EL KHACHTOUF into a
        # given name EL and a surname KHACHTOUF.
        "NAME", "FULL NAME", "HOLDER", "NOMBRE", "NOME", "NAAM", "NAME DES INHABERS",
        # From PRADO specimens: Vietnam "Họ và tên", Thailand ชื่อ, and the
        # Philippine card's single combined header over one printed row.
        "HỌ VÀ TÊN", "HO VA TEN", "ชื่อ", "LAST NAME, FIRST NAME, MIDDLE NAME",
        # Lebanon labels the row "الاسم والشهرة" (name and surname). Matching
        # only "الاسم" left "والشهرة" inside the value, and the word for "and
        # surname" was stored as the holder's first name. The longer label is
        # tried first, so listing it here is what fixes that.
        "الاسم والشهرة", "الاسم و الشهرة",
        # Syria "الاسم والنسبة"; Sudan writes the bare label with a hamza,
        # "الإسم", which is a different string from the plain "الاسم".
        "الاسم والنسبة", "الإسم",
        "الاسم", "姓名", "氏名", "성명", "שם", "AD SOYAD",
    ),
    "personal_info.first_name": (
        "FIRST NAME", "GIVEN NAME", "GIVEN NAMES", "PRÉNOM", "VORNAME", "NOME", "ADI",
        "NOMBRES",
    ),
    "personal_info.last_name": (
        "SURNAME", "FAMILY NAME", "LAST NAME", "NOM", "NACHNAME", "COGNOME", "SOYADI",
        # Latin America prints the surname as "Apellido(s)" and the given names
        # as "Nombre(s)" -- the reverse of the European "Nome" for a given name,
        # so the two must not be conflated.
        "APELLIDO", "APELLIDOS",
    ),
    "personal_info.date_of_birth": (
        "DATE OF BIRTH", "DOB", "BIRTH DATE", "NÉ(E) LE", "GEBURTSDATUM", "FECHA DE NACIMIENTO",
        "DATA DI NASCITA", "DATA NASCIMENTO", "DATA DE NASCIMENTO",
        # From PRADO specimens: China labels this row "Birthday" in English,
        # Thailand "Birth Date", Vietnam "Năm sinh" (which carries the year
        # only on the older paper model).
        "BIRTHDAY", "BIRTH DATE", "NĂM SINH", "NAM SINH", "เกิดวันที่",
        "FECHA DE NAC", "FECHA DE NACIMIENTO",
        # Nigeria abbreviates to "D of B"; Morocco combines date and place in
        # one row, so the place must not be read as part of the date.
        "D OF B", "DATE ET LIEU DE NAISSANCE", "DATE DE NAISSANCE",
        # Iran (Persian) and Iraq, whose card is Arabic and Kurdish only.
        "تاریخ تولد", "التولد",
        "تاريخ الميلاد", "出生日期", "生年月日", "생년월일", "תאריך לידה", "DOĞUM TARİHİ",
    ),
    "national_driving_licence.number": (
        # The word spelled out. Only the abbreviations were listed, and
        # "LICENCE NO" does not sit inside "LICENCE NUMBER", so a card
        # captioning its field in full matched nothing.
        "LICENCE NUMBER", "LICENSE NUMBER",
        "DRIVER LICENCE NUMBER", "DRIVER LICENSE NUMBER",
        "LICENCE NO", "LICENSE NO", "LICENCE #", "LICENSE #",
        "DRIVER NO", "DRIVER NUMBER", "DL NO", "DL NUMBER", "DLN",
        "PERMIT NO", "NUMÉRO DU PERMIS", "FÜHRERSCHEINNUMMER", "PERMISO N", "RIJBEWIJSNUMMER",
        "KÖRKORTSNUMMER", "KØREKORT NR",
        # Brazil keys the CNH by "Nº REGISTRO". The two other long numbers on
        # the card -- the security/mirror number under ASSINADO DIGITALMENTE and
        # the CPF -- are not the licence number and must not bind here.
        "Nº REGISTRO", "N REGISTRO", "NO REGISTRO", "REGISTRO",
        # From PRADO specimens: India "DL. No.", Thailand ฉบับที่, Vietnam "Số",
        # Hong Kong 檔號 (Ref.), Philippines "License No.".
        #
        # A bare "REF" was listed here beside 檔號 and is deliberately gone. On
        # an AAMVA card that word labels the document discriminator, which is
        # the one number on the licence that is explicitly not the licence
        # number: the Ontario card prints "5 DD/RÉF  IS0831624", the row
        # matched, and the reader reported the licence number as "6DD/R" -- a
        # slice of the label itself, at 0.90 confidence, in the field the
        # rental contract is keyed on. Hong Kong is still read from 檔號.
        "DL. NO", "DL NO.", "ฉบับที่", "SỐ", "檔號",
        # Latin American specimens: Argentina "Licencia Nº", Chile "Nº de
        # Licencia", Peru "No de Licencia", Colombia a bare "No.".
        # Mexico City abbreviates it on the card face: "Lic. No R8946342".
        # The full word was listed, the abbreviation was not, and the one
        # number the rental is keyed on was reported as absent.
        "LIC NO", "LIC. NO", "LIC N",
        "LICENCIA N", "LICENCIA NO", "N DE LICENCIA", "NO DE LICENCIA",
        "NUMERO DE LICENCIA", "NÚMERO DE LICENCIA",
        # Morocco "Permis N°"; Nigeria abbreviates to "L/NO"; Australia and
        # Hong Kong print "Licence No." in full.
        "PERMIS N", "PERMIS NO", "L/NO", "LNO", "LICENCE NO", "LICENSE NO",
        # A British Columbia licence heads the number with the abbreviation
        # alone -- "DL:9126623" -- and the fuller spellings above matched none
        # of it. The label boundary is a letter boundary, so this cannot fire
        # inside a longer word.
        "DL",
        # El Salvador abbreviates to "Nº LIC"; Botswana and Bangladesh spell
        # "Licence Number" out in full.
        "N LIC", "NO LIC", "LICENCE NUMBER", "LICENSE NUMBER",
        # Iran prints only Persian. "شماره ملی" beside it is the national
        # number and is deliberately not listed, so it cannot take this field.
        "شماره گواهینامه",
        # Lebanon labels the licence number simply "الرقم"; Syria prints
        # "رقم الإجازة" beside the English "Licence No".
        "رقم الرخصة", "الرقم", "رقم الإجازة", "رقم الاجازة",
        "رقم الرخصة", "رقم رخصة القيادة", "证号", "驾驶证号",
        "番号", "운전면허번호", "מספר רשיון", "SÜRÜCÜ BELGE NO",
    ),
    "national_driving_licence.issued_by_name": (
        "ISSUED BY", "ISSUING AUTHORITY", "AUTHORITY", "DÉLIVRÉ PAR", "AUSSTELLENDE BEHÖRDE",
        "LOCAL", "جهة الإصدار", "发证机关", "公安局", "교부기관", "VEREN MAKAM",
    ),
    "national_driving_licence.issue_date": (
        "ISSUE DATE", "DATE OF ISSUE", "ISS", "DÉLIVRÉ LE", "AUSSTELLUNGSDATUM",
        # Canada labels the row with the past participle alone: a British
        # Columbia licence prints "Issued: 2023-Mar-03" and "Expires:
        # 2024-Sep-02". "EXPIRES" was already listed; its pair was not, so the
        # card gave up an expiry and no issue date.
        "ISSUED",
        # "DATA EMISSÃO" is the date this card was issued. "1ª HABILITAÇÃO" is
        # the date the holder first qualified, years earlier, and is a different
        # fact that must not land in the issue date.
        "DATA EMISSÃO", "DATA DE EMISSÃO", "DATA EMISSAO",
        # From PRADO specimens: Thailand วันอนุญาต / "Issue Date", China's
        # 初次领证日期 is labelled "Issue Date" in English on the same row.
        "วันอนุญาต",
        # Latin American specimens name the issue date four different ways.
        "FECHA DE EXPEDICION", "FECHA DE EXPEDICIÓN", "OTORGAMIENTO",
        "FECHA ULTIMO CONTROL", "FECHA DE OTORGAMIENTO",
        # Australia (Queensland) prints "Effective" where others print "Date of
        # Issue"; Morocco prints "délivré à <place> Le <date>"; Nigeria "ISS".
        "EFFECTIVE", "DELIVRE LE", "DÉLIVRÉ LE", "VALIDE LE",
        # Quebec dates the licence from the day it takes effect: "4a Valide
        # le : 2023-07-30", against "4b Expire le" for its pair. Both are
        # listed here and in the expiry group below; neither existed, and the
        # only reason the issue date was read at all is that "VALIDE LE" had
        # been added for another card. New Brunswick and Ontario print the
        # same wording on their French half.
        # Morocco writes the issuance as a sentence rather than a caption:
        # "délivré à Tanger" on one row and "Le 17/04/2018" on the next,
        # with بتاريخ -- "dated" -- closing it on the Arabic side. Neither
        # half was listed. "Délivré le" was, but it never matches, because
        # the place stands between the two words. The place cannot be
        # mistaken for the value: only a date-shaped row can fill a date.
        "DÉLIVRÉ À", "DELIVRE A", "بتاريخ",
        "DATE DE DELIVRANCE", "DATE DE DÉLIVRANCE", "DATE D'ÉMISSION",
        "DATE D'EMISSION", "EMIS LE", "ÉMIS LE",
        "تاریخ صدور", "تاريخ المنح", "تاريخ أول إصدار", "تاريخ اول اصدار",
        # El Salvador prints a bare "EXPEDICION"; Bangladesh and Botswana date
        # the licence from its "First Issue".
        "EXPEDICION", "EXPEDICIÓN", "FIRST ISSUE", "RENEWAL DATE", "REFRENDA",
        "تاريخ الإصدار", "تاريخ التسليم", "初次领证日期", "交付", "발급일자",
        "VERİLİŞ TARİHİ",
    ),
    "national_driving_licence.expiry_date": (
        # "EXPIRY" alone is needed as well as "EXPIRY DATE": the boundary check
        # stops the shorter "EXP" from matching inside it, so the Queensland
        # card's bare "Expiry" row was invisible.
        "EXPIRY", "EXPIRY DATE", "EXPIRATION DATE", "EXPIRES", "EXP",
        "VALID UNTIL", "VALID TO", "VALID THRU",
        "DATE D'EXPIRATION", "ABLAUFDATUM", "VALIDADE",
        # The pair of "Valide le" on the Quebec card, printed against
        # designator 4b. Its absence is why a licence whose expiry the
        # recogniser read at 0.97 -- "Expire le : 2028-04-14", in plain sight
        # beside the issue date that *was* read -- was reported as having no
        # expiry date at all. "Valable jusqu'au" is the wording used on the
        # French half of several other bilingual cards.
        "EXPIRE LE", "EXPIRE LE :", "VALABLE JUSQU'AU", "VALIDE JUSQU'AU",
        # Read off specimens in the EU Council's PRADO register: India prints
        # "Valid Till", Thailand "Expire Date", Vietnam "Có giá trị đến",
        # Hong Kong 有效期至, Korea 기간.
        "VALID TILL", "EXPIRE DATE", "CÓ GIÁ TRỊ ĐẾN", "CO GIA TRI DEN",
        # The same Indian card is also printed "Validity (NT) 02-Dec-2042" --
        # the date the licence is valid until, for non-transport vehicles --
        # and against "Valid Till" alone that row named nothing, so a licence
        # stating plainly when it expires was read as having no expiry.
        # "VALIDITY" is a date wherever it captions one; where a card writes a
        # duration under it instead, the value fails to normalise as a date
        # and nothing is bound, exactly as for 有效期限 below.
        "VALIDITY", "VALIDITY DATE", "VALID UPTO", "VALID UP TO",
        "有效期至", "기간",
        # Latin America rarely writes "expiry". Argentina abbreviates
        # vencimiento to "VTO", Peru renews by "Fecha de Revalidación", and
        # Chile prints the next medical control date instead.
        # Syria "صالحة لغاية" beside "Expiry date"; Sudan "تسري حتى".
        "صالحة لغاية", "تسري حتى",
        # Morocco and much of Francophone Africa caption the card's own
        # expiry "Fin de validité" -- end of validity -- on the reverse,
        # beside its Arabic equivalent. Neither was listed, so a Moroccan
        # licence printing its expiry in plain figures under that caption
        # was left to an ordering guess against the entitlement dates in
        # the category table beside it.
        "FIN DE VALIDITE", "FIN DE VALIDITÉ", "نهاية الصلاحية",
        "VTO", "VENCIMIENTO", "FECHA DE VENCIMIENTO",
        "FECHA DE REVALIDACION", "FECHA DE REVALIDACIÓN",
        "FECHA DE CONTROL", "VALIDA HASTA", "VÁLIDA HASTA",
        # 有效期限 is deliberately NOT here. On the Chinese licence that row
        # reads "6年" -- a validity *duration*, not a date. Binding it produced
        # an expiry of "6". The Japanese 有効期限 below is a real date and stays.
        "تاريخ الانتهاء", "有効期限", "만료일자",
        "בתוקף עד", "SON GEÇERLİLİK TARİHİ",
    ),
    # Rows that state how long a licence lasts rather than when it ends. They
    # are recognised so that a date-hungry parser cannot mistake them for an
    # expiry; nothing is bound from them.
    "_validity_duration_labels": (
        # China prints "6年" against 有效期限 and Iran a number of years against
        # مدت اعتبار. Neither is a date, and neither may reach the expiry field.
        "有效期限", "VALID FOR", "مدت اعتبار",
    ),
}


def policy_payload(country: str | None) -> dict[str, Any] | None:
    policy = policy_for_country(country)
    if policy is None:
        return None
    return {
        "country": policy.country,
        "iso3": policy.iso3,
        "requirement": policy.requirement.value,
        "accepted_document": (
            "INTERNATIONAL_DRIVING_PERMIT"
            if policy.requirement == LicenceRequirement.NEED_IDL
            else "NATIONAL_DRIVING_LICENCE"
        ),
        "template_family": policy.template_family.value,
        "nationality_match_required": policy.requirement == LicenceRequirement.NATIONAL_ONLY,
        "layout_sources": list(LAYOUT_SOURCES[policy.template_family]),
    }
