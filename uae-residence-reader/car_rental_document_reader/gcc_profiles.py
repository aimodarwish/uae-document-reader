from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from .schemas import DocumentType


# The five GCC states served by the GCC National workflow. The United Arab
# Emirates is deliberately absent: a UAE holder is a UAE Resident and has its
# own Emirates ID / UAE licence route, so listing it here would create two
# competing paths for the same customer.
GCC_ISO3 = ("SAU", "QAT", "BHR", "KWT", "OMN")


# Every field the GCC workflow is allowed to read. Anything outside this set
# (place of birth, licence class, blood group, card serial, traffic file) is
# neither extracted nor exported.
GCC_CUSTOMER_FIELDS = frozenset({
    "personal_info.first_name",
    "personal_info.last_name",
    "personal_info.gender",
    "personal_info.date_of_birth",
    "personal_info.nationality_name",
    "gcc_driving_licence.number",
    "gcc_driving_licence.issued_by_name",
    "gcc_driving_licence.issue_date",
    "gcc_driving_licence.expiry_date",
    "gcc_identity.number",
    "gcc_identity.issued_by_name",
    "gcc_identity.issue_date",
    "gcc_identity.expiry_date",
})


# Read to derive first/last name and to cross-check the holder between the two
# uploads. Never exported on its own.
GCC_DERIVATION_FIELDS = frozenset({
    "personal_info.full_name",
    "personal_info.full_name_arabic",
    "personal_info.nationality_code",
})


GCC_EXTRACTION_FIELDS = GCC_CUSTOMER_FIELDS | GCC_DERIVATION_FIELDS


@dataclass(frozen=True)
class GCCDocumentProfile:
    """One GCC state's identity-card and driving-licence document model.

    ``*_front_fields`` / ``*_back_fields`` record which values are actually
    printed on which physical side of each card, transcribed from the sources
    in ``source_urls``. They are what stops the reader from spending a fallback
    OCR pass or a VLM generation hunting for a field the card never carries,
    and what stops a stray date on the wrong side from binding to a label.
    """

    country: str
    iso3: str
    identity_titles: tuple[str, ...]
    identity_number_labels: tuple[str, ...]
    identity_number_pattern: str
    licence_titles: tuple[str, ...]
    licence_number_labels: tuple[str, ...]
    identity_layout: str
    licence_layout: str
    source_urls: tuple[str, ...]
    identity_front_fields: frozenset[str]
    identity_back_fields: frozenset[str]
    licence_front_fields: frozenset[str]
    licence_back_fields: frozenset[str]
    # Cards whose reverse carries a TD1 machine-readable zone. Its checksummed
    # birth date, expiry, sex and nationality are the strongest evidence the
    # reader can obtain, and they cost nothing extra to read.
    identity_mrz_side: str | None = None
    # Only set where the MRZ document-number field is documented as carrying the
    # holder's national/personal number rather than a card serial. Everywhere
    # else the identity number must come from the visible labelled row.
    identity_mrz_id_source: str | None = None
    # Most GCC TD1 zones use ICAO's ``SURNAME<<GIVEN NAMES`` order.  A profile
    # can disable those unchecked name fields where the state's card uses a
    # different order; the checksum-protected birth/expiry/sex fields remain
    # available and the visible front supplies the holder name.
    identity_mrz_names: bool = True
    # Rows the card prints in Hijri only. They are genuinely on the document,
    # so they stay in the side maps, but no amount of re-reading will produce a
    # Gregorian value, so the reader must not spend a retry or a model
    # generation chasing one.
    hijri_only_fields: frozenset[str] = frozenset()


COMMON_IDENTITY_TITLES = (
    "IDENTITY CARD", "IDENTIFICATION CARD", "NATIONAL ID", "NATIONAL IDENTITY CARD",
    "ID CARD", "بطاقة الهوية", "البطاقة الشخصية", "البطاقة المدنية",
)

COMMON_LICENCE_TITLES = (
    "DRIVING LICENCE", "DRIVING LICENSE", "DRIVER LICENCE", "DRIVER LICENSE",
    "رخصة القيادة", "رخصة قيادة", "رخصة السياقة", "رخصة سياقة",
    "رخصة سوق", "رخصه سوق", "إجازة السوق",
)

COMMON_FIELD_LABELS: dict[str, tuple[str, ...]] = {
    "personal_info.full_name": (
        "FULL NAME", "NAME OF HOLDER", "NAME", "الاسم", "اسم حامل البطاقة",
    ),
    "personal_info.date_of_birth": (
        "DATE OF BIRTH", "BIRTH DATE", "DOB", "تاريخ الميلاد", "تاريخ الولادة",
    ),
    "personal_info.nationality_name": ("NATIONALITY", "الجنسية"),
    "personal_info.gender": ("SEX", "GENDER", "الجنس", "النوع"),
    "gcc_identity.issue_date": (
        "ISSUING DATE", "ISSUE DATE", "DATE OF ISSUE", "تاريخ الإصدار", "تاريخ الاصدار",
    ),
    "gcc_identity.expiry_date": (
        "EXPIRY DATE", "EXPIRATION DATE", "DATE OF EXPIRY", "EXPIRY",
        # "DOE" is the Latin-side abbreviation on the Saudi card, the mirror of
        # "DOB" beside it. Without it the expiry survived only through the
        # Arabic label alongside, so a weak Arabic read lost the field.
        "DOE", "VALID UNTIL", "VALID TILL",
        "VALID TO", "VALIDITY DATE", "VALIDITY",
        "تاريخ الانتهاء", "تاريخ انتهاء البطاقة", "تاريخ الصلاحية", "الصلاحية",
        "صالحة لغاية",
    ),
    "gcc_driving_licence.issued_by_name": (
        "ISSUED BY", "ISSUING AUTHORITY", "PLACE OF ISSUE", "AUTHORITY",
        "جهة الإصدار", "مكان الإصدار",
    ),
    "gcc_driving_licence.issue_date": (
        "ISSUING DATE", "ISSUE DATE", "DATE OF ISSUE", "FIRST ISSUE",
        "FIRST ISSUE DATE", "DATE OF FIRST ISSUE", "1ST ISSUE",
        "تاريخ الإصدار", "تاريخ الاصدار", "تاريخ أول إصدار", "تاريخ أول اصدار",
        "تاريخ الإصدار الأول", "تاريخ الاصدار الأول",
        # Qatar prints the compact caption without the word "date". On the
        # reported card this was the clearest Arabic row while the stylised
        # English caption was split, so omitting it forced an unnecessary
        # repair pass and made the identity layout outscore the licence.
        "أول إصدار", "اول إصدار", "الإصدار الأول", "الاصدار الأول",
    ),
    "gcc_driving_licence.expiry_date": (
        "EXPIRY DATE", "EXPIRATION DATE", "DATE OF EXPIRY", "EXPIRY",
        "DOE", "VALID UNTIL", "VALID TILL",
        "VALID TO", "VALID THRU", "VALIDITY DATE", "VALIDITY",
        # The Saudi card abbreviates the Latin-side row to "Exp".
        "EXP", "EXPIRES",
        "تاريخ الانتهاء", "تاريخ انتهاء الرخصة", "تاريخ الصلاحية", "الصلاحية",
        "صالحة لغاية",
    ),
}


# Every supported GCC layout identifies the holder through the national, civil
# or personal ID number. Card serials, traffic-file numbers and barcode
# payloads are never that identifier.
SHARED_GCC_IDENTIFIER_ISO3 = frozenset(GCC_ISO3)


_NAME_FIELDS = frozenset({
    "personal_info.full_name", "personal_info.full_name_arabic",
    "personal_info.first_name", "personal_info.last_name",
})


GCC_PROFILES: dict[str, GCCDocumentProfile] = {
    "Bahrain": GCCDocumentProfile(
        country="Bahrain", iso3="BHR",
        identity_titles=("IDENTITY CARD", "بطاقة الهوية"),
        identity_number_labels=(
            "NO", "PERSONAL NUMBER", "PERSONAL NO", "CPR NUMBER", "CPR NO", "CPR",
            "الرقم", "الرقم الشخصي", "رقم الهوية",
        ),
        # Central Population Registry number: exactly nine digits, and a card
        # issued to a holder born in 2000-2009 legitimately starts with a zero,
        # which is why the value is carried as a string end to end.
        identity_number_pattern=r"\d{9}",
        licence_titles=COMMON_LICENCE_TITLES,
        licence_number_labels=(
            "LICENCE NO", "LICENSE NO", "LICENCE NUMBER", "LICENSE NUMBER",
            "PERSONAL NUMBER", "PERSONAL NO", "CPR", "CPR NUMBER", "CPR NO",
            "رقم الرخصة", "الرقم الشخصي", "الرقم السكاني",
        ),
        identity_front_fields=frozenset({
            *_NAME_FIELDS, "personal_info.nationality_name",
            "personal_info.nationality_code",
            "gcc_identity.number", "gcc_identity.expiry_date",
        }),
        # The reverse carries the birth date, sex and a TD1 MRZ. Those two
        # holder fields are not printed on the front at all, so a GCC Bahrain
        # bundle without the ID back legitimately returns them as null.
        identity_back_fields=frozenset({
            "personal_info.date_of_birth", "personal_info.gender",
            "personal_info.nationality_code",
            "gcc_identity.expiry_date",
        }),
        licence_front_fields=frozenset({
            *_NAME_FIELDS, "personal_info.date_of_birth", "personal_info.gender",
            "personal_info.nationality_name", "personal_info.nationality_code",
            "gcc_driving_licence.number",
        }),
        # First issue date and expiry are printed on the reverse of the Bahrain
        # licence, not beside the holder details on the front.
        licence_back_fields=frozenset({
            "gcc_driving_licence.issue_date", "gcc_driving_licence.expiry_date",
        }),
        identity_mrz_side="BACK",
        identity_mrz_id_source="document_number",
        identity_layout=(
            "Bahrain identity card. FRONT: portrait, the nine-digit Personal Number/الرقم الشخصي "
            "(CPR, may begin with 0), the holder name in Latin and Arabic, nationality and the "
            "card expiry. The front carries no birth date and no sex field. BACK: date of birth, "
            "sex, card serial and a three-line TD1 machine-readable zone. Read birth date and sex "
            "only from the back or its MRZ, never infer them from the CPR digits, and never use the "
            "card serial as the Personal Number. The card prints no issue date, so ID Issue Date "
            "must stay null."
        ),
        licence_layout=(
            "Bahrain driving licence. FRONT: portrait, holder name in Latin and Arabic, date of "
            "birth, nationality, sex, address and the licence/document number, which is the "
            "holder's nine-digit CPR. BACK: First Issue Date and Date of Expiry. Bind issue and "
            "expiry only to their own labels on the reverse and never substitute the printed card "
            "serial for the licence number."
        ),
        source_urls=(
            "https://bh.bh/new/en/identity-idcard_en.html",
            "https://docs.uqudo.com/docs/kyc/uqudo-api/scan/country-specific-ids/bhr_id-bahrain-id",
            "https://docs.uqudo.com/docs/kyc/uqudo-api/scan/country-specific-ids/bhr_dl-bahrain-driving-licence",
            "https://www.consilium.europa.eu/prado/en/BHR-BO-01001/index.html",
        ),
    ),
    "Saudi Arabia": GCCDocumentProfile(
        country="Saudi Arabia", iso3="SAU",
        identity_titles=(
            "NATIONAL IDENTITY CARD", "NATIONAL ID", "بطاقة الهوية الوطنية", "الهوية الوطنية",
        ),
        identity_number_labels=(
            # The current card labels the row simply "No:" beside "الرقم"; older
            # cards spell out "ID Number / رقم الهوية". Both are accepted, and
            # the ten-digit 1-prefixed pattern below is what actually gates the
            # value, so the short label cannot bind a serial or a version.
            "NO", "NATIONAL ID NUMBER", "NATIONAL ID NO", "ID NUMBER", "IDENTITY NO",
            "الرقم", "رقم الهوية الوطنية", "رقم الهوية", "رقم السجل المدني",
        ),
        # A Saudi national's record number is ten digits beginning with 1;
        # a 2-prefixed number is an Iqama, which is not a GCC national.
        identity_number_pattern=r"1\d{9}",
        licence_titles=COMMON_LICENCE_TITLES,
        licence_number_labels=(
            # The current card labels the holder identifier simply "No." on the
            # Latin side and "الرقم" on the Arabic side. Older cards printed
            # "ID Number/رقم الهوية"; both are accepted.
            "NO", "ID NUMBER", "ID NO", "IDENTITY NUMBER", "IDENTITY NO",
            "الرقم", "رقم الهوية", "رقم الهوية الوطنية",
        ),
        identity_front_fields=frozenset({
            *_NAME_FIELDS, "personal_info.date_of_birth",
            "gcc_identity.number", "gcc_identity.expiry_date",
        }),
        identity_back_fields=frozenset({
            "personal_info.date_of_birth", "personal_info.gender",
            "personal_info.nationality_code", "gcc_identity.expiry_date",
        }),
        licence_front_fields=frozenset({
            *_NAME_FIELDS, "personal_info.date_of_birth",
            "personal_info.nationality_name", "personal_info.nationality_code",
            "gcc_driving_licence.number", "gcc_driving_licence.issue_date",
            "gcc_driving_licence.expiry_date",
        }),
        licence_back_fields=frozenset(),
        identity_mrz_side="BACK",
        # تاريخ الاصدار appears on the Arabic side only, as a Hijri date, with
        # no Gregorian counterpart anywhere on the card.
        hijri_only_fields=frozenset({"gcc_driving_licence.issue_date"}),
        identity_layout=(
            "Saudi national-ID front headed الهوية الوطنية: portrait at left; Arabic name above a Latin "
            "surname-first name (SURNAME, GIVEN NAMES); the 10-digit ID is only beside ID Number/رقم الهوية. "
            "Gregorian Date of Birth and Expiry Date are the left-hand DD/MM/YYYY rows; Arabic Hijri dates "
            "may also appear at right. The card has no printed issue date and no sex field. Never use "
            "رقم النسخة (version number), QR content, or barcode digits as the ID. Current citizen cards "
            "carry a TD1 MRZ on the reverse."
        ),
        licence_layout=(
            "Saudi driving licence headed رخصة سياقة / DRIVING LICENSE, Kingdom of Saudi Arabia, "
            "Ministry of Interior. Portrait at left; Arabic name above a Latin surname-first name "
            "(SURNAME, GIVEN NAMES). The Latin column carries No. (the holder identifier, the same "
            "ten-digit national ID), DOB and Exp as Gregorian DD/MM/YYYY. The Arabic column repeats "
            "them as الرقم, تاريخ الميلاد and تاريخ الانتهاء in Hijri, and adds تاريخ الاصدار, فصيلة الدم "
            "and النوع. Always take dates from the Latin Gregorian column. تاريخ الاصدار is printed in "
            "Hijri only and has no Gregorian counterpart, so the licence issue date must be null. "
            "Never use portrait, QR, barcode, card-serial or version-number data."
        ),
        source_urls=(
            "https://saudipedia.com/بطاقة-الهوية-الوطنية",
            "https://www.moi.gov.sa/wps/portal/Home/sectors/publicsecurity/traffic/",
            "https://docs.uqudo.com/docs/kyc/uqudo-api/scan/country-specific-ids/sau_id-saudi-arabia-id",
            "https://docs.uqudo.com/docs/kyc/uqudo-api/scan/country-specific-ids/sau_dl-saudi-arabia-driving-licence",
        ),
    ),
    "Qatar": GCCDocumentProfile(
        country="Qatar", iso3="QAT",
        # The card heads itself "ID. Card" over "بطاقة إثبات شخصية".
        identity_titles=(
            "ID CARD", "IDENTITY CARD", "QATAR ID",
            "بطاقة إثبات شخصية", "بطاقة اثبات شخصية",
            "بطاقة شخصية", "البطاقة الشخصية",
        ),
        identity_number_labels=(
            "NO", "ID NO", "ID NUMBER", "PERSONAL NUMBER", "PERSONAL NO", "QID",
            "الرقم", "الرقم الشخصي", "رقم البطاقة الشخصية",
        ),
        # Qatar ID: 11 digits, leading century digit 2 (1900s) or 3 (2000s).
        identity_number_pattern=r"[23]\d{10}",
        licence_titles=COMMON_LICENCE_TITLES,
        licence_number_labels=(
            "LICENCE NO", "LICENSE NO", "LICENCE NUMBER", "LICENSE NUMBER",
            "ID NO", "ID NUMBER", "QID", "PERSONAL NUMBER", "PERSONAL NO",
            "رقم الرخصة", "رقم البطاقة", "رقم البطاقة الشخصية", "الرقم الشخصي",
        ),
        identity_front_fields=frozenset({
            *_NAME_FIELDS, "personal_info.date_of_birth",
            "personal_info.nationality_name", "personal_info.nationality_code",
            "gcc_identity.number", "gcc_identity.expiry_date",
        }),
        # Qatar is the one supported layout that prints the ID issue date, and
        # it prints it on the reverse together with the card serial.
        identity_back_fields=frozenset({"gcc_identity.issue_date"}),
        licence_front_fields=frozenset({
            *_NAME_FIELDS, "personal_info.date_of_birth",
            "personal_info.nationality_name", "personal_info.nationality_code",
            "gcc_driving_licence.number", "gcc_driving_licence.issue_date",
            "gcc_driving_licence.expiry_date",
        }),
        licence_back_fields=frozenset(),
        identity_layout=(
            "Qatar identity card. FRONT: portrait, bilingual holder name, nationality, date of "
            "birth, the 11-digit ID No/الرقم الشخصي and the card expiry. BACK: Issuing Date, "
            "passport number, card serial and blood group. The card prints no sex field, so gender "
            "stays null unless an operator supplies it. A card or barcode serial is never the ID No."
        ),
        licence_layout=(
            "Qatar driving licence, State of Qatar / Traffic Department heading. FRONT carries every "
            "customer field: Name, Nationality, Date of Birth, the 11-digit ID No holder identifier, "
            "First Issue and Validity (treat Validity as the licence expiry). The reverse lists only "
            "permitted vehicle categories and notes. Never use a card or traffic serial as the holder "
            "identifier."
        ),
        source_urls=(
            "https://www.consilium.europa.eu/prado/en/QAT-BO-01001/index.html",
            "https://docs.uqudo.com/docs/kyc/uqudo-api/scan/country-specific-ids/qat_id-qatar-id",
            "https://docs.uqudo.com/docs/kyc/uqudo-api/scan/country-specific-ids/qat_dl-qatar-driving-licence",
            "https://portal.moi.gov.qa/",
        ),
    ),
    "Kuwait": GCCDocumentProfile(
        country="Kuwait", iso3="KWT",
        identity_titles=(
            "CIVIL IDENTITY CARD", "CIVIL ID", "البطاقة المدنية",
        ),
        identity_number_labels=(
            "NO", "CIVIL ID", "CIVIL NO", "CIVIL NUMBER", "ID NUMBER",
            "الرقم", "الرقم المدني", "رقم البطاقة المدنية",
        ),
        # PACI civil number: 12 digits, leading century digit 2 or 3.
        identity_number_pattern=r"[23]\d{11}",
        licence_titles=COMMON_LICENCE_TITLES,
        licence_number_labels=(
            "LICENCE NO", "LICENSE NO", "LICENCE NUMBER", "LICENSE NUMBER",
            "CIVIL ID", "CIVIL ID NO", "CIVIL ID NUMBER", "CIVIL NO", "CIVIL NUMBER",
            "رقم الرخصة", "رقم رخصة القيادة", "الرقم المدني",
        ),
        identity_front_fields=frozenset({
            *_NAME_FIELDS, "personal_info.date_of_birth", "personal_info.gender",
            "personal_info.nationality_name", "personal_info.nationality_code",
            "gcc_identity.number", "gcc_identity.expiry_date",
        }),
        identity_back_fields=frozenset({
            "personal_info.date_of_birth", "personal_info.gender",
            "personal_info.nationality_code", "gcc_identity.expiry_date",
        }),
        licence_front_fields=frozenset({
            *_NAME_FIELDS, "personal_info.date_of_birth",
            "personal_info.nationality_name", "personal_info.nationality_code",
            "gcc_driving_licence.number", "gcc_driving_licence.issue_date",
            "gcc_driving_licence.expiry_date",
        }),
        licence_back_fields=frozenset({
            "gcc_driving_licence.issue_date", "gcc_driving_licence.expiry_date",
        }),
        identity_mrz_side="BACK",
        identity_layout=(
            "Kuwait civil identity card. FRONT: portrait, bilingual holder name, the 12-digit Civil "
            "ID/الرقم المدني, nationality, sex, date of birth and expiry. BACK: card serial, address, "
            "blood group, sponsor and a TD1 machine-readable zone that repeats birth date, sex, "
            "nationality and expiry. The card prints no issue date, so ID Issue Date must stay null, "
            "and the back's serial number is never the Civil ID."
        ),
        licence_layout=(
            "Kuwait driving licence, bilingual General Traffic Department card. The holder is keyed "
            "by the 12-digit Civil ID printed on the licence itself, alongside the name, nationality, "
            "date of birth, issue date and expiry. Ignore reference, traffic-file and physical-card "
            "serial numbers."
        ),
        source_urls=(
            "https://www.e.gov.kw/sites/kgoenglish/Pages/ApplicationPages/Application-PACI.aspx",
            "https://www.e.gov.kw/sites/kgoenglish/Pages/Services/MOI/UpdateLicense.aspx",
            "https://docs.uqudo.com/docs/kyc/uqudo-api/scan/country-specific-ids/kwt_id-kuwait-id",
        ),
    ),
    "Oman": GCCDocumentProfile(
        country="Oman", iso3="OMN",
        identity_titles=(
            "IDENTITY CARD", "CIVIL STATUS CARD",
            "البطاقة الشخصية", "بطاقة الأحوال المدنية",
        ),
        identity_number_labels=(
            "NO", "CIVIL NUMBER", "CIVIL NO", "ID NUMBER", "ID NO",
            "الرقم", "الرقم المدني", "الرقم الشخصي",
        ),
        # Royal Oman Police civil number: eight digits.
        identity_number_pattern=r"\d{8}",
        licence_titles=COMMON_LICENCE_TITLES,
        licence_number_labels=(
            "LICENCE NO", "LICENSE NO", "LICENCE NUMBER", "LICENSE NUMBER",
            "CIVIL NO", "CIVIL NUMBER", "CIVIL ID", "CIVIL ID NO", "ID NO", "ID NUMBER",
            "رقم الرخصة", "رقم رخصة القيادة", "الرقم المدني", "الرقم الشخصي",
        ),
        identity_front_fields=frozenset({
            *_NAME_FIELDS, "personal_info.date_of_birth",
            "gcc_identity.number", "gcc_identity.expiry_date",
        }),
        identity_back_fields=frozenset({
            "personal_info.date_of_birth", "personal_info.gender",
            "personal_info.nationality_name", "personal_info.nationality_code",
            "gcc_identity.expiry_date",
        }),
        licence_front_fields=frozenset({
            *_NAME_FIELDS, "gcc_driving_licence.number",
            "gcc_driving_licence.issue_date", "gcc_driving_licence.expiry_date",
            "gcc_driving_licence.issued_by_name",
        }),
        # The Omani licence prints the holder's birth date and nationality on
        # the reverse, the mirror image of the Bahraini card.
        licence_back_fields=frozenset({
            "personal_info.date_of_birth", "personal_info.nationality_name",
            "personal_info.nationality_code",
        }),
        identity_mrz_side="BACK",
        # The Omani card's third TD1 row places the given-name sequence before
        # ``<<`` and a sometimes-truncated family name after it.  Feeding that
        # row through the standard ICAO surname-first parser reverses the name
        # and can cancel a complete, explicitly labelled front-side reading.
        identity_mrz_names=False,
        identity_layout=(
            "Oman civil status / identity card. FRONT: portrait, bilingual holder name, date of "
            "birth, the eight-digit Civil Number/الرقم المدني and the card expiry. BACK: address, "
            "nationality, card serial and a TD1 machine-readable zone carrying birth date, sex, "
            "nationality and expiry. The card prints no issue date, so ID Issue Date must stay null."
        ),
        licence_layout=(
            "Royal Oman Police driving licence. FRONT: holder name in Latin and Arabic, First Issue "
            "Date, Expiry Date, the licence/document number and the place of issue. BACK: date of "
            "birth, address, nationality, blood group and the permitted licence classes. Barcode and "
            "physical-card serial data are not customer identifiers."
        ),
        source_urls=(
            "https://www.rop.gov.om/english/omani_id.aspx",
            "https://omanportal.gov.om/wps/wcm/connect/en/site/home/gov/gov1/gov5governmentorganizations/rop/residencecard",
            "https://docs.uqudo.com/docs/kyc/uqudo-api/scan/country-specific-ids/omn_id-oman-id",
            "https://docs.uqudo.com/docs/kyc/uqudo-api/scan/country-specific-ids/omn_dl-oman-driving-licence",
            "https://www.consilium.europa.eu/prado/en/OMN-BO-01001/index.html",
        ),
    ),
}


# The UI allow-list, the upload validator and the document router all read this
# tuple, so a country is either fully supported or not offered at all.
GCC_COUNTRY_NAMES = tuple(GCC_PROFILES)


def printed_fields(
    profile: GCCDocumentProfile | None, doc_type: DocumentType,
) -> frozenset[str]:
    """Return the customer fields printed on this exact card side."""
    if profile is None:
        return frozenset()
    return {
        DocumentType.GCC_IDENTITY_FRONT: profile.identity_front_fields,
        DocumentType.GCC_IDENTITY_BACK: profile.identity_back_fields,
        DocumentType.GCC_DRIVING_LICENCE_FRONT: profile.licence_front_fields,
        DocumentType.GCC_DRIVING_LICENCE_BACK: profile.licence_back_fields,
    }.get(doc_type, frozenset())


def identity_issue_date_printed(profile: GCCDocumentProfile | None) -> bool:
    """True only where an identity card actually prints an issue date.

    Of the five supported states only Qatar does. Everywhere else an OCR value
    landing in that field came from a birth date, an expiry, a Hijri row or a
    card version, so the field is left null instead of guessed.
    """
    if profile is None:
        return True
    return "gcc_identity.issue_date" in (
        profile.identity_front_fields | profile.identity_back_fields
    )


# The naming of a state and its authorities, and nothing else. A card carrying
# one cannot belong to a different state, which is what lets the reader notice
# that the operator picked the wrong customer type before it writes another
# country's number into a UAE field.
#
# Document titles are deliberately absent: every one of these states prints its
# own name on both its identity card and its driving licence, so a state name
# says which country issued a page and never which of the two it is. Confusing
# the two made a Qatari licence read as an identity card whenever a glared
# capture lost its banner.
ISSUER_MARKERS: dict[str, tuple[str, ...]] = {
    "Saudi Arabia": ("KINGDOM OF SAUDI ARABIA", "المملكة العربية السعودية"),
    "Qatar": ("STATE OF QATAR", "دولة قطر"),
    "Kuwait": ("STATE OF KUWAIT", "دولة الكويت"),
    "Bahrain": ("KINGDOM OF BAHRAIN", "مملكة البحرين"),
    "Oman": ("SULTANATE OF OMAN", "سلطنة عمان", "شرطة عمان السلطانية"),
    "United Arab Emirates": (
        "UNITED ARAB EMIRATES", "الإمارات العربية المتحدة",
        "FEDERAL AUTHORITY FOR IDENTITY",
    ),
}


def issuing_states(text: str) -> set[str]:
    """Return every state whose own document wording appears in this page."""
    upper = text.upper()
    return {
        country for country, markers in ISSUER_MARKERS.items()
        if any(marker.upper() in upper for marker in markers)
    }


def profile_for_gcc_country(value: str | None) -> GCCDocumentProfile | None:
    if not value:
        return None
    cleaned = " ".join(value.strip().upper().split())
    aliases = {
        "SAUDI": "Saudi Arabia", "KSA": "Saudi Arabia", "SAU": "Saudi Arabia",
        "KINGDOM OF SAUDI ARABIA": "Saudi Arabia",
        "KUWAIT": "Kuwait", "KWT": "Kuwait", "STATE OF KUWAIT": "Kuwait",
        "BAHRAIN": "Bahrain", "BHR": "Bahrain", "KINGDOM OF BAHRAIN": "Bahrain",
        "QATAR": "Qatar", "QAT": "Qatar", "STATE OF QATAR": "Qatar",
        "OMAN": "Oman", "OMN": "Oman", "SULTANATE OF OMAN": "Oman",
    }
    country = aliases.get(cleaned)
    if country:
        return GCC_PROFILES[country]
    return next((profile for name, profile in GCC_PROFILES.items() if name.upper() == cleaned), None)


def gcc_labels_for_path(path: str, country: str | None) -> tuple[str, ...]:
    profile = profile_for_gcc_country(country)
    if path == "gcc_identity.number":
        return profile.identity_number_labels if profile else ()
    if path == "gcc_driving_licence.number":
        return profile.licence_number_labels if profile else ()
    return COMMON_FIELD_LABELS.get(path, ())


def ascii_digits(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value)
    output: list[str] = []
    for character in normalized:
        if character.isdigit():
            try:
                output.append(str(unicodedata.digit(character)))
            except (TypeError, ValueError):
                output.append(character)
    return "".join(output)


def ascii_numerals(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value)
    output: list[str] = []
    for character in normalized:
        if character.isdigit():
            try:
                output.append(str(unicodedata.digit(character)))
            except (TypeError, ValueError):
                output.append(character)
        else:
            output.append(character)
    return "".join(output)


def normalize_gcc_number(value: str, country: str | None, identity: bool) -> str | None:
    """Return the holder identifier as a digit string, or None.

    The value stays a string for its whole life: a Bahraini CPR issued to a
    holder born in 2003 begins with a zero, and parsing it as a number would
    silently destroy it.
    """
    digits = ascii_digits(value)
    profile = profile_for_gcc_country(country)
    if profile is None:
        return None
    # Both the identity card and the licence identify the holder by the same
    # national/civil/personal number, so a value offered for either document
    # must satisfy that country's format. This is what keeps a traffic-file
    # number, reference code or card serial out of the field.
    return digits if re.fullmatch(profile.identity_number_pattern, digits) else None


def gcc_profile_payload(country: str | None) -> dict[str, object] | None:
    profile = profile_for_gcc_country(country)
    if profile is None:
        return None
    return {
        "country": profile.country,
        "iso3": profile.iso3,
        "identity_layout": profile.identity_layout,
        "licence_layout": profile.licence_layout,
        "source_urls": list(profile.source_urls),
    }
