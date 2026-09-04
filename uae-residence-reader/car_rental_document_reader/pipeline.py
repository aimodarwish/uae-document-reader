from __future__ import annotations

import hashlib
import json
import re
import time
import unicodedata
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field, replace
from datetime import date
from functools import wraps
from itertools import combinations, permutations
from pathlib import Path
from typing import Any

from PIL import Image

from .barcode import decode_barcodes
from .classify import (
    classify_document, has_uae_driving_licence_title,
    is_overseas_citizen_of_india_certificate,
)
from .config import (
    AppConfig, RuntimeInfo, clear_accelerators, detect_runtime,
    installed_versions, select_vlm,
)
from .extract import (
    VALIDITY_CAPTIONS,
    FIELD_LABELS, _numbered_national_licence_candidates, barcode_candidates,
    american_unlabelled_licence_number_candidates,
    augment_gcc_ocr_lines,
    california_licence_layout_candidates,
    is_california_driver_licence,
    canadian_licence_candidates, compact_label, idp_layout_candidates,
    emirates_id_partial_mrz_gender_candidates,
    labelled_ocr_candidates, licence_category_table_dates, mrz_candidates,
    national_licence_date_sequence, passport_issue_date_from_mrz,
    prints_american_licence_layout,
    private_international_driver_licence_candidates,
)
from .files import DocumentInputError, load_document
from .gcc_profiles import (
    GCC_COUNTRY_NAMES, GCC_EXTRACTION_FIELDS, GCCDocumentProfile,
    ascii_numerals, gcc_profile_payload, identity_issue_date_printed, issuing_states,
    printed_fields, profile_for_gcc_country,
)
from .image_processing import (
    PreprocessedImage, analyze_and_preprocess, dot_matrix_number,
    ensure_ocr_variants, red_ink_boxes, split_card_sides, zoom_repair,
)
from .licence_profiles import (
    COMMON_NATIONAL_LABELS, NATIONAL_ID_IN_LICENCE_4D, CountryLicencePolicy,
    LicenceRequirement,
    policy_for_country, policy_payload,
)
from .mrz import (
    CONFUSIONS, ParsedMRZ, normalize_mrz_line, parse_mrz,
    passport_name_row_present, td1_filler_repairs, validate_check,
)
from .normalize import (
    fold_for_match, name_similarity, nationality_country, normalize_country,
    validate_date_relationships,
)
from .ocr import (
    OCRLine, OCRResult, PaddleOCREngine, find_clipped_mrz_lines, find_mrz_lines,
    merge_ocr_lines, merge_ocr_results,
)
from .privacy import logger
from .gender_hint import gender_from_name
from .tourist_detect import (
    country_from_us_state,
    IDP_WEAK_MARKERS, CountryEvidence, DetectionSource, country_from_barcode,
    country_from_card_zone, country_from_eu_distinguishing_sign,
    country_from_idp_permit_label,
    country_from_passport_lines, country_from_text, idp_convention,
    idp_is_non_government_translation, idp_is_private_translation_document,
    looks_like_idp, resolve_licence_country,
)
from .reconcile import reconcile_all, reconcile_field
from .schemas import (
    CustomerType, DocumentRecord, DocumentType,
    ExtractionResult, FieldCandidate, FieldMetadata, FieldStatus, MRZChecks,
    GCCProfileDecision, LicencePolicyDecision, MRZData, ProcessingInfo,
    REQUIRED_SLOTS,
)
from .vlm import LocalVLM, vlm_candidates


FIELD_PATHS = [
    "personal_info.first_name", "personal_info.middle_name", "personal_info.last_name",
    "personal_info.full_name", "personal_info.full_name_arabic", "personal_info.gender",
    "personal_info.date_of_birth", "personal_info.nationality_code", "personal_info.nationality_name",
    "personal_info.place_of_birth",
    "emirates_id.number", "emirates_id.issue_date", "emirates_id.expiry_date",
    "passport.number", "passport.issued_by_code", "passport.issued_by_name",
    "passport.issue_date", "passport.expiry_date", "passport.holder_id",
    "uae_driving_licence.number", "uae_driving_licence.issued_by_code",
    "uae_driving_licence.issued_by_name", "uae_driving_licence.issue_date", "uae_driving_licence.expiry_date",
    "gcc_identity.number", "gcc_identity.issued_by_code", "gcc_identity.issued_by_name",
    "gcc_identity.issue_date", "gcc_identity.expiry_date",
    "gcc_driving_licence.number", "gcc_driving_licence.issued_by_code",
    "gcc_driving_licence.issued_by_name", "gcc_driving_licence.issue_date",
    "gcc_driving_licence.expiry_date",
    "international_driving_permit.number", "international_driving_permit.issued_by_code",
    "international_driving_permit.issued_by_name", "international_driving_permit.issue_date",
    "international_driving_permit.expiry_date",
    "national_driving_licence.number", "national_driving_licence.issued_by_code",
    "national_driving_licence.issued_by_name", "national_driving_licence.issue_date",
    "national_driving_licence.expiry_date",
    "national_driving_licence.holder_id",
]


SLOT_TYPES = {
    "document_bundle": DocumentType.UNKNOWN,
    "emirates_id_front": DocumentType.EMIRATES_ID_FRONT,
    "emirates_id_back": DocumentType.EMIRATES_ID_BACK,
    "uae_licence_front": DocumentType.UAE_DRIVING_LICENCE_FRONT,
    "uae_licence_back": DocumentType.UAE_DRIVING_LICENCE_BACK,
    "gcc_identity_front": DocumentType.GCC_IDENTITY_FRONT,
    "gcc_identity_back": DocumentType.GCC_IDENTITY_BACK,
    "gcc_licence_front": DocumentType.GCC_DRIVING_LICENCE_FRONT,
    "gcc_licence_back": DocumentType.GCC_DRIVING_LICENCE_BACK,
    "passport": DocumentType.PASSPORT_BIODATA,
    "idp_pages": DocumentType.INTERNATIONAL_DRIVING_PERMIT,
    "national_licence_front": DocumentType.NATIONAL_DRIVING_LICENCE_FRONT,
    "national_licence_back": DocumentType.NATIONAL_DRIVING_LICENCE_BACK,
}


CRITICAL_PATHS_BY_DOCUMENT = {
    DocumentType.EMIRATES_ID_FRONT: {
        "personal_info.full_name", "personal_info.date_of_birth", "emirates_id.number",
        "emirates_id.issue_date", "emirates_id.expiry_date",
    },
    DocumentType.EMIRATES_ID_BACK: {"emirates_id.issue_date", "emirates_id.expiry_date"},
    DocumentType.UAE_DRIVING_LICENCE_FRONT: {
        "uae_driving_licence.number", "uae_driving_licence.issue_date",
        "uae_driving_licence.expiry_date",
    },
    DocumentType.PASSPORT_BIODATA: {
        "personal_info.full_name", "personal_info.date_of_birth",
        "personal_info.nationality_name", "passport.number", "passport.expiry_date",
        # The one passport field the machine-readable zone can never supply.
        # Every other entry here is also encoded in the zone, so a page whose
        # zone decoded looked complete and no second pass ever ran -- which is
        # precisely backwards: the issue date is the only field that always
        # needs the page itself to be read, and it was the only one excluded
        # from the list that decides whether to look again.
        "passport.issue_date",
    },
    DocumentType.INTERNATIONAL_DRIVING_PERMIT: {
        "international_driving_permit.number", "international_driving_permit.issue_date",
        "international_driving_permit.expiry_date",
    },
    DocumentType.NATIONAL_DRIVING_LICENCE_FRONT: {
        "personal_info.full_name", "personal_info.date_of_birth",
        "national_driving_licence.number", "national_driving_licence.issue_date",
        "national_driving_licence.expiry_date",
    },
}


# Values worth a second, more expensive pass when the first read misses them.
# A retry here can cost a ten-second generative OCR pass, so a field earns its
# place only if it is hard to supply by hand and the run is not usable without
# it. Nationality and the two "issued by" fields are excluded because the
# operator already chose the issuing country. Gender is excluded too: the GCC
# workflow refuses to confirm without an operator-selected gender regardless of
# what was read, so spending ten seconds hunting a sex row that several of
# these cards never print buys nothing.
GCC_CRITICAL_FIELDS = frozenset({
    "personal_info.full_name", "personal_info.date_of_birth",
    "gcc_identity.number", "gcc_identity.issue_date", "gcc_identity.expiry_date",
    "gcc_driving_licence.number", "gcc_driving_licence.issue_date",
    "gcc_driving_licence.expiry_date",
})


GCC_DOCUMENT_TYPES = frozenset({
    DocumentType.GCC_IDENTITY_FRONT, DocumentType.GCC_IDENTITY_BACK,
    DocumentType.GCC_DRIVING_LICENCE_FRONT, DocumentType.GCC_DRIVING_LICENCE_BACK,
})


PROVEN_STATUSES = frozenset({
    FieldStatus.VERIFIED, FieldStatus.HIGH_CONFIDENCE, FieldStatus.MANUALLY_EDITED,
})


# Sources that are a reading of the document rather than a guess about it. A
# model's uncorroborated answer is worth having when nothing else is available
# and worth discarding the moment the bundle proves the value some other way.
_GUESSED_SOURCE_METHODS = frozenset({"", "vlm", "name_inference"})


# The letter/digit pairs an OCR engine trades for one another, folded to one
# side so two readings of the same printed number compare equal.
_CONFUSABLE_CHARACTERS = str.maketrans(CONFUSIONS)


# NFKD removes marks from most Latin letters (``é`` -> ``e``), but a small
# group used in European names is encoded as a distinct letter and does not
# decompose.  These are the spellings used by the Latin/MRZ representation of
# those letters.  Keeping the table here makes the output operation constant
# time and avoids another model or network call on the tourist path.
_TOURIST_LATIN_NAME_TRANSLITERATION = str.maketrans({
    "Æ": "AE", "æ": "ae", "Œ": "OE", "œ": "oe", "Ø": "O", "ø": "o",
    "Ł": "L", "ł": "l", "Đ": "D", "đ": "d", "Ð": "D", "ð": "d",
    "Þ": "TH", "þ": "th", "ß": "ss", "ẞ": "SS", "Ħ": "H", "ħ": "h",
    "Ŧ": "T", "ŧ": "t", "Ŋ": "N", "ŋ": "n", "ı": "i", "ĸ": "k",
    "Ə": "E", "ə": "e",
})
_TOURIST_NAME_COMMAS = str.maketrans({
    ",": " ", "،": " ", "，": " ", "﹐": " ", "、": " ",
})
_TOURIST_NAME_PATHS = frozenset({
    "personal_info.first_name", "personal_info.last_name",
})


def _normalize_tourist_name_value(value: str | None) -> str | None:
    """Return a CRM-safe tourist name without accents or comma separators.

    Tourist passports already provide the authoritative Latin spelling in the
    MRZ, including for Arabic, Cyrillic and Asian scripts.  This final, cheap
    normalization handles the printed Latin spelling too, so a visual read of
    ``Aurélien, Valentin, Enzo`` is exported exactly like its MRZ counterpart:
    ``Aurelien Valentin Enzo``.  Apostrophes and hyphens inside a real name are
    preserved; only comma variants are separators.
    """
    if not value:
        return None
    decomposed = unicodedata.normalize(
        "NFKD", value.translate(_TOURIST_LATIN_NAME_TRANSLITERATION),
    )
    without_marks = "".join(
        character for character in decomposed
        if not unicodedata.combining(character)
    )
    cleaned = " ".join(without_marks.translate(_TOURIST_NAME_COMMAS).split())
    return cleaned or None


def _latin_only_name_candidates(
    candidates: list[FieldCandidate],
) -> list[FieldCandidate]:
    """Drop a name written in a script the rental contract cannot carry.

    A passport prints the holder's name twice -- once in the issuing state's
    own script and once in Latin, and the machine-readable zone repeats the
    Latin spelling because that is the one every border system, airline and
    rental agreement is keyed on. An Algerian booklet's Arabic row sits nearer
    the "الإسم / Given names" caption than the Latin row underneath it, so the
    label bound to it and "عبد الحق" was presented as the given name of a
    customer whose contract, licence and permit all read ABDELHAK.

    A name with no Latin letter in it is therefore not a reading of this field,
    whatever its confidence. The Latin spelling on the same page, or in the
    zone, is what remains; where the capture lost both, the field stays empty
    for an operator rather than arriving in a script the contract cannot use.
    """
    kept: list[FieldCandidate] = []
    for candidate in candidates:
        value = candidate.normalized_value or candidate.value or ""
        if (
            candidate.field_path in _TOURIST_NAME_PATHS
            and any(character.isalpha() for character in value)
            and not any(
                "A" <= character.upper() <= "Z"
                for character in unicodedata.normalize("NFKD", value)
            )
        ):
            continue
        kept.append(candidate)
    return kept


def _normalize_tourist_name_candidates(
    candidates: list[FieldCandidate], customer: CustomerType,
) -> None:
    """Normalize only tourist first/surname candidates before reconciliation."""
    if customer != CustomerType.TOURIST:
        return
    for candidate in candidates:
        if candidate.field_path not in _TOURIST_NAME_PATHS:
            continue
        normalized = _normalize_tourist_name_value(candidate.normalized_value)
        if normalized == candidate.normalized_value:
            continue
        candidate.normalized_value = normalized
        if "TOURIST_NAME_LATIN_SPACING_NORMALIZED" not in candidate.warnings:
            candidate.warnings.append("TOURIST_NAME_LATIN_SPACING_NORMALIZED")


def _normalize_tourist_result_names(result: ExtractionResult) -> None:
    """Apply the tourist name contract after all review-value recoveries."""
    if result.customer_type != CustomerType.TOURIST:
        return
    for path in _TOURIST_NAME_PATHS:
        section, attribute = path.split(".", 1)
        current = getattr(getattr(result, section), attribute)
        normalized = _normalize_tourist_name_value(current)
        if normalized != current:
            _set_path(result, path, normalized)


def _is_partial_name(one: str, other: str) -> bool:
    """True when one reading is the other with whole names missing."""
    tokens = [
        {part for part in name.upper().replace(",", " ").split() if part}
        for name in (one, other)
    ]
    if not all(tokens) or tokens[0] == tokens[1]:
        return False
    return tokens[0] <= tokens[1] or tokens[1] <= tokens[0]


def _is_character_completion(shorter: str, longer: str) -> bool:
    """Whether ``longer`` restores at most two OCR-dropped name characters."""
    compact = [
        re.sub(r"[^A-Z0-9]", "", fold_for_match(value))
        for value in (shorter, longer)
    ]
    if not compact[0] or not 1 <= len(compact[1]) - len(compact[0]) <= 2:
        return False
    iterator = iter(compact[1])
    return all(any(character == existing for existing in iterator) for character in compact[0])


def _check_holder_id(result: ExtractionResult, licence_country: str | None) -> None:
    """Compare the national number the passport and the licence both carry.

    Row 4d of an EU or Vienna licence and the "personal no." row of a passport
    hold the same lifelong citizen number. When both are read, agreement is the
    strongest evidence in the bundle that the two documents describe one
    person -- stronger than the name, which is transliterated differently by
    every issuer -- and disagreement is worth an operator's attention.
    """
    licence = (result.national_driving_licence.holder_id or "").upper()
    passport = (result.passport.holder_id or "").upper()
    if not licence or not passport:
        return
    # Both readings come from OCR, so two identical numbers can differ by a
    # letter the recognizer confuses for a digit. Folding those pairs before
    # comparing keeps a scanning artefact from being reported as two different
    # people; numbers belonging to different holders do not differ only there.
    folded = [
        value.translate(_CONFUSABLE_CHARACTERS) for value in (licence, passport)
    ]
    matched = licence == passport or folded[0] == folded[1]
    result.cross_document_checks.holder_id_match = matched
    if matched:
        note = (
            "CROSS_DOCUMENT_HOLDER_ID_MATCH" if licence == passport
            else "CROSS_DOCUMENT_HOLDER_ID_MATCH_AFTER_OCR_CONFUSION"
        )
        for path in ("national_driving_licence.holder_id", "passport.holder_id"):
            metadata = result.field_metadata.get(path)
            if metadata is not None:
                metadata.validation_results.append(note)
        return
    if licence_country not in NATIONAL_ID_IN_LICENCE_4D:
        # Designator 4d is only defined as an administrative number, and on
        # most cards it is not the number the passport prints. Two different
        # values there are the norm, so the reading is recorded and nothing is
        # claimed from it.
        return
    result.cross_document_checks.conflicts.append({
        "field": "personal_info.holder_id",
        "documents": ["passport", "national_driving_licence"],
        "reason": "HOLDER_ID_MISMATCH", "similarity": None,
    })
    result.warnings.append("CROSS_DOCUMENT_HOLDER_ID_MISMATCH")


# Whole-year terms passports are issued for. A date pair that lands on one of
# them to the day did not do so by accident.
_PASSPORT_TERM_YEARS = (3, 5, 7, 10)


def _corroborate_issue_against_term(result: ExtractionResult) -> None:
    """Check a read issue date against the term implied by a proven expiry.

    A machine-readable zone carries its own check digits for the expiry date
    but no issue date at all, so the issue date is always a plain reading and
    always arrives unproven. It is not unverifiable, though: when it sits a
    whole passport term before an expiry the zone has proved, the two confirm
    each other. Left alone, the field was reported as needing review on a page
    where the arithmetic settles it.
    """
    issue = result.field_metadata.get("passport.issue_date")
    expiry = result.field_metadata.get("passport.expiry_date")
    if (
        issue is None or expiry is None
        or issue.status != FieldStatus.NEEDS_REVIEW
        or expiry.status not in PROVEN_STATUSES
        or not result.passport.issue_date or not result.passport.expiry_date
    ):
        return
    try:
        issued = date.fromisoformat(result.passport.issue_date)
        expires = date.fromisoformat(result.passport.expiry_date)
    except ValueError:
        return
    for years in _PASSPORT_TERM_YEARS:
        try:
            term = issued.replace(year=issued.year + years)
        except ValueError:                     # 29 February
            continue
        if 0 <= (term - expires).days <= 1:
            issue.status = FieldStatus.HIGH_CONFIDENCE
            issue.reason_for_review = None
            issue.validation_results.append(f"VALIDITY_TERM_CONSISTENT:{years}Y")
            return


def _unproven_by_documents(metadata: FieldMetadata | None) -> bool:
    if metadata is None:
        return True
    if metadata.status in PROVEN_STATUSES:
        return False
    return metadata.source_method in _GUESSED_SOURCE_METHODS


def _critical_paths(
    document_type: DocumentType,
    gcc_profile: GCCDocumentProfile | None = None,
) -> set[str]:
    """Fields this exact card side is documented to carry.

    For GCC documents the answer comes from the country profile rather than a
    single shared list, because the states disagree about which side prints
    what: a Bahraini card puts the birth date and sex on the reverse of the ID
    and the issue and expiry on the reverse of the licence, while an Omani
    licence prints issue and expiry on the front and the birth date on the
    back. Chasing a field the card never prints is what used to cost a wasted
    OCR pass and a wasted model generation per page.
    """
    if gcc_profile is not None and document_type in GCC_DOCUMENT_TYPES:
        # A row printed only in Hijri is on the card but can never yield the
        # Gregorian value the workflow stores, so it is not worth a retry.
        return set(
            printed_fields(gcc_profile, document_type)
            & GCC_CRITICAL_FIELDS - gcc_profile.hijri_only_fields
        )
    return set(CRITICAL_PATHS_BY_DOCUMENT.get(document_type, set()))


def _ocr_recovery_paths(
    document_type: DocumentType,
    customer: CustomerType,
    gcc_profile: GCCDocumentProfile | None = None,
    lines: list[OCRLine] | None = None,
) -> set[str]:
    """Return the fields worth another expensive page-scale OCR pass.

    A Tourist's passport is the identity document.  A national licence can
    corroborate its holder, but is never allowed to replace passport names;
    its name/DOB rows must therefore not trigger Cyrillic OCR plus three image
    repairs after the licence number and validity have already been read.  The
    latter three fields remain mandatory recovery targets, so a genuinely
    unread licence still gets every established fallback.
    """
    paths = _critical_paths(document_type, gcc_profile)
    if (
        customer == CustomerType.TOURIST
        and document_type == DocumentType.NATIONAL_DRIVING_LICENCE_FRONT
    ):
        paths -= {
            "personal_info.full_name",
            "personal_info.date_of_birth",
        }
        # Singapore's physical driving licence is open-ended: its front states
        # the number and issue date but prints no card-expiry row.  Sending
        # three repaired renderings (and then a visual model) to look for a
        # date the document does not carry adds seconds and risks inventing a
        # value from a barcode, class table or the passport beside it.  Scope
        # this to the unmistakable Singapore heading and only when the page
        # itself contains no expiry caption; other national licences retain
        # the normal recovery path.
        page_text = " ".join(line.text.upper() for line in lines or ())
        is_singapore_licence = (
            "REPUBLIC OF SINGAPORE" in page_text
            and ("DRIVING LICENCE" in page_text or "DRIVING LICENSE" in page_text)
        )
        has_expiry_caption = any(marker in page_text for marker in (
            "EXPIRY", "EXPIRES", "VALID UNTIL", "VALID TO",
        ))
        if is_singapore_licence and not has_expiry_caption:
            paths.discard("national_driving_licence.expiry_date")
    return paths


# Values filled from a field that was already read, rather than by looking at
# the page again. A passport's machine-readable zone encodes the nationality and
# the issuing state as ISO codes, and ``_enrich_country_names`` turns each code
# into its country name once the bundle is in. Counting the name as unread is
# what used to send every passport page for a second OCR pass and a model
# generation, hunting a value the zone had already given.
DERIVED_FROM_FIELD = {
    "personal_info.nationality_name": "personal_info.nationality_code",
    "passport.issued_by_name": "passport.issued_by_code",
    "uae_driving_licence.issued_by_name": "uae_driving_licence.issued_by_code",
    "gcc_identity.issued_by_name": "gcc_identity.issued_by_code",
    "gcc_driving_licence.issued_by_name": "gcc_driving_licence.issued_by_code",
    "international_driving_permit.issued_by_name": "international_driving_permit.issued_by_code",
    "national_driving_licence.issued_by_name": "national_driving_licence.issued_by_code",
}


def _with_derivable(read_paths: set[str]) -> set[str]:
    """Add the fields that are filled from a field already read."""
    return read_paths | {
        name for name, code in DERIVED_FROM_FIELD.items() if code in read_paths
    }


PERSONAL_RELEVANT_PATHS = {
    "personal_info.first_name", "personal_info.last_name",
    "personal_info.full_name", "personal_info.full_name_arabic",
    "personal_info.gender", "personal_info.date_of_birth",
    "personal_info.nationality_code", "personal_info.nationality_name",
    "personal_info.place_of_birth",
}


# The UAE Resident workflow uses an Emirates ID and UAE driving licence, not a
# passport. Keep this explicit -- the generic personal-info prefix also
# includes place of birth, which this workflow must neither read nor present.
UAE_RESIDENT_CUSTOMER_FIELD_PATHS = [
    "personal_info.first_name",
    "personal_info.last_name",
    "personal_info.full_name",
    "personal_info.full_name_arabic",
    "personal_info.gender",
    "personal_info.date_of_birth",
    "personal_info.nationality_code",
    "personal_info.nationality_name",
    "emirates_id.number",
    "emirates_id.issue_date",
    "emirates_id.expiry_date",
    "uae_driving_licence.number",
    "uae_driving_licence.issued_by_code",
    "uae_driving_licence.issued_by_name",
    "uae_driving_licence.issue_date",
    "uae_driving_licence.expiry_date",
]

# A UAE Resident route can only contain a UAE driving licence. Its issuing
# country is a workflow constant, not an OCR task: wording such as ``RTA`` or
# ``Place of Issue Dubai`` describes an authority or city, not the customer
# field's country value.
UAE_DRIVING_LICENCE_STATIC_ISSUER_PATHS = frozenset({
    "uae_driving_licence.issued_by_code",
    "uae_driving_licence.issued_by_name",
})

# Middle name remains an internal parsing aid for the existing UAE name
# handling. The licence issuer is populated from the UAE workflow selection,
# so neither it nor its ISO code is sent through label extraction or VLM.
UAE_RESIDENT_EXTRACTION_FIELDS = frozenset({
    *UAE_RESIDENT_CUSTOMER_FIELD_PATHS,
    "personal_info.middle_name",
}) - UAE_DRIVING_LICENCE_STATIC_ISSUER_PATHS


# The GCC workflow's complete output, in the order it is presented. Nothing
# outside this list is extracted, stored or exported for a GCC customer.
GCC_CUSTOMER_FIELD_PATHS = [
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
]

# Backward-compatible names for callers created against earlier revisions of
# the GCC workflow.
GCC_SHARED_IDENTIFIER_FIELD_PATHS = GCC_CUSTOMER_FIELD_PATHS
SAUDI_CUSTOMER_FIELD_PATHS = GCC_CUSTOMER_FIELD_PATHS


# The tourist workflow's complete output, in the order it is presented: the
# passport's personal and document rows, the driving licence, and the
# international permit that the country policy may require alongside it.
# Nothing outside this list is shown, stored or exported for a tourist.
TOURIST_CUSTOMER_FIELD_PATHS = [
    "personal_info.first_name",
    "personal_info.last_name",
    "personal_info.gender",
    "personal_info.date_of_birth",
    "personal_info.nationality_name",
    "passport.number",
    "passport.issued_by_name",
    "passport.issue_date",
    "passport.expiry_date",
    "national_driving_licence.number",
    "national_driving_licence.issued_by_name",
    "national_driving_licence.issue_date",
    "national_driving_licence.expiry_date",
    "international_driving_permit.number",
    "international_driving_permit.issued_by_name",
    "international_driving_permit.issue_date",
    "international_driving_permit.expiry_date",
]


# The JSON key each tourist field is exported under.
TOURIST_EXPORT_KEYS = {
    "personal_info.first_name": "first_name",
    "personal_info.last_name": "last_name",
    "personal_info.gender": "gender",
    "personal_info.date_of_birth": "date_of_birth",
    "personal_info.nationality_name": "nationality",
    "passport.number": "passport_number",
    "passport.issued_by_name": "passport_issued_by_country",
    "passport.issue_date": "passport_issue_date",
    "passport.expiry_date": "passport_expiry_date",
    "national_driving_licence.number": "driving_license_number",
    "national_driving_licence.issued_by_name": "driving_license_issued_by_country",
    "national_driving_licence.issue_date": "driving_license_issue_date",
    "national_driving_licence.expiry_date": "driving_license_expiry_date",
    "international_driving_permit.number": "international_license_number",
    "international_driving_permit.issued_by_name": "international_license_issued_by_country",
    "international_driving_permit.issue_date": "international_license_issue_date",
    "international_driving_permit.expiry_date": "international_license_expiry_date",
}


# Read to derive the exported values, to settle which acceptance rule applies,
# and to check that the passport and the licence describe one person. Never
# exported and never shown on the tourist page.
#
# The two ``holder_id`` rows are the national number the passport and an EU or
# Vienna licence both print, which is the strongest evidence in the bundle that
# the two documents belong to one person. The ``issued_by_code`` rows fill their
# ``issued_by_name`` counterpart when only the code was read.
TOURIST_DERIVATION_FIELDS = frozenset({
    "personal_info.full_name",
    "personal_info.full_name_arabic",
    "personal_info.middle_name",
    "personal_info.nationality_code",
    "passport.issued_by_code",
    "passport.holder_id",
    "national_driving_licence.issued_by_code",
    "national_driving_licence.holder_id",
    "international_driving_permit.issued_by_code",
})


# What the extractor is allowed to look for on a tourist's pages. Restricting it
# is both what keeps another workflow's fields out of the result and what stops
# every page being scanned for Emirates ID, UAE licence and GCC card labels that
# a tourist bundle never carries.
TOURIST_EXTRACTION_FIELDS = frozenset(TOURIST_CUSTOMER_FIELD_PATHS) | TOURIST_DERIVATION_FIELDS


# The JSON key each GCC field is exported under.
GCC_EXPORT_KEYS = {
    "personal_info.first_name": "first_name",
    "personal_info.last_name": "last_name",
    "personal_info.gender": "gender",
    "personal_info.date_of_birth": "date_of_birth",
    "personal_info.nationality_name": "nationality",
    "gcc_driving_licence.number": "licence_number",
    "gcc_driving_licence.issued_by_name": "licence_issued_by",
    "gcc_driving_licence.issue_date": "licence_issue_date",
    "gcc_driving_licence.expiry_date": "licence_expiry",
    "gcc_identity.number": "id_number",
    "gcc_identity.issued_by_name": "id_issued_by",
    "gcc_identity.issue_date": "id_issue_date",
    "gcc_identity.expiry_date": "id_expiry",
}


def relevant_field_paths(
    customer_type: CustomerType | str,
    licence_policy: CountryLicencePolicy | None = None,
    gcc_profile: GCCDocumentProfile | None = None,
) -> list[str]:
    """Return only fields that belong to the selected customer workflow."""
    customer = CustomerType(customer_type)
    if customer == CustomerType.GCC_NATIONAL:
        return list(GCC_CUSTOMER_FIELD_PATHS)
    if customer == CustomerType.TOURIST:
        # The permit rows stay in the list whatever the policy says. A country
        # that needs no permit is not a country whose visitors never carry one,
        # and a permit that was presented and read has to be reportable.
        return list(TOURIST_CUSTOMER_FIELD_PATHS)
    return list(UAE_RESIDENT_CUSTOMER_FIELD_PATHS)


def _required_review_paths(
    customer: CustomerType,
    licence_policy: CountryLicencePolicy | None,
    gcc_profile: GCCDocumentProfile | None,
) -> set[str]:
    if customer == CustomerType.UAE_RESIDENT:
        return {
            *_critical_paths(DocumentType.EMIRATES_ID_FRONT),
            *_critical_paths(DocumentType.UAE_DRIVING_LICENCE_FRONT),
        }
    if customer == CustomerType.GCC_NATIONAL:
        return set(GCC_CUSTOMER_FIELD_PATHS)
    if licence_policy and licence_policy.requirement == LicenceRequirement.NEED_IDL:
        return {
            *_critical_paths(DocumentType.PASSPORT_BIODATA),
            *_critical_paths(DocumentType.INTERNATIONAL_DRIVING_PERMIT),
        }
    return {
        *_critical_paths(DocumentType.PASSPORT_BIODATA),
        *_critical_paths(DocumentType.NATIONAL_DRIVING_LICENCE_FRONT),
    }


def _required_review_paths_for_result(result: ExtractionResult) -> set[str]:
    policy = policy_for_country(result.licence_policy.country)
    profile = profile_for_gcc_country(result.gcc_profile.country)
    return _required_review_paths(result.customer_type, policy, profile)


# The only fields a national licence's reverse may contribute, and the ceiling
# they are held to, so a front reading always outranks one taken from the back.
_LICENCE_BACK_CARRIED_PATHS = frozenset({
    "national_driving_licence.issue_date",
    "national_driving_licence.expiry_date",
})
_LICENCE_BACK_MAXIMUM_CONFIDENCE = 0.75


MULTILINGUAL_DOCUMENTS = {
    DocumentType.PASSPORT_BIODATA,
    DocumentType.INTERNATIONAL_DRIVING_PERMIT,
    DocumentType.NATIONAL_DRIVING_LICENCE_FRONT,
    DocumentType.NATIONAL_DRIVING_LICENCE_BACK,
    DocumentType.GCC_IDENTITY_FRONT,
    DocumentType.GCC_IDENTITY_BACK,
    DocumentType.GCC_DRIVING_LICENCE_FRONT,
    DocumentType.GCC_DRIVING_LICENCE_BACK,
}


DRIVING_DOCUMENT_TYPES = {
    DocumentType.UAE_DRIVING_LICENCE_FRONT,
    DocumentType.UAE_DRIVING_LICENCE_BACK,
    DocumentType.INTERNATIONAL_DRIVING_PERMIT,
    DocumentType.NATIONAL_DRIVING_LICENCE_FRONT,
    DocumentType.NATIONAL_DRIVING_LICENCE_BACK,
    DocumentType.GCC_DRIVING_LICENCE_FRONT,
    DocumentType.GCC_DRIVING_LICENCE_BACK,
}


VLM_FIELD_PATHS_BY_DOCUMENT = {
    DocumentType.EMIRATES_ID_FRONT: {
        "emirates_id.number", "emirates_id.issue_date", "emirates_id.expiry_date",
    },
    DocumentType.EMIRATES_ID_BACK: {
        "emirates_id.issue_date", "emirates_id.expiry_date",
    },
    DocumentType.UAE_DRIVING_LICENCE_FRONT: {
        "personal_info.first_name", "personal_info.middle_name", "personal_info.last_name",
        "personal_info.full_name", "personal_info.full_name_arabic",
        "personal_info.date_of_birth", "personal_info.nationality_name",
        "uae_driving_licence.number", "uae_driving_licence.issued_by_code",
        "uae_driving_licence.issued_by_name", "uae_driving_licence.issue_date",
        "uae_driving_licence.expiry_date",
    },
    # The four GCC entries are placeholders: the paths actually requested from
    # the model are narrowed per country and per card side by the profile, so
    # nothing is asked for that the card does not print.
    DocumentType.GCC_IDENTITY_FRONT: GCC_EXTRACTION_FIELDS,
    DocumentType.GCC_IDENTITY_BACK: GCC_EXTRACTION_FIELDS,
    DocumentType.GCC_DRIVING_LICENCE_FRONT: GCC_EXTRACTION_FIELDS,
    DocumentType.GCC_DRIVING_LICENCE_BACK: GCC_EXTRACTION_FIELDS,
    DocumentType.PASSPORT_BIODATA: {
        "personal_info.first_name", "personal_info.middle_name", "personal_info.last_name",
        "personal_info.full_name", "personal_info.gender", "personal_info.date_of_birth",
        "personal_info.nationality_code", "personal_info.nationality_name",
        "passport.number", "passport.issued_by_code", "passport.issued_by_name",
        "passport.issue_date", "passport.expiry_date",
    },
    DocumentType.INTERNATIONAL_DRIVING_PERMIT: {
        "personal_info.first_name", "personal_info.middle_name", "personal_info.last_name",
        "personal_info.full_name", "personal_info.date_of_birth",
        "international_driving_permit.number", "international_driving_permit.issued_by_code",
        "international_driving_permit.issued_by_name", "international_driving_permit.issue_date",
        "international_driving_permit.expiry_date",
    },
    DocumentType.NATIONAL_DRIVING_LICENCE_FRONT: {
        "personal_info.first_name", "personal_info.middle_name", "personal_info.last_name",
        "personal_info.full_name", "personal_info.date_of_birth",
        "national_driving_licence.number", "national_driving_licence.issued_by_code",
        "national_driving_licence.issued_by_name", "national_driving_licence.issue_date",
        "national_driving_licence.expiry_date",
    },
    # A reverse carries categories and restrictions, not fields belonging to
    # the passport or international permit. Keeping it in the typed map with
    # no generative fields prevents the generic UNKNOWN-page fallback from
    # asking for every tourist field and copying a Ghanaian card reference into
    # international_driving_permit.number.
    DocumentType.NATIONAL_DRIVING_LICENCE_BACK: set(),
}


def validate_required_uploads(customer_type: CustomerType, uploads: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if "document_bundle" in uploads:
        bundle = uploads.get("document_bundle")
        if bundle is None or bundle == []:
            errors.append("MISSING_REQUIRED_DOCUMENT:document_bundle")
        # A tourist no longer has to name their country: the reader derives it
        # from the documents themselves. A country that *is* supplied still has
        # to be one the policy list knows, otherwise the operator has selected
        # an acceptance rule that does not exist.
        if (
            customer_type == CustomerType.TOURIST
            and uploads.get("licence_country")
            and policy_for_country(uploads.get("licence_country")) is None
        ):
            errors.append("MISSING_OR_UNSUPPORTED_LICENCE_COUNTRY")
        if customer_type == CustomerType.GCC_NATIONAL and uploads.get("gcc_country") not in GCC_COUNTRY_NAMES:
            errors.append("MISSING_OR_UNSUPPORTED_GCC_COUNTRY")
        return errors
    if customer_type in {CustomerType.UAE_RESIDENT, CustomerType.GCC_NATIONAL}:
        required_slots = REQUIRED_SLOTS[customer_type]
        if customer_type == CustomerType.GCC_NATIONAL and uploads.get("gcc_country") not in GCC_COUNTRY_NAMES:
            errors.append("MISSING_OR_UNSUPPORTED_GCC_COUNTRY")
    elif "licence_country" not in uploads:
        # Backward-compatible API behaviour. The UI always sends a country.
        required_slots = REQUIRED_SLOTS[customer_type]
    else:
        policy = policy_for_country(uploads.get("licence_country"))
        if policy is None and uploads.get("licence_country"):
            errors.append("MISSING_OR_UNSUPPORTED_LICENCE_COUNTRY")
            required_slots = ("passport",)
        elif policy is None:
            # Country left blank on purpose. The passport is still required,
            # and one of the two driving documents has to be there for the
            # reader to have anything to read the country off.
            required_slots = ("passport",)
            if not uploads.get("idp_pages") and not uploads.get("national_licence_front"):
                errors.append(
                    "MISSING_REQUIRED_DOCUMENT:idp_pages_or_national_licence_front"
                )
        elif policy.requirement == LicenceRequirement.NEED_IDL:
            required_slots = ("passport", "idp_pages")
        else:
            required_slots = ("passport", "national_licence_front")
    for slot in required_slots:
        value = uploads.get(slot)
        if value is None or value == []:
            errors.append(f"MISSING_REQUIRED_DOCUMENT:{slot}")
    return errors


def _set_path(result: ExtractionResult, path: str, value: str | None) -> None:
    section, attribute = path.split(".", 1)
    setattr(getattr(result, section), attribute, value)


def _flatten_uploads(uploads: dict[str, Any]) -> list[tuple[str, Any]]:
    flattened: list[tuple[str, Any]] = []
    for slot, value in uploads.items():
        if slot in {"licence_country", "gcc_country"}:
            continue
        if value is None: continue
        values = value if isinstance(value, list) else [value]
        flattened.extend((slot, item) for item in values if item is not None)
    return flattened


def _split_both_sides(loaded: Any) -> list[Any]:
    """Present a capture holding both sides of a card as two pages.

    Each half is then routed, classified and read on its own, so the reverse
    is recognised as the reverse and its category table is read as one rather
    than as stray rows on the front.
    """
    parts = split_card_sides(loaded.image)
    if len(parts) < 2:
        return [loaded]
    return [
        replace(loaded, image=part, page_index=loaded.page_index * len(parts) + offset)
        for offset, part in enumerate(parts)
    ]


def _best_mrz_from_ocr(
    lines: list[OCRLine], *,
    allow_printed_given_name_separator_repair: bool = False,
) -> ParsedMRZ | None:
    """Select the strongest MRZ, allowing rows recognized by different languages.

    ``allow_printed_given_name_separator_repair`` belongs to the tourist route.
    It lets a clearly labelled passport Given names row restore one dropped
    surname/given-name separator when OCR failed to read the printed surname
    label. Other customer routes retain the stricter two-labelled-row rule.
    """
    candidate_sets = [
        find_mrz_lines([line for line in lines if line.language == "en"]),
        find_mrz_lines(lines),
    ]
    parsed_options: list[ParsedMRZ] = []
    for candidate_lines in candidate_sets:
        distinct = list(dict.fromkeys(candidate_lines))
        for size in (2, 3):
            # Every order, not the page's. The rows are offered top to bottom,
            # and a zone read twice defeats that: an Australian passport had its
            # second row bleed through near the head of the page, which sorted
            # ahead of the real first row, and the pair was handed to the parser
            # reversed. Both rows were perfect and the zone was reported unread.
            # Order is a property of the format, so the format decides it -- the
            # check digits and the identity-row guard still throw out the rest.
            for subset in permutations(distinct, size):
                parsed = parse_mrz(list(subset))
                if parsed.mrz_type and _credible_mrz_identity_row(parsed):
                    parsed_options.append(parsed)

    def right_clipped_tourist_td3_options() -> list[ParsedMRZ]:
        """Recover the checksummed core of a passport row cut on the right.

        A TD3 passport's document number, birth date and expiry date all end
        before column 29 and each has its own check digit. Phone captures can
        therefore lose the eleven trailing optional/final-check columns while
        still preserving those three values completely. Padding arbitrary OCR
        rows to 44 characters would recreate the old failure where a page
        caption became a holder name, so this recovery is intentionally much
        narrower:

        * the complete 44-character name row must begin with a credible
          passport header;
        * the data fragment must be the next physical row and align with it;
        * all three visible field check digits must pass; and
        * every surviving character after column 28 must be filler ``<``.

        The missing optional-data and composite checks remain ``None``. They
        are never manufactured, and the whole zone remains not-fully-valid,
        while the three independently checksummed fields can still be used as
        verified evidence by ``mrz_candidates``.
        """
        if not allow_printed_given_name_separator_repair:
            return []

        rows: list[tuple[str, str, float, float, float, float]] = []
        for line in lines:
            compact = normalize_mrz_line(line.text)
            if not line.bounding_box or "<" not in compact:
                continue
            xs = [point[0] for point in line.bounding_box]
            ys = [point[1] for point in line.bounding_box]
            rows.append((
                line.text, compact,
                min(xs), max(xs), min(ys), max(ys),
            ))

        name_rows = [
            row for row in rows
            if len(row[1]) == 44
            and row[1].startswith("P<")
            and "<<" in row[1][5:]
        ]
        data_rows = [
            row for row in rows
            if 28 <= len(row[1]) < 44
            and row[1][0].isalnum()
            and re.fullmatch(r"<*", row[1][28:]) is not None
            and re.fullmatch(r"[A-Z0-9<]{9}[0-9][A-Z]{3}[0-9]{6}[0-9][MFX<][0-9]{6}[0-9]", row[1][:28])
        ]

        recovered: list[ParsedMRZ] = []
        for raw_name, name_row, name_left, name_right, name_top, name_bottom in name_rows:
            for raw_data, data_fragment, data_left, data_right, data_top, data_bottom in data_rows:
                name_height = max(1.0, name_bottom - name_top)
                data_height = max(1.0, data_bottom - data_top)
                # MRZ rows are consecutive, left-aligned lines. This prevents
                # a valid data fragment elsewhere on the page from borrowing
                # an unrelated P-prefixed row as its holder name.
                if data_top < name_top:
                    continue
                if data_top - name_bottom > max(80.0, 1.5 * max(name_height, data_height)):
                    continue
                if abs(data_left - name_left) > max(60.0, max(name_height, data_height)):
                    continue
                if data_right <= data_left or name_right <= name_left:
                    continue
                if not (
                    validate_check(data_fragment[0:9], data_fragment[9])
                    and validate_check(data_fragment[13:19], data_fragment[19])
                    and validate_check(data_fragment[21:27], data_fragment[27])
                ):
                    continue

                # Only unknown right-edge columns are filled for the parser;
                # they are immediately marked unknown below and cannot become
                # checksum evidence. The visible 0:28 core is unchanged.
                data_row = data_fragment.ljust(44, "<")
                parsed = parse_mrz([name_row, data_row])
                if (
                    parsed.mrz_type != "TD3"
                    or not _credible_mrz_identity_row(parsed)
                    or parsed.fields.get("issuing_country_code")
                    != parsed.fields.get("nationality_code")
                    or parsed.checks.get("document_number") is not True
                    or parsed.checks.get("date_of_birth") is not True
                    or parsed.checks.get("expiry_date") is not True
                ):
                    continue
                parsed.raw_lines = [raw_name, raw_data]
                parsed.checks["optional_data"] = None
                parsed.checks["composite"] = None
                parsed.fields["optional_data"] = None
                parsed.valid = False
                parsed.warnings = [
                    warning for warning in parsed.warnings
                    if warning != "MRZ_CHECKSUM_FAILURE"
                ]
                parsed.warnings.append("MRZ_RIGHT_CROP_CORE_FIELDS_CHECKSUMMED")
                parsed.corrections.insert(0, {
                    "field": "mrz_right_crop",
                    "from": [name_row, data_fragment],
                    "to": [name_row, data_row],
                    "evidence": "DOCUMENT_NUMBER_AND_BIRTH_AND_EXPIRY_CHECK_DIGITS",
                })
                recovered.append(parsed)
        return recovered

    def left_clipped_tourist_td3_options() -> list[ParsedMRZ]:
        """Recover a TD3 zone whose left edge was cut off by the photograph.

        The recovery is deliberately evidence-bound. A passport number read
        from its labelled printed row supplies only the missing prefix of the
        data row; every surviving character remains OCR evidence. The name row
        must begin with the remainder of the standard ``P<ISO3`` header, both
        rows must physically touch the image's left edge, and the birth and
        expiry check digits must pass after alignment. This is enough to read
        the fixed sex position without inventing any identity value.
        """
        if not allow_printed_given_name_separator_repair:
            return []
        printed_numbers = {
            re.sub(r"[^A-Z0-9]", "", str(candidate.normalized_value).upper())
            for candidate in labelled_ocr_candidates(
                lines, DocumentType.PASSPORT_BIODATA, "mrz_crop_evidence",
            )
            if candidate.field_path == "passport.number"
            and candidate.normalized_value
            and candidate.confidence >= 0.80
        }
        printed_numbers = {
            value for value in printed_numbers if len(value) == 9
        }
        if not printed_numbers:
            return []

        edge_rows: list[tuple[float, str]] = []
        for line in lines:
            compact = normalize_mrz_line(line.text)
            if len(compact) < 20 or "<" not in compact or not line.bounding_box:
                continue
            xs = [point[0] for point in line.bounding_box]
            ys = [point[1] for point in line.bounding_box]
            height = max(ys) - min(ys)
            if min(xs) > max(12.0, height * 0.25):
                continue
            edge_rows.append((min(ys), compact))

        name_fragments = {
            compact for _, compact in edge_rows
            if re.fullmatch(r"<[A-Z]{3}[A-Z0-9<]+", compact)
            and compact.endswith("<")
            and len(compact) < 44
        }
        data_fragments = {
            compact for _, compact in edge_rows
            if compact[0].isalnum()
            and compact.endswith(("<", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9"))
            and 32 <= len(compact) < 44
            and sum(character.isdigit() for character in compact[:28]) >= 12
        }
        if not name_fragments or not data_fragments:
            return []

        def equivalent(one: str, other: str) -> bool:
            if len(one) != len(other):
                return False
            return all(
                left == right
                or CONFUSIONS.get(left) == right
                or CONFUSIONS.get(right) == left
                for left, right in zip(one, other)
            )

        recovered: list[ParsedMRZ] = []
        for name_fragment in name_fragments:
            name_row = ("P" + name_fragment).ljust(44, "<")
            if len(name_row) != 44:
                continue
            for data_fragment in data_fragments:
                for printed_number in printed_numbers:
                    for missing in range(1, 4):
                        overlap = 9 - missing
                        if (
                            overlap <= 0
                            or len(data_fragment) < overlap + 20
                            or not equivalent(
                                printed_number[missing:], data_fragment[:overlap],
                            )
                        ):
                            continue
                        rebuilt = printed_number[:missing] + data_fragment
                        number_corrections: list[dict[str, Any]] = []
                        if name_row[2:5] == "FRA" and len(rebuilt) >= 10:
                            # French ordinary-passport numbers are two digits,
                            # two letters and five digits. That field class plus
                            # its surviving check digit uniquely restores I/1,
                            # O/0 and the other standard MRZ confusions.
                            reverse_confusions = {
                                digit: letter for letter, digit in CONFUSIONS.items()
                            }
                            raw_number = rebuilt[:9]
                            repaired_chars = list(raw_number)
                            for position, character in enumerate(repaired_chars):
                                expects_letter = position in {2, 3}
                                if expects_letter and character.isdigit():
                                    repaired_chars[position] = reverse_confusions.get(
                                        character, character,
                                    )
                                elif not expects_letter and character.isalpha():
                                    repaired_chars[position] = CONFUSIONS.get(
                                        character, character,
                                    )
                            repaired_number = "".join(repaired_chars)
                            if (
                                re.fullmatch(r"\d{2}[A-Z]{2}\d{5}", repaired_number)
                                and validate_check(repaired_number, rebuilt[9])
                            ):
                                number_corrections = [{
                                    "field": "document_number",
                                    "position": position,
                                    "from": before,
                                    "to": after,
                                    "reason": "French passport field class and check digit",
                                } for position, (before, after) in enumerate(zip(
                                    raw_number, repaired_number,
                                )) if before != after]
                                rebuilt = repaired_number + rebuilt[9:]
                        deficit = 44 - len(rebuilt)
                        if not 0 <= deficit <= 8:
                            continue
                        completions = {rebuilt.ljust(44, "<")}
                        if deficit and rebuilt[-1].isdigit():
                            # A right crop commonly removes filler immediately
                            # before the final check digit, rather than after it.
                            completions.add(
                                rebuilt[:-1] + "<" * deficit + rebuilt[-1]
                            )
                        for data_row in completions:
                            parsed = parse_mrz([name_row, data_row])
                            if (
                                parsed.mrz_type != "TD3"
                                or not _credible_mrz_identity_row(parsed)
                                or parsed.checks.get("date_of_birth") is not True
                                or parsed.checks.get("expiry_date") is not True
                            ):
                                continue
                            parsed.warnings.append(
                                "MRZ_LEFT_CROP_RESTORED_FROM_PRINTED_PASSPORT_NUMBER"
                            )
                            parsed.corrections.insert(0, {
                                "field": "mrz_left_crop",
                                "from": [name_fragment, data_fragment],
                                "to": [name_row, data_row],
                                "evidence": "PRINTED_PASSPORT_NUMBER_AND_DATE_CHECK_DIGITS",
                            })
                            parsed.corrections[1:1] = number_corrections
                            recovered.append(parsed)
        return recovered

    if not parsed_options:
        parsed_options.extend(right_clipped_tourist_td3_options())
    if not parsed_options:
        parsed_options.extend(left_clipped_tourist_td3_options())
    if not parsed_options:
        # Nothing parsed cleanly: retry with the row filler a capture commonly
        # clips. Check digits still decide whether the repair is accepted.
        clipped = list(dict.fromkeys([
            *find_clipped_mrz_lines([line for line in lines if line.language == "en"]),
            *find_clipped_mrz_lines(lines),
        ]))
        for subset in combinations(clipped, 3):
                for repaired in td1_filler_repairs(list(subset)):
                    parsed = parse_mrz(repaired)
                    if parsed.mrz_type and _credible_mrz_identity_row(parsed) and any(
                        check is True for check in parsed.checks.values()
                    ):
                        parsed.warnings.append("MRZ_FILLER_RESTORED")
                        parsed_options.append(parsed)
    if not parsed_options:
        return None
    # The English-only and all-language candidate sets can produce the same
    # zone twice. Collapse those before asking whether there is a real tie.
    distinct_options: dict[tuple[str, ...], ParsedMRZ] = {}
    for parsed in parsed_options:
        distinct_options.setdefault(tuple(parsed.normalized_lines), parsed)
    options = list(distinct_options.values())

    printed_names: dict[str, dict[str, float]] | None = None

    def passport_printed_names() -> dict[str, dict[str, float]]:
        """Return independently labelled names once, only if needed."""
        nonlocal printed_names
        if printed_names is not None:
            return printed_names
        printed_names = {}
        for candidate in labelled_ocr_candidates(
            lines, DocumentType.PASSPORT_BIODATA, "mrz_name_evidence",
        ):
            if candidate.field_path not in {
                "personal_info.first_name", "personal_info.last_name",
                "personal_info.full_name",
            } or not candidate.normalized_value:
                continue
            value = str(candidate.normalized_value).strip().upper()
            values = printed_names.setdefault(candidate.field_path, {})
            values[value] = max(values.get(value, 0.0), candidate.confidence)
        return printed_names

    def restore_missing_name_separator(parsed: ParsedMRZ) -> ParsedMRZ:
        """Restore one lost ``<`` only when printed name rows prove the split.

        No TD2/TD3 check digit covers the name row.  A dropped filler in
        ``ANI<<OLIVER`` therefore leaves an otherwise fully valid zone reading
        ``ANI<OLIVER`` as the surname ``ANI OLIVER``.  The labelled Surname and
        Given names rows are independent evidence for exactly where that one
        separator belongs.  Requiring exact agreement with both rows keeps
        compound surnames intact and refuses an ambiguous reconstruction.
        """
        partially_aligned = (
            allow_printed_given_name_separator_repair
            and "MRZ_LEFT_CROP_RESTORED_FROM_PRINTED_PASSPORT_NUMBER"
            in parsed.warnings
            and parsed.checks.get("date_of_birth") is True
            and parsed.checks.get("expiry_date") is True
        )
        if (
            parsed.mrz_type not in {"TD2", "TD3"}
            or (not parsed.valid and not partially_aligned)
            or parsed.fields.get("first_name")
            or not parsed.fields.get("last_name")
            or len(parsed.normalized_lines) != 2
        ):
            return parsed
        evidence = passport_printed_names()
        last_values = {
            value for value, confidence in evidence.get("personal_info.last_name", {}).items()
            if confidence >= 0.80
        }
        first_values = {
            value for value, confidence in evidence.get("personal_info.first_name", {}).items()
            if confidence >= 0.80
        }
        if not first_values:
            return parsed

        def encoded(value: str) -> str:
            return re.sub(r"[^A-Z0-9]+", "<", value.upper()).strip("<")

        current_last = " ".join(str(parsed.fields["last_name"]).upper().split())
        name_row, data_row = parsed.normalized_lines
        name_area = name_row[5:]

        def repaired_with(
            last_name: str, first_name: str, last_encoded: str,
            first_encoded: str, evidence_source: str,
        ) -> ParsedMRZ | None:
            prefix = f"{last_encoded}<{first_encoded}"
            if not last_encoded or not first_encoded or not name_area.startswith(prefix):
                return None
            separator = 5 + len(last_encoded)
            if name_row[separator] != "<" or not name_row.endswith("<"):
                return None
            repaired_row = (
                name_row[:separator] + "<<" + name_row[separator + 1:-1]
            )
            repaired = parse_mrz([repaired_row, data_row])
            repaired_partially_aligned = (
                partially_aligned
                and repaired.checks.get("date_of_birth") is True
                and repaired.checks.get("expiry_date") is True
            )
            if not (
                (repaired.valid or repaired_partially_aligned)
                and repaired.fields.get("last_name") == last_name
                and repaired.fields.get("first_name") == first_name
            ):
                return None
            repaired.warnings = list(dict.fromkeys([
                *parsed.warnings, *repaired.warnings,
            ]))
            repaired.corrections = [
                *parsed.corrections, *repaired.corrections,
            ]
            repaired.warnings.append(
                "MRZ_NAME_SEPARATOR_RESTORED_FROM_PRINTED_LABELS"
            )
            if evidence_source == "PRINTED_GIVEN_NAMES_AND_MRZ_STRUCTURE":
                repaired.warnings.append(
                    "MRZ_NAME_SEPARATOR_RESTORED_FROM_PRINTED_GIVEN_NAMES"
                )
            repaired.corrections.append({
                "field": "name_separator",
                "from": "<", "to": "<<",
                "evidence": evidence_source,
            })
            return repaired

        for last_name in last_values:
            for first_name in first_values:
                if current_last != f"{last_name} {first_name}":
                    continue
                last_encoded, first_encoded = encoded(last_name), encoded(first_name)
                repaired = repaired_with(
                    last_name, first_name, last_encoded, first_encoded,
                    "PRINTED_SURNAME_AND_GIVEN_NAMES",
                )
                if repaired is not None:
                    return repaired

        if not allow_printed_given_name_separator_repair:
            return parsed

        # A tourist passport can lose the tiny surname label while preserving
        # the much clearer Given names label and both value rows. When the MRZ
        # parser consequently reports no given name at all, an exact labelled
        # given-name suffix identifies the one existing filler that must have
        # been the missing ``<<`` boundary. The fully checksum-valid data row,
        # fixed TD2/TD3 length and trailing filler are still required above.
        compact_name = name_area.rstrip("<")
        for first_name in first_values:
            first_encoded = encoded(first_name)
            suffix = f"<{first_encoded}"
            if not first_encoded or not compact_name.endswith(suffix):
                continue
            last_encoded = compact_name[:-len(suffix)].strip("<")
            last_name = re.sub(r"<+", " ", last_encoded).strip()
            if not last_name or current_last != f"{last_name} {first_name}":
                continue
            if last_values and not (
                partially_aligned
                and any(
                    _is_character_completion(printed_last, last_name)
                    for printed_last in last_values
                )
            ):
                continue
            repaired = repaired_with(
                last_name, first_name, last_encoded, first_encoded,
                "PRINTED_GIVEN_NAMES_AND_MRZ_STRUCTURE",
            )
            if repaired is not None:
                return repaired
        return parsed

    def base_rank(parsed: ParsedMRZ) -> tuple[bool, int, int, int]:
        return (
            parsed.valid,
            _zone_plausibility(parsed),
            sum(check is True for check in parsed.checks.values()),
            -len(parsed.warnings),
        )

    strongest_rank = max(map(base_rank, options))
    strongest = [parsed for parsed in options if base_rank(parsed) == strongest_rank]
    if len(strongest) == 1:
        return restore_missing_name_separator(strongest[0])

    # No check digit covers a TD2/TD3 name row. Two OCR renderings can therefore
    # produce equally valid zones whose only difference is a missing letter in
    # the name: the reported Canadian passport yielded both FARAZ and FAAZ,
    # while every digit on the second row passed. The printed, explicitly
    # labelled Surname/Given names rows are independent evidence and settle
    # exactly this tie without weakening any checksum-protected field.
    printed_names = passport_printed_names()

    field_paths = {
        "first_name": "personal_info.first_name",
        "last_name": "personal_info.last_name",
        "full_name": "personal_info.full_name",
    }

    def printed_name_agreement(parsed: ParsedMRZ) -> float:
        return sum(
            printed_names.get(path, {}).get(
                str(parsed.fields.get(key) or "").strip().upper(), 0.0,
            )
            for key, path in field_paths.items()
        )

    return restore_missing_name_separator(
        max(strongest, key=printed_name_agreement),
    )


_NON_STATE_TRAVEL_DOCUMENT_ISSUERS = frozenset({
    # ICAO-assigned codes used by UN organisations and refugee/stateless travel
    # documents. They are valid issuing authorities even though they are not a
    # rental-country policy and therefore do not appear in ``policy_for_country``.
    "UNO", "UNA", "UNK", "XOM", "XXA", "XXB", "XXC", "XXX",
    # ICAO's official specimen-state code, retained for conformance fixtures.
    "UTO",
})


def _credible_mrz_identity_row(parsed: ParsedMRZ) -> bool:
    """Reject a label row paired with a genuine passport data row.

    TD3 puts every check digit on its second row. Consequently an arbitrary
    forty-four-character caption can be paired with a real second row and the
    resulting *names* still appear fully checksum-valid. A genuine TD3 first
    row has two structural facts independent of those checks: its document
    code starts with ``P`` and positions 3-5 name a real issuing authority.

    TD1 is also used by driving and identity cards, so its document code is
    intentionally left to the format parser. TD2 and TD3 are the two-row
    formats vulnerable to an unrelated caption replacing their first row, so
    both require the document-code class and issuing authority their standard
    reserves for identity/travel documents.
    """
    if parsed.mrz_type not in {"TD2", "TD3"}:
        return True
    fields = parsed.fields or {}
    document_code = str(fields.get("document_code") or "").upper()
    issuing = str(fields.get("issuing_country_code") or "").upper()
    known_issuer = (
        policy_for_country(issuing) is not None
        or issuing in _NON_STATE_TRAVEL_DOCUMENT_ISSUERS
    )
    if parsed.mrz_type == "TD2":
        # ICAO two-row travel documents use I/A/C (identity or travel
        # document), P (passport) or V (visa). A 36-character printed caption
        # paired with a clipped genuine data row can satisfy every field check
        # while yielding a prose document code such as ET and names such as
        # DATE OF EXPIRY. The code class and issuer are facts that caption
        # cannot borrow from the second row.
        return document_code[:1] in {"I", "A", "C", "P", "V"} and known_issuer
    return (
        document_code.startswith("P")
        and known_issuer
    )


def _zone_plausibility(parsed: ParsedMRZ) -> int:
    """How much the upper row looks like a passport zone rather than a label.

    Every check digit in a TD3 zone lives on the second row. The first carries
    the document code, the issuing state and the name, and not one character of
    arithmetic -- so any forty-four character row can stand in for it and the
    pair still validates perfectly.

    The trilingual Dutch passport prints exactly such a row: "Documentnummer /
    Document no. / N° de document" folds to forty-four characters, paired with
    the genuine second row, and passed all four check digits. It outranked the
    real zone on nothing but page order, and the holder was reported as
    DOCUMENT ENTNUMMER -- with the status VERIFIED, because the arithmetic
    really had passed.

    What the label row cannot fake is the shape of the first row: a passport
    zone opens with P, names its issuing state in three letters, and that state
    is the one the second row gives as the holder's nationality. The Dutch label
    row offers "DO" and "CUM" against a second row that says NLD.
    """
    fields = parsed.fields or {}
    code = (fields.get("document_code") or "").upper()
    issuing = (fields.get("issuing_country_code") or "").upper()
    nationality = (fields.get("nationality_code") or "").upper()
    score = 0
    if code.startswith("P"):
        score += 2
    if _ISO3_CODE.fullmatch(issuing):
        score += 1
    # A passport is issued by the state whose nationality it certifies. Kept as
    # a ranking signal rather than a rule, so the rare document that separates
    # the two is still read when it is the only zone on the page.
    if issuing and issuing == nationality:
        score += 2
    return score


_ISO3_CODE = re.compile(r"[A-Z]{3}")


def _candidate_score(candidates: list[FieldCandidate]) -> float:
    paths = {
        candidate.field_path for candidate in candidates
        if candidate.normalized_value and candidate.validation_passed is not False
    }
    score = 0.0
    for path in paths:
        if path.endswith(".number"):
            score += 3.0
        elif path.endswith(("issue_date", "expiry_date", "date_of_birth")):
            score += 1.0
        elif path.endswith(("full_name", "first_name", "last_name")):
            score += 0.75
        else:
            score += 0.5
    return score


def _reject_cross_document_identifiers(
    candidates: list[FieldCandidate], documents: list[DocumentRecord],
) -> None:
    """Keep each document's values inside that document.

    Several GCC states print the same national number on both the identity card
    and the driving licence, which makes it tempting to copy the ID's value into
    the licence field when the licence row is unreadable. That copy is a guess
    about a document the reader could not actually read, so it is refused: a
    licence number is reported only when it was read off the licence, and an ID
    number only when it was read off the ID. Anything unread stays null.
    """
    identity_documents = {
        document.upload_id for document in documents
        if document.detected_type in {
            DocumentType.GCC_IDENTITY_FRONT.value, DocumentType.GCC_IDENTITY_BACK.value,
        }
    }
    licence_documents = {
        document.upload_id for document in documents
        if document.detected_type in {
            DocumentType.GCC_DRIVING_LICENCE_FRONT.value,
            DocumentType.GCC_DRIVING_LICENCE_BACK.value,
        }
    }
    if not identity_documents and not licence_documents:
        return
    candidates[:] = [
        candidate for candidate in candidates
        if not (
            candidate.field_path.startswith("gcc_identity.")
            and candidate.source_document in licence_documents
        )
        and not (
            candidate.field_path.startswith("gcc_driving_licence.")
            and candidate.source_document in identity_documents
        )
    ]


_FIRST_NAME_SOURCE_TYPES = {
    CustomerType.TOURIST: {
        DocumentType.PASSPORT_BIODATA.value,
    },
    CustomerType.UAE_RESIDENT: {
        DocumentType.EMIRATES_ID_FRONT.value,
        DocumentType.EMIRATES_ID_BACK.value,
    },
    CustomerType.GCC_NATIONAL: {
        DocumentType.GCC_IDENTITY_FRONT.value,
        DocumentType.GCC_IDENTITY_BACK.value,
    },
}


def _restrict_first_name_to_identity_source(
    candidates: list[FieldCandidate], documents: list[DocumentRecord],
    customer: CustomerType,
) -> None:
    """Accept ``first_name`` only from the workflow's identity document.

    Tourist licences and permits often abbreviate the holder's given names;
    resident and GCC driving licences can do the same.  They remain useful for
    whole-name cross checks, but they may never replace the CRM first name.  A
    page explicitly uploaded in the correct slot is allowed when classification
    is UNKNOWN, because glare can hide the title while its labelled name remains
    readable.
    """
    allowed_types = _FIRST_NAME_SOURCE_TYPES[customer]
    allowed_documents = {
        document.upload_id for document in documents
        if document.detected_type in allowed_types
        or (
            document.detected_type == DocumentType.UNKNOWN.value
            and document.expected_type in allowed_types
        )
    }
    candidates[:] = [
        candidate for candidate in candidates
        if candidate.field_path != "personal_info.first_name"
        or candidate.source_document in allowed_documents
    ]


def _prefer_emirates_id_mrz_name_components(
    candidates: list[FieldCandidate], documents: list[DocumentRecord],
    customer: CustomerType,
) -> None:
    """Use a valid Emirates-ID MRZ when a UAE name row was only guessed.

    The printed ``Name`` line has no divider between family and given names.
    Its first/last-name projections are therefore guesses based on token order,
    whereas the card's TD1 line explicitly separates ``SURNAME<<GIVEN NAMES``.
    An explicit First Name or Surname row remains authoritative. A generic
    full-name projection on either the Emirates ID *or UAE driving licence*
    must yield to the ID's valid MRZ; otherwise the licence's token-order
    guess can incorrectly turn the MRZ's family name into a conflict.
    """
    if customer != CustomerType.UAE_RESIDENT:
        return
    emirates_documents = {
        document.upload_id for document in documents
        if document.detected_type in {
            DocumentType.EMIRATES_ID_FRONT.value,
            DocumentType.EMIRATES_ID_BACK.value,
        }
        or (
            document.detected_type == DocumentType.UNKNOWN.value
            and document.expected_type in {
                DocumentType.EMIRATES_ID_FRONT.value,
                DocumentType.EMIRATES_ID_BACK.value,
            }
        )
    }
    mrz_paths = {
        candidate.field_path for candidate in candidates
        if candidate.source_document in emirates_documents
        and candidate.source_method == "mrz"
        and "EMIRATES_ID_MRZ_NAME_FALLBACK" in candidate.warnings
    }
    if not mrz_paths:
        return
    uae_name_documents = {
        document.upload_id for document in documents
        if document.detected_type in {
            DocumentType.EMIRATES_ID_FRONT.value,
            DocumentType.EMIRATES_ID_BACK.value,
            DocumentType.UAE_DRIVING_LICENCE_FRONT.value,
            DocumentType.UAE_DRIVING_LICENCE_BACK.value,
        }
        or (
            document.detected_type == DocumentType.UNKNOWN.value
            and document.expected_type in {
                DocumentType.EMIRATES_ID_FRONT.value,
                DocumentType.EMIRATES_ID_BACK.value,
                DocumentType.UAE_DRIVING_LICENCE_FRONT.value,
                DocumentType.UAE_DRIVING_LICENCE_BACK.value,
            }
        )
    }
    candidates[:] = [
        candidate for candidate in candidates
        if not (
            candidate.source_document in uae_name_documents
            and candidate.field_path in mrz_paths
            and "DERIVED_FROM_VISIBLE_FULL_NAME" in candidate.warnings
        )
    ]


def _restrict_tourist_last_name_to_passport(
    candidates: list[FieldCandidate], documents: list[DocumentRecord],
) -> None:
    """Prevent a driving document from replacing a tourist's passport surname.

    The licence and permit still remain in the original evidence collection for
    cross-document matching and operator review.  This filter is applied only
    to the copy sent to reconciliation: the passport is the tourist's identity
    document, so it is the only document allowed to populate the CRM surname.
    """
    passport_documents = {
        document.upload_id for document in documents
        if document.detected_type == DocumentType.PASSPORT_BIODATA.value
        or (
            document.detected_type == DocumentType.UNKNOWN.value
            and document.expected_type == DocumentType.PASSPORT_BIODATA.value
        )
    }
    candidates[:] = [
        candidate for candidate in candidates
        if candidate.field_path != "personal_info.last_name"
        or candidate.source_document in passport_documents
    ]


def _recover_tourist_last_name_from_verified_names(
    result: ExtractionResult, candidates: list[FieldCandidate],
) -> None:
    """Resolve an OCR-only surname conflict from independent bundle evidence.

    Passport name rows carry no check digit. A single OCR substitution can
    therefore affect both the printed surname and the MRZ name while every
    numeric MRZ check still passes. If the already-verified full name and given
    names isolate one family name, and a *different uploaded document* carries
    that exact family name (commonly its PDF417 barcode), the bundle has two
    independent reasons for the value and no reason to leave the CRM field
    blank.

    The cross-document requirement is deliberate. Without it, deriving one
    field from two other readings of the same page would merely repeat the same
    OCR mistake and disguise it as corroboration.
    """
    if result.customer_type != CustomerType.TOURIST:
        return
    last_metadata = result.field_metadata.get("personal_info.last_name")
    first_metadata = result.field_metadata.get("personal_info.first_name")
    full_metadata = result.field_metadata.get("personal_info.full_name")
    if (
        last_metadata is None
        or last_metadata.status not in {FieldStatus.CONFLICTING, FieldStatus.MISSING}
        or first_metadata is None or first_metadata.status not in PROVEN_STATUSES
        or full_metadata is None or full_metadata.status not in PROVEN_STATUSES
        or not result.personal_info.first_name
        or not result.personal_info.full_name
    ):
        return

    def words(value: str) -> tuple[list[str], list[str]]:
        original = [
            token.strip(" ,.;:/\\|()[]{}") for token in value.split()
            if token.strip(" ,.;:/\\|()[]{}")
        ]
        return original, [fold_for_match(token) for token in original]

    full_words, folded_full = words(result.personal_info.full_name)
    _first_words, folded_first = words(result.personal_info.first_name)
    if not folded_first or len(folded_full) <= len(folded_first):
        return
    if folded_full[:len(folded_first)] == folded_first:
        family_words = full_words[len(folded_first):]
    elif folded_full[-len(folded_first):] == folded_first:
        family_words = full_words[:-len(folded_first)]
    else:
        return
    family_name = " ".join(family_words).strip()
    folded_family = " ".join(words(family_name)[1])
    if not folded_family:
        return

    corroborating = [
        candidate for candidate in candidates
        if candidate.field_path == "personal_info.last_name"
        and candidate.normalized_value
        and " ".join(words(str(candidate.normalized_value))[1]) == folded_family
        and candidate.validation_passed is not False
        and candidate.source_document != full_metadata.source_document
    ]
    if not corroborating:
        return
    support = max(corroborating, key=lambda candidate: candidate.confidence)
    confidence = min(
        first_metadata.confidence or 0.0,
        full_metadata.confidence or 0.0,
        support.confidence,
    )
    result.personal_info.last_name = family_name
    result.field_metadata["personal_info.last_name"] = FieldMetadata(
        status=FieldStatus.VERIFIED,
        confidence=max(0.85, confidence),
        confidence_components={
            "verified_full_name": full_metadata.confidence,
            "verified_given_names": first_metadata.confidence,
            "independent_document": support.confidence,
        },
        source_document=full_metadata.source_document,
        source_method="document_parser",
        evidence_text=full_metadata.evidence_text,
        bounding_box=full_metadata.bounding_box,
        alternate_candidates=last_metadata.alternate_candidates,
        validation_results=[
            "DERIVED_FROM_VERIFIED_FULL_NAME_AND_GIVEN_NAMES",
            "CROSS_DOCUMENT_SURNAME_MATCH",
        ],
    )
    result.warnings.append("LAST_NAME_RECOVERED_FROM_CROSS_DOCUMENT_NAME_AGREEMENT")


def _complete_tourist_surname_from_partial_mrz(
    result: ExtractionResult, candidates: list[FieldCandidate],
) -> None:
    """Restore characters hidden by glare from a tourist's printed surname.

    A left-cropped MRZ cannot prove its complete zone, but its name row remains
    an independent reading. Use it only when the labelled given name agrees
    exactly and the printed surname is the MRZ surname with no more than two
    characters dropped. The completed value stays ``NEEDS_REVIEW`` because no
    passport checksum covers a name.
    """
    current = result.personal_info.last_name
    first_name = result.personal_info.first_name
    metadata = result.field_metadata.get("personal_info.last_name")
    if (
        result.customer_type != CustomerType.TOURIST
        or not current or not first_name or metadata is None
        or metadata.source_method != "labelled_ocr"
    ):
        return
    marker = "MRZ_LEFT_CROP_RESTORED_FROM_PRINTED_PASSPORT_NUMBER"
    partial_first_names = {
        (candidate.source_document, fold_for_match(str(candidate.normalized_value)))
        for candidate in candidates
        if candidate.field_path == "personal_info.first_name"
        and candidate.source_method == "mrz"
        and candidate.normalized_value
        and marker in candidate.warnings
    }
    options = [
        candidate for candidate in candidates
        if candidate.field_path == "personal_info.last_name"
        and candidate.source_method == "mrz"
        and candidate.normalized_value
        and marker in candidate.warnings
        and (
            candidate.source_document, fold_for_match(first_name),
        ) in partial_first_names
        and _is_character_completion(
            current, str(candidate.normalized_value),
        )
    ]
    if not options:
        return
    selected = max(
        options,
        key=lambda candidate: (
            len(str(candidate.normalized_value)), candidate.confidence,
        ),
    )
    completed = str(selected.normalized_value).strip()
    result.personal_info.last_name = completed
    result.field_metadata["personal_info.last_name"] = FieldMetadata(
        status=FieldStatus.NEEDS_REVIEW,
        confidence=max(metadata.confidence or 0.0, selected.confidence),
        confidence_components={
            "printed_surname": metadata.confidence,
            "partial_mrz_name_row": selected.confidence,
            "printed_given_name_match": True,
        },
        source_document=selected.source_document,
        source_method="document_parser",
        evidence_text=(
            f"printed:{metadata.evidence_text or current};mrz:{completed}"
        ),
        bounding_box=metadata.bounding_box,
        alternate_candidates=[
            *metadata.alternate_candidates,
            {
                "field_path": "personal_info.last_name",
                "value": current,
                "normalized_value": current,
                "source_document": metadata.source_document,
                "source_method": metadata.source_method,
                "confidence": metadata.confidence,
                "evidence_text": metadata.evidence_text,
                "bounding_box": metadata.bounding_box,
                "validation_passed": True,
                "warnings": ["PRINTED_SURNAME_MISSING_CHARACTERS"],
            },
        ],
        validation_results=[
            "PARTIAL_MRZ_COMPLETES_PRINTED_SURNAME",
            "PRINTED_GIVEN_NAME_MATCH",
        ],
        reason_for_review=(
            "Partial MRZ restored characters obscured in the printed surname; "
            "operator confirmation is required because names have no checksum"
        ),
    )
    result.warnings.append("TOURIST_SURNAME_COMPLETED_FROM_PARTIAL_MRZ")


def _show_tourist_passport_last_name_for_review(
    result: ExtractionResult, candidates: list[FieldCandidate],
    documents: list[DocumentRecord],
) -> None:
    """Show the best passport surname when disagreement prevents verification.

    ``CONFLICTING`` used to mean both "do not trust this automatically" and
    "erase the value from the operator's form". The latter is counterproductive:
    the evidence inspector can already show several readings, but the operator
    has to retype the surname even when the passport's printed row and MRZ both
    saw it. Keep the review requirement and alternates, while putting the best
    identity-document reading in the editable field.

    A driving-licence parser is deliberately excluded here. It can still
    corroborate a passport through the stricter recovery above, but when the
    two documents disagree the passport is the tourist identity source.
    """
    metadata = result.field_metadata.get("personal_info.last_name")
    if (
        result.customer_type != CustomerType.TOURIST
        or metadata is None
        or metadata.status != FieldStatus.CONFLICTING
        or result.personal_info.last_name
    ):
        return
    passport_documents = {
        document.upload_id for document in documents
        if document.detected_type == DocumentType.PASSPORT_BIODATA.value
    }
    options = [
        candidate for candidate in candidates
        if candidate.field_path == "personal_info.last_name"
        and candidate.source_document in passport_documents
        and candidate.normalized_value
        and candidate.validation_passed is not False
        and candidate.source_method not in _GUESSED_SOURCE_METHODS
    ]
    if not options:
        return

    def clean(value: str) -> str:
        # OCR often appends the passport background's bullet or a comma to the
        # final letter. Remove only non-alphanumeric edge marks; punctuation
        # inside a real family name remains untouched.
        return re.sub(r"^\W+|\W+$", "", " ".join(value.split()), flags=re.UNICODE)

    def method_group(method: str) -> str:
        return "barcode" if method.startswith("barcode:") else method

    def support(candidate: FieldCandidate) -> set[str]:
        value = clean(str(candidate.normalized_value))
        return {
            method_group(other.source_method) for other in options
            if other is not candidate
            and (
                similarity := name_similarity(
                    value, clean(str(other.normalized_value)),
                )
            ) is not None
            and similarity >= 0.88
        }

    selected = max(
        options,
        key=lambda candidate: (
            len(support(candidate)),
            candidate.source_method == "mrz",
            candidate.confidence,
        ),
    )
    value = clean(str(selected.normalized_value))
    if len(value) < 2 or any(character.isdigit() for character in value):
        return
    supporters = support(selected)
    result.personal_info.last_name = value
    result.field_metadata["personal_info.last_name"] = FieldMetadata(
        status=FieldStatus.NEEDS_REVIEW,
        confidence=selected.confidence,
        confidence_components={
            "identity_document_read": selected.confidence,
            "independent_name_methods": sorted(supporters),
        },
        source_document=selected.source_document,
        source_method=selected.source_method,
        evidence_text=selected.evidence_text,
        bounding_box=selected.bounding_box,
        alternate_candidates=[
            candidate.model_dump() for candidate in candidates
            if candidate.field_path == "personal_info.last_name"
            and candidate is not selected
            and candidate.normalized_value
        ],
        validation_results=[
            "PASSPORT_LAST_NAME_SHOWN_FOR_OPERATOR_REVIEW",
            *(["PRINTED_NAME_AND_MRZ_AGREE"] if supporters else []),
        ],
        reason_for_review=(
            "Best passport reading shown; another uploaded document produced "
            "a competing surname, so operator confirmation is required"
        ),
    )
    result.warnings.append("LAST_NAME_SHOWN_FROM_PASSPORT_REQUIRES_REVIEW")


def _show_tourist_passport_first_name_for_review(
    result: ExtractionResult, candidates: list[FieldCandidate],
    documents: list[DocumentRecord],
) -> None:
    """Keep a readable passport given name visible when its readings conflict.

    Printed names and MRZ names carry no check digit, so disagreement must stay
    reviewable.  It must not, however, erase every reading from the operator's
    form.  As with the surname rule above, no licence or permit value can fill
    this identity field.
    """
    metadata = result.field_metadata.get("personal_info.first_name")
    if (
        result.customer_type != CustomerType.TOURIST
        or metadata is None
        or metadata.status != FieldStatus.CONFLICTING
        or result.personal_info.first_name
    ):
        return
    passport_documents = {
        document.upload_id for document in documents
        if document.detected_type == DocumentType.PASSPORT_BIODATA.value
    }
    options = [
        candidate for candidate in candidates
        if candidate.field_path == "personal_info.first_name"
        and candidate.source_document in passport_documents
        and candidate.normalized_value
        and candidate.validation_passed is not False
        and candidate.source_method not in _GUESSED_SOURCE_METHODS
    ]
    if not options:
        return

    def clean(value: str) -> str:
        return re.sub(r"^\W+|\W+$", "", " ".join(value.split()), flags=re.UNICODE)

    def mrz_repeated_suffix_is_artifact(candidate: FieldCandidate) -> bool:
        """Reject a three-letter MRZ tail contradicted by the printed row.

        The reported Spanish passport's otherwise usable name row ended
        ``KENNETH<<<<<<<<<SSS``.  Names have no MRZ check digit, so the parser
        faithfully exposed ``KENNETH SSS`` although the labelled Given Names
        row directly above said ``KENNETH``.  A labelled name may settle only
        this unmistakable bleed: it must be the exact token prefix and the MRZ
        may add exactly one three-times-repeated token.  Real extra names and
        every ordinary MRZ/printed-name disagreement remain reviewable.
        """
        if candidate.source_method != "mrz" or not candidate.normalized_value:
            return False
        mrz_tokens = clean(str(candidate.normalized_value)).upper().split()
        if len(mrz_tokens) < 2 or re.fullmatch(r"([A-Z])\1\1", mrz_tokens[-1]) is None:
            return False
        return any(
            other is not candidate
            and other.source_method != "mrz"
            and other.normalized_value
            and (
                printed_tokens := clean(str(other.normalized_value)).upper().split()
            ) == mrz_tokens[:-1]
            for other in options
        )

    filtered = [
        candidate for candidate in options
        if not mrz_repeated_suffix_is_artifact(candidate)
    ]
    if filtered:
        options = filtered

    def method_group(method: str) -> str:
        return "barcode" if method.startswith("barcode:") else method

    def support(candidate: FieldCandidate) -> set[str]:
        value = clean(str(candidate.normalized_value))
        return {
            method_group(other.source_method) for other in options
            if other is not candidate
            and (
                similarity := name_similarity(
                    value, clean(str(other.normalized_value)),
                )
            ) is not None
            and similarity >= 0.88
        }

    selected = max(
        options,
        key=lambda candidate: (
            len(support(candidate)), candidate.source_method == "mrz",
            candidate.confidence,
        ),
    )
    value = clean(str(selected.normalized_value))
    if not value or any(character.isdigit() for character in value):
        return
    supporters = support(selected)
    result.personal_info.first_name = value
    result.field_metadata["personal_info.first_name"] = FieldMetadata(
        status=FieldStatus.NEEDS_REVIEW,
        confidence=selected.confidence,
        confidence_components={
            "identity_document_read": selected.confidence,
            "independent_name_methods": sorted(supporters),
        },
        source_document=selected.source_document,
        source_method=selected.source_method,
        evidence_text=selected.evidence_text,
        bounding_box=selected.bounding_box,
        alternate_candidates=[
            candidate.model_dump() for candidate in candidates
            if candidate.field_path == "personal_info.first_name"
            and candidate is not selected
            and candidate.normalized_value
        ],
        validation_results=[
            "PASSPORT_FIRST_NAME_SHOWN_FOR_OPERATOR_REVIEW",
            *(["PRINTED_NAME_AND_MRZ_AGREE"] if supporters else []),
        ],
        reason_for_review=(
            "Best passport reading shown; competing given-name evidence "
            "requires operator confirmation"
        ),
    )
    result.warnings.append("FIRST_NAME_SHOWN_FROM_PASSPORT_REQUIRES_REVIEW")


def _complete_tourist_first_name_from_matching_licence(
    result: ExtractionResult, candidates: list[FieldCandidate],
    documents: list[DocumentRecord],
) -> None:
    """Complete a truncated passport given-name read from a matching licence.

    The tourist passport remains the identity source.  This very narrow repair
    is for a passport that supplies the beginning of the given-name sequence
    while the matching national licence visibly supplies the remaining names:
    ``KENNETH`` on the passport and ``KENNETH YANNICK`` on the Andorran card.
    It never replaces a passport value with a different licence name.  The
    passport surname and the licence surname must agree exactly, and the
    passport tokens must be a strict prefix of the licence tokens.  Names carry
    no MRZ check digit, so the completed value remains reviewable.
    """
    if (
        result.customer_type != CustomerType.TOURIST
        or not result.personal_info.first_name
        or not result.personal_info.last_name
    ):
        return
    metadata = result.field_metadata.get("personal_info.first_name")
    if metadata is None:
        return
    passport_documents = {
        document.upload_id for document in documents
        if document.detected_type == DocumentType.PASSPORT_BIODATA.value
    }
    licence_documents = {
        document.upload_id for document in documents
        if document.detected_type in {
            DocumentType.NATIONAL_DRIVING_LICENCE_FRONT.value,
            DocumentType.NATIONAL_DRIVING_LICENCE_BACK.value,
        }
    }
    if metadata.source_document not in passport_documents or not licence_documents:
        return

    def tokens(value: str | None) -> list[str]:
        return [
            fold_for_match(token)
            for token in (value or "").replace(",", " ").split()
            if fold_for_match(token)
        ]

    passport_value = result.personal_info.first_name
    passport_given = tokens(passport_value)
    passport_surname = tokens(result.personal_info.last_name)
    if not passport_given or not passport_surname:
        return
    licence_surnames = {
        candidate.source_document
        for candidate in candidates
        if candidate.source_document in licence_documents
        and candidate.field_path == "personal_info.last_name"
        and candidate.normalized_value
        and candidate.validation_passed is not False
        and tokens(str(candidate.normalized_value)) == passport_surname
    }
    options = [
        candidate for candidate in candidates
        if candidate.source_document in licence_surnames
        and candidate.field_path == "personal_info.first_name"
        and candidate.normalized_value
        and candidate.validation_passed is not False
        and len(licence_given := tokens(str(candidate.normalized_value))) > len(passport_given)
        and licence_given[:len(passport_given)] == passport_given
    ]
    if not options:
        return
    selected = max(
        options,
        key=lambda candidate: (
            len(tokens(str(candidate.normalized_value))), candidate.confidence,
        ),
    )
    value = _normalize_tourist_name_value(str(selected.normalized_value))
    if not value:
        return
    result.personal_info.first_name = value
    result.field_metadata["personal_info.first_name"] = FieldMetadata(
        status=FieldStatus.NEEDS_REVIEW,
        confidence=min(metadata.confidence or 0.0, selected.confidence),
        confidence_components={
            "passport_given_name": metadata.confidence,
            "matching_licence_given_names": selected.confidence,
            "passport_and_licence_surname_match": True,
        },
        source_document=selected.source_document,
        source_method="document_parser",
        evidence_text=(
            f"passport:{passport_value};"
            f"licence:{selected.evidence_text or selected.normalized_value}"
        ),
        bounding_box=selected.bounding_box,
        alternate_candidates=[
            *metadata.alternate_candidates,
            {
                "field_path": "personal_info.first_name",
                "value": passport_value,
                "normalized_value": passport_value,
                "source_document": metadata.source_document,
                "source_method": metadata.source_method,
                "confidence": metadata.confidence,
                "evidence_text": metadata.evidence_text,
                "bounding_box": metadata.bounding_box,
                "validation_passed": True,
                "warnings": ["PARTIAL_PASSPORT_GIVEN_NAME"],
            },
        ],
        validation_results=[
            "PASSPORT_GIVEN_NAME_PREFIX_MATCHES_LICENCE",
            "PASSPORT_AND_LICENCE_SURNAME_MATCH",
            "LICENCE_COMPLETES_PASSPORT_GIVEN_NAMES",
        ],
        reason_for_review=(
            "A matching licence completed given names that the passport read only partially"
        ),
    )
    result.warnings.append("TOURIST_GIVEN_NAMES_COMPLETED_FROM_MATCHING_LICENCE")


def _suggest_gender_from_name(result: ExtractionResult) -> None:
    """Pre-position the operator's gender choice when no document carries one.

    Several GCC cards print no sex field anywhere, so on those the value exists
    on no page and the operator starts from an empty dropdown every time. The
    suggestion pre-positions that choice from the holder's own name. It costs
    no model call and no measurable time, and it is still written at
    NEEDS_REVIEW: ``ProcessingSession.confirm`` refuses it until an operator
    has confirmed it, so nothing inferred reaches a contract unreviewed.
    """
    if result.personal_info.gender: return
    suggestion = next(
        (
            guess for guess in (
                gender_from_name(result.personal_info.first_name),
                gender_from_name(result.personal_info.full_name),
                gender_from_name(result.personal_info.full_name_arabic),
            ) if guess
        ),
        None,
    )
    if suggestion is None: return
    result.personal_info.gender = suggestion
    result.field_metadata["personal_info.gender"] = FieldMetadata(
        status=FieldStatus.NEEDS_REVIEW, confidence=0.2,
        confidence_components={"source_reliability": 0.2},
        source_document="derived", source_method="name_inference",
        validation_results=["GENDER_GUESSED_FROM_GIVEN_NAME"],
        reason_for_review=(
            "Guessed from the given name because no uploaded document carries a "
            "sex field; confirm with the customer before use"
        ),
    )
    result.warnings.append("GENDER_INFERRED_FROM_NAME")


def _promote_agreeing_gcc_identifier(
    result: ExtractionResult, candidates: list[FieldCandidate],
) -> None:
    """Verify an identifier that two independent reads agree on.

    Where a state prints the holder's national number on both cards, the two
    documents having been read separately and having produced the same digits
    is genuine corroboration — unlike copying one into the other, which proves
    nothing. Only then is the pair promoted to VERIFIED.
    """
    identifier = result.gcc_identity.number
    if not identifier or result.gcc_driving_licence.number != identifier:
        return
    sources: dict[str, set[str]] = {
        "gcc_identity.number": set(), "gcc_driving_licence.number": set(),
    }
    for candidate in candidates:
        if (
            candidate.field_path in sources
            and candidate.normalized_value == identifier
            and candidate.validation_passed is not False
        ):
            sources[candidate.field_path].add(candidate.source_document)
    if not all(sources.values()):
        return
    if not any(
        identity_source != licence_source
        for identity_source in sources["gcc_identity.number"]
        for licence_source in sources["gcc_driving_licence.number"]
    ):
        return
    validation = f"CROSS_DOCUMENT_IDENTIFIER_MATCH:{result.gcc_profile.iso3}"
    for path in sources:
        metadata = result.field_metadata.get(path)
        if metadata is None:
            continue
        metadata.status = FieldStatus.VERIFIED
        metadata.confidence = max(metadata.confidence or 0.0, 0.90)
        if validation not in metadata.validation_results:
            metadata.validation_results.append(validation)
        metadata.reason_for_review = None


# Retained for callers written against the previous function name.
def _promote_shared_gcc_identifier_agreement(
    result: ExtractionResult,
    candidates: list[FieldCandidate],
    gcc_profile: GCCDocumentProfile | None = None,
) -> None:
    _promote_agreeing_gcc_identifier(result, candidates)


def _format_uk_driving_licence_number(
    value: str | None, country: str | None,
) -> str | None:
    """Keep the British two-digit issue number visibly separated.

    A UK photocard prints a 16-character driver number followed by a space and
    its two-digit issue number (for example ``DEGHA009131Z99LM 52``). Generic
    serial cleanup deliberately removes OCR-inserted spaces, so restore this
    one country-defined separator only after the licence country is known.
    """
    if not value or country not in {"United Kingdom", "GBR"}:
        return value
    compact = re.sub(r"\s+", "", value.strip().upper())
    match = re.fullmatch(r"([A-Z0-9]{16})(\d{2})", compact)
    if match is None:
        return value
    return f"{match.group(1)} {match.group(2)}"


def _preserve_uk_driving_licence_issue_number_space(
    result: ExtractionResult, country: str | None,
) -> None:
    """Apply the UK display/export format without changing extracted digits."""
    current = result.national_driving_licence.number
    formatted = _format_uk_driving_licence_number(current, country)
    if not formatted or formatted == current:
        return
    result.national_driving_licence.number = formatted
    metadata = result.field_metadata.get("national_driving_licence.number")
    validation = "UK_LICENCE_ISSUE_NUMBER_SPACE_PRESERVED"
    if metadata is not None and validation not in metadata.validation_results:
        metadata.validation_results.append(validation)


def workflow_export_payload(result: ExtractionResult) -> dict[str, Any]:
    """Return the customer-safe final payload for the selected workflow.

    Thirteen keys for a GCC customer and eighteen for a tourist, always present,
    always a string or null. Dates are ISO ``YYYY-MM-DD``; identity, licence and
    passport numbers stay strings so a Bahraini CPR or a passport number
    beginning with a zero survives the round trip.
    """
    if result.customer_type == CustomerType.UAE_RESIDENT:
        # A UAE resident supplies an Emirates ID and UAE driving licence. Do
        # not expose a blank passport section, its metadata, or place of birth
        # in the final JSON for that workflow.
        return result.model_dump(mode="json", exclude={
            "personal_info": {"place_of_birth"},
            "passport": True,
            "field_metadata": {
                "personal_info.place_of_birth",
                "passport.number",
                "passport.issued_by_code",
                "passport.issued_by_name",
                "passport.issue_date",
                "passport.expiry_date",
                "passport.holder_id",
            },
        })
    workflow: tuple[list[str], dict[str, str]] | None = {
        CustomerType.GCC_NATIONAL: (GCC_CUSTOMER_FIELD_PATHS, GCC_EXPORT_KEYS),
        CustomerType.TOURIST: (TOURIST_CUSTOMER_FIELD_PATHS, TOURIST_EXPORT_KEYS),
    }.get(result.customer_type)
    if workflow is None:
        return result.model_dump(mode="json")
    paths, keys = workflow
    payload: dict[str, Any] = {}
    for path in paths:
        section, attribute = path.split(".", 1)
        value = getattr(getattr(result, section), attribute)
        if (
            result.customer_type == CustomerType.TOURIST
            and path in _TOURIST_NAME_PATHS
        ):
            value = _normalize_tourist_name_value(value)
        if (
            result.customer_type == CustomerType.TOURIST
            and path == "national_driving_licence.number"
        ):
            value = _format_uk_driving_licence_number(
                value, result.licence_policy.country,
            )
        payload[keys[path]] = str(value) if value else None
    return payload


WRONG_CUSTOMER_TYPE = "WRONG_CUSTOMER_TYPE_FOR_DOCUMENT"


_NUMBERED_LICENCE_FRONT_SIGNATURE = frozenset({
    ("personal_info.date_of_birth", "STANDARD_FIELD_DESIGNATOR:3"),
    ("national_driving_licence.issue_date", "STANDARD_FIELD_DESIGNATOR:4A"),
    ("national_driving_licence.expiry_date", "STANDARD_FIELD_DESIGNATOR:4B"),
    ("national_driving_licence.number", "STANDARD_FIELD_DESIGNATOR:5"),
})


def _has_numbered_licence_front_signature(
    candidates: list[FieldCandidate],
) -> bool:
    """Whether a page proves the EU/Vienna numbered front is present.

    A combined front/back photograph can score as the reverse because the
    lower half contains the very clear words CATEGORIES, CLASS and
    RESTRICTIONS while glare or OCR damage loses the title on the upper half.
    The values explicitly anchored to 3, 4a, 4b and 5 are stronger structural
    evidence than those generic reverse headings. Requiring all four prevents
    a category table, field legend or barcode-only back from being promoted.
    """
    observed = {
        (candidate.field_path, warning)
        for candidate in candidates
        if candidate.normalized_value
        and candidate.validation_passed is not False
        for warning in candidate.warnings
    }
    return _NUMBERED_LICENCE_FRONT_SIGNATURE <= observed


def _mismatched_issuer(
    lines: list[OCRLine], customer: CustomerType,
    gcc_profile: GCCDocumentProfile | None,
) -> str | None:
    """Return the state that issued this page when it is the wrong workflow.

    A Saudi national ID and an Emirates ID both print رقم الهوية, and both
    licences print DRIVING LICENSE, so on wording alone the UAE Resident route
    will happily read a Saudi card and write its ten-digit record number into
    an Emirates ID field. The issuing authority named on the card settles it.
    """
    if customer == CustomerType.TOURIST:
        return None
    states = issuing_states(" ".join(line.text for line in lines))
    if not states:
        return None
    expected = (
        "United Arab Emirates" if customer == CustomerType.UAE_RESIDENT
        else (gcc_profile.country if gcc_profile else None)
    )
    if expected is None or expected in states:
        return None
    return sorted(states)[0]


_GCC_LICENCE_BACK_MARKERS = (
    "AUTHORIZED VEHICLES", "LICENSING AUTHORITY",
    "المركبات المصرح بقيادتها", "سلطة الترخيص",
)
_GCC_IDENTITY_BACK_MARKERS = (
    "HOLDER'S SIGNATURE", "AUTHORITY'S SIGNATURE", "SERIAL NO",
    "توقيع حامل البطاقة", "الرقم المسلسل",
)


def _gcc_back_only_page(lines: list[OCRLine]) -> bool:
    """Avoid spending front-routing retries on an unmistakable GCC reverse."""
    text = " ".join(line.text for line in lines).upper()
    return (
        any(marker.upper() in text for marker in _GCC_LICENCE_BACK_MARKERS)
        or sum(
            marker.upper() in text for marker in _GCC_IDENTITY_BACK_MARKERS
        ) >= 2
    )


def _driving_document_prefixes(
    presented_idp: bool, presented_national: bool,
) -> list[str]:
    prefixes = []
    if presented_idp:
        prefixes.append("international_driving_permit")
    if presented_national:
        prefixes.append("national_driving_licence")
    return prefixes or ["national_driving_licence"]


def _clear_non_country_issuer(result: ExtractionResult, prefix: str) -> None:
    name_path = f"{prefix}.issued_by_name"
    section, attribute = name_path.split(".")
    value = getattr(getattr(result, section), attribute)
    if not value:
        return
    code, _, _ = normalize_country(value)
    if code is not None:
        return
    _set_path(result, name_path, None)
    _set_path(result, f"{prefix}.issued_by_code", None)
    result.field_metadata[name_path] = FieldMetadata(
        status=FieldStatus.MISSING,
        confidence=None,
        confidence_components={},
        source_document="",
        source_method="",
        evidence_text=value,
        validation_results=["ISSUING_COUNTRY_NOT_PROVEN_BY_DOCUMENT"],
        reason_for_review=(
            "The page names an issuing authority, not a country; "
            "operator confirmation is required"
        ),
    )


def _mrz_names_a_passport(mrz: Any) -> bool:
    """Whether the page's own machine-readable zone says it is a passport.

    A page is routed to the passport reader on the strength of the word
    printed at its head, and on an Indian booklet whose captions came back as
    "Paupert No", "Surmame" and "Given Name(s)" that word was nowhere on the
    page. Every passport field was read and every one of them was thrown away:
    the page was filed as a driving licence on the strength of its file
    number, whose "DL" prefix is a passport office, not a licence.

    The zone settles it without any of that. ICAO fixes the first character of
    a TD2/TD3 zone as the document class, and the composite check digit covers
    the row it stands in, so a P that passes it is the document saying what it
    is. The same reasoning already decides an identity card against a licence
    where a GCC card prints both titles.
    """
    return bool(
        mrz is not None
        and getattr(mrz, "mrz_type", None) in {"TD2", "TD3"}
        and getattr(mrz, "checks", {}).get("composite")
        and str(getattr(mrz, "fields", {}).get("document_code", "")).upper().startswith("P")
    )


def _auto_document_type(
    lines: list[OCRLine], detected: DocumentType, customer: CustomerType,
    policy: CountryLicencePolicy | None, source: str,
    barcodes: list[Any] | None = None,
    gcc_profile: GCCDocumentProfile | None = None,
    cache: _CandidateCache | None = None,
    mrz: Any = None,
) -> DocumentType:
    """Route a page from the unified upload without trusting file names/order."""
    if cache is None:
        cache = _CandidateCache(source, policy.country if policy else None)

    def candidates_for(doc_type: DocumentType) -> list[FieldCandidate]:
        return cache.candidates(lines, doc_type)

    # OCR frequently mixes Latin and Cyrillic lookalikes in bilingual passport
    # headings (``Рassрort``).  Keep the original text too: permit markers
    # such as ``НОМЕР МВУ`` are deliberately Cyrillic and must still reach the
    # IDP detector.  The second folded copy only supplies tolerant matching
    # for headings whose individual glyphs have been mixed by OCR.
    raw_text = " ".join(line.text for line in lines).upper()
    text = f"{raw_text}\n{fold_for_match(raw_text)}"
    passport_markers = (
        "PASSPORT", "PASSEPORT", "REISEPASS", "PASAPORTE", "PASSAPORTO",
        "ПАСПОРТ", "جواز سفر", "护照", "旅券", "여권",
    )
    if customer == CustomerType.GCC_NATIONAL:
        profile = gcc_profile or profile_for_gcc_country(policy.country if policy else None)
        if profile is None:
            return DocumentType.UNKNOWN
        # GCC card banners and their Arabic labels are often returned as two
        # adjacent OCR boxes. Rejoin only profile-known fragments before the
        # page is scored, and tolerate whitespace dropped inside a single title
        # (``DRIVINGLICENSE``). This is deliberately local to GCC routing.
        lines = augment_gcc_ocr_lines(lines, profile.country)
        raw_text = " ".join(line.text for line in lines).upper()
        text = f"{raw_text}\n{fold_for_match(raw_text)}"

        def title_present(markers: tuple[str, ...]) -> bool:
            compact_rows = [
                compact_label(line.text) for line in lines
                if compact_label(line.text)
            ]
            for marker in markers:
                compact_marker = compact_label(marker)
                if (
                    marker.upper() in text
                    or any(
                        compact_marker in compact_row
                        for compact_row in compact_rows
                    )
                ):
                    return True
                # Large GCC card banners are also printed as two or three
                # stacked words. Paddle can assign different OCR languages to
                # those Latin boxes (the Omani report returned IDENTITY as
                # Arabic and CARD as English), so the fragment join cannot
                # safely rely on the language tag. Exact consecutive pieces of
                # a known title are still deterministic and cost no reread.
                for width in (2, 3):
                    if any(
                        "".join(compact_rows[start:start + width])
                        == compact_marker
                        for start in range(len(compact_rows) - width + 1)
                    ):
                        return True
            return False

        licence_title = title_present(profile.licence_titles)
        identity_title = title_present(profile.identity_titles)
        identity_candidates = candidates_for(DocumentType.GCC_IDENTITY_FRONT)
        licence_candidates = candidates_for(DocumentType.GCC_DRIVING_LICENCE_FRONT)
        identity_score = _candidate_score(identity_candidates)
        licence_score = _candidate_score(licence_candidates)
        if licence_title and (identity_title or identity_score >= 3.0):
            # Some GCC identity cards also print driving entitlements.  The
            # reported Omani card therefore contains both ``IDENTITY CARD``
            # and ``VEHICLE DRIVING LICENCE`` plus a three-line TD1 zone.  A
            # checksum-valid TD1 whose document class begins with I is direct
            # evidence that this is the identity document. The identity title
            # itself may be split between OCR language streams, so a strong
            # identity-field score can admit the same decisive check; title
            # order must not turn its civil number and expiry into a licence.
            mrz_rows = {
                *find_mrz_lines(lines), *find_clipped_mrz_lines(lines),
            }
            # Ordinary GCC licences have no card MRZ. Avoid even the in-memory
            # parser unless three plausible TD1 rows are present, so the common
            # clear-front path retains its previous latency.
            card_mrz = (
                _best_mrz_from_ocr(lines) if len(mrz_rows) >= 3 else None
            )
            document_code = str(
                card_mrz.fields.get("document_code") if card_mrz else ""
            ).upper()
            if (
                card_mrz is not None
                and card_mrz.mrz_type == "TD1"
                and card_mrz.valid
                and document_code.startswith("I")
            ):
                return DocumentType.GCC_IDENTITY_FRONT
        if licence_title:
            return DocumentType.GCC_DRIVING_LICENCE_FRONT
        if identity_title:
            return DocumentType.GCC_IDENTITY_FRONT
        if profile.iso3 == "SAU":
            saudi_licence_markers = (
                "LICENSE TYPE", "LICENCE TYPE", "BLOOD TYPE", "BLOOD GROUP", "ISSUE DATE",
            )
            if any(marker in text for marker in saudi_licence_markers):
                return DocumentType.GCC_DRIVING_LICENCE_FRONT
            identity_paths = {
                candidate.field_path for candidate in identity_candidates
                if candidate.normalized_value and candidate.validation_passed is not False
            }
            if {
                "personal_info.full_name", "personal_info.date_of_birth",
                "gcc_identity.expiry_date",
            }.issubset(identity_paths):
                return DocumentType.GCC_IDENTITY_FRONT
        if detected in {
            DocumentType.GCC_DRIVING_LICENCE_FRONT, DocumentType.GCC_DRIVING_LICENCE_BACK,
            DocumentType.UAE_DRIVING_LICENCE_FRONT, DocumentType.UAE_DRIVING_LICENCE_BACK,
            DocumentType.NATIONAL_DRIVING_LICENCE_FRONT, DocumentType.NATIONAL_DRIVING_LICENCE_BACK,
        } and licence_score >= 2.0:
            return (
                DocumentType.GCC_DRIVING_LICENCE_BACK
                if detected.value.endswith("BACK") else DocumentType.GCC_DRIVING_LICENCE_FRONT
            )
        if detected in {
            DocumentType.GCC_IDENTITY_FRONT, DocumentType.GCC_IDENTITY_BACK,
            DocumentType.EMIRATES_ID_FRONT, DocumentType.EMIRATES_ID_BACK,
        } and identity_score >= 2.0:
            return (
                DocumentType.GCC_IDENTITY_BACK
                if detected.value.endswith("BACK") else DocumentType.GCC_IDENTITY_FRONT
            )
        barcode_keys = {
            key for barcode in (barcodes or [])
            for key in barcode.structured_candidate
        }
        if barcode_keys & {"license_number", "licence_number"}:
            return DocumentType.GCC_DRIVING_LICENCE_BACK
        if barcode_keys & {"id_number", "emirates_id"}:
            return DocumentType.GCC_IDENTITY_BACK
        if identity_score >= 3.0 and identity_score >= licence_score:
            return DocumentType.GCC_IDENTITY_FRONT
        if licence_score >= 3.0:
            return DocumentType.GCC_DRIVING_LICENCE_FRONT
        return DocumentType.UNKNOWN
    if customer == CustomerType.UAE_RESIDENT:
        # A resident identity-card back can contain a valid TD1 MRZ. That MRZ
        # proves machine readability, not that the page is a passport. Visible
        # UAE ID evidence must win before the generic MRZ/passport route.
        resident_eid_candidates = candidates_for(DocumentType.EMIRATES_ID_FRONT)
        has_valid_eid_number = any(
            candidate.field_path == "emirates_id.number"
            and candidate.normalized_value
            and candidate.validation_passed is not False
            for candidate in resident_eid_candidates
        )
        has_eid_title = any(marker in text for marker in (
            "EMIRATES ID", "IDENTITY CARD", "RESIDENT IDENTITY CARD",
            "FEDERAL AUTHORITY FOR IDENTITY", "بطاقة هوية", "رقم الهوية",
        ))
        has_uae_licence_title = has_uae_driving_licence_title(raw_text)
        if has_uae_licence_title:
            return DocumentType.UAE_DRIVING_LICENCE_FRONT
        if has_valid_eid_number or has_eid_title:
            return DocumentType.EMIRATES_ID_FRONT
    if customer == CustomerType.TOURIST:
        # A rental photo can show a driving licence resting on an open passport
        # page.  Both titles then appear in the same upload, but the passport
        # has a full biodata layout (number + birth + expiry) lower down the
        # page.  The generic title score favours DRIVING LICENCE because it is
        # large and high contrast, which made every passport field disappear
        # even though OCR had read them.  Three bound biodata rows are enough
        # to identify the actual document page; a passport cover or a random
        # passport word beside a licence cannot satisfy this gate.
        passport_paths = {
            candidate.field_path
            for candidate in candidates_for(DocumentType.PASSPORT_BIODATA)
            if candidate.normalized_value
            and candidate.validation_passed is not False
        }
        has_passport_biodata = (
            (
                any(marker in text for marker in passport_markers)
                or _mrz_names_a_passport(mrz)
            )
            and {
                "passport.number",
                "personal_info.date_of_birth",
                "passport.expiry_date",
            } <= passport_paths
        )
        if has_passport_biodata:
            return DocumentType.PASSPORT_BIODATA
    if detected == DocumentType.PASSPORT_BIODATA:
        has_passport_title = (
            any(marker in text for marker in passport_markers)
            or _mrz_names_a_passport(mrz)
        )
        if customer == CustomerType.TOURIST and not has_passport_title:
            # A permit or a licence misread as a passport. Which of the two it
            # is comes from the page itself: only a permit prints its
            # convention. That is what lets this run before any country is
            # known, and it stays correct when one has been selected.
            # Naming the convention is already decisive document evidence; do
            # not require all three handwritten permit fields to be readable
            # before honoring it. Ghana's booklet clearly said INTERNATIONAL
            # CONVENTION OF 1968 and ISSUE OF PERMIT, but one handwritten date
            # was missed and the other was read with an invalid month. The old
            # all-fields gate therefore left the page classified as a passport,
            # which meant the visual model was never asked for IDP fields.
            if looks_like_idp(text):
                return DocumentType.INTERNATIONAL_DRIVING_PERMIT
            permit_page = (
                policy is not None
                and policy.requirement == LicenceRequirement.NEED_IDL
            )
            if permit_page:
                layout_paths = {
                    candidate.field_path
                    for candidate in idp_layout_candidates(lines, source)
                    if candidate.normalized_value and candidate.validation_passed is not False
                }
                if {
                    "international_driving_permit.number",
                    "international_driving_permit.issue_date",
                    "international_driving_permit.expiry_date",
                } <= layout_paths:
                    return DocumentType.INTERNATIONAL_DRIVING_PERMIT
            if not looks_like_idp(text):
                national_score = _candidate_score(
                    candidates_for(DocumentType.NATIONAL_DRIVING_LICENCE_FRONT),
                )
                if national_score >= 4.5:
                    return DocumentType.NATIONAL_DRIVING_LICENCE_FRONT
        return detected
    if detected in {
        DocumentType.EMIRATES_ID_FRONT, DocumentType.EMIRATES_ID_BACK,
    } and customer != CustomerType.TOURIST:
        # A tourist bundle carries no Emirates ID. Routing a page there would
        # spend a full extraction pass on a card whose fields the workflow does
        # not collect, and would lose whatever the page really is; the driving
        # and passport routing below decides instead.
        return detected
    if detected in DRIVING_DOCUMENT_TYPES:
        if customer == CustomerType.UAE_RESIDENT:
            return (
                DocumentType.UAE_DRIVING_LICENCE_BACK
                if detected in {DocumentType.UAE_DRIVING_LICENCE_BACK, DocumentType.NATIONAL_DRIVING_LICENCE_BACK}
                else DocumentType.UAE_DRIVING_LICENCE_FRONT
            )
        if customer == CustomerType.TOURIST:
            # The page decides. A permit names its convention; a national card
            # does not. Where the country is already known to require a permit,
            # the weaker fragments below are also allowed to settle which page
            # of an otherwise unlabelled booklet this is.
            if detected == DocumentType.INTERNATIONAL_DRIVING_PERMIT or looks_like_idp(text):
                return DocumentType.INTERNATIONAL_DRIVING_PERMIT
            if (
                policy is not None
                and policy.requirement == LicenceRequirement.NEED_IDL
                and any(marker in text for marker in IDP_WEAK_MARKERS)
            ):
                return DocumentType.INTERNATIONAL_DRIVING_PERMIT
            if (
                detected in {
                    DocumentType.UAE_DRIVING_LICENCE_BACK,
                    DocumentType.NATIONAL_DRIVING_LICENCE_BACK,
                }
                and (
                    _has_numbered_licence_front_signature(
                        candidates_for(DocumentType.NATIONAL_DRIVING_LICENCE_FRONT),
                    )
                    or (
                        # AAMVA fronts can lose their title and 4d but still
                        # retain DOB plus the explicitly labelled 4a/4b dates.
                        # That is front evidence, whereas a real reverse is
                        # deliberately ignored by the Tourist workflow.
                        "USA" in text
                        and re.search(r"\b4\s*A(?:\b|\s)*(?:ISS|ISSUED)", text)
                        and re.search(r"\b4\s*B(?:\b|\s)*(?:EXP|EXPIRES)", text)
                        and (
                            bool(american_unlabelled_licence_number_candidates(
                                lines, source,
                            ))
                            # Or the holder's birth date beside them. A real
                            # reverse carries none of the three -- it is a
                            # table of classes and endorsements -- so the
                            # trio is front evidence by itself. A Florida
                            # licence labelled its number "DLN", which left
                            # the unlabelled recovery above nothing to find,
                            # and the page was routed as a reverse: every
                            # value on it was then deliberately ignored and
                            # the licence came back blank.
                            or re.search(
                                r"(?<![A-Z0-9])3\s*[.):\-]?\s*DOB\b",
                                text, re.I,
                            ) is not None
                        )
                    )
                )
            ):
                # Both sides are present, so route as the front: that is where
                # the customer's number and dated identity fields live. The
                # category parser still sees the reverse half on the same page.
                return DocumentType.NATIONAL_DRIVING_LICENCE_FRONT
            return (
                DocumentType.NATIONAL_DRIVING_LICENCE_BACK
                if detected in {DocumentType.UAE_DRIVING_LICENCE_BACK, DocumentType.NATIONAL_DRIVING_LICENCE_BACK}
                else DocumentType.NATIONAL_DRIVING_LICENCE_FRONT
            )
        return detected

    barcode_keys = {
        key for barcode in (barcodes or [])
        for key in barcode.structured_candidate
    }
    if "passport_number" in barcode_keys:
        return DocumentType.PASSPORT_BIODATA
    if barcode_keys & {"id_number", "emirates_id"} and customer != CustomerType.TOURIST:
        return DocumentType.EMIRATES_ID_BACK
    if barcode_keys & {"license_number", "licence_number"}:
        if customer == CustomerType.UAE_RESIDENT:
            return DocumentType.UAE_DRIVING_LICENCE_BACK
        if policy and policy.requirement != LicenceRequirement.NEED_IDL:
            return DocumentType.NATIONAL_DRIVING_LICENCE_BACK

    if customer == CustomerType.TOURIST and (
        looks_like_idp(text)
        or (policy is not None and policy.requirement == LicenceRequirement.NEED_IDL)
    ):
        layout_paths = {
            candidate.field_path
            for candidate in idp_layout_candidates(lines, source)
            if candidate.normalized_value and candidate.validation_passed is not False
        }
        required_layout = {
            "international_driving_permit.number",
            "international_driving_permit.issue_date",
            "international_driving_permit.expiry_date",
        }
        if required_layout <= layout_paths:
            return DocumentType.INTERNATIONAL_DRIVING_PERMIT

    plausible: list[DocumentType] = []
    if any(marker in text for marker in passport_markers):
        plausible.append(DocumentType.PASSPORT_BIODATA)
    if customer == CustomerType.UAE_RESIDENT:
        plausible.extend([
            DocumentType.EMIRATES_ID_FRONT, DocumentType.EMIRATES_ID_BACK,
            DocumentType.UAE_DRIVING_LICENCE_FRONT,
        ])
    elif policy and policy.requirement == LicenceRequirement.NEED_IDL:
        plausible.append(DocumentType.INTERNATIONAL_DRIVING_PERMIT)
    elif policy is None and looks_like_idp(text):
        plausible.append(DocumentType.INTERNATIONAL_DRIVING_PERMIT)
    else:
        # With no country selected, a permit has to name itself to be scored as
        # one. Letting the permit compete here on evidence alone made every
        # ordinary licence a candidate, because the permit scorer falls back to
        # "three or more dates in sequence" -- which is simply what a driving
        # licence looks like. A Brazilian CNH was read as a permit that way, and
        # since Brazil requires a permit, the licence silently satisfied the very
        # rule it was supposed to fail.
        plausible.append(DocumentType.NATIONAL_DRIVING_LICENCE_FRONT)
    scored: list[tuple[float, DocumentType]] = []
    for doc_type in plausible:
        scored.append((_candidate_score(candidates_for(doc_type)), doc_type))
    best_score, best_type = max(scored, key=lambda item: item[0])
    return best_type if best_score >= 2.0 else DocumentType.UNKNOWN


TOURIST_DRIVING_TYPES = {
    DocumentType.NATIONAL_DRIVING_LICENCE_FRONT,
    DocumentType.NATIONAL_DRIVING_LICENCE_BACK,
    DocumentType.INTERNATIONAL_DRIVING_PERMIT,
}


_NON_ISSUER_COUNTRY_ROW_LABELS = (
    "NATIONALITY", "NATIONALITE", "NATIONALITAT", "STAATSANGEHORIGKEIT",
    "PLACE OF BIRTH", "COUNTRY OF BIRTH", "RESIDENCE", "ADDRESS",
    # A bilingual Gulf card captions the row in Arabic beside its English
    # half, and the English half is what glare takes first: the reported Saudi
    # licence returned "الجنسية فلسطين" whole and its Latin twin as the stump
    # "NAL PALESTINE". With only English wording listed, the holder's
    # nationality counted as a second issuer, cancelled the card's own
    # "KINGDOM OF SAUDI ARABIA" heading, and the country fell back to the
    # passport -- naming Palestine as the state that issued a Saudi licence.
    "الجنسية", "الجنسيه", "مكان الميلاد", "محل الميلاد", "العنوان",
    "NACIONALIDAD", "NACIONALIDADE", "CITTADINANZA", "UYRUĞU", "UYRUGU",
)

# An EU-model licence prints its field labels as numbers rather than words, so
# the word-label table above never sees them. Designator 3 is "date and place
# of birth" and 8 is the holder's address; 4c is the issuing authority. A row
# opening with a holder designator therefore states the holder's country, not
# the issuer's. A UK card reading "3. 13.05.1991 IRELAND" beside "4c. DVLA"
# was read as two issuers, cancelling both and sending the bundle to the
# passport-nationality fallback, which then named Ireland as the licence
# country. A row that also reaches 4c still names an authority and is kept.
_HOLDER_FIELD_DESIGNATOR = re.compile(r"^\s*[38]\s*[.,):\-]\s*\S")
_ISSUER_FIELD_DESIGNATOR = re.compile(r"\b4\s*[.,):\-]?\s*C\b", re.I)


def _is_non_issuer_country_row(line: OCRLine, lines: list[OCRLine]) -> bool:
    """Whether a country OCR row is the holder's data, not the licence issuer.

    OCR commonly separates a printed key and its value into two boxes.  A
    whole-page country scan must therefore not treat a standalone ``Pakistan``
    box as issuer evidence merely because ``Nationality`` was read beside it.
    """
    line_text = compact_label(line.text)
    if any(compact_label(label) in line_text for label in _NON_ISSUER_COUNTRY_ROW_LABELS):
        return True
    designator_text = ascii_numerals(line.text)
    if (
        _HOLDER_FIELD_DESIGNATOR.match(designator_text)
        and not _ISSUER_FIELD_DESIGNATOR.search(designator_text)
    ):
        return True
    line_ys = [point[1] for point in line.bounding_box]
    line_top, line_bottom = min(line_ys), max(line_ys)
    line_height = max(line_bottom - line_top, 1.0)
    for peer in lines:
        if peer is line:
            continue
        peer_text = compact_label(peer.text)
        if not any(
            compact_label(label) in peer_text
            for label in _NON_ISSUER_COUNTRY_ROW_LABELS
        ):
            continue
        peer_ys = [point[1] for point in peer.bounding_box]
        peer_top, peer_bottom = min(peer_ys), max(peer_ys)
        overlap = max(0.0, min(line_bottom, peer_bottom) - max(line_top, peer_top))
        peer_height = max(peer_bottom - peer_top, 1.0)
        if overlap >= 0.45 * min(line_height, peer_height):
            return True
    return False


# Two rows are one heading when the second sits directly under the first and
# both start from the same edge: that is what stacked display type looks like
# after OCR, and it is how a card's own state name arrives when it is set over
# two lines. The gap is measured against the type's own height, so ordinary
# body rows -- which are set further apart relative to their size -- are not
# joined into each other.
_STACKED_HEADING_MAXIMUM_GAP = 0.6
_STACKED_HEADING_ALIGNMENT = 0.05


def _stacked_heading_pairs(
    lines: list[OCRLine],
) -> list[tuple[OCRLine, OCRLine]]:
    """The row pairs that are one heading broken across two OCR boxes."""
    boxed = [line for line in lines if line.bounding_box]
    page_width = max(
        (max(point[0] for point in line.bounding_box) for line in boxed),
        default=0.0,
    )
    tolerance = max(12.0, _STACKED_HEADING_ALIGNMENT * page_width)
    pairs: list[tuple[OCRLine, OCRLine]] = []
    for first in boxed:
        first_xs = [point[0] for point in first.bounding_box]
        first_ys = [point[1] for point in first.bounding_box]
        first_height = max(1.0, max(first_ys) - min(first_ys))
        for second in boxed:
            if second is first:
                continue
            second_xs = [point[0] for point in second.bounding_box]
            second_ys = [point[1] for point in second.bounding_box]
            second_height = max(1.0, max(second_ys) - min(second_ys))
            gap = min(second_ys) - max(first_ys)
            if gap < 0 or gap > _STACKED_HEADING_MAXIMUM_GAP * min(
                first_height, second_height,
            ):
                continue
            if abs(min(second_xs) - min(first_xs)) > tolerance:
                continue
            pairs.append((first, second))
    return pairs


def _tourist_country_evidence(
    lines: list[OCRLine], detected: DocumentType, barcodes: list[Any] | None,
    customer: CustomerType,
) -> list[CountryEvidence]:
    """Which country this page says issued the driving document on it.

    Only driving documents are consulted. The passport names a nationality, not
    a licence issuer, and letting it in here would make the NATIONAL_ONLY
    nationality check compare a value against itself.
    """
    if customer != CustomerType.TOURIST or detected not in TOURIST_DRIVING_TYPES:
        return []
    evidence: list[CountryEvidence] = []
    for barcode in barcodes or []:
        found = country_from_barcode(getattr(barcode, "structured_candidate", None))
        if found is not None:
            evidence.append(found)
    if not evidence and detected in {
        DocumentType.NATIONAL_DRIVING_LICENCE_FRONT,
        DocumentType.NATIONAL_DRIVING_LICENCE_BACK,
    }:
        # Russian cards often print only the compact ``RUS`` code in the Latin
        # strip; the state heading can be Cyrillic and slightly damaged even
        # though the unmistakable driving-licence title survives.  This is
        # issuer evidence from the licence itself, not a passport-nationality
        # inference.
        russian_text = " ".join(line.text.upper() for line in lines)
        russian_layout = (
            ("УДОСТОВЕРЕНИЕ" in russian_text and "ВОДИ" in russian_text)
            or "ГИБДД" in russian_text
            or "GIBDD" in russian_text
        )
        has_rus_code = any(
            re.fullmatch(r"RUS", ascii_numerals(line.text).strip().upper())
            for line in lines
        )
        if russian_layout and has_rus_code:
            evidence.append(CountryEvidence(
                country="Russia", source=DetectionSource.LICENCE_TEXT,
                confidence=0.95, evidence_text="RUS:RUSSIAN_LICENCE_LAYOUT",
            ))
    if not evidence and detected in {
        DocumentType.NATIONAL_DRIVING_LICENCE_FRONT,
        DocumentType.NATIONAL_DRIVING_LICENCE_BACK,
    }:
        # Before any country name is looked for: an American card names the
        # state that issued it and, usually, no country at all. Read first so
        # that a New Mexico or Washington heading is the state it is rather
        # than the country whose name it contains.
        found = country_from_us_state(lines)
        if found is not None:
            evidence.append(found)
    if not evidence and detected in {
        DocumentType.NATIONAL_DRIVING_LICENCE_FRONT,
        DocumentType.NATIONAL_DRIVING_LICENCE_BACK,
    }:
        # Prefer a country named in its own heading/authority row. Unlike the
        # old whole-page scan, this keeps a country value aligned with
        # ``Nationality`` (or birthplace/address) out of issuer evidence.
        line_evidence = [
            found for line in lines
            if not _is_non_issuer_country_row(line, lines)
            for found in [country_from_text(line.text)]
            if found is not None
        ]
        if not line_evidence:
            # A state that sets its name in two stacked lines of display type
            # never reaches a whole-line test. The Portuguese licence heads its
            # card "REPÚBLICA" over "PORTUGUESA", and each half on its own
            # names nothing, so a card that states its issuer in the largest
            # type on the page was read as naming no country at all and fell
            # back to the passport's nationality.
            line_evidence = [
                found for first, second in _stacked_heading_pairs(lines)
                if not _is_non_issuer_country_row(first, lines)
                and not _is_non_issuer_country_row(second, lines)
                for found in [country_from_text(f"{first.text} {second.text}")]
                if found is not None
            ]
        line_countries = {item.country for item in line_evidence}
        if len(line_countries) == 1:
            evidence.append(max(line_evidence, key=lambda item: item.confidence))
    page_text = " ".join(line.text for line in lines)
    if not evidence:
        # The card's own zone ranks with the barcode and above any wording: both
        # are printed by the issuing authority rather than read off a title.
        found = country_from_card_zone(page_text)
        if found is not None:
            evidence.append(found)
    if not evidence:
        # An AAMVA payload is printed by the issuing authority, so a page that
        # carries one is not re-read for a country name that could only be
        # weaker evidence. Strip holder-data rows first: their country values
        # are often in a separate OCR box from ``Nationality`` and do not name
        # the authority that issued the licence.
        issuer_text = (
            " ".join(
                line.text for line in lines
                if not _is_non_issuer_country_row(line, lines)
            )
            if detected in {
                DocumentType.NATIONAL_DRIVING_LICENCE_FRONT,
                DocumentType.NATIONAL_DRIVING_LICENCE_BACK,
                # A permit states the issuing state in its heading and the
                # holder's place of birth in row 3, and a holder born abroad
                # therefore puts two states on one page. The whole-page scan
                # reads that as a translation panel and names neither, so a
                # Zimbabwean permit issued to a holder born in Karachi
                # reported no issuer at all -- while the same booklet read
                # correctly for a holder born in Harare. The rows that carry
                # holder data are excluded here for the same reason they are
                # excluded on a licence: they name where someone is from, not
                # who issued the document.
                DocumentType.INTERNATIONAL_DRIVING_PERMIT,
            }
            else page_text
        )
        found = country_from_text(issuer_text)
        if found is not None:
            evidence.append(found)
    if not evidence and detected in {
        DocumentType.NATIONAL_DRIVING_LICENCE_FRONT,
        DocumentType.NATIONAL_DRIVING_LICENCE_BACK,
    }:
        # Last, because a state that writes its name out says more than its
        # sign. But a card can print the sign and nothing else, and then this
        # is the only thing on the page that names the issuer at all.
        found = country_from_eu_distinguishing_sign(lines)
        if found is not None:
            evidence.append(found)
    return evidence


def _page_fingerprint(image: Image.Image, slot: str) -> str:
    """Identify a page by its pixels, so identical uploads are read once."""
    rgb = image.convert("RGB")
    digest = hashlib.sha256()
    digest.update(f"{slot}|{rgb.width}x{rgb.height}|".encode())
    digest.update(rgb.tobytes())
    return digest.hexdigest()


_ORIENTATION_READABILITY_MARKERS = tuple(
    fold_for_match(marker) for marker in (
        "PASSPORT", "SURNAME", "GIVEN NAME", "DATE OF BIRTH",
        "DATE OF ISSUE", "EXPIRY", "PLACE OF ISSUE", "NATIONALITY",
        "DRIVER LICENCE", "DRIVING LICENCE", "DRIVER LICENSE",
        "DRIVING LICENSE", "LICENCE NO", "LICENSE NO", "CLASS",
        "INTERNATIONAL", "CONVENTION", "PERMIT", "CERTIFICATE",
        "PARTICULARS CONCERNING THE DRIVER", "SIGNATURE",
    )
)


def _ocr_orientation_readability(result: OCRResult) -> float:
    """Score whether an OCR pass resembles an upright identity document.

    The score intentionally uses only evidence already returned by OCR. Common
    document labels carry most of the weight; confident alphanumeric rows add
    a smaller amount so a document in an unlisted language can still win.
    """
    folded = fold_for_match(" ".join(line.text for line in result.lines))
    marker_score = 28.0 * sum(
        marker in folded for marker in _ORIENTATION_READABILITY_MARKERS
    )
    row_score = 0.0
    for line in result.lines:
        text = " ".join(line.text.split())
        alphanumeric = sum(character.isalnum() for character in text)
        if line.confidence < 0.65 or alphanumeric < 2:
            continue
        row_score += min(alphanumeric, 32) * line.confidence
        if len(re.findall(r"[A-Za-z]{2,}", text)) >= 2:
            row_score += 4.0 * line.confidence
    return marker_score + row_score


def _orientation_reread_is_better(
    original: OCRResult, corrected: OCRResult,
) -> bool:
    """Accept a classifier rotation only when it materially improves OCR."""
    original_score = _ocr_orientation_readability(original)
    corrected_score = _ocr_orientation_readability(corrected)
    return corrected_score > original_score + max(5.0, original_score * 0.05)


def _relabel_page(
    page: tuple[PageArtifact, list[FieldCandidate], DocumentRecord, list[str]],
    slot: str, page_number: int, source_index: int,
) -> tuple[PageArtifact, list[FieldCandidate], DocumentRecord, list[str]]:
    """Re-present an already-read page under the repeat upload's identity.

    The evidence is the same evidence, so it keeps pointing at the first read's
    upload id; only the record identifying the duplicate upload is new. That
    keeps the reconciler from mistaking one photograph for two agreeing
    documents.
    """
    artifact, candidates, record, warnings = page
    auto_detect = SLOT_TYPES[slot] == DocumentType.UNKNOWN
    upload_id = (
        f"{slot}:{source_index + 1}:{page_number + 1}"
        if auto_detect else f"{slot}:{page_number + 1}"
    )
    duplicate_record = record.model_copy(deep=True)
    duplicate_record.upload_id = upload_id
    duplicate_artifact = PageArtifact(
        upload_id=upload_id, expected_type=artifact.expected_type,
        detected_type=artifact.detected_type, preprocessed=artifact.preprocessed,
        ocr=artifact.ocr, mrz=artifact.mrz, preview=artifact.preview,
    )
    return duplicate_artifact, [], duplicate_record, []


class _CandidateCache:
    """Bind labels to values at most once per (line set, document type).

    Routing a page, checking whether a second OCR pass is worth running and
    building the final candidate list all ask the same question of the same
    lines. Answering it once per page instead of up to eight times is the
    single largest saving in the read path, and it cannot change a result
    because the extractor is a pure function of its inputs.
    """

    def __init__(
        self, source: str, licence_country: str | None,
        allowed_paths: frozenset[str] | None = None,
    ):
        self.source = source
        self.licence_country = licence_country
        self.allowed_paths = allowed_paths
        self.calls = 0
        # Each entry keeps its line list alive, so the id() used in the key
        # cannot be recycled by another list while the entry exists.
        self._entries: dict[
            tuple[int, DocumentType], tuple[list[OCRLine], list[FieldCandidate]]
        ] = {}

    def candidates(
        self, lines: list[OCRLine], doc_type: DocumentType,
    ) -> list[FieldCandidate]:
        key = (id(lines), doc_type)
        entry = self._entries.get(key)
        if entry is None:
            self.calls += 1
            entry = (lines, labelled_ocr_candidates(
                lines, doc_type, self.source,
                licence_country=self.licence_country,
                allowed_paths=self.allowed_paths,
            ))
            self._entries[key] = entry
        # Callers cap confidences and append warnings in place, so hand out
        # copies rather than the cached originals.
        return [candidate.model_copy(deep=True) for candidate in entry[1]]


@dataclass
class PageArtifact:
    upload_id: str
    expected_type: DocumentType
    detected_type: DocumentType
    preprocessed: PreprocessedImage
    ocr: OCRResult
    mrz: ParsedMRZ | None
    preview: Image.Image
    # Countries this page names as the issuer of the driving document on it.
    # Collected per page so that the acceptance rule can be settled from the
    # bundle as a whole once every page has been read.
    country_evidence: list[CountryEvidence] = field(default_factory=list)


@dataclass
class ProcessingSession:
    session_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    result: ExtractionResult | None = None
    artifacts: list[PageArtifact] = field(default_factory=list)
    temporary_outputs: set[Path] = field(default_factory=set)
    confirmed_json: str | None = None

    def confirm(self, manual_values: dict[str, str | None] | None = None) -> str:
        if self.result is None: raise ValueError("No processed result is available")
        manual_values = manual_values or {}
        for path, value in manual_values.items():
            if path not in FIELD_PATHS: continue
            if (
                self.result.customer_type == CustomerType.UAE_RESIDENT
                and path not in UAE_RESIDENT_CUSTOMER_FIELD_PATHS
            ):
                continue
            current_section, attribute = path.split(".", 1)
            old = getattr(getattr(self.result, current_section), attribute)
            cleaned = value.strip() if isinstance(value, str) else value
            if (
                self.result.customer_type == CustomerType.TOURIST
                and path in _TOURIST_NAME_PATHS
            ):
                cleaned = _normalize_tourist_name_value(cleaned)
            if (
                self.result.customer_type == CustomerType.TOURIST
                and path == "national_driving_licence.number"
            ):
                cleaned = _format_uk_driving_licence_number(
                    cleaned, self.result.licence_policy.country,
                )
            # Re-submitting a suggested gender unchanged is still the operator
            # accepting it, so it has to clear the guess metadata as an edit does.
            confirms_guess = (
                path == "personal_info.gender" and bool(cleaned)
                and (metadata := self.result.field_metadata.get(path)) is not None
                and metadata.source_method == "name_inference"
            )
            if cleaned != old or confirms_guess:
                _set_path(self.result, path, cleaned or None)
                self.result.field_metadata[path] = FieldMetadata(
                    status=FieldStatus.MANUALLY_EDITED, confidence=1.0,
                    confidence_components={"human_confirmation": True},
                    source_document="human_confirmation", source_method="manual_edit",
                    validation_results=["CONFIRMED_BY_USER"], manually_edited=True,
                )
        gender_metadata = self.result.field_metadata.get("personal_info.gender")
        gender_is_unconfirmed_guess = (
            gender_metadata is not None
            and gender_metadata.source_method == "name_inference"
        )
        if (
            self.result.customer_type == CustomerType.GCC_NATIONAL
            and (
                self.result.personal_info.gender not in {"M", "F", "X"}
                or gender_is_unconfirmed_guess
            )
        ):
            raise ValueError(
                "GCC_GENDER_REQUIRED: select M, F, or X before confirmation; a "
                "value guessed from the holder name is a suggestion only, and "
                "gender is never taken from a portrait"
            )
        self.result.confirmed_by_user = True
        required_review_paths = _required_review_paths_for_result(self.result)
        self.result.manual_review_required = any(
            self.result.field_metadata.get(path) is None
            or self.result.field_metadata[path].status in {
                    FieldStatus.NEEDS_REVIEW, FieldStatus.CONFLICTING,
                    FieldStatus.MISSING,
                }
            for path in required_review_paths
        )
        self.confirmed_json = json.dumps(
            workflow_export_payload(self.result), indent=2, ensure_ascii=False,
        )
        return self.confirmed_json

    def final_json(self) -> str | None:
        return self.confirmed_json

    def compact_for_transport(self, max_preview_side: int = 900) -> ProcessingSession:
        """Drop working images the caller will never display.

        When the reader runs on a remote GPU worker, everything reachable from
        this object is serialised and sent back over the network. Each page
        carries its source capture, its normalised copy, every OCR variant and
        the orientation-corrected image -- around seventy megabytes per page,
        against a caller that renders one thumbnail per page and reads the
        quality figures. Shipping the rest cost far longer than the extraction
        it was reporting on.

        The quality report, the document records and every extracted value are
        untouched; only pixel buffers already consumed by OCR are released.
        """
        placeholder = Image.new("RGB", (1, 1))
        for artifact in self.artifacts:
            preview = artifact.preview
            if max(preview.size) > max_preview_side:
                preview = preview.copy()
                preview.thumbnail(
                    (max_preview_side, max_preview_side), Image.Resampling.LANCZOS,
                )
            artifact.preview = preview
            artifact.preprocessed.original = placeholder
            artifact.preprocessed.normalized = placeholder
            artifact.preprocessed.variants = {}
            artifact.ocr.corrected_images = {}
        return self

    def reset(self) -> None:
        for path in list(self.temporary_outputs):
            try: path.unlink(missing_ok=True)
            except OSError: pass
        self.temporary_outputs.clear()
        self.result = None
        self.artifacts.clear()
        self.confirmed_json = None
        clear_accelerators()


# Where a read's seconds went, held per call rather than per reader: one worker
# can serve two jobs at once, and a timings dictionary living on the instance
# would blend them into a number that describes neither. Unset outside a read,
# so a pipeline driven piecemeal by a test measures and allocates nothing.
_STAGE_SECONDS: ContextVar[dict[str, float] | None] = ContextVar(
    "stage_seconds", default=None,
)
_STAGE_ACTIVE: ContextVar[frozenset[str]] = ContextVar(
    "stage_active", default=frozenset(),
)


@contextmanager
def stage(name: str) -> Iterator[None]:
    """Attribute the time spent inside to one named stage of the current read.

    Re-entrant by name, because the stages nest: an engine's ``run`` delegates
    to its own ``run_languages``, and counting the same wait twice would report
    an OCR cost larger than the read that contains it.
    """
    timings = _STAGE_SECONDS.get()
    active = _STAGE_ACTIVE.get()
    if timings is None or name in active:
        yield
        return
    token = _STAGE_ACTIVE.set(active | {name})
    started = time.perf_counter()
    try:
        yield
    finally:
        _STAGE_ACTIVE.reset(token)
        timings[name] = round(
            timings.get(name, 0.0) + time.perf_counter() - started, 3,
        )


def _instrument(owner: Any, attribute: str, name: str) -> None:
    """Time one engine call in place, once, leaving its signature alone."""
    function = getattr(owner, attribute, None)
    if function is None or getattr(function, "stage_name", None) is not None:
        return

    @wraps(function)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        with stage(name):
            return function(*args, **kwargs)

    wrapper.stage_name = name
    setattr(owner, attribute, wrapper)


class DocumentReader:
    def __init__(
        self, config: AppConfig | None = None,
        ocr_engine: PaddleOCREngine | None = None,
        initialize_vlm: bool = False,
        shadow_ocr_engine: Any | None = None,
        runtime_info: RuntimeInfo | None = None,
    ):
        self.config = config or AppConfig()
        self.runtime = runtime_info or detect_runtime()
        self.choice = self.config.model_choice or select_vlm(self.runtime)
        self.ocr = ocr_engine or PaddleOCREngine(
            self.config.ocr_languages,
            enable_document_vl=self.config.enable_document_vl,
            document_vl_pipeline_version=self.config.document_vl_pipeline_version,
            document_vl_max_new_tokens=self.config.document_vl_max_new_tokens,
            document_vl_detail_crops=self.config.document_vl_detail_crops,
        )
        # An optional observer may read the same in-memory pixels, but its
        # output never enters classification, extraction, or reconciliation.
        # Paddle therefore remains authoritative throughout a shadow rollout.
        self.shadow_ocr = shadow_ocr_engine
        self.vlm = LocalVLM(self.choice, self.runtime, self.config)
        if initialize_vlm and self.config.enable_vlm:
            self.vlm.initialize(download=True)

    # How far past a label its value can sit, as multiples of the label's own
    # height. A card prints the value on the label's row or the row beneath it.
    _ZOOM_ROWS_BELOW = 2.6
    # Enlargement applied to the crop. The crop is a few hundred pixels, so
    # even four times its size is a fraction of a page and costs milliseconds.
    _ZOOM_SCALE = 4.0
    # How many places on one page are worth a second, closer look. Each costs a
    # recognition pass per language, and a page with more than two unreadable
    # rows needs recapturing, not more passes over the same pixels.
    _ZOOM_MAX_REGIONS = 2

    def _zoom_reread(
        self, missing: set[str], ocr: OCRResult, preprocessed: PreprocessedImage,
        detected: DocumentType, ocr_languages: tuple[str, ...],
    ) -> OCRResult | None:
        """Read again, closely, where a label was found but its value was not.

        This is the shape of nearly every failure in this project's bug
        reports: the label survives the capture and the value beside it does
        not. "LICENCE NO. / CRN" was read off a Queensland card whose
        "130 750 802" underneath it was not, and the reader went looking
        elsewhere and found a class table. The page was not unreadable -- one
        row of it was, and that row is a few hundred pixels the recogniser saw
        at the same scale as everything else.

        So it is shown that row on its own, enlarged, with the glare and blur
        repairs applied to the crop rather than the page. Nothing is inferred
        here: the value still has to be read, and still has to pass every guard
        that a first-pass reading passes. What changes is how many pixels the
        recogniser gets for the characters that matter.

        Costs nothing on a clean capture, because nothing is missing.
        """
        engine_ready = hasattr(self.ocr, "run") or hasattr(self.ocr, "run_languages")
        if (
            not missing
            or not engine_ready
            or not getattr(self.ocr, "supports_repair_passes", True)
        ):
            return None
        page = preprocessed.normalized
        labels: list[str] = []
        for path in missing:
            labels.extend(FIELD_LABELS.get(path, ()))
            labels.extend(COMMON_NATIONAL_LABELS.get(path, ()))
            # A licence can date itself with a validity cell instead of an
            # issue caption: the South African card heads one row "Valid" and
            # prints both dates in it. That caption names no field on its own,
            # so it is not in the tables -- but it is exactly the row worth a
            # closer look when a licence's dates were not read, and on this
            # card the row under the diagonal print is the one the page pass
            # loses.
            if path in {
                "national_driving_licence.issue_date",
                "national_driving_licence.expiry_date",
            }:
                labels.extend(VALIDITY_CAPTIONS)
        if not labels:
            return None
        regions: list[tuple[tuple[int, int, int, int], OCRLine]] = []
        for line in ocr.lines:
            upper = line.text.upper()
            if not any(
                compact_label(label) in compact_label(upper) for label in labels
            ):
                continue
            xs = [point[0] for point in line.bounding_box]
            ys = [point[1] for point in line.bounding_box]
            x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
            height = max(y2 - y1, 1.0)
            # The label's own row plus the rows under it, and rightwards as far
            # as a value is ever printed from its key.
            regions.append(((
                max(0, int(x1 - height)),
                max(0, int(y1 - height * 0.4)),
                min(page.width, int(x2 + max((x2 - x1) * 2.5, 320))),
                min(page.height, int(y2 + height * self._ZOOM_ROWS_BELOW)),
            ), line))
        if not regions:
            return None
        # One place, one crop.
        #
        # The page is read as several renderings, so the same printed label
        # comes back once per rendering and each copy asked for its own crop of
        # the same few hundred pixels. Six of those, each recognised in two
        # renderings and two languages, is twenty-four extra passes to read one
        # row -- which is not a repair, it is a stall, and it is what put two
        # minutes between an upload and an answer.
        deduplicated: list[tuple[tuple[int, int, int, int], OCRLine]] = []
        for region, anchor in regions:
            x1, y1, x2, y2 = region
            for kept, _ in deduplicated:
                overlap = (
                    max(0, min(x2, kept[2]) - max(x1, kept[0]))
                    * max(0, min(y2, kept[3]) - max(y1, kept[1]))
                )
                if overlap >= 0.5 * min(
                    max(1, (x2 - x1) * (y2 - y1)),
                    max(1, (kept[2] - kept[0]) * (kept[3] - kept[1])),
                ):
                    break
            else:
                deduplicated.append((region, anchor))
        merged: list[OCRLine] = []
        zoom_requests: list[
            tuple[str, Image.Image, OCRLine, int, int, float]
        ] = []
        for index, ((x1, y1, x2, y2), anchor) in enumerate(
            deduplicated[:self._ZOOM_MAX_REGIONS]
        ):
            if x2 - x1 < 24 or y2 - y1 < 12:
                continue
            crop = page.crop((x1, y1, x2, y2))
            enlarged = crop.resize(
                (round(crop.width * self._ZOOM_SCALE), round(crop.height * self._ZOOM_SCALE)),
                Image.Resampling.LANCZOS,
            )
            # The repaired crop only. Reading the plain one as well doubled the
            # cost to re-read what the page pass had already failed on at that
            # exposure, which is the reading least likely to say anything new.
            variant_name = f"zoom_{index}_flat"
            zoom_requests.append((
                variant_name, zoom_repair(enlarged), anchor, x1, y1,
                self._ZOOM_SCALE,
            ))

        if not zoom_requests:
            return None

        def append_mapped(
            result: OCRResult, request: tuple[
                str, Image.Image, OCRLine, int, int, float
            ],
        ) -> None:
            variant_name, _image, anchor, x1, y1, scale = request
            for line in result.lines:
                if (
                    len(zoom_requests) > 1
                    and line.variant != variant_name
                    and not line.variant.startswith(f"{variant_name}:")
                ):
                    continue
                # Back into page coordinates, and into the label's own variant
                # and language.
                #
                # An enlarged crop is not another rendering of the page, it is
                # a closer look at the same one, and its rows carry page
                # coordinates to prove it. Filed under a variant of their own
                # they could never bind: a value may only answer a label read
                # in the same variant, so "130 750 802" recovered at four times
                # the size sat one row under the "LICENCE NO / CRN" that
                # summoned it and was refused for having been read by a
                # different pass.
                merged.append(replace(
                    line,
                    variant=anchor.variant,
                    language=anchor.language,
                    # Marked as what it is. Filed under the anchor's variant it
                    # is indistinguishable from a page-scale reading, and the
                    # rule that a box lying inside a label's own box is part of
                    # that label -- true of every page-scale reading -- is
                    # false of this one by construction: the crop was taken
                    # around the label, so what it recovers maps back onto it.
                    # The suffix keeps any startswith() test on this field
                    # working as before.
                    model_name=f"{line.model_name}+zoom",
                    bounding_box=[
                        [point[0] / scale + x1, point[1] / scale + y1]
                        for point in line.bounding_box
                    ],
                ))

        # The production engine can submit all independent crops in one GPU
        # batch. It returns the originating variant on every row, so each one
        # is mapped through its own crop and anchor exactly as in the previous
        # per-crop loop. Test/custom engines retain that loop for compatibility.
        if type(self.ocr) is PaddleOCREngine:
            variants = {
                variant_name: image
                for variant_name, image, *_rest in zoom_requests
            }
            try:
                result = self.ocr.run_languages(
                    variants, ocr_languages, merge_variants=False,
                )
            except (OSError, ValueError, RuntimeError):
                return None
            for request in zoom_requests:
                append_mapped(result, request)
        else:
            for request in zoom_requests:
                variant_name, image, *_rest = request
                variants = {variant_name: image}
                try:
                    if hasattr(self.ocr, "run_languages"):
                        result = self.ocr.run_languages(variants, ocr_languages)
                    else:
                        result = self.ocr.run(variants)
                except (OSError, ValueError, RuntimeError):
                    continue
                append_mapped(result, request)
        return OCRResult(lines=merged) if merged else None

    def _preprocess_config(self, customer: CustomerType) -> AppConfig:
        """The size this workflow renders a page at before recognition."""
        if (
            customer == CustomerType.UAE_RESIDENT
            and self.config.uae_fast_path_enabled
        ):
            # The Emirates ID's Latin fields and TD1 zone remain comfortably
            # readable at this size.  Rendering it at the broader workflow's
            # 1600-pixel floor costs substantial GPU time on every card.
            return replace(
                self.config,
                ocr_target_min_side=min(
                    self.config.ocr_target_min_side,
                    self.config.uae_fast_ocr_target_min_side,
                ),
            )
        return self.config

    def _process_page(
        self, slot: str, source: Any, page_number: int, loaded: Any,
        licence_policy: CountryLicencePolicy | None = None,
        customer: CustomerType = CustomerType.UAE_RESIDENT,
        source_index: int = 0,
        gcc_profile: GCCDocumentProfile | None = None,
        previously_read_paths: set[str] | None = None,
        prepared: PreprocessedImage | None = None,
    ) -> tuple[PageArtifact, list[FieldCandidate], DocumentRecord, list[str]]:
        expected = SLOT_TYPES[slot]
        auto_detect = expected == DocumentType.UNKNOWN
        uae_fast_path = (
            customer == CustomerType.UAE_RESIDENT
            and self.config.uae_fast_path_enabled
        )
        upload_id = (
            f"{slot}:{source_index + 1}:{page_number + 1}"
            if auto_detect else f"{slot}:{page_number + 1}"
        )
        licence_country = licence_policy.country if licence_policy else None
        # Every workflow has an explicit extraction allowlist. Restricting the
        # lookup here keeps fields from another workflow out of the result and
        # stops the extractor scanning for labels the workflow ignores.
        allowed_paths = {
            CustomerType.UAE_RESIDENT: UAE_RESIDENT_EXTRACTION_FIELDS,
            CustomerType.GCC_NATIONAL: frozenset(GCC_EXTRACTION_FIELDS),
            CustomerType.TOURIST: TOURIST_EXTRACTION_FIELDS,
        }.get(customer)
        # A supplementary crop or reverse may corroborate a card, but it must
        # not trigger costly repair OCR/model passes for fields an earlier page
        # has already read. Every page still receives its primary OCR pass and
        # retains all evidence; this set only suppresses redundant retries.
        previously_read = _with_derivable(set(previously_read_paths or ()))
        cache = _CandidateCache(upload_id, licence_country, allowed_paths)
        with stage("preprocess"):
            preprocessed = prepared if prepared is not None else analyze_and_preprocess(
                loaded.image, self._preprocess_config(customer),
            )
        selected_variants = {
            name: image for name, image in preprocessed.variants.items()
            if name in self.config.ocr_variant_names
        }
        if not selected_variants:
            selected_variants = {"original_normalized": preprocessed.normalized}
        # GCC may spend one or two fallback views to identify a stylised front
        # before it knows which field set is missing. Remember those views so
        # the extraction repair does not pay for the same GPU pass twice.
        gcc_attempted_variants = set(selected_variants)
        deferred_languages: tuple[str, ...] = ()
        if uae_fast_path:
            # UAE cards carry English labels and a Latin TD1 zone.  An Arabic
            # full-page pass is retained for GCC but is not needed to recover
            # the UAE workflow fields, and doubles the dominant OCR cost.
            ocr_languages = ("en",)
        elif customer in {CustomerType.UAE_RESIDENT, CustomerType.GCC_NATIONAL}:
            # GCC cards are commonly Arabic-first.  Routing must see both
            # scripts on the first pass; waiting until after classification to
            # run Arabic OCR creates a circular failure where an Arabic title
            # is required to decide whether Arabic OCR should run.
            ocr_languages = ("en", "ar")
        elif (auto_detect and customer == CustomerType.TOURIST) or expected in MULTILINGUAL_DOCUMENTS:
            # One compact PP-OCRv5 Latin model covers the alphabets used by the
            # great majority of tourist licences. Cyrillic remains a deferred
            # pass, and scripts outside those two are handled by the single
            # multilingual visual fallback instead of paying for dozens of OCR
            # recognizers on every upload.
            #
            # Test/dummy engines written before the broad model use the old
            # ``en`` key. Keep that compatibility without changing production,
            # where the concrete Paddle engine always selects ``latin``.
            primary_tourist_language = (
                "latin" if isinstance(self.ocr, PaddleOCREngine) else "en"
            )
            # Latin first, Cyrillic only if the Latin pass leaves the page
            # unclassified or a critical row unread. Running both recognizers
            # up front is a second full recognition pass over every page, and
            # it buys nothing on the Latin-script documents that are the great
            # majority of this route -- a passport carries a Latin
            # transliteration of every row the workflow wants, and its
            # machine-readable zone is Latin by definition. The deferred pass
            # still runs for the Cyrillic-only licence it exists for.
            ocr_languages = (primary_tourist_language,)
            deferred_languages = ("ru",)
        else:
            ocr_languages = ("en",)

        def run_ocr(
            languages: tuple[str, ...],
            variants: dict[str, Image.Image] | None = None, *,
            use_orientation_classifier: bool = True,
        ) -> OCRResult:
            active_variants = variants if variants is not None else selected_variants
            if hasattr(self.ocr, "run_languages"):
                if isinstance(self.ocr, PaddleOCREngine):
                    return self.ocr.run_languages(
                        active_variants, languages,
                        use_orientation_classifier=use_orientation_classifier,
                    )
                return self.ocr.run_languages(active_variants, languages)
            return self.ocr.run(active_variants)

        def merged_with(current: OCRResult, extra: OCRResult) -> OCRResult:
            return OCRResult(
                lines=merge_ocr_lines([*current.lines, *extra.lines]),
                model_names=sorted({*current.model_names, *extra.model_names}),
                warnings=[*current.warnings, *extra.warnings],
                corrected_images={**extra.corrected_images, **current.corrected_images},
                orientation_angles={**extra.orientation_angles, **current.orientation_angles},
            )

        ocr = run_ocr(
            ocr_languages,
            use_orientation_classifier=not uae_fast_path,
        )
        if self.shadow_ocr is not None:
            try:
                # Submit exactly one full-page view. Repair variants, zoom
                # crops, orientation retries and deferred-language passes must
                # not create more provider requests (or page charges).
                self.shadow_ocr.submit(
                    ocr, preprocessed.normalized, ocr_languages,
                )
            except Exception as exc:
                # Shadow availability can never change the customer result.
                # Log only the exception type; provider messages may echo
                # request metadata.
                logger.warning("GDA shadow submit failed kind=%s", type(exc).__name__)
        if (
            not hasattr(self.ocr, "run_languages")
            or getattr(self.ocr, "auto_detects_languages", False)
        ):
            # An engine that cannot be asked for one language at a time has
            # already read the page in every language it has.
            deferred_languages = ()
        orientation_angle = ocr.orientation_angles.get("original_normalized", 0)
        corrected_primary = ocr.corrected_images.get("original_normalized")
        if (
            corrected_primary is not None
            and getattr(self.ocr, "returns_canonical_image", False)
        ):
            # Online OCR already annotates its deskewed output image. Adopt
            # those pixels without the Paddle-specific orientation reread.
            # All later geometry, barcode checks and previews use the same
            # coordinate system as the provider's OCR boxes.
            preprocessed.normalized = corrected_primary
            preprocessed.variants = {"original_normalized": corrected_primary}
            selected_variants = dict(preprocessed.variants)
            preprocessed.quality.width, preprocessed.quality.height = corrected_primary.size
            preprocessed.quality.orientation = (
                "LANDSCAPE" if corrected_primary.width >= corrected_primary.height
                else "PORTRAIT"
            )
            preprocessed.transformations.append({
                "operation": "provider_content_normalization",
                "reason": "ocr_coordinates_match_provider_image",
            })
        if (
            not uae_fast_path
            and orientation_angle
            and corrected_primary is not None
        ):
            # Paddle exposes the content-corrected pixels alongside the first
            # prediction, but the text in that prediction can still belong to
            # the pre-rotation page. An upside-down passport therefore reached
            # the report with rotation_degrees=180 while every OCR row remained
            # upside down. Replace that evidence with one bounded reread of the
            # corrected image, then keep every later crop and fallback in the
            # same coordinate system.
            original_primary = preprocessed.normalized
            original_variants = preprocessed.variants
            original_selected_variants = selected_variants
            original_transformations = list(preprocessed.transformations)
            original_quality = (
                preprocessed.quality.rotation_degrees,
                preprocessed.quality.width,
                preprocessed.quality.height,
                preprocessed.quality.orientation,
            )
            # Re-read the untouched pixels with orientation classification
            # disabled. This is the control that prevents a low-confidence
            # 180-degree guess from destroying a page that was already upright.
            original_reread = (
                run_ocr(
                    ocr_languages, original_selected_variants,
                    use_orientation_classifier=False,
                )
                if isinstance(self.ocr, PaddleOCREngine) else None
            )
            preprocessed.quality.rotation_degrees = round(
                (preprocessed.quality.rotation_degrees or 0.0)
                + orientation_angle,
                2,
            )
            preprocessed.quality.width, preprocessed.quality.height = (
                corrected_primary.size
            )
            preprocessed.quality.orientation = (
                "LANDSCAPE"
                if corrected_primary.width >= corrected_primary.height
                else "PORTRAIT"
            )
            preprocessed.transformations.append({
                "operation": "content_orientation_rotation",
                "degrees": orientation_angle,
                "reason": "ocr_document_orientation_classifier",
            })
            selected_names = set(selected_variants) or {"original_normalized"}
            preprocessed.normalized = corrected_primary
            # Every existing variant was derived from the old orientation.
            # Rebuild lazily from the corrected primary so OCR boxes, zoom
            # crops, colour reads and preview pixels all share one geometry.
            preprocessed.variants = {}
            selected_variants = ensure_ocr_variants(
                preprocessed, selected_names,
            )
            first_pass = ocr
            reread = run_ocr(
                ocr_languages, use_orientation_classifier=False,
            )
            reread.model_names = sorted({
                *first_pass.model_names, *reread.model_names,
            })
            reread.warnings = list(dict.fromkeys([
                *first_pass.warnings, *reread.warnings,
                "OCR_RERUN_AFTER_ORIENTATION_CORRECTION",
            ]))
            reread.corrected_images = {
                **reread.corrected_images,
                "original_normalized": corrected_primary,
            }
            reread.orientation_angles = {
                **reread.orientation_angles,
                "original_normalized": orientation_angle,
            }
            if (
                original_reread is not None
                and not _orientation_reread_is_better(original_reread, reread)
            ):
                # The Ghanaian permit in the reported bundle was already
                # upright. Paddle guessed 180 degrees from the two-page portrait
                # composition, and the unconditional correction turned every
                # printed label and handwritten value upside down. Restore the
                # original coordinate system and keep the demonstrably clearer
                # control read.
                preprocessed.normalized = original_primary
                preprocessed.variants = original_variants
                selected_variants = original_selected_variants
                preprocessed.transformations = original_transformations
                (
                    preprocessed.quality.rotation_degrees,
                    preprocessed.quality.width,
                    preprocessed.quality.height,
                    preprocessed.quality.orientation,
                ) = original_quality
                original_reread.model_names = sorted({
                    *first_pass.model_names, *reread.model_names,
                    *original_reread.model_names,
                })
                original_reread.warnings = list(dict.fromkeys([
                    *first_pass.warnings, *reread.warnings,
                    *original_reread.warnings,
                    "OCR_ORIENTATION_CORRECTION_REJECTED_BY_READABILITY",
                ]))
                original_reread.corrected_images = {}
                original_reread.orientation_angles = {}
                ocr = original_reread
            else:
                ocr = reread
        mrz = _best_mrz_from_ocr(
            ocr.lines,
            allow_printed_given_name_separator_repair=(
                customer == CustomerType.TOURIST
            ),
        )

        def read_paths_for(doc_type: DocumentType) -> set[str]:
            """Every field this page has already yielded for that document type.

            The machine-readable zone was decoded out of the text the first pass
            returned, at no extra cost, and on a passport it carries most of the
            page. Counting only the labelled rows is what used to send a
            perfectly readable passport for a second recognizer pass and a model
            generation, hunting values its zone had already proved.
            """
            evidence = [
                *cache.candidates(ocr.lines, doc_type),
                *(
                    mrz_candidates(
                        mrz, upload_id, doc_type, licence_country=licence_country,
                    ) if mrz and mrz.mrz_type else []
                ),
            ]
            # California's compact ISS/EXP/SEX layout is a deterministic
            # document parser, not a best-effort inference.  Count its values
            # while deciding whether another page-scale recognition pass is
            # needed; otherwise the retry controller cannot see fields it has
            # already read and needlessly launches more variants.
            if doc_type == DocumentType.NATIONAL_DRIVING_LICENCE_FRONT:
                evidence.extend(california_licence_layout_candidates(
                    ocr.lines, upload_id,
                ))
            return _with_derivable({
                candidate.field_path for candidate in evidence
                if (
                    candidate.normalized_value
                    and candidate.validation_passed is not False
                    # A chronological ordering fallback is useful operator
                    # evidence, but it does not prove the card's own 4a/4b
                    # rows were read.  Treating it as complete suppressed the
                    # repair that sees the tiny headings and, on the Swiss
                    # reverse table, retained a category date as licence issue.
                    and not (
                        doc_type == DocumentType.NATIONAL_DRIVING_LICENCE_FRONT
                        and "LICENCE_DATE_ORDER_FALLBACK_REQUIRES_REVIEW"
                        in candidate.warnings
                    )
                )
            })

        def merge_zoom_if_safe(
            document_type: DocumentType, missing: set[str],
        ) -> bool:
            """Keep a close reread only when it fills a gap without changing evidence.

            A crop is much cheaper than another complete rendering of the page,
            but it remains just OCR evidence: it may never displace a value that
            was already accepted at page scale.  This preserves that invariant
            while letting the Tourist passport path try the one labelled row it
            is missing before it queues a full-page repair.
            """
            nonlocal ocr
            zoomed = self._zoom_reread(
                missing, ocr, preprocessed, document_type, ocr_languages,
            )
            if zoomed is None:
                return False

            def bound_values(result: OCRResult) -> dict[str, frozenset[str]]:
                found: dict[str, set[str]] = {}
                for candidate in cache.candidates(result.lines, document_type):
                    if (
                        candidate.normalized_value
                        and candidate.validation_passed is not False
                    ):
                        found.setdefault(candidate.field_path, set()).add(
                            candidate.normalized_value,
                        )
                return {key: frozenset(value) for key, value in found.items()}

            before = bound_values(ocr)
            previous, ocr = ocr, merged_with(ocr, zoomed)
            after = bound_values(ocr)
            unchanged = all(
                after.get(path) == values for path, values in before.items()
            )
            if unchanged and (set(after) - set(before)) & missing:
                return True
            ocr = previous
            return False

        def has_visible_latin_passport_issue_label() -> bool:
            """Whether Latin OCR has located the row the crop is about to read.

            Do not suppress Cyrillic OCR merely because a passport has a valid
            Latin MRZ.  The fast path is sound only where the first pass has
            actually found an ASCII issue-date caption; Russian-only (and other
            non-Latin) pages continue through their existing script fallback.
            """
            labels = (
                *FIELD_LABELS.get("passport.issue_date", ()),
                *COMMON_NATIONAL_LABELS.get("passport.issue_date", ()),
            )
            latin_labels = [label for label in labels if label.isascii()]
            return any(
                compact_label(label) in compact_label(line.text)
                for line in ocr.lines for label in latin_labels
            )

        def has_visible_latin_passport_date_layout() -> bool:
            """Whether the first pass can recover a Latin passport by repair.

            A Tunisian passport keeps its data values in Latin digits but the
            small field labels are often partly Arabic.  If the base Latin OCR
            has already located its passport and birth-date captions, a whole
            Russian page pass cannot help those rows; the local repair views
            are the appropriate next step.  Actual Cyrillic passports retain
            their deferred-language fallback.
            """
            page = "\n".join(fold_for_match(line.text) for line in ocr.lines)
            has_cyrillic = any("\u0400" <= character <= "\u052f" for character in page)
            return (
                not has_cyrillic
                and "PASSPORT" in page
                and "BIRTH" in page
            )

        def has_glared_english_aamva_front_signature() -> bool:
            """Identify a U.S.-style front whose state heading glare hid.

            A California front can lose its large state/title row in the normal
            rendering but retain the compact ``DL``, ``ISS``, ``EXP`` and
            ``SEX`` labels. Those four English AAMVA cues do not occur on a
            Cyrillic licence, so they safely justify trying the batched Latin
            repair before loading and running the deferred Russian recognizer.
            """
            page = "\n".join(line.text.upper() for line in ocr.lines)
            has_cyrillic = any("\u0400" <= character <= "\u052f" for character in page)
            return (
                not has_cyrillic
                and re.search(r"\bDL\b", page) is not None
                and re.search(r"\bISS(?:UED)?\b", page) is not None
                and re.search(r"\bEXP(?:IR(?:Y|ATION|ES)?)?\b", page) is not None
                and re.search(r"\bSEX\b", page) is not None
            )

        def repair_ocr_until_complete(
            document_type: DocumentType, *, batch_all: bool = False,
        ) -> OCRResult | None:
            """Retry only as far as the outstanding visible fields require.

            The previous implementation sent contrast, illumination-flattened
            and deblurred full-page images to each recognizer in one batch as
            soon as any critical field was absent.  For a warm bilingual
            request that turns one missed row into six full-page recognitions,
            even when contrast alone recovers it.  Run the established variants
            in their deterministic order and stop immediately once the page
            has supplied every field it is documented to print.  Later views
            remain available for genuinely difficult captures, so this is a
            latency optimisation rather than a reduction in evidence quality.
            """
            nonlocal ocr, mrz

            if not getattr(self.ocr, "supports_repair_passes", True):
                return None

            # Custom/legacy OCR integrations receive their repair variants as
            # one opaque batch.  Apart from retaining their existing API
            # contract, this keeps controlled test engines deterministic.  The
            # production Paddle implementation is the only engine whose
            # per-variant cost is known and whose results can be safely stopped
            # as soon as the documented fields are complete.
            if not isinstance(self.ocr, PaddleOCREngine):
                fallback_variants = {
                    name: image for name, image in ensure_ocr_variants(
                        preprocessed, self.config.ocr_fallback_variant_names,
                    ).items()
                    if name not in selected_variants
                    and not (
                        customer == CustomerType.GCC_NATIONAL
                        and name in gcc_attempted_variants
                    )
                }
                if not fallback_variants:
                    return None
                if hasattr(self.ocr, "run_languages"):
                    repair_result = self.ocr.run_languages(
                        fallback_variants, ocr_languages,
                    )
                else:
                    repair_result = self.ocr.run(fallback_variants)
                if customer == CustomerType.GCC_NATIONAL:
                    gcc_attempted_variants.update(fallback_variants)
                ocr = merged_with(ocr, repair_result)
                mrz = _best_mrz_from_ocr(
                    ocr.lines,
                    allow_printed_given_name_separator_repair=(
                        customer == CustomerType.TOURIST
                    ),
                )
                return repair_result

            if (
                document_type == DocumentType.NATIONAL_DRIVING_LICENCE_FRONT
                and preprocessed.quality.glare_detected
                and (batch_all or is_california_driver_licence(ocr.lines))
            ):
                # A California DL is English-only, but glare splits its compact
                # ISS/EXP/SEX rows across different repair views.  Give the
                # Latin recognizer all three images in one batched call: this
                # preserves their combined evidence while avoiding the three
                # separate GPU pipeline launches the generic retry loop needs.
                fallback_variants = {
                    name: image for name, image in ensure_ocr_variants(
                        preprocessed, self.config.ocr_fallback_variant_names,
                    ).items()
                    if name not in selected_variants
                }
                if not fallback_variants:
                    return None
                additional = self.ocr.run_languages(
                    fallback_variants, ocr_languages,
                )
                ocr = merged_with(ocr, additional)
                mrz = _best_mrz_from_ocr(
                    ocr.lines,
                    allow_printed_given_name_separator_repair=(
                        customer == CustomerType.TOURIST
                    ),
                )
                return additional

            repair_result: OCRResult | None = None
            # Contrast is the least destructive repair and fixes ordinary
            # low-contrast print.  It is also the first view the previous
            # batched implementation supplied (the image builder's stable
            # order), so keep that evidence order while avoiding the unused
            # full-page recognitions.  Any custom variant remains reachable
            # after the standard repairs in its configured order.
            configured_variants = self.config.ocr_fallback_variant_names
            preferred_variants = (
                # On a glared passport the document number is normally the
                # lost row, and deblurring has proven the successful repair.
                # Starting there avoids two full-page OCR passes when it is
                # the only image transformation that can recover the number.
                ("deblurred", "illumination_flattened", "contrast_enhanced")
                if (
                    document_type == DocumentType.PASSPORT_BIODATA
                    and preprocessed.quality.glare_detected
                )
                else ("contrast_enhanced", "illumination_flattened", "deblurred")
            )
            repair_order = tuple(dict.fromkeys([
                *(name for name in preferred_variants if name in configured_variants),
                *configured_variants,
            ]))
            for variant_name in repair_order:
                if variant_name in selected_variants or (
                    customer == CustomerType.GCC_NATIONAL
                    and variant_name in gcc_attempted_variants
                ):
                    continue
                variants = ensure_ocr_variants(preprocessed, (variant_name,))
                image = variants.get(variant_name)
                if image is None:
                    continue
                if hasattr(self.ocr, "run_languages"):
                    additional = self.ocr.run_languages(
                        {variant_name: image}, ocr_languages,
                    )
                else:
                    additional = self.ocr.run({variant_name: image})
                if customer == CustomerType.GCC_NATIONAL:
                    gcc_attempted_variants.add(variant_name)
                repair_result = (
                    additional if repair_result is None
                    else merge_ocr_results(repair_result, additional)
                )
                ocr = merged_with(ocr, additional)
                mrz = _best_mrz_from_ocr(
                    ocr.lines,
                    allow_printed_given_name_separator_repair=(
                        customer == CustomerType.TOURIST
                    ),
                )
                remaining = (
                    _ocr_recovery_paths(
                        document_type, customer, gcc_profile, ocr.lines,
                    )
                    - read_paths_for(document_type) - previously_read
                )
                if not remaining:
                    break
            return repair_result

        page_notes: list[str] = []
        missing_critical = (
            set() if auto_detect
            else (
                _ocr_recovery_paths(expected, customer, gcc_profile, ocr.lines)
                - read_paths_for(expected) - previously_read
            )
        )
        if missing_critical and deferred_languages:
            # On a page whose type is already known, the postponed script is the
            # first thing to try for an unread row: it is a whole recognizer the
            # page has not been shown to yet.
            # A passport's MRZ has already supplied every other critical
            # value.  Its visibly printed issue date is commonly the only
            # remaining row, so a complete Russian pass is not useful on this
            # Latin document.  The targeted reread below gets the same OCR
            # evidence at a fraction of the page-wide cost; a failure still
            # reaches every established fallback.
            only_passport_issue_date = (
                customer == CustomerType.TOURIST
                and expected == DocumentType.PASSPORT_BIODATA
                and missing_critical <= {"passport.issue_date"}
                and has_visible_latin_passport_issue_label()
            )
            if not only_passport_issue_date:
                ocr = merged_with(ocr, run_ocr(deferred_languages))
                deferred_languages = ()
                mrz = _best_mrz_from_ocr(
                    ocr.lines,
                    allow_printed_given_name_separator_repair=(
                        customer == CustomerType.TOURIST
                    ),
                )
                missing_critical = (
                    _ocr_recovery_paths(expected, customer, gcc_profile, ocr.lines)
                    - read_paths_for(expected) - previously_read
                )
        fallback_ocr: OCRResult | None = None
        if (
            customer == CustomerType.TOURIST
            and expected == DocumentType.PASSPORT_BIODATA
            and missing_critical
            and merge_zoom_if_safe(expected, missing_critical)
        ):
            page_notes.append(f"ZOOM_REREAD_APPLIED:{upload_id}")
            missing_critical = (
                _critical_paths(expected, gcc_profile)
                - read_paths_for(expected) - previously_read
            )
        if missing_critical and not uae_fast_path:
            fallback_ocr = repair_ocr_until_complete(expected)
        preview = ocr.corrected_images.get(
            "original_normalized", preprocessed.normalized,
        )
        mismatched_issuer = _mismatched_issuer(ocr.lines, customer, gcc_profile)
        # Re-read after any merge above: a zone can be split across the two
        # passes, and repairing it needs both halves.
        mrz = _best_mrz_from_ocr(
            ocr.lines,
            allow_printed_given_name_separator_repair=(
                customer == CustomerType.TOURIST
            ),
        )
        with stage("barcode"):
            barcodes = decode_barcodes(
                preview, upload_id,
                # The tourist route accepts national licences, which are often
                # uploaded as one front-and-back photograph.  Restrict the
                # bounded panel retry to that route so UAE and GCC barcode
                # behaviour and latency remain unchanged.
                scan_composite=customer == CustomerType.TOURIST,
            )
        detected, _ = classify_document(
            ocr.lines, expected,
            bool(mrz and mrz.valid) or passport_name_row_present(
                find_mrz_lines(ocr.lines)
            ),
            [barcode.barcode_type for barcode in barcodes],
        )
        if auto_detect:
            detected = _auto_document_type(
                ocr.lines, detected, customer, licence_policy, upload_id, barcodes,
                gcc_profile=gcc_profile, cache=cache, mrz=mrz,
            )
            if (
                customer == CustomerType.GCC_NATIONAL
                and detected == DocumentType.UNKNOWN
                and isinstance(self.ocr, PaddleOCREngine)
                and not _gcc_back_only_page(ocr.lines)
            ):
                # A GCC front that lost its large stylised banner cannot name
                # its field schema, so the ordinary "repair missing fields"
                # path is circular: it needs a type before it can decide what
                # is missing. Spend at most two established views, cheapest and
                # most useful first, and stop the moment the page routes. Clear
                # fronts never enter this branch; unmistakable category/serial
                # backs are excluded because a front-title retry cannot help
                # classify those layouts.
                routing_variants = (
                    "illumination_flattened", "deblurred",
                )
                for variant_name in routing_variants:
                    if (
                        variant_name not in self.config.ocr_fallback_variant_names
                        or variant_name in gcc_attempted_variants
                    ):
                        continue
                    variant = ensure_ocr_variants(
                        preprocessed, (variant_name,),
                    ).get(variant_name)
                    if variant is None:
                        continue
                    additional = run_ocr(
                        ocr_languages, {variant_name: variant},
                    )
                    gcc_attempted_variants.add(variant_name)
                    ocr = merged_with(ocr, additional)
                    mrz = _best_mrz_from_ocr(ocr.lines)
                    classified, _ = classify_document(
                        ocr.lines, expected, bool(mrz and mrz.valid),
                        [barcode.barcode_type for barcode in barcodes],
                    )
                    detected = _auto_document_type(
                        ocr.lines, classified, customer, licence_policy,
                        upload_id, barcodes, gcc_profile=gcc_profile,
                        cache=cache,
                    )
                    page_notes.append(
                        f"GCC_ROUTING_REPAIR_APPLIED:{upload_id}:{variant_name}",
                    )
                    if detected != DocumentType.UNKNOWN:
                        mismatched_issuer = (
                            _mismatched_issuer(
                                ocr.lines, customer, gcc_profile,
                            ) or mismatched_issuer
                        )
                        break
            if deferred_languages:
                # Do the routing repair before the deferred Russian recognizer
                # only for the unmistakable English AAMVA signature above. The
                # reported California card lost its state heading in the first
                # view, was called a back, then paid for Russian OCR and three
                # serial repair calls. A batched Latin repair restores those
                # compact rows; Russian remains available if they are still
                # incomplete, so a genuinely Cyrillic card loses no fallback.
                if (
                    customer == CustomerType.TOURIST
                    and detected in {
                        DocumentType.NATIONAL_DRIVING_LICENCE_FRONT,
                        DocumentType.NATIONAL_DRIVING_LICENCE_BACK,
                    }
                    and preprocessed.quality.glare_detected
                    and has_glared_english_aamva_front_signature()
                ):
                    fallback_ocr = repair_ocr_until_complete(
                        DocumentType.NATIONAL_DRIVING_LICENCE_FRONT,
                        batch_all=True,
                    )
                    detected, _ = classify_document(
                        ocr.lines, expected, bool(mrz and mrz.valid),
                        [barcode.barcode_type for barcode in barcodes],
                    )
                    detected = _auto_document_type(
                        ocr.lines, detected, customer, licence_policy, upload_id,
                        barcodes, gcc_profile=gcc_profile, cache=cache,
                    )
                    if (
                        detected == DocumentType.NATIONAL_DRIVING_LICENCE_FRONT
                        and not (
                            _ocr_recovery_paths(
                                detected, customer, gcc_profile, ocr.lines,
                            )
                            - read_paths_for(detected) - previously_read
                        )
                    ):
                        deferred_languages = ()
                read_paths = (
                    read_paths_for(detected)
                    if detected != DocumentType.UNKNOWN else set()
                )
                missing_after_latin = (
                    _ocr_recovery_paths(detected, customer, gcc_profile, ocr.lines)
                    - read_paths - previously_read
                ) if detected != DocumentType.UNKNOWN else set()
                only_passport_issue_date = (
                    customer == CustomerType.TOURIST
                    and detected == DocumentType.PASSPORT_BIODATA
                    and missing_after_latin <= {"passport.issue_date"}
                    and has_visible_latin_passport_issue_label()
                )
                passport_latin_repair = (
                    customer == CustomerType.TOURIST
                    and detected == DocumentType.PASSPORT_BIODATA
                    and has_visible_latin_passport_date_layout()
                )
                california_latin_repair = (
                    customer == CustomerType.TOURIST
                    and detected == DocumentType.NATIONAL_DRIVING_LICENCE_FRONT
                    and is_california_driver_licence(ocr.lines)
                )
                if detected == DocumentType.UNKNOWN or (
                    missing_after_latin
                    and not only_passport_issue_date
                    and not passport_latin_repair
                    and not california_latin_repair
                ):
                    # A page the Latin pass could not name, or could name but
                    # not read out, is the page the postponed script exists for.
                    # Routing and the machine-readable zone are both settled
                    # again over the merged evidence, because a Cyrillic title
                    # can be what identifies the document.
                    ocr = merged_with(ocr, run_ocr(deferred_languages))
                    deferred_languages = ()
                    mrz = _best_mrz_from_ocr(
                        ocr.lines,
                        allow_printed_given_name_separator_repair=(
                            customer == CustomerType.TOURIST
                        ),
                    )
                    detected, _ = classify_document(
                        ocr.lines, expected, bool(mrz and mrz.valid),
                        [barcode.barcode_type for barcode in barcodes],
                    )
                    detected = _auto_document_type(
                        ocr.lines, detected, customer, licence_policy, upload_id,
                        barcodes, gcc_profile=gcc_profile, cache=cache,
                    )
            if detected != DocumentType.UNKNOWN:
                # Tourist passports normally derive every critical field but
                # their printed issue date from the Latin MRZ.  Read that
                # labelled row closely before full-page repairs or the
                # deferred Cyrillic recognizer.  Other customer routes retain
                # their exact existing order and latency behaviour.
                missing_after_classification = (
                    _ocr_recovery_paths(detected, customer, gcc_profile, ocr.lines)
                    - read_paths_for(detected) - previously_read
                )
                if (
                    customer == CustomerType.TOURIST
                    and detected == DocumentType.PASSPORT_BIODATA
                    and missing_after_classification
                    and merge_zoom_if_safe(detected, missing_after_classification)
                ):
                    page_notes.append(f"ZOOM_REREAD_APPLIED:{upload_id}")
                # The whole-page repair variants run once, and only if an
                # earlier stage has not already spent them.
                if fallback_ocr is None and not uae_fast_path:
                    missing_after_classification = (
                        _ocr_recovery_paths(
                            detected, customer, gcc_profile, ocr.lines,
                        )
                        - read_paths_for(detected) - previously_read
                    )
                    if missing_after_classification:
                        fallback_ocr = repair_ocr_until_complete(detected)
                # The close re-read is judged on what is still missing, not on
                # whether a repair pass happened to have run already. Nested
                # under that condition it never fired on the page that needed
                # it most: a capture poor enough to have spent its whole-page
                # variants earlier is exactly the capture with a row left to
                # recover, and the Queensland licence reached an operator with
                # no number while the pass written to find it was unreachable.
                still_missing = (
                    _ocr_recovery_paths(detected, customer, gcc_profile, ocr.lines)
                    - read_paths_for(detected) - previously_read
                )
                if still_missing and not uae_fast_path:
                    if merge_zoom_if_safe(detected, still_missing):
                        page_notes.append(f"ZOOM_REREAD_APPLIED:{upload_id}")
        elif detected == DocumentType.UNKNOWN and expected in DRIVING_DOCUMENT_TYPES:
            driving_section = {
                DocumentType.UAE_DRIVING_LICENCE_FRONT: "uae_driving_licence.",
                DocumentType.UAE_DRIVING_LICENCE_BACK: "uae_driving_licence.",
                DocumentType.INTERNATIONAL_DRIVING_PERMIT: "international_driving_permit.",
                DocumentType.NATIONAL_DRIVING_LICENCE_FRONT: "national_driving_licence.",
                DocumentType.NATIONAL_DRIVING_LICENCE_BACK: "national_driving_licence.",
                DocumentType.GCC_DRIVING_LICENCE_FRONT: "gcc_driving_licence.",
                DocumentType.GCC_DRIVING_LICENCE_BACK: "gcc_driving_licence.",
            }[expected]
            evidenced_fields = {
                candidate.field_path
                for candidate in cache.candidates(ocr.lines, expected)
                if candidate.field_path.startswith(driving_section)
                and candidate.validation_passed is not False
            }
            if expected == DocumentType.INTERNATIONAL_DRIVING_PERMIT:
                evidenced_fields.update(
                    candidate.field_path
                    for candidate in idp_layout_candidates(ocr.lines, upload_id)
                    if candidate.validation_passed is not False
                )
            # Two independently labelled licence fields (for example number +
            # expiry) are stronger evidence than a title OCR miss on a
            # multilingual or patterned card.
            if len(evidenced_fields) >= 2:
                detected = expected
        if (
            auto_detect
            and customer == CustomerType.GCC_NATIONAL
            and fallback_ocr is not None
        ):
            # The first readable GCC view can establish the wrong schema when
            # a shared personal-number/validity row survives but the stylised
            # document banner does not. Repair OCR may then recover the title;
            # settle the type again before any candidate is bound. Without this
            # second decision a Qatari licence expiry becomes an ID expiry and
            # all licence fields are deliberately rejected downstream.
            classified, _ = classify_document(
                ocr.lines, expected, bool(mrz and mrz.valid),
                [barcode.barcode_type for barcode in barcodes],
            )
            repaired_type = _auto_document_type(
                ocr.lines, classified, customer, licence_policy, upload_id,
                barcodes, gcc_profile=gcc_profile, cache=cache,
            )
            if (
                repaired_type != DocumentType.UNKNOWN
                and repaired_type != detected
            ):
                previous_type = detected
                detected = repaired_type
                page_notes.append(
                    "GCC_DOCUMENT_RECLASSIFIED_AFTER_REPAIR:"
                    f"{upload_id}:{previous_type.value}:{detected.value}",
                )
                still_missing = (
                    _ocr_recovery_paths(
                        detected, customer, gcc_profile, ocr.lines,
                    )
                    - read_paths_for(detected) - previously_read
                )
                if still_missing:
                    additional_repair = repair_ocr_until_complete(detected)
                    if additional_repair is not None:
                        fallback_ocr = merge_ocr_results(
                            fallback_ocr, additional_repair,
                        )
                # One final cheap routing decision over any remaining repair
                # rows; no further OCR is launched here.
                classified, _ = classify_document(
                    ocr.lines, expected, bool(mrz and mrz.valid),
                    [barcode.barcode_type for barcode in barcodes],
                )
                final_type = _auto_document_type(
                    ocr.lines, classified, customer, licence_policy,
                    upload_id, barcodes, gcc_profile=gcc_profile, cache=cache,
                )
                if final_type != DocumentType.UNKNOWN:
                    detected = final_type
                mismatched_issuer = (
                    _mismatched_issuer(ocr.lines, customer, gcc_profile)
                    or mismatched_issuer
                )
        type_match = detected != DocumentType.UNKNOWN if auto_detect else detected == expected
        extraction_type = detected if auto_detect else (
            expected if detected == DocumentType.UNKNOWN or (
                expected in DRIVING_DOCUMENT_TYPES and detected in DRIVING_DOCUMENT_TYPES
            ) else detected
        )
        private_translation_document = (
            customer == CustomerType.TOURIST
            and idp_is_private_translation_document(
                "\n".join(line.text for line in ocr.lines),
            )
        )
        if private_translation_document:
            # Business policy accepts this commercial international-licence
            # translation. It is deliberately routed through the
            # international-document fields so the card's own number and dates
            # can be shown, while its issuer country is never guessed from the
            # artwork or nationality.
            detected = extraction_type = DocumentType.INTERNATIONAL_DRIVING_PERMIT
            type_match = True
            page_notes.append(
                f"PRIVATE_DRIVER_LICENCE_TRANSLATION_ACCEPTED_BY_POLICY:{upload_id}",
            )
        if (
            customer == CustomerType.GCC_NATIONAL
            and detected != DocumentType.UNKNOWN
            and hasattr(self.ocr, "run_document_vl")
        ):
            # A generative OCR pass over one card costs well over ten seconds,
            # so it must be weighed against every piece of evidence already in
            # hand -- not against the labelled rows alone. The machine-readable
            # zone and any barcode were decoded before this point at no extra
            # cost, and on a GCC identity card the zone routinely carries the
            # birth date and sex the pass would have been sent to look for.
            free_evidence = [
                *cache.candidates(ocr.lines, extraction_type),
                *(
                    mrz_candidates(
                        mrz, upload_id, extraction_type,
                        licence_country=licence_country,
                    ) if mrz and mrz.mrz_type else []
                ),
                *barcode_candidates(
                    barcodes, upload_id, set(FIELD_PATHS), extraction_type,
                    licence_country=licence_country,
                ),
            ]
            supported_paths = {
                candidate.field_path for candidate in free_evidence
                if candidate.normalized_value and candidate.validation_passed is not False
            }
            still_missing = _critical_paths(extraction_type, gcc_profile) - supported_paths
            if still_missing:
                # Record what the pass was spent on. A generative read costs
                # more than the rest of the page put together, so a recurring
                # entry here is the signal that this country's profile is
                # missing a label rather than that the card is hard to read.
                page_notes.append(
                    "DOCUMENT_VL_INVOKED:" + ",".join(sorted(still_missing))
                )
                document_vl_variants = {
                    name: ocr.corrected_images.get(name, image)
                    for name, image in selected_variants.items()
                }
                ocr = merge_ocr_results(
                    ocr, self.ocr.run_document_vl(document_vl_variants),
                )
        if mismatched_issuer is not None:
            record = DocumentRecord(
                upload_id=upload_id,
                expected_type="AUTO_DETECT" if auto_detect else expected.value,
                detected_type=DocumentType.UNKNOWN.value, type_match=False,
                side="BACK" if expected.value.endswith("BACK") else "FRONT",
                quality=preprocessed.quality,
            )
            return (
                PageArtifact(
                    upload_id, expected, DocumentType.UNKNOWN, preprocessed,
                    ocr, mrz, preview,
                ),
                [],
                record,
                [f"{WRONG_CUSTOMER_TYPE}:{mismatched_issuer}:{upload_id}"],
            )
        record = DocumentRecord(
            upload_id=upload_id,
            expected_type="AUTO_DETECT" if auto_detect else expected.value,
            detected_type=detected.value,
            type_match=type_match, side="BACK" if expected.value.endswith("BACK") else "FRONT",
            quality=preprocessed.quality,
        )
        if extraction_type.value.endswith("BACK"):
            record.side = "BACK"
        if licence_policy and extraction_type in {
            DocumentType.INTERNATIONAL_DRIVING_PERMIT,
            DocumentType.NATIONAL_DRIVING_LICENCE_FRONT,
            DocumentType.NATIONAL_DRIVING_LICENCE_BACK,
            DocumentType.GCC_IDENTITY_FRONT,
            DocumentType.GCC_IDENTITY_BACK,
            DocumentType.GCC_DRIVING_LICENCE_FRONT,
            DocumentType.GCC_DRIVING_LICENCE_BACK,
        }:
            record.issuing_country_code = licence_policy.iso3
        warnings = [*ocr.warnings, *page_notes]
        compatible_driving_type = expected in DRIVING_DOCUMENT_TYPES and detected in DRIVING_DOCUMENT_TYPES
        if not auto_detect and not type_match and detected != DocumentType.UNKNOWN and not compatible_driving_type:
            warnings.append(f"WRONG_DOCUMENT_TYPE:{upload_id}:{expected.value}:{detected.value}")
            return PageArtifact(upload_id, expected, detected, preprocessed, ocr, mrz, preview), [], record, warnings
        if is_overseas_citizen_of_india_certificate(ocr.lines):
            # Shaped like a passport and like a licence, and neither. Keep the
            # page on the record so the operator can see it was uploaded, and
            # let nothing on it reach a customer field.
            warnings.append(
                f"UNSUPPORTED_DOCUMENT_PAGE:{upload_id}:"
                "OVERSEAS_CITIZEN_OF_INDIA_CERTIFICATE"
            )
            record.detected_type = DocumentType.UNKNOWN.value
            record.type_match = False
            record.issuing_country_code = None
            return (
                PageArtifact(upload_id, expected, detected, preprocessed, ocr, mrz, preview),
                [], record, warnings,
            )
        if (
            customer == CustomerType.TOURIST
            and extraction_type == DocumentType.NATIONAL_DRIVING_LICENCE_BACK
        ):
            # The Tourist workflow gets every licence value from the front.
            # A reverse can contain a barcode, category table, serial number
            # and entitlement dates; none is a substitute for the holder,
            # licence-number or validity rows on the front.  Retain it as a
            # diagnostic document record, but never let it fill, corroborate
            # or override a customer field.
            #
            # That holds for every card whose front carries its validity, and
            # it is not every card. The Moroccan licence prints the holder and
            # the licence number on the front and its validity only on the
            # reverse: "Date de délivrance" heads the category table and "Fin
            # de validité" captions the card's own expiry beneath it. Dropping
            # the page wholesale reported both dates as having no evidence,
            # with both printed and read at 0.98.
            #
            # So the reverse may carry the two validity dates, and only where
            # it captions them. The failure the rule was built to stop is a
            # guess made from the entitlement dates in the category table --
            # so a reading that came from ordering dates rather than from a
            # caption is still discarded here, and what survives is capped
            # below any reading from the front, which continues to decide.
            carried = [
                candidate
                for candidate in cache.candidates(ocr.lines, extraction_type)
                if candidate.field_path in _LICENCE_BACK_CARRIED_PATHS
                and candidate.validation_passed is not False
                and not any(
                    warning.startswith("LICENCE_DATE_ORDER_FALLBACK")
                    for warning in candidate.warnings
                )
            ]
            for candidate in carried:
                candidate.warnings.append(
                    f"NATIONAL_LICENCE_VALIDITY_READ_FROM_BACK:{upload_id}"
                )
                candidate.confidence = min(
                    candidate.confidence, _LICENCE_BACK_MAXIMUM_CONFIDENCE,
                )
            warnings.append(
                f"NATIONAL_LICENCE_BACK_VALIDITY_USED:{upload_id}" if carried
                else f"NATIONAL_LICENCE_BACK_VALUES_IGNORED:{upload_id}"
            )
            # Which state issued the card is not a holder value, so the rule
            # above does not reach it: a reverse that heads itself "ZIMBABWE
            # DRIVERS LICENCE" has named its issuer as plainly as any front,
            # and returning before the page was asked threw that reading away.
            # The bundle then had no licence evidence at all and settled its
            # acceptance rule from the passport's nationality -- the inference
            # the NATIONAL_ONLY check exists to catch -- leaving the issuing
            # country empty on a page that prints it in its title.
            back_country_evidence = _tourist_country_evidence(
                ocr.lines, detected, barcodes, customer,
            )
            return (
                PageArtifact(
                    upload_id, expected, detected, preprocessed, ocr, mrz,
                    preview, back_country_evidence,
                ),
                carried, record, warnings,
            )
        # A weak/partial OCR result can miss the exact classifier keywords while
        # still reading strongly labelled fields such as ID NUMBER or DATE OF
        # BIRTH. Keep those visible candidates when the classifier is UNKNOWN,
        # but cap their confidence so they always require human review. A
        # positively detected *different* document type is still rejected above.
        if auto_detect and detected == DocumentType.UNKNOWN:
            if (
                customer != CustomerType.TOURIST
                or not self.config.automatic_vlm_fallback
                or not self.vlm.state.loaded
            ):
                warnings.append(f"UNRECOGNIZED_DOCUMENT_PAGE:{upload_id}")
                return PageArtifact(upload_id, expected, detected, preprocessed, ocr, mrz, preview), [], record, warnings
            # A licence in a script the fast Latin/Cyrillic path cannot read is
            # exactly what the multilingual visual router exists for. Continue
            # with no deterministic candidates and let the one model pass both
            # classify the page and recover only the requested tourist fields.
            warnings.append(f"MULTILINGUAL_VISUAL_ROUTING_REQUIRED:{upload_id}")
        selected_lines = (
            # GCC labels and long bilingual names are frequently divided
            # across the original and one bounded repair view. Bind them once
            # over the merged in-memory evidence; this adds no OCR/model call
            # and is the only way a label from one view can name the value the
            # other recovered. Other workflows retain their established
            # per-pass candidate policy.
            list(ocr.lines)
            if customer == CustomerType.GCC_NATIONAL
            else [
                line for line in ocr.lines
                if any(
                    line.variant == variant
                    or line.variant.startswith(f"{variant}:paddleocr_vl")
                    for variant in selected_variants
                )
            ]
        )
        candidates = cache.candidates(selected_lines, extraction_type)
        if fallback_ocr is not None:
            present_paths = {candidate.field_path for candidate in candidates}
            # Each pass decides on its own evidence, so a pass that read fewer
            # rows can reach a conclusion the first pass had already refused.
            # On a Dutch licence the first pass read 4b from its designator and
            # correctly left the covered 4a empty; the contrast pass, run to
            # hunt the licence number, missed 4b entirely, fell back to ordering
            # three numbers, and filled the issue date from the category table
            # on the reverse -- into the very field the first pass had declined
            # to guess at. A guess is only admissible where the page as a whole
            # has no read date to contradict it.
            read_licence_dates = {
                candidate.field_path for candidate in candidates
                if candidate.field_path.startswith("national_driving_licence.")
                and candidate.field_path.endswith(("issue_date", "expiry_date"))
            }
            candidates.extend(
                candidate
                for candidate in cache.candidates(fallback_ocr.lines, extraction_type)
                if candidate.field_path not in present_paths
                and not (
                    read_licence_dates
                    and "LICENCE_DATE_ORDER_FALLBACK_REQUIRES_REVIEW" in candidate.warnings
                )
            )
            sensitive_paths: set[str] = set()
            if extraction_type in {
                DocumentType.EMIRATES_ID_FRONT,
                DocumentType.EMIRATES_ID_BACK,
                DocumentType.UAE_DRIVING_LICENCE_FRONT,
                DocumentType.UAE_DRIVING_LICENCE_BACK,
                DocumentType.GCC_IDENTITY_FRONT,
                DocumentType.GCC_IDENTITY_BACK,
                DocumentType.GCC_DRIVING_LICENCE_FRONT,
                DocumentType.GCC_DRIVING_LICENCE_BACK,
            }:
                sensitive_paths = {
                    "emirates_id.number", "emirates_id.issue_date",
                    "emirates_id.expiry_date", "uae_driving_licence.number",
                    "gcc_identity.number", "gcc_identity.issue_date",
                    "gcc_identity.expiry_date", "gcc_driving_licence.number",
                    "gcc_driving_licence.issue_date", "gcc_driving_licence.expiry_date",
                }
            elif extraction_type == DocumentType.PASSPORT_BIODATA:
                sensitive_paths = {
                    "personal_info.date_of_birth", "personal_info.gender",
                    "personal_info.nationality_name", "passport.number",
                    "passport.issue_date", "passport.expiry_date",
                }
            elif extraction_type in {
                DocumentType.NATIONAL_DRIVING_LICENCE_FRONT,
                DocumentType.NATIONAL_DRIVING_LICENCE_BACK,
            }:
                sensitive_paths = {
                    "personal_info.date_of_birth",
                    "national_driving_licence.number",
                    "national_driving_licence.issue_date",
                    "national_driving_licence.expiry_date",
                }
            elif extraction_type == DocumentType.INTERNATIONAL_DRIVING_PERMIT:
                sensitive_paths = {
                    "personal_info.date_of_birth",
                    "international_driving_permit.number",
                    "international_driving_permit.issue_date",
                    "international_driving_permit.expiry_date",
                }
            if sensitive_paths:
                # Labels and their small numeric values are often recovered by
                # different renderings of the same page.  Deciding each OCR
                # pass separately leaves a plausible but wrong first-pass
                # value in place even when the repair pass read the right row.
                # That is what happened on the reported Ontario card: the
                # original pass attached DOB 1998/04/22 to expiry, while the
                # fallback read 4a=2024/10/29 and 4b=2028/11/28 exactly.  Bind
                # these fields once over the already-merged lines and replace
                # the per-pass candidates.  This launches no OCR/model work;
                # it is one in-memory pass over the roughly 100 returned rows.
                combined_sensitive = [
                    candidate
                    for candidate in cache.candidates(ocr.lines, extraction_type)
                    if candidate.field_path in sensitive_paths
                ]
                combined_paths = {
                    candidate.field_path for candidate in combined_sensitive
                    if candidate.normalized_value
                }
                if combined_paths:
                    candidates = [
                        candidate for candidate in candidates
                        if candidate.field_path not in combined_paths
                    ]
                    candidates.extend(combined_sensitive)
        if extraction_type in {
            DocumentType.NATIONAL_DRIVING_LICENCE_FRONT,
            DocumentType.NATIONAL_DRIVING_LICENCE_BACK,
        }:
            # The first OCR pass uses only the normal image. It can read the
            # category dates while missing the tiny 9/10/11/12 headings, so its
            # ordering fallback cannot know those dates belong to the reverse
            # table. A later repair pass reads the headings, but the merge used
            # to replace only fields the combined pass populated; because the
            # combined pass correctly produced no document expiry, the wrong
            # first-pass expiry survived and then caused the true 4a issue date
            # to be rejected as chronologically impossible.
            category_dates = licence_category_table_dates(ocr.lines)
            refused_category_dates = [
                candidate for candidate in candidates
                if candidate.field_path in {
                    "national_driving_licence.issue_date",
                    "national_driving_licence.expiry_date",
                }
                and candidate.normalized_value in category_dates
                and "LICENCE_DATE_ORDER_FALLBACK_REQUIRES_REVIEW"
                in candidate.warnings
            ]
            if refused_category_dates:
                candidates = [
                    candidate for candidate in candidates
                    if candidate not in refused_category_dates
                ]
                warning = f"LICENCE_CATEGORY_TABLE_DATE_REFUSED:{upload_id}"
                if warning not in warnings:
                    warnings.append(warning)
        if extraction_type == DocumentType.INTERNATIONAL_DRIVING_PERMIT:
            # Numeric rows can be split between the normal and contrast OCR
            # passes. Run the IDP layout parser over their merged evidence.
            # Use the raw multilingual rows here, before duplicate suppression:
            # a clean Russian numeric line can overlap a garbled English line.
            #
            # The page's red print is passed with them. A booklet sets its own
            # number in red and everything else in black, so the colour says
            # which of the page's long numbers is the permit's -- the one cue
            # that does not depend on a 5pt label surviving a phone capture.
            # Only the normalized image is measured; the OCR boxes are in its
            # coordinate space, and every greyscale variant has thrown the
            # colour away by definition.
            layout_candidates = (
                private_international_driver_licence_candidates(ocr.lines, upload_id)
                if private_translation_document
                else idp_layout_candidates(
                    ocr.lines, upload_id,
                    red_boxes=red_ink_boxes(preprocessed.normalized),
                )
            )
            visible_idp_country = country_from_text(
                "\n".join(line.text for line in ocr.lines),
            )
            if (
                licence_country == "Algeria"
                or (
                    visible_idp_country is not None
                    and visible_idp_country.country == "Algeria"
                )
            ):
                dotted = dot_matrix_number(preprocessed.normalized)
                if dotted is not None:
                    number, (x1, y1, x2, y2), confidence = dotted
                    layout_candidates = [
                        candidate for candidate in layout_candidates
                        if candidate.field_path != "international_driving_permit.number"
                    ]
                    layout_candidates.append(FieldCandidate(
                        field_path="international_driving_permit.number",
                        value=number, normalized_value=number,
                        source_document=upload_id,
                        source_method="document_parser",
                        confidence=confidence,
                        evidence_text=f"dot-matrix:{number}",
                        bounding_box=[
                            [x1, y1], [x2, y1], [x2, y2], [x1, y2],
                        ],
                        validation_passed=True,
                        warnings=["ALGERIAN_IDP_DOT_MATRIX_NUMBER"],
                    ))
            if customer == CustomerType.TOURIST and not private_translation_document:
                idp_text = "\n".join(line.text for line in ocr.lines)
                if idp_is_non_government_translation(idp_text):
                    warning = f"NON_GOVERNMENT_IDP_TRANSLATION_REQUIRES_REVIEW:{upload_id}"
                    if warning not in warnings:
                        warnings.append(warning)
            layout_paths = {candidate.field_path for candidate in layout_candidates}
            if layout_paths:
                candidates = [candidate for candidate in candidates if candidate.field_path not in layout_paths]
                candidates.extend(layout_candidates)
            combined_candidates = cache.candidates(ocr.lines, extraction_type)
            present_paths = {candidate.field_path for candidate in candidates}
            candidates.extend(
                candidate for candidate in combined_candidates
                if candidate.field_path not in present_paths
            )
        if extraction_type == DocumentType.NATIONAL_DRIVING_LICENCE_FRONT and any(
            "LICENCE_CATEGORY_TABLE" in candidate.warnings for candidate in candidates
        ):
            # The category table is printed on the reverse. Reading one from a
            # page classified as the front means both sides were captured into
            # a single image, as this Albanian bundle was, and the reverse was
            # being reported as never supplied while its rows sat in the same
            # picture as the front's.
            record.side = "FRONT_AND_BACK"
            warnings.append(f"BOTH_LICENCE_SIDES_IN_ONE_IMAGE:{upload_id}")
        if mrz and mrz.mrz_type:
            # A GCC identity back carries a TD1 zone whose birth date, expiry,
            # sex and nationality are checksum-protected.
            candidates.extend(mrz_candidates(
                mrz, upload_id, extraction_type, licence_country=licence_country,
            ))
        # A UAE ID can lose the right-hand filler/name tail from its TD1 zone.
        # Its middle row still provides sex once the two independently
        # checksummed dates prove the row's alignment.
        candidates.extend(emirates_id_partial_mrz_gender_candidates(
            ocr.lines, upload_id, extraction_type,
        ))
        if extraction_type == DocumentType.PASSPORT_BIODATA and not any(
            candidate.field_path == "passport.issue_date"
            and candidate.normalized_value
            for candidate in candidates
        ):
            # Only where the printed row was not read. A label bound to its own
            # value states the issue date; the zone can only bracket it.
            candidates.extend(
                passport_issue_date_from_mrz(ocr.lines, upload_id, mrz),
            )
        if (
            customer == CustomerType.TOURIST
            and extraction_type == DocumentType.PASSPORT_BIODATA
            and not any(
                candidate.field_path == "passport.issued_by_code"
                and candidate.normalized_value
                and policy_for_country(candidate.normalized_value) is not None
                for candidate in candidates
            )
        ):
            visible_issuer = country_from_passport_lines(
                [line.text for line in ocr.lines],
            )
            issuer_policy = (
                policy_for_country(visible_issuer.country)
                if visible_issuer is not None else None
            )
            if visible_issuer is not None and issuer_policy is not None:
                # A structurally parsed but non-country token (for example raw
                # filler from an incomplete MRZ issuer slot) must neither block
                # this fallback nor survive beside it as a false conflict.
                candidates = [
                    candidate for candidate in candidates
                    if not (
                        candidate.field_path == "passport.issued_by_code"
                        and candidate.normalized_value
                        and policy_for_country(candidate.normalized_value) is None
                    )
                ]
                evidence_line = next((
                    line for line in ocr.lines
                    if line.text.strip() == visible_issuer.evidence_text
                ), None)
                candidates.append(FieldCandidate(
                    field_path="passport.issued_by_code",
                    value=issuer_policy.iso3,
                    normalized_value=issuer_policy.iso3,
                    source_document=upload_id,
                    source_method="document_parser",
                    confidence=visible_issuer.confidence,
                    evidence_text=visible_issuer.evidence_text,
                    bounding_box=(
                        evidence_line.bounding_box if evidence_line else None
                    ),
                    validation_passed=True,
                    warnings=["PASSPORT_ISSUER_FROM_VISIBLE_DOCUMENT_EVIDENCE"],
                ))
                record.issuing_country_code = issuer_policy.iso3
        candidates.extend(barcode_candidates(
            barcodes, upload_id, set(FIELD_PATHS), extraction_type,
            licence_country=licence_policy.country if licence_policy else None,
        ))
        if (
            customer == CustomerType.TOURIST
            and extraction_type == DocumentType.NATIONAL_DRIVING_LICENCE_FRONT
        ):
            # California's visible short captions (ISS/EXP/SEX) and its one
            # letter/seven digit customer number remain usable when the tiny
            # AAMVA designators disappear in glare.  Apply these candidates
            # before deciding whether Qwen needs a second, costly reading.
            california = california_licence_layout_candidates(ocr.lines, upload_id)
            california_paths = {candidate.field_path for candidate in california}
            if california_paths:
                candidates = [
                    candidate for candidate in candidates
                    if candidate.field_path not in california_paths
                ]
                candidates.extend(california)
        if not auto_detect and compatible_driving_type and not type_match:
            warnings.append(f"COMPATIBLE_DRIVING_DOCUMENT_TYPE:{upload_id}:{expected.value}:{detected.value}")
            for candidate in candidates:
                candidate.confidence = min(candidate.confidence, 0.78)
                if "DRIVING_DOCUMENT_SUBTYPE_UNCONFIRMED" not in candidate.warnings:
                    candidate.warnings.append("DRIVING_DOCUMENT_SUBTYPE_UNCONFIRMED")
        elif not auto_detect and detected == DocumentType.UNKNOWN:
            warnings.append(f"DOCUMENT_TYPE_UNCONFIRMED:{upload_id}:{expected.value}")
            for candidate in candidates:
                candidate.confidence = min(candidate.confidence, 0.60)
                if "DOCUMENT_TYPE_UNCONFIRMED" not in candidate.warnings:
                    candidate.warnings.append("DOCUMENT_TYPE_UNCONFIRMED")
        deterministic_paths = {candidate.field_path for candidate in candidates if candidate.confidence >= 0.80}
        visibly_supported_paths = {
            candidate.field_path for candidate in candidates
            if candidate.validation_passed is not False and candidate.normalized_value
        }
        # Which state this page names as the issuer of the driving document on
        # it. Read here rather than at the end of the page, because a card that
        # states its own country has answered the one question the model would
        # otherwise be sent to answer.
        country_evidence = _tourist_country_evidence(
            ocr.lines, detected, barcodes, customer,
        )
        if (
            customer == CustomerType.TOURIST
            and extraction_type == DocumentType.INTERNATIONAL_DRIVING_PERMIT
        ):
            permit_country = country_from_idp_permit_label(
                "\n".join(line.text for line in ocr.lines),
            )
            if permit_country is not None and not any(
                item.country == permit_country.country
                and item.source == permit_country.source
                for item in country_evidence
            ):
                country_evidence.append(permit_country)
        vlm_paths = VLM_FIELD_PATHS_BY_DOCUMENT.get(extraction_type, set())
        if extraction_type == DocumentType.PASSPORT_BIODATA:
            # A failed MRZ says that its checksums could not prove the page;
            # it does not say every visible field is absent. Calling Qwen on
            # every such passport costs seconds and made otherwise complete
            # multi-page tourist bundles miss the ten-second warm target.
            # Reserve it for a genuinely unread critical passport field; the
            # existing recapture warning still tells the operator that the MRZ
            # was not independently validated.
            requested_vlm_paths = (
                _critical_paths(extraction_type, gcc_profile)
                - _with_derivable(visibly_supported_paths)
                - previously_read
            )
            should_use_vlm = bool(requested_vlm_paths)
        elif (
            customer == CustomerType.TOURIST
            and extraction_type == DocumentType.NATIONAL_DRIVING_LICENCE_FRONT
            and not (
                _ocr_recovery_paths(
                    extraction_type, customer, gcc_profile, ocr.lines,
                )
                - _with_derivable(visibly_supported_paths)
            )
            and country_evidence
        ):
            # The card itself has proved its issuer, number and validity.
            # Passport identity is authoritative for a Tourist, so generating
            # a second visual reading merely to copy a licence name/DOB cannot
            # improve the exported record.  It did add a full Qwen generation
            # after the concrete Russian front reader had already settled every
            # licence value.  Keep Qwen for any missing licence value or
            # unknown issuer; this is only the fully-read fast path.
            requested_vlm_paths = set()
            should_use_vlm = False
        elif extraction_type in VLM_FIELD_PATHS_BY_DOCUMENT:
            # Avoid an unnecessary second GPU generation when labelled OCR has
            # already recovered every critical permit field. Russian booklet
            # pages still invoke Qwen whenever a name, DOB, number, or date is
            # absent.
            critical = _critical_paths(extraction_type, gcc_profile)
            missing_critical = critical - visibly_supported_paths - previously_read
            requested_vlm_paths = set(missing_critical)
            if "personal_info.full_name" in missing_critical:
                is_saudi_gcc = bool(
                    gcc_profile and gcc_profile.iso3 == "SAU"
                    and extraction_type in {
                        DocumentType.GCC_IDENTITY_FRONT, DocumentType.GCC_IDENTITY_BACK,
                        DocumentType.GCC_DRIVING_LICENCE_FRONT, DocumentType.GCC_DRIVING_LICENCE_BACK,
                    }
                )
                requested_vlm_paths.add("personal_info.full_name")
                if not is_saudi_gcc:
                    requested_vlm_paths.update({
                        "personal_info.first_name", "personal_info.last_name",
                    })
                if extraction_type not in {
                    DocumentType.GCC_IDENTITY_FRONT, DocumentType.GCC_IDENTITY_BACK,
                    DocumentType.GCC_DRIVING_LICENCE_FRONT, DocumentType.GCC_DRIVING_LICENCE_BACK,
                }:
                    requested_vlm_paths.add("personal_info.middle_name")
            if (
                extraction_type == DocumentType.NATIONAL_DRIVING_LICENCE_FRONT
                and not country_evidence
                and "national_driving_licence.issued_by_code" not in visibly_supported_paths
            ):
                # Designator 4c names the authority, and on many cards that is a
                # city rather than a state, so the model is asked which country
                # issued the licence. It is asked only where the page named no
                # state of its own: every European card prints its state across
                # the top, and asking anyway is what put the card's title,
                # "PATENTE DI GUIDA", into the issuing-country field of an
                # Italian licence that says REPUBBLICA ITALIANA beside it.
                requested_vlm_paths.update({
                    "national_driving_licence.issued_by_code",
                    "national_driving_licence.issued_by_name",
                })
            elif extraction_type == DocumentType.UAE_DRIVING_LICENCE_FRONT:
                requested_vlm_paths.update({
                    "uae_driving_licence.issued_by_code",
                    "uae_driving_licence.issued_by_name",
                })
            elif extraction_type in {
                DocumentType.EMIRATES_ID_FRONT,
                DocumentType.EMIRATES_ID_BACK,
            }:
                ocr_text = " ".join(line.text.upper() for line in ocr.lines)
                issue_markers = (
                    "ISSUE DATE", "DATE OF ISSUE", "ISSUING DATE",
                    "تاريخ الإصدار", "تاريخ إصدار البطاقة",
                )
                if (
                    "emirates_id.issue_date" not in visibly_supported_paths
                    and any(marker in ocr_text for marker in issue_markers)
                ):
                    requested_vlm_paths.add("emirates_id.issue_date")
            if extraction_type in GCC_DOCUMENT_TYPES:
                # Never ask the model for a field this country does not print
                # on this side of this card. On a Bahraini identity front that
                # removes the birth date, the sex and the issue date from the
                # request, which is usually the whole request.
                requested_vlm_paths &= set(printed_fields(gcc_profile, extraction_type))
            should_use_vlm = bool(requested_vlm_paths)
        else:
            should_use_vlm = (
                len(deterministic_paths) < 3
                or (customer == CustomerType.TOURIST and not country_evidence)
            )
            requested_vlm_paths = (
                set(TOURIST_EXTRACTION_FIELDS)
                if customer == CustomerType.TOURIST else set(FIELD_PATHS)
            )
        if customer == CustomerType.UAE_RESIDENT:
            # A misclassified or supplementary page must not make the UAE
            # workflow ask the visual model for passport or place-of-birth
            # data.  On the fast path it also cannot turn a one-pass read into
            # a multi-second generation; absent fields stay marked for review.
            requested_vlm_paths &= UAE_RESIDENT_EXTRACTION_FIELDS
            should_use_vlm = bool(requested_vlm_paths) and not uae_fast_path
        if self.config.automatic_vlm_fallback and self.vlm.state.loaded and should_use_vlm:
            expected_label = extraction_type.value
            if licence_policy and extraction_type in {
                DocumentType.INTERNATIONAL_DRIVING_PERMIT,
                DocumentType.NATIONAL_DRIVING_LICENCE_FRONT,
                DocumentType.NATIONAL_DRIVING_LICENCE_BACK,
            }:
                expected_label = (
                    f"{extraction_type.value} issued by {licence_policy.country}; "
                    f"template family {licence_policy.template_family.value}"
                )
            if gcc_profile and extraction_type in {
                DocumentType.GCC_IDENTITY_FRONT, DocumentType.GCC_IDENTITY_BACK,
                DocumentType.GCC_DRIVING_LICENCE_FRONT, DocumentType.GCC_DRIVING_LICENCE_BACK,
            }:
                layout = (
                    gcc_profile.identity_layout
                    if extraction_type in {DocumentType.GCC_IDENTITY_FRONT, DocumentType.GCC_IDENTITY_BACK}
                    else gcc_profile.licence_layout
                )
                expected_label = (
                    f"{extraction_type.value} issued by {gcc_profile.country}; "
                    f"verified layout: {layout}"
                )
            payload = self.vlm.extract(
                preview, expected_label, ocr.lines,
                {path: "string|null" for path in sorted(requested_vlm_paths)},
            )
            accepted_vlm_paths = set(requested_vlm_paths)
            if (
                gcc_profile and gcc_profile.iso3 == "SAU"
                and "personal_info.full_name" in requested_vlm_paths
            ):
                accepted_vlm_paths.update({"personal_info.first_name", "personal_info.last_name"})
            grounded = [
                FieldCandidate.model_validate(item)
                for item in vlm_candidates(
                    payload, upload_id,
                    licence_country=licence_policy.country if licence_policy else None,
                )
                if item.get("field_path") in accepted_vlm_paths
            ]
            if extraction_type in {
                DocumentType.NATIONAL_DRIVING_LICENCE_FRONT,
                DocumentType.NATIONAL_DRIVING_LICENCE_BACK,
            }:
                # The reverse's columns 9-12 describe vehicle entitlements,
                # not the card's own validity. When 4b is unreadable, the model
                # is asked only for expiry and can return the clearest date it
                # sees in that table. Do not ground a document-level date whose
                # only visible source is inside the category table.
                category_dates = licence_category_table_dates(ocr.lines)
                refused_category_dates = [
                    candidate for candidate in grounded
                    if candidate.field_path in {
                        "national_driving_licence.issue_date",
                        "national_driving_licence.expiry_date",
                    }
                    and candidate.normalized_value in category_dates
                ]
                if refused_category_dates:
                    grounded = [
                        candidate for candidate in grounded
                        if candidate not in refused_category_dates
                    ]
                    warning = f"LICENCE_CATEGORY_TABLE_DATE_REFUSED:{upload_id}"
                    if warning not in warnings:
                        warnings.append(warning)
            if grounded:
                grounded_paths = {candidate.field_path for candidate in grounded}
                # For multilingual identity documents, Qwen resolves the field
                # semantics while MRZ/barcode remain authoritative. Remove only
                # heuristic OCR candidates for fields Qwen actually returned;
                # untouched OCR fields remain available when Qwen returns null.
                candidates = [
                    candidate for candidate in candidates
                    if candidate.field_path not in grounded_paths
                    or candidate.source_method == "mrz"
                    or candidate.source_method == "document_parser"
                    or candidate.source_method.startswith("barcode:")
                ]
                candidates.extend(grounded)
            routing = payload.get("routing")
            if customer == CustomerType.TOURIST and isinstance(routing, dict):
                routed_languages = routing.get("languages")
                if isinstance(routed_languages, list):
                    record.detected_languages = [
                        str(language) for language in routed_languages[:4]
                    ]
                routed_type_value = routing.get("document_type")
                try:
                    routed_type = DocumentType(str(routed_type_value))
                except ValueError:
                    routed_type = DocumentType.UNKNOWN
                if (
                    detected == DocumentType.UNKNOWN
                    and routed_type in {
                        DocumentType.PASSPORT_BIODATA,
                        DocumentType.NATIONAL_DRIVING_LICENCE_FRONT,
                        DocumentType.NATIONAL_DRIVING_LICENCE_BACK,
                        DocumentType.INTERNATIONAL_DRIVING_PERMIT,
                    }
                ):
                    detected = routed_type
                    extraction_type = routed_type
                    record.detected_type = routed_type.value
                    record.type_match = True
                    warnings = [
                        warning for warning in warnings
                        if not warning.startswith("UNRECOGNIZED_DOCUMENT_PAGE:")
                    ]
                    warnings.append(
                        f"DOCUMENT_TYPE_FROM_MULTILINGUAL_VISUAL_ROUTER:{upload_id}"
                    )
                routed_country = policy_for_country(
                    routing.get("issuing_country_code")
                    or routing.get("issuing_country_name")
                )
                if (
                    routed_country is not None
                    and detected in TOURIST_DRIVING_TYPES
                    and not country_evidence
                ):
                    vlm_country = CountryEvidence(
                        country=routed_country.country,
                        source=DetectionSource.VLM_VISUAL,
                        confidence=0.72,
                        evidence_text=(
                            f"visual:{routed_country.iso3}:"
                            + ",".join(record.detected_languages)
                        ),
                    )
                    country_evidence.append(vlm_country)
                    record.issuing_country_code = routed_country.iso3
            if payload.get("warning"): warnings.append(payload["warning"])
        # Country auto-detection necessarily happens after the first generic
        # extraction pass.  Once this page has established a U.S. issuer, apply
        # the incompatible AAMVA numbering semantics to the final merged OCR:
        # 4d is the licence/customer number, 5 is only the document
        # discriminator, 4a is issue and 4b expiry.  Printed designators are
        # authoritative, so discard earlier proximity-bound guesses for paths
        # that the AAMVA pass recovered.
        is_us_licence = (
            customer == CustomerType.TOURIST
            and extraction_type in {
                DocumentType.NATIONAL_DRIVING_LICENCE_FRONT,
                DocumentType.NATIONAL_DRIVING_LICENCE_BACK,
            }
            and (
                licence_country == "United States"
                or any(item.country == "United States" for item in country_evidence)
            )
        )
        if is_us_licence:
            aamva = _numbered_national_licence_candidates(
                ocr.lines, upload_id, licence_country="United States",
            )
            california_number = next((
                candidate for candidate in california_licence_layout_candidates(
                    ocr.lines, upload_id,
                )
                if candidate.field_path == "national_driving_licence.number"
            ), None)
            if california_number is not None:
                # A glare-damaged California ``4d`` is routinely recognised
                # with duplicated leading letters (for example ``ABC...``),
                # producing several apparently strong but impossible AAMVA
                # numbers.  Its state-specific one-letter/seven-digit number
                # is the better evidence and was already attached above;
                # never reintroduce the malformed generic 4d readings here.
                aamva = [
                    candidate for candidate in aamva
                    if candidate.field_path != "national_driving_licence.number"
                ]
            if not any(
                candidate.field_path == "national_driving_licence.number"
                for candidate in aamva
            ):
                aamva.extend(american_unlabelled_licence_number_candidates(
                    ocr.lines, upload_id,
                ))
            aamva_paths = {candidate.field_path for candidate in aamva}
            if aamva_paths:
                candidates = [
                    candidate for candidate in candidates
                    if candidate.field_path not in aamva_paths
                ]
                candidates.extend(aamva)
        # Canada is the other half of the same problem and needs the opposite
        # treatment. A U.S. card labels its number with an abbreviation the
        # reader knows; several Canadian cards label it with nothing at all but
        # a four-point designator, and the province decides both the shape of
        # the number and which designator carries it. The province is on the
        # card, so it is read from the page rather than asked of the operator:
        # a bundle whose selected country is Canada says nothing about which of
        # thirteen documents was uploaded.
        is_canadian_licence = (
            customer == CustomerType.TOURIST
            and extraction_type in {
                DocumentType.NATIONAL_DRIVING_LICENCE_FRONT,
                DocumentType.NATIONAL_DRIVING_LICENCE_BACK,
            }
            and (
                licence_country == "Canada"
                or any(item.country == "Canada" for item in country_evidence)
            )
        )
        if is_canadian_licence:
            # The card prints the holder's birth date, and on the two provinces
            # that encode it in the number that is what turns a shape match
            # into a confirmed read. Taken from this page only: a date carried
            # over from another upload would be confirming the licence against
            # a document the licence has not yet been matched to.
            printed_birth_date = next(
                (
                    candidate.normalized_value
                    for candidate in sorted(
                        candidates, key=lambda item: -item.confidence,
                    )
                    if candidate.field_path == "personal_info.date_of_birth"
                    and candidate.normalized_value
                ),
                None,
            )
            printed_surname = next(
                (
                    candidate.normalized_value
                    for candidate in sorted(
                        candidates, key=lambda item: -item.confidence,
                    )
                    if candidate.field_path == "personal_info.last_name"
                    and candidate.normalized_value
                ),
                None,
            )
            provincial = canadian_licence_candidates(
                ocr.lines, upload_id, known_birth_date=printed_birth_date,
                known_surname=printed_surname,
            )
            # Anything the card states in words outranks this pass, which reads
            # a number by its shape and an issuer by its name. It fills the
            # fields the labelled pass left empty; it does not overrule it.
            already = {
                candidate.field_path for candidate in candidates
                if candidate.normalized_value and candidate.confidence >= 0.80
            }
            candidates.extend(
                candidate for candidate in provincial
                if candidate.field_path not in already
            )
        if extraction_type in {
            DocumentType.NATIONAL_DRIVING_LICENCE_FRONT,
            DocumentType.NATIONAL_DRIVING_LICENCE_BACK,
        }:
            # A licence is not issued on the day it expires. Where the two
            # dated fields share one printed row -- Ontario sets "4a ISS/DÉL
            # 2025/10/22" beside "4b EXP/ EXP. 2028/01/12" -- a pass that read
            # only one of them binds the issue label to the expiry date, and
            # the reading is as confident as a correct one.
            #
            # Refusing it is half the answer. The other half is that the card
            # still states its issue date, and the ordering fallback below
            # reads it correctly from nothing but the fact that a holder is
            # born, then licensed, then expires. That fallback is skipped
            # whenever the labelled pass produced an issue date, so a wrong one
            # suppressed it; the field then came back empty from a card whose
            # issue date the recogniser had read at 1.00.
            expiries = [
                candidate.normalized_value for candidate in candidates
                if candidate.field_path == "national_driving_licence.expiry_date"
                and candidate.normalized_value
            ]
            if expiries:
                latest = max(expiries)
                refused = [
                    candidate for candidate in candidates
                    if candidate.field_path == "national_driving_licence.issue_date"
                    and candidate.normalized_value
                    and candidate.normalized_value >= latest
                ]
                if refused:
                    candidates = [
                        candidate for candidate in candidates
                        if candidate not in refused
                    ]
                    warnings.append(
                        f"LICENCE_ISSUE_DATE_NOT_BEFORE_EXPIRY_REFUSED:{upload_id}"
                    )
                # Whether a wrong issue date was just removed or none was ever
                # read, the card still states one and the ordering below is
                # what reads it. Requiring a refusal first meant a Michigan
                # licence -- whose issue date shares an OCR box with its
                # number, so no caption ever reached it -- reported no issue
                # date at all while printing it beside the number.
                if (
                    refused or prints_american_licence_layout(ocr.lines)
                ) and not any(
                    candidate.field_path == "national_driving_licence.issue_date"
                    and candidate.normalized_value
                    for candidate in candidates
                ):
                    birth = next(
                        (
                            candidate.normalized_value
                            for candidate in sorted(
                                candidates, key=lambda item: -item.confidence,
                            )
                            if candidate.field_path == "personal_info.date_of_birth"
                            and candidate.normalized_value
                        ),
                        None,
                    )
                    candidates.extend(
                        candidate
                        for candidate in national_licence_date_sequence(
                            ocr.lines, upload_id, known_birth_date=birth,
                        )
                        if candidate.field_path
                        == "national_driving_licence.issue_date"
                    )
        if record.quality.unreadable:
            # The page has already been measured as one this reader cannot
            # read. Everything that survives here is either something the
            # document proved -- a checksummed zone, a barcode, a value bound
            # to its own printed label -- or something inferred from where ink
            # happened to land. On a page this poor the second kind is not
            # weak evidence, it is noise with a field name attached: the
            # Queensland licence in this project's bug report came back with a
            # class-table row as its issue date, from a capture the reader had
            # itself flagged UNREADABLE.
            #
            # Matched on the naming convention rather than on a list, so a
            # fallback added later is covered the day it is written instead of
            # the day it first reaches an operator.
            # Flagged, not discarded. Discarding was the first thing tried here
            # and it was the wrong trade: every candidate from a page this poor
            # is already forced to review by the low-quality penalty, so
            # removing it took a value an operator could see on the card and
            # correct in a second, and gave back an empty box and no clue what
            # belonged in it. A page worth recapturing is worth saying so about;
            # it is not a reason to withhold what was read.
            warnings.append(f"RECAPTURE_REQUIRED:{upload_id}")
        if customer == CustomerType.UAE_RESIDENT:
            # MRZ, barcode and model candidates are produced outside the
            # labelled-OCR cache. Apply the workflow boundary once more before
            # candidates can be reconciled into a customer field.
            candidates = [
                candidate for candidate in candidates
                if candidate.field_path in UAE_RESIDENT_EXTRACTION_FIELDS
            ]
        return PageArtifact(
            upload_id, expected, detected, preprocessed, ocr, mrz, preview,
            country_evidence,
        ), candidates, record, warnings

    def process(self, customer_type: CustomerType | str, uploads: dict[str, Any], session: ProcessingSession | None = None) -> ProcessingSession:
        started = time.perf_counter()
        stage_seconds: dict[str, float] = {}
        _STAGE_SECONDS.set(stage_seconds)
        # The stages that leave this process or occupy it for whole seconds.
        # Everything unaccounted for is the extraction and reconciliation
        # between them, which is arithmetic on text and costs milliseconds.
        _instrument(self.ocr, "run_languages", "ocr")
        _instrument(self.ocr, "run", "ocr")
        _instrument(self.ocr, "run_document_vl", "ocr_document_vl")
        _instrument(self.vlm, "extract", "vlm")
        customer = CustomerType(customer_type)
        uae_fast_deadline = (
            started + self.config.uae_processing_budget_seconds
            if (
                customer == CustomerType.UAE_RESIDENT
                and self.config.uae_fast_path_enabled
                and self.config.uae_processing_budget_seconds > 0
            )
            else None
        )
        session = session or ProcessingSession()
        session.reset()
        result = ExtractionResult(customer_type=customer)
        gcc_profile = profile_for_gcc_country(uploads.get("gcc_country"))
        selected_country = (
            gcc_profile.country
            if customer == CustomerType.GCC_NATIONAL and gcc_profile
            else uploads.get("licence_country")
        )
        licence_policy = policy_for_country(selected_country)
        payload = policy_payload(uploads.get("licence_country")) if customer == CustomerType.TOURIST else None
        if payload:
            result.licence_policy = LicencePolicyDecision(**payload)
        gcc_payload = gcc_profile_payload(uploads.get("gcc_country")) if customer == CustomerType.GCC_NATIONAL else None
        if gcc_payload:
            result.gcc_profile = GCCProfileDecision(**gcc_payload)
        result.errors.extend(validate_required_uploads(customer, uploads))
        candidates: list[FieldCandidate] = []
        if customer == CustomerType.UAE_RESIDENT:
            # The route itself proves the issuing country of the UAE driving
            # licence. Never let local authority text (for example RTA) or a
            # city of issue displace the CRM's required country value.
            candidates.extend([
                FieldCandidate(
                    field_path="uae_driving_licence.issued_by_code",
                    value="ARE", normalized_value="ARE",
                    source_document="workflow_selection",
                    source_method="selected_uae_workflow", confidence=1.0,
                    evidence_text="United Arab Emirates", validation_passed=True,
                ),
                FieldCandidate(
                    field_path="uae_driving_licence.issued_by_name",
                    value="United Arab Emirates",
                    normalized_value="United Arab Emirates",
                    source_document="workflow_selection",
                    source_method="selected_uae_workflow", confidence=1.0,
                    evidence_text="United Arab Emirates", validation_passed=True,
                ),
            ])
        if customer == CustomerType.GCC_NATIONAL and gcc_profile:
            candidates.extend([
                FieldCandidate(
                    field_path="personal_info.nationality_code", value=gcc_profile.iso3,
                    normalized_value=gcc_profile.iso3, source_document="customer_selection",
                    source_method="selected_gcc_country", confidence=1.0,
                    evidence_text=gcc_profile.country, validation_passed=True,
                ),
                FieldCandidate(
                    field_path="personal_info.nationality_name", value=gcc_profile.country,
                    normalized_value=gcc_profile.country, source_document="customer_selection",
                    source_method="selected_gcc_country", confidence=1.0,
                    evidence_text=gcc_profile.country, validation_passed=True,
                ),
            ])
            for section in ("gcc_identity", "gcc_driving_licence"):
                candidates.extend([
                    FieldCandidate(
                        field_path=f"{section}.issued_by_code", value=gcc_profile.iso3,
                        normalized_value=gcc_profile.iso3, source_document="customer_selection",
                        source_method="selected_gcc_country", confidence=1.0,
                        evidence_text=gcc_profile.country, validation_passed=True,
                    ),
                    FieldCandidate(
                        field_path=f"{section}.issued_by_name", value=gcc_profile.country,
                        normalized_value=gcc_profile.country, source_document="customer_selection",
                        source_method="selected_gcc_country", confidence=1.0,
                        evidence_text=gcc_profile.country, validation_passed=True,
                    ),
                ])
        low_quality: set[str] = set()
        best_mrz: ParsedMRZ | None = None
        # Scoped to this call: extracted document content is never retained
        # across jobs.
        processed_pages: dict[str, tuple[PageArtifact, list[FieldCandidate], DocumentRecord, list[str]]] = {}
        # Every page is decoded and normalized before any of them is read, so
        # that a recognizer able to start its requests early can have them all
        # in flight at once. Reading a page is almost entirely a wait on a
        # remote recognizer, and taking those waits one after another is what
        # made a four-page bundle cost four round trips end to end.
        #
        # The reading below is unchanged: still page by page, in upload order,
        # each page seeing what the earlier ones read. Only the waiting moved.
        # A failed upload is not reported here either -- the exception is kept
        # and raised in the loop, where an upload's failure has always been
        # recorded, in the same words.
        loaded_uploads: dict[tuple[int, str], list[Any] | Exception] = {}
        prepared_pages: dict[str, PreprocessedImage] = {}
        preprocess_config = self._preprocess_config(customer)
        for source_index, (slot, source) in enumerate(_flatten_uploads(uploads)):
            if slot not in SLOT_TYPES:
                continue
            try:
                with stage("load"):
                    loaded_uploads[(source_index, slot)] = [
                        part
                        for loaded in load_document(source, self.config)
                        for part in _split_both_sides(loaded)
                    ]
            except (DocumentInputError, OSError, ValueError, RuntimeError) as exc:
                loaded_uploads[(source_index, slot)] = exc
                continue
            for loaded in loaded_uploads[(source_index, slot)]:
                fingerprint = _page_fingerprint(loaded.image, slot)
                if fingerprint in prepared_pages:
                    # The same photograph attached twice is one page, and is
                    # read, rendered and paid for once.
                    continue
                try:
                    with stage("preprocess"):
                        prepared_pages[fingerprint] = analyze_and_preprocess(
                            loaded.image, preprocess_config,
                        )
                except (OSError, ValueError, RuntimeError):
                    # Left for the reading loop to raise and report in place.
                    continue
        prefetch_pages = getattr(self.ocr, "prefetch_pages", None)
        if prefetch_pages is not None and len(prepared_pages) > 1:
            try:
                prefetch_pages([
                    prepared.variants.get("original_normalized", prepared.normalized)
                    for prepared in prepared_pages.values()
                ])
            except Exception as exc:
                # A recognizer that cannot start early still reads normally.
                logger.warning(
                    "Page prefetch unavailable kind=%s", type(exc).__name__,
                )
        for source_index, (slot, source) in enumerate(_flatten_uploads(uploads)):
            if (
                uae_fast_deadline is not None
                and time.perf_counter() >= uae_fast_deadline
            ):
                if "UAE_FAST_PATH_TIME_BUDGET_EXCEEDED" not in result.warnings:
                    result.warnings.append("UAE_FAST_PATH_TIME_BUDGET_EXCEEDED")
                break
            if slot not in SLOT_TYPES:
                result.warnings.append(f"IGNORED_UNKNOWN_UPLOAD_SLOT:{slot}")
                continue
            try:
                pages = loaded_uploads.get((source_index, slot), [])
                if isinstance(pages, Exception):
                    raise pages
                for page_number, loaded in enumerate(pages):
                    if (
                        uae_fast_deadline is not None
                        and time.perf_counter() >= uae_fast_deadline
                    ):
                        if "UAE_FAST_PATH_TIME_BUDGET_EXCEEDED" not in result.warnings:
                            result.warnings.append("UAE_FAST_PATH_TIME_BUDGET_EXCEEDED")
                        break
                    # An operator who attaches the same photograph twice, or a
                    # PDF that repeats a page, must not pay for a second full
                    # preprocess, OCR and model pass over identical pixels.
                    fingerprint = _page_fingerprint(loaded.image, slot)
                    cached_page = processed_pages.get(fingerprint)
                    if cached_page is None:
                        previously_read_paths = {
                            candidate.field_path for candidate in candidates
                            if (
                                candidate.normalized_value
                                and candidate.validation_passed is not False
                            )
                        }
                        page_output = self._process_page(
                            slot, source, page_number, loaded, licence_policy,
                            customer, source_index, gcc_profile,
                            previously_read_paths,
                            prepared_pages.pop(fingerprint, None),
                        )
                        processed_pages[fingerprint] = page_output
                    else:
                        page_output = _relabel_page(
                            cached_page, slot, page_number, source_index,
                        )
                        result.warnings.append(
                            f"DUPLICATE_PAGE_REUSED:{page_output[2].upload_id}"
                        )
                    artifact, page_candidates, record, warnings = page_output
                    session.artifacts.append(artifact)
                    result.documents.append(record)
                    result.warnings.extend(warnings)
                    if record.quality.unreadable or len(record.quality.warnings) >= 2:
                        low_quality.add(record.upload_id)
                    candidates.extend(page_candidates)
                    if (
                        artifact.detected_type == DocumentType.PASSPORT_BIODATA
                        and artifact.mrz and artifact.mrz.mrz_type
                        and (best_mrz is None or artifact.mrz.valid)
                    ):
                        best_mrz = artifact.mrz
            except DocumentInputError as exc:
                result.errors.append(f"INVALID_UPLOAD:{slot}:{exc}")
            except (OSError, ValueError, RuntimeError) as exc:
                result.errors.append(f"PROCESSING_FAILED:{slot}:{type(exc).__name__}")
        # No page content outlives the job that read it. A read started for a
        # page this loop never reached is dropped here rather than left waiting
        # in a worker that is about to serve somebody else.
        discard_prefetched = getattr(self.ocr, "discard_prefetched", None)
        if discard_prefetched is not None:
            discard_prefetched()
        mismatched = [
            warning for warning in result.warnings
            if warning.startswith(f"{WRONG_CUSTOMER_TYPE}:")
        ]
        for warning in mismatched:
            country = warning.split(":")[1]
            error = (
                f"{WRONG_CUSTOMER_TYPE}:{country}:select "
                + (
                    f"GCC National and {country}"
                    if country in GCC_COUNTRY_NAMES else "the matching customer type"
                )
            )
            if error not in result.errors:
                result.errors.append(error)
        ocr_initialization_warnings = list(
            getattr(self.ocr, "initialization_warnings", [])
        )
        if getattr(self.ocr, "engines", None) == {} and ocr_initialization_warnings:
            result.errors.append(
                "OCR_ENGINE_UNAVAILABLE:" + ",".join(ocr_initialization_warnings)
            )
        if customer == CustomerType.GCC_NATIONAL:
            _reject_cross_document_identifiers(candidates, result.documents)
        # The contract is written in Latin script, so a name that is not
        # is refused before anything competes to be the value.
        candidates = _latin_only_name_candidates(candidates)
        _normalize_tourist_name_candidates(candidates, customer)
        # Keep every reading for cross-document checks, but reconcile the CRM
        # identity names from the workflow's identity document only. A licence
        # can corroborate a passport name; it can never replace it.
        reconciliation_candidates = list(candidates)
        _restrict_first_name_to_identity_source(
            reconciliation_candidates, result.documents, customer,
        )
        _prefer_emirates_id_mrz_name_components(
            reconciliation_candidates, result.documents, customer,
        )
        if customer == CustomerType.TOURIST:
            _restrict_tourist_last_name_to_passport(
                reconciliation_candidates, result.documents,
            )
        reconciled = reconcile_all(
            reconciliation_candidates, FIELD_PATHS, low_quality,
        )
        for path, resolved in reconciled.items():
            _set_path(result, path, resolved.value)
            result.field_metadata[path] = resolved.metadata
        _show_tourist_passport_first_name_for_review(
            result, candidates, result.documents,
        )
        _complete_tourist_first_name_from_matching_licence(
            result, candidates, result.documents,
        )
        _recover_tourist_last_name_from_verified_names(result, candidates)
        _show_tourist_passport_last_name_for_review(
            result, candidates, result.documents,
        )
        _complete_tourist_surname_from_partial_mrz(result, candidates)
        _normalize_tourist_result_names(result)
        if customer == CustomerType.GCC_NATIONAL:
            _promote_agreeing_gcc_identifier(result, candidates)
        _suggest_gender_from_name(result)
        self._validate_emirates_id_dates(result)
        self._validate_gcc_dates(result)
        self._enrich_country_names(result)
        if best_mrz is None and any(
            document.detected_type == DocumentType.PASSPORT_BIODATA.value
            for document in result.documents
        ):
            # Every passport issued since November 2015 carries a machine
            # readable zone, and its check digits are the strongest evidence in
            # the whole bundle. Without it every passport field falls back to a
            # model reading and the result is a page of unverified guesses, as
            # happened on an Albanian bundle whose capture cut the two bottom
            # lines off. Say so, because a re-photograph fixes it and nothing
            # else will.
            result.warnings.append("PASSPORT_MRZ_NOT_READ_RECAPTURE_FULL_PAGE")
        if not result.international_driving_permit.expiry_date and any(
            document.detected_type == DocumentType.INTERNATIONAL_DRIVING_PERMIT.value
            for document in result.documents
        ):
            # The 1968 booklet prints where and when it was delivered on its
            # cover and how long it is valid for inside, so a bundle holding
            # only the cover cannot state the date the rental turns on. Say
            # which page is missing: an operator who is told the field is
            # empty photographs the same cover again.
            result.warnings.append("IDP_VALIDITY_PAGE_NOT_SUPPLIED")
        if best_mrz:
            result.passport.mrz = MRZData(
                type=best_mrz.mrz_type, raw_lines=best_mrz.raw_lines,
                normalized_lines=best_mrz.normalized_lines, valid=best_mrz.valid,
                checks=MRZChecks(**best_mrz.checks), corrections=best_mrz.corrections,
            )
        if customer == CustomerType.TOURIST:
            # Everything the country selection used to answer is now read off
            # the documents. An operator choice, when one was made, still wins.
            licence_policy = self._resolve_tourist_policy(
                result, session, uploads.get("licence_country"),
            )
        self._cross_checks(result, candidates)
        self._licence_policy_checks(result, licence_policy)
        if "document_bundle" in uploads:
            self._validate_auto_detected_evidence(result, licence_policy)
        self._date_checks(result)
        required_review_paths = _required_review_paths(
            customer, licence_policy, gcc_profile,
        )
        result.manual_review_required = (
            bool(result.errors)
            or any(
                result.field_metadata[path].status in {
                    FieldStatus.NEEDS_REVIEW, FieldStatus.CONFLICTING,
                    FieldStatus.MISSING,
                }
                for path in required_review_paths
            )
            # A tourist whose acceptance rule could not be established is not a
            # result anyone should be able to confirm unattended.
            or (customer == CustomerType.TOURIST and licence_policy is None)
        )
        document_vl = getattr(self.ocr, "document_vl", None)
        ocr_model = self.config.ocr_model
        if document_vl is not None and getattr(document_vl, "loaded", False):
            ocr_model = f"{ocr_model} + PaddleOCR-VL-{document_vl.pipeline_version.lstrip('v')}"
        result.processing = ProcessingInfo(
            duration_seconds=round(time.perf_counter() - started, 3),
            stage_seconds=dict(stage_seconds),
            device=self.runtime.gpu_name or "CPU", ocr_model=ocr_model,
            vlm_model=self.choice.model_id if self.vlm.state.loaded else None,
            versions=installed_versions(["paddleocr", "paddlepaddle", "torch", "transformers", "gradio", "opencv-python-headless", "zxing-cpp"]),
        )
        session.result = result
        logger.info("Document job completed without emitting extracted values")
        return session

    @staticmethod
    def _validate_emirates_id_dates(result: ExtractionResult) -> None:
        issue = result.emirates_id.issue_date
        expiry = result.emirates_id.expiry_date
        if not issue or not expiry or issue != expiry:
            return
        # Issuing and expiry cannot be proven by the same date. Preserve the
        # well-established expiry value and reject the newly introduced issue
        # field instead of silently presenting a serious false autofill.
        result.emirates_id.issue_date = None
        result.field_metadata["emirates_id.issue_date"] = FieldMetadata(
            status=FieldStatus.MISSING,
            confidence=None,
            source_document="",
            source_method="",
            validation_results=["REJECTED_DUPLICATE_OF_EXPIRY_DATE"],
            reason_for_review=(
                "Issue Date reused the same evidence/value as Expiry Date; "
                "independent visible evidence is required"
            ),
        )
        if "EMIRATES_ID_ISSUE_EQUALS_EXPIRY_REJECTED" not in result.warnings:
            result.warnings.append("EMIRATES_ID_ISSUE_EQUALS_EXPIRY_REJECTED")

    @staticmethod
    def _validate_gcc_dates(result: ExtractionResult) -> None:
        profile = profile_for_gcc_country(result.gcc_profile.country)
        if (
            profile is not None
            and not identity_issue_date_printed(profile)
            and result.gcc_identity.issue_date
        ):
            # Only the Qatari identity card prints an issue date. On the other
            # four a value in this field can only have come from a birth date,
            # an expiry, a Hijri row, a card version or a QR payload, so it is
            # rejected rather than shown.
            result.gcc_identity.issue_date = None
            result.field_metadata["gcc_identity.issue_date"] = FieldMetadata(
                status=FieldStatus.MISSING, confidence=None,
                validation_results=[f"NOT_PRINTED_ON_{profile.iso3}_IDENTITY_CARD"],
                reason_for_review=(
                    f"The {profile.country} identity card prints no issue date"
                ),
            )
            warning = f"{profile.iso3}_ID_ISSUE_DATE_NOT_PRINTED"
            if warning not in result.warnings:
                result.warnings.append(warning)
        for section, label in (
            ("gcc_identity", "GCC_IDENTITY"),
            ("gcc_driving_licence", "GCC_DRIVING_LICENCE"),
        ):
            document = getattr(result, section)
            if not document.issue_date or document.issue_date != document.expiry_date:
                continue
            document.issue_date = None
            result.field_metadata[f"{section}.issue_date"] = FieldMetadata(
                status=FieldStatus.MISSING, confidence=None,
                validation_results=["REJECTED_DUPLICATE_OF_EXPIRY_DATE"],
                reason_for_review=(
                    "Issue Date reused the same value/evidence as Expiry Date; "
                    "independent visible evidence is required"
                ),
            )
            warning = f"{label}_ISSUE_EQUALS_EXPIRY_REJECTED"
            if warning not in result.warnings:
                result.warnings.append(warning)

    @staticmethod
    def _validate_auto_detected_evidence(
        result: ExtractionResult, policy: CountryLicencePolicy | None,
    ) -> None:
        if result.customer_type == CustomerType.UAE_RESIDENT:
            required = {
                "emirates_id": result.emirates_id.number,
                "uae_driving_licence": result.uae_driving_licence.number,
            }
        elif result.customer_type == CustomerType.GCC_NATIONAL:
            required = {
                "gcc_identity": result.gcc_identity.number,
                "gcc_driving_licence": result.gcc_driving_licence.number,
            }
        else:
            required = {"passport": result.passport.number}
            if policy and policy.requirement == LicenceRequirement.NEED_IDL:
                required["international_driving_permit"] = result.international_driving_permit.number
            elif policy:
                required["national_driving_licence"] = result.national_driving_licence.number
        existing = set(result.errors)
        for document, evidence in required.items():
            error = f"MISSING_REQUIRED_DOCUMENT_EVIDENCE:{document}"
            if not evidence and error not in existing:
                result.errors.append(error)

    @staticmethod
    def _resolve_tourist_policy(
        result: ExtractionResult, session: ProcessingSession,
        operator_choice: str | None,
    ) -> CountryLicencePolicy | None:
        """Settle the acceptance rule from the bundle, and record how.

        The country is taken from the driving document; the passport is used
        only when the licence named nobody, and then it is flagged. Whichever
        way it went is written onto the result so a reviewer can see whether the
        rule that was applied was proven or inferred.
        """
        evidence: list[CountryEvidence] = []
        conventions: list[str] = []
        presented_idp = False
        presented_national = False
        for artifact in session.artifacts:
            evidence.extend(artifact.country_evidence)
            if artifact.detected_type in {
                DocumentType.NATIONAL_DRIVING_LICENCE_FRONT,
                DocumentType.NATIONAL_DRIVING_LICENCE_BACK,
            }:
                presented_national = True
            if artifact.detected_type == DocumentType.INTERNATIONAL_DRIVING_PERMIT:
                presented_idp = True
                model = idp_convention(
                    " ".join(line.text for line in artifact.ocr.lines)
                )
                if model is not None:
                    conventions.append(model)
        # A booklet that names its convention outranks one recognised only by
        # its title, so the specific reading is the one reported.
        convention = next(
            (model for model in conventions if model != "UNSPECIFIED"),
            next(iter(conventions), None),
        )
        winner, warnings = resolve_licence_country(
            operator_choice, evidence, result.personal_info.nationality_code,
        )
        for warning in warnings:
            if warning not in result.warnings:
                result.warnings.append(warning)
        policy = policy_for_country(winner.country) if winner else None
        if policy is None:
            result.licence_policy = LicencePolicyDecision(idp_convention=convention)
            return None
        result.licence_policy = LicencePolicyDecision(
            **(policy_payload(policy.country) or {}),
            detected_from=winner.source,
            detection_confidence=winner.confidence,
            detection_evidence=winner.evidence_text,
            idp_convention=convention,
        )
        # The field is the issuing *country*, so once the country is settled it
        # is the answer -- not the authority printed in row 4c, and certainly
        # not a model's reading of the card's title. An Italian licence states
        # REPUBBLICA ITALIANA across the top and prints "MC-CH" in 4c; the
        # country is Italy either way, and it is what decides whether a permit
        # is required.
        #
        # Only a country the licence itself proved, or one the operator chose,
        # may fill it. A country inferred from the passport says nothing about
        # who issued the licence -- that is the very gap the NATIONAL_ONLY rule
        # exists to catch -- so it is left empty rather than guessed at.
        if winner.source in {
            DetectionSource.LICENCE_TEXT,
            DetectionSource.LICENCE_BARCODE,
            DetectionSource.VLM_VISUAL,
            DetectionSource.IDP_PERMIT_LABEL,
            DetectionSource.OPERATOR,
        }:
            # The bundle-level decision supersedes a page-level visual guess.
            # Keep the diagnostic records consistent with the fields the user
            # sees; otherwise a Danish licence can correctly resolve to DNK
            # while its category-table reverse still reports the discarded ARE
            # guess in the same JSON report.
            for document in result.documents:
                if document.detected_type in {
                    DocumentType.NATIONAL_DRIVING_LICENCE_FRONT.value,
                    DocumentType.NATIONAL_DRIVING_LICENCE_BACK.value,
                    DocumentType.INTERNATIONAL_DRIVING_PERMIT.value,
                }:
                    document.issuing_country_code = policy.iso3
            # Every driving document in the bundle, not one of them. A tourist
            # who needs a permit presents both the permit and the national
            # licence it was issued against, and the two name one country: the
            # convention has the permit issued by the state that issued the
            # licence, which is why a single country is resolved for the bundle
            # at all. Writing it to the permit alone left the Argentine licence
            # beside it reporting no issuing country -- the card says "República
            # Argentina" across its foot, and the reader had already read it.
            prefixes = _driving_document_prefixes(presented_idp, presented_national)
            for path, value in [
                pair for prefix in prefixes for pair in (
                    (f"{prefix}.issued_by_code", policy.iso3),
                    (f"{prefix}.issued_by_name", policy.country),
                )
            ]:
                _set_path(result, path, value)
                review_inferred = winner.source in {
                    DetectionSource.VLM_VISUAL,
                    DetectionSource.IDP_PERMIT_LABEL,
                } or any(
                    # A booklet printed by a private association is a
                    # translation of a licence, not a state's permit, and the
                    # country it names is shown for an operator to confirm
                    # however it was read. That had been holding only by
                    # accident: such a card names its holder's country of
                    # birth beside the country in its title, the whole-page
                    # scan refused a page naming two states, and the weaker
                    # permit-label reading was what asked for review. Reading
                    # the title correctly must not turn a private card into a
                    # proven one.
                    note.startswith("NON_GOVERNMENT_IDP_TRANSLATION_REQUIRES_REVIEW:")
                    for note in result.warnings
                )
                result.field_metadata[path] = FieldMetadata(
                    status=(
                        FieldStatus.NEEDS_REVIEW
                        if review_inferred else FieldStatus.VERIFIED
                    ),
                    confidence=winner.confidence,
                    confidence_components={"issuing_country_established": True},
                    source_document="",
                    source_method=(
                        "vlm"
                        if winner.source == DetectionSource.VLM_VISUAL
                        else "document_parser"
                    ),
                    evidence_text=winner.evidence_text,
                    validation_results=[
                        f"ISSUING_COUNTRY_FROM_{winner.source.upper()}",
                    ],
                    reason_for_review=(
                        "Issuing country was identified by multilingual visual routing"
                        if winner.source == DetectionSource.VLM_VISUAL
                        else (
                            "Country is stated on a non-standard IDP permit label; operator confirmation is required"
                            if winner.source == DetectionSource.IDP_PERMIT_LABEL
                            else None
                        )
                    ),
                )
        else:
            # The field is the issuing country on this branch too. Where no
            # country was proved, what is left standing in it is whatever the
            # page put there -- designator 4c, which names an authority: an
            # Austrian card reported "LPD Wien VA" and a Swiss one "TG-CH", each
            # beside an empty ISO code. A rental contract is keyed on the field
            # above; an authority read as a country is worse than a blank the
            # operator is asked to fill.
            for prefix in _driving_document_prefixes(
                presented_idp, presented_national,
            ):
                _clear_non_country_issuer(result, prefix)
        _preserve_uk_driving_licence_issue_number_space(
            result, policy.country,
        )
        return policy

    @staticmethod
    def _licence_policy_checks(
        result: ExtractionResult, policy: CountryLicencePolicy | None,
    ) -> None:
        if result.customer_type != CustomerType.TOURIST or policy is None:
            return
        if policy.requirement == LicenceRequirement.NEED_IDL:
            if not result.international_driving_permit.number:
                result.warnings.append("REQUIRED_IDP_FIELDS_MISSING")
            return
        if not result.national_driving_licence.number:
            result.warnings.append("REQUIRED_NATIONAL_LICENCE_FIELDS_MISSING")
        if policy.requirement != LicenceRequirement.NATIONAL_ONLY:
            return
        passport_country = result.personal_info.nationality_code
        if not passport_country:
            result.licence_policy.nationality_match = None
            result.warnings.append("NATIONAL_ONLY_NATIONALITY_UNPROVEN")
        else:
            matches = passport_country == policy.iso3
            result.licence_policy.nationality_match = matches
            if not matches:
                result.errors.append(
                    f"NATIONAL_ONLY_NATIONALITY_MISMATCH:{policy.iso3}:{passport_country}"
                )

    @staticmethod
    def _enrich_country_names(result: ExtractionResult) -> None:
        # A value the documents never proved must not block one they did. The
        # Albanian bundle encoded ALB in a checksummed machine-readable zone
        # while the model had guessed "Shqiptare/Albanian" into the nationality
        # name; because that guess was present, the proven code was never used
        # to fill the name, and a field the passport states outright was
        # reported as needing review.
        pairs = [
            ("personal_info.nationality_code", "personal_info.nationality_name"),
            ("passport.issued_by_code", "passport.issued_by_name"),
            ("uae_driving_licence.issued_by_code", "uae_driving_licence.issued_by_name"),
            ("gcc_identity.issued_by_code", "gcc_identity.issued_by_name"),
            ("gcc_driving_licence.issued_by_code", "gcc_driving_licence.issued_by_name"),
            ("international_driving_permit.issued_by_code", "international_driving_permit.issued_by_name"),
            ("national_driving_licence.issued_by_code", "national_driving_licence.issued_by_name"),
        ]
        for code_path, name_path in pairs:
            section, attr = code_path.split(".")
            code = getattr(getattr(result, section), attr)
            name_section, name_attr = name_path.split(".")
            existing_name = getattr(getattr(result, name_section), name_attr)
            code_metadata = result.field_metadata.get(code_path)
            # A reading no country table can place is not a country. A US
            # passport prints "E PLURIBUS UNUM" around its seal, and the arc of
            # it came back as "CPIORIBUS" on the nationality caption's own row
            # -- a labelled reading, so it counted as read from the document
            # and stood in front of the USA the zone had checksummed. A word
            # that names neither a state nor a people states no nationality.
            unplaceable_name = bool(
                existing_name
                and normalize_country(existing_name)[0] is None
                and nationality_country(existing_name)[0] is None
            )
            if code and (
                not existing_name
                or (
                    code_metadata is not None
                    and code_metadata.status in PROVEN_STATUSES
                    and (
                        _unproven_by_documents(result.field_metadata.get(name_path))
                        or unplaceable_name
                    )
                )
            ):
                normalized_code, name, warnings = normalize_country(code)
                if normalized_code and name:
                    setattr(getattr(result, name_section), name_attr, name)
                    result.field_metadata[name_path] = FieldMetadata(
                        status=result.field_metadata[code_path].status,
                        confidence=result.field_metadata[code_path].confidence,
                        confidence_components={"iso_lookup_from_evidenced_code": True},
                        source_document=result.field_metadata[code_path].source_document,
                        source_method="iso_country_normalization", evidence_text=code,
                        validation_results=["ISO_3166_1_MATCH"],
                    )

    @staticmethod
    def _cross_checks(result: ExtractionResult, candidates: list[FieldCandidate]) -> None:
        name_candidates: dict[str, list[FieldCandidate]] = {}
        for candidate in candidates:
            if candidate.field_path == "personal_info.full_name" and candidate.normalized_value:
                source_parts = candidate.source_document.split(":")
                document = (
                    ":".join(source_parts[:2])
                    if source_parts[0] == "document_bundle" and len(source_parts) >= 2
                    else source_parts[0]
                )
                name_candidates.setdefault(document, []).append(candidate)
        names = {
            document: reconciled.value
            for document, options in name_candidates.items()
            if (reconciled := reconcile_field(options)).value
        }
        if len(names) >= 2:
            values = list(names.items())
            pairs = [
                (a[1], b[1], name_similarity(a[1], b[1]))
                for index, a in enumerate(values) for b in values[index + 1:]
            ]
            pairs = [pair for pair in pairs if pair[2] is not None]
            scores = [pair[2] for pair in pairs]
            result.cross_document_checks.name_similarity = min(scores) if scores else None
            disagreeing = [pair for pair in pairs if pair[2] < 0.86]
            if disagreeing:
                # A name one document spells in full and another carries only
                # part of is not two different people. The Albanian licence
                # reported "DANILO" against the passport's "DANILO GECAJ" and
                # the bundle was flagged as a name conflict, which is the one
                # finding an operator must never learn to ignore. Say instead
                # that one reading is incomplete, which is what it is.
                reason = (
                    "NAME_PARTIALLY_READ"
                    if all(_is_partial_name(a, b) for a, b, _ in disagreeing)
                    else "NAME_MISMATCH"
                )
                result.cross_document_checks.conflicts.append({
                    "field": "personal_info.full_name", "documents": list(names),
                    "reason": reason, "similarity": min(scores),
                })
                result.warnings.append(
                    "CROSS_DOCUMENT_NAME_PARTIAL"
                    if reason == "NAME_PARTIALLY_READ" else "CROSS_DOCUMENT_NAME_CONFLICT"
                )
        _check_holder_id(result, result.licence_policy.country)
        for path, output_attr in [
            ("personal_info.date_of_birth", "date_of_birth_match"),
            ("personal_info.nationality_code", "nationality_match"),
            ("personal_info.gender", "gender_match"),
        ]:
            observations = [c.normalized_value for c in candidates if c.field_path == path and c.normalized_value]
            if len(observations) >= 2:
                setattr(result.cross_document_checks, output_attr, len(set(observations)) == 1)

    @staticmethod
    def _date_checks(result: ExtractionResult) -> None:
        documents = [
            result.emirates_id, result.passport, result.uae_driving_licence,
            result.gcc_identity, result.gcc_driving_licence,
            result.international_driving_permit, result.national_driving_licence,
        ]
        for document in documents:
            warnings = validate_date_relationships(result.personal_info.date_of_birth, document.issue_date, document.expiry_date)
            result.warnings.extend(warnings)
            if "DOCUMENT_EXPIRED" in warnings and document.expiry_date:
                for path, metadata in result.field_metadata.items():
                    if path.endswith("expiry_date") and getattr(getattr(result, path.split(".")[0]), path.split(".")[1]) == document.expiry_date:
                        metadata.validation_results.append("DOCUMENT_EXPIRED")
        _corroborate_issue_against_term(result)


def process_callback(customer_type: str, uploads: dict[str, Any], reader: DocumentReader | None = None, session: ProcessingSession | None = None) -> tuple[ProcessingSession, dict[str, Any]]:
    reader = reader or DocumentReader()
    processed = reader.process(CustomerType(customer_type), uploads, session)
    return processed, workflow_export_payload(processed.result) if processed.result else {}


def write_confirmed_json(session: ProcessingSession, path: Path) -> Path:
    if session.final_json() is None:
        raise ValueError("Final JSON is unavailable before confirmation")
    path.write_text(session.final_json() or "", encoding="utf-8")
    session.temporary_outputs.add(path)
    return path


def write_processing_report(session: ProcessingSession, path: Path) -> Path:
    if session.result is None: raise ValueError("No processing result")
    report_policy = policy_for_country(session.result.licence_policy.country)
    report_profile = profile_for_gcc_country(session.result.gcc_profile.country)
    report_paths = relevant_field_paths(
        session.result.customer_type, report_policy, report_profile,
    )
    status_counts: dict[str, int] = {}
    for field_path in report_paths:
        status = session.result.field_metadata[field_path].status.value
        status_counts[status] = status_counts.get(status, 0) + 1
    quality_warnings = [
        f"DOCUMENT_QUALITY:{document.upload_id}:{warning}"
        for document in session.result.documents
        for warning in document.quality.warnings
    ]
    report_warnings = list(dict.fromkeys([
        *session.result.warnings,
        *quality_warnings,
    ]))
    report = {
        "summary": {
            "customer_type": session.result.customer_type.value,
            "licence_policy": session.result.licence_policy.model_dump(mode="json"),
            "gcc_profile": session.result.gcc_profile.model_dump(mode="json"),
            "document_count": len(session.result.documents),
            "warning_count": len(report_warnings), "error_count": len(session.result.errors),
            "manual_review_required": session.result.manual_review_required,
            "field_status_counts": status_counts,
        },
        "documents": [doc.model_dump(mode="json") for doc in session.result.documents],
        "field_metadata": {
            key: session.result.field_metadata[key].model_dump(mode="json")
            for key in report_paths
        },
        "cross_document_checks": session.result.cross_document_checks.model_dump(mode="json"),
        "processing": session.result.processing.model_dump(mode="json"),
        # Every row the recogniser returned, per page, whether or not anything
        # bound to it.
        #
        # Without this a report says which values were chosen and is silent on
        # what there was to choose from, so diagnosing a missing field means
        # guessing at the recogniser's output and testing the guess -- which,
        # over the Queensland licence in this project's bug reports, was wrong
        # five times running and cost a day an upload. A field that is missing
        # because its row was never recognised and a field that is missing
        # because a rule rejected the row look identical from the outside and
        # need opposite fixes. This is the line that tells them apart.
        "ocr_transcript": {
            artifact.upload_id: [
                {
                    "text": line.text,
                    "confidence": round(line.confidence, 4),
                    "language": line.language,
                    "variant": line.variant,
                    "box": [[round(x), round(y)] for x, y in line.bounding_box],
                }
                for line in sorted(
                    artifact.ocr.lines,
                    key=lambda item: (
                        min(point[1] for point in item.bounding_box),
                        min(point[0] for point in item.bounding_box),
                    ),
                )
            ]
            for artifact in session.artifacts
        },
        "warnings": report_warnings,
        "errors": session.result.errors,
        "limitations": ["Not legal identity verification", "Not forgery or biometric verification", "Synthetic evaluation is not production accuracy"],
    }
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    session.temporary_outputs.add(path)
    return path
