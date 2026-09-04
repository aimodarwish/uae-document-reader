from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from .normalize import name_similarity
from .schemas import FieldCandidate, FieldMetadata, FieldStatus


SOURCE_RELIABILITY = {
    "mrz": 1.0, "barcode": 0.92, "labelled_ocr": 0.84,
    "spatial_ocr": 0.78, "regex": 0.70, "document_parser": 0.78,
    "selected_gcc_country": 1.0, "selected_uae_workflow": 1.0,
    "vlm": 0.45, "manual_edit": 1.0,
    # A guess from the given name, not evidence read off a document. Ranked
    # below every real source so any printed or encoded value outranks it.
    "name_inference": 0.20,
}


@dataclass
class ReconciledField:
    value: str | None
    metadata: FieldMetadata


def _method_group(method: str) -> str:
    return "barcode" if method.startswith("barcode:") else method


def _score(candidate: FieldCandidate, low_quality: bool) -> tuple[float, dict[str, Any]]:
    source = SOURCE_RELIABILITY.get(_method_group(candidate.source_method), 0.5)
    validation = 1.0 if candidate.validation_passed is True else 0.55 if candidate.validation_passed is None else 0.0
    quality = 0.72 if low_quality else 1.0
    score = candidate.confidence * 0.55 + source * 0.25 + validation * 0.15 + quality * 0.05
    return round(score, 4), {"ocr_or_source_confidence": candidate.confidence, "source_reliability": source, "format_or_checksum": validation, "image_quality": quality}


def _equivalent_to_top(top: FieldCandidate, value: str) -> bool:
    if value == top.normalized_value:
        return True
    if top.field_path in {
        "personal_info.first_name", "personal_info.middle_name",
        "personal_info.last_name", "personal_info.full_name",
    }:
        similarity = name_similarity(top.normalized_value, value)
        return similarity is not None and similarity >= 0.88
    return False


def reconcile_field(candidates: list[FieldCandidate], low_quality_documents: set[str] | None = None) -> ReconciledField:
    low_quality_documents = low_quality_documents or set()
    usable = [candidate for candidate in candidates if candidate.normalized_value]
    if not usable:
        return ReconciledField(None, FieldMetadata(status=FieldStatus.MISSING, reason_for_review="No supported document evidence"))
    ranked = sorted(usable, key=lambda c: (_score(c, c.source_document in low_quality_documents)[0], SOURCE_RELIABILITY.get(_method_group(c.source_method), 0)), reverse=True)
    top = ranked[0]
    score, components = _score(top, top.source_document in low_quality_documents)
    distinct = defaultdict(list)
    for candidate in ranked: distinct[candidate.normalized_value].append(candidate)
    conflicts = [value for value in distinct if not _equivalent_to_top(top, value)]
    high_conflict = any(max(_score(c, c.source_document in low_quality_documents)[0] for c in distinct[value]) >= 0.78 for value in conflicts)
    independent_agreement = len({_method_group(c.source_method) for c in distinct[top.normalized_value]}) >= 2
    valid_mrz = top.source_method == "mrz" and top.validation_passed is True
    manual = top.source_method == "manual_edit"
    if valid_mrz:
        # Arithmetic beats proximity. A machine-readable value carrying its own
        # passing check digit is proof, while a competing reading is a label
        # bound to the nearest plausible box; letting the latter cancel the
        # former emptied fields the document had encoded correctly. The rival
        # readings stay visible as alternates for the operator.
        high_conflict = False
    if high_conflict:
        status, value, reason = FieldStatus.CONFLICTING, None, "Competing well-supported values; human selection required"
    elif manual:
        status, value, reason = FieldStatus.MANUALLY_EDITED, top.normalized_value, None
    elif valid_mrz or (independent_agreement and score >= 0.85):
        status, value, reason = FieldStatus.VERIFIED, top.normalized_value, None
    elif top.source_method == "vlm":
        status, value, reason = FieldStatus.NEEDS_REVIEW, top.normalized_value, "Uncorroborated VLM evidence"
    elif top.source_document in low_quality_documents or score < 0.78 or top.validation_passed is False:
        status, value, reason = FieldStatus.NEEDS_REVIEW, top.normalized_value, "Evidence quality or validation is insufficient"
    else:
        status, value, reason = FieldStatus.HIGH_CONFIDENCE, top.normalized_value, None
    metadata = FieldMetadata(
        status=status, confidence=score, confidence_components=components,
        source_document=top.source_document, source_method=top.source_method,
        evidence_text=top.evidence_text, bounding_box=top.bounding_box,
        alternate_candidates=[candidate.model_dump() for candidate in ranked[1:]],
        validation_results=["VALIDATION_PASSED" if top.validation_passed else "VALIDATION_NOT_PROVEN"],
        reason_for_review=reason, manually_edited=manual,
    )
    return ReconciledField(value, metadata)


# The documents that print an issue date and an expiry date as a pair. On every
# one of them the first precedes the second; none is issued on the day it runs
# out.
_DATED_DOCUMENTS = (
    "national_driving_licence", "international_driving_permit", "passport",
    "emirates_id", "uae_driving_licence", "gcc_identity", "gcc_driving_licence",
)


# A driving document cannot be issued on the holder's date of birth.  More
# importantly, a date box that has already been identified as the holder's
# birth-date row must never be reused as an issue-date row merely because OCR
# lost the final character of a nearby ``4a`` designator.  Keep this at
# reconciliation, after every OCR/VLM/barcode candidate has arrived, so no
# individual extraction path can bypass the protection.
_DRIVING_DOCUMENTS = (
    "national_driving_licence", "international_driving_permit",
    "uae_driving_licence", "gcc_driving_licence",
)


def _boxes_overlap(left: FieldCandidate, right: FieldCandidate) -> bool:
    """Whether two candidates point to the same printed region."""
    if not left.bounding_box or not right.bounding_box:
        return False
    try:
        left_x = [point[0] for point in left.bounding_box]
        left_y = [point[1] for point in left.bounding_box]
        right_x = [point[0] for point in right.bounding_box]
        right_y = [point[1] for point in right.bounding_box]
    except (IndexError, TypeError):
        return False
    left_edge, top_edge = max(min(left_x), min(right_x)), max(min(left_y), min(right_y))
    right_edge, bottom_edge = min(max(left_x), max(right_x)), min(max(left_y), max(right_y))
    intersection = max(0.0, right_edge - left_edge) * max(0.0, bottom_edge - top_edge)
    left_area = max(left_x) - min(left_x)
    left_area *= max(left_y) - min(left_y)
    right_area = max(right_x) - min(right_x)
    right_area *= max(right_y) - min(right_y)
    smaller_area = min(left_area, right_area)
    return smaller_area > 0 and intersection / smaller_area >= 0.60


def _printed_date(candidate: FieldCandidate) -> str:
    """The date as the card prints it, stripped of separators and case."""
    return re.sub(r"[^0-9A-Z]", "", str(candidate.value or "").upper())


def _reuses_birth_date_evidence(
    issue: FieldCandidate, birth: FieldCandidate,
) -> bool:
    """Reject an issue-date candidate derived from its own DOB evidence.

    The failure this refuses is one printed date being reported twice: the
    birth row read as the issue row because OCR lost the tail of a nearby
    designator. Overlapping boxes identify it even where two OCR paths
    normalized that one ambiguous date differently, which is why the boxes
    matter and not just the values. A visual model has no bounding box, so an
    identical normalized date from the same page is refused as well. A human
    correction remains authoritative.

    Sharing a box is not by itself the failure. A British Columbia licence
    prints both dates inside one line -- "Issued: 2021-Feb-10 DOB:
    1991-Mar-02" -- which the recogniser returns as a single box, so the two
    fields legitimately point at the same region. Refusing on the region alone
    threw away an issue date the card states plainly, and the field came back
    empty from a page read at 0.97. What identifies the real failure is that
    the characters are the same date, printed once.
    """
    if (
        issue.source_method == "manual_edit"
        or not issue.source_document
        or issue.source_document != birth.source_document
    ):
        return False
    printed = _printed_date(issue)
    if _boxes_overlap(issue, birth) and printed and printed == _printed_date(birth):
        return True
    return bool(
        issue.normalized_value
        and issue.normalized_value == birth.normalized_value
    )


def _drop_driving_issue_dates_that_reuse_birth_evidence(
    grouped: dict[str, list[FieldCandidate]],
) -> None:
    """Keep a date of birth from ever being surfaced as a licence issue date."""
    births = grouped.get("personal_info.date_of_birth") or []
    if not births:
        return
    for section in _DRIVING_DOCUMENTS:
        path = f"{section}.issue_date"
        issues = grouped.get(path) or []
        if not issues:
            continue
        grouped[path] = [
            issue for issue in issues
            if not any(_reuses_birth_date_evidence(issue, birth) for birth in births)
        ]


def _drop_issue_dates_that_are_not_before_expiry(
    grouped: dict[str, list[FieldCandidate]],
) -> None:
    """Refuse an issue date that the same document says is its expiry date.

    A card whose two dated fields share one printed row can hand the same value
    to both. The Ontario licence sets "4a ISS/DÉL 2025/10/22" and "4b EXP/ EXP.
    2028/01/12" side by side, and only one pass over the page read the issue
    date at all; in the passes that missed it the issue label reached across the
    card and took the expiry. The licence was reported as issued and expiring on
    2028-01-12 -- the same day -- at 0.96, with nothing marked for review.

    Geometry catches that where the intervening label was read cleanly enough to
    be recognised as one. Here it was not: "4b EXP/ EXP." came back as
    "4b EXPJEXP.". So the arithmetic has to catch it too, and the arithmetic
    does not depend on OCR having read anything correctly except the two dates
    themselves.

    Only the issue date is dropped. The expiry is the date a rental is refused
    on, so where the two cannot both be right the one that survives is the one
    whose loss is visible rather than silent.
    """
    for section in _DATED_DOCUMENTS:
        issues = grouped.get(f"{section}.issue_date") or []
        expiries = grouped.get(f"{section}.expiry_date") or []
        expiry_values = {
            candidate.normalized_value for candidate in expiries
            if candidate.normalized_value
        }
        if not expiry_values or not issues:
            continue
        latest = max(expiry_values)
        kept = [
            candidate for candidate in issues
            if not candidate.normalized_value
            or candidate.normalized_value < latest
        ]
        # Every reading refused means the row was never read; leaving the field
        # empty says so, where keeping one would state a date the document
        # contradicts.
        grouped[f"{section}.issue_date"] = kept


def reconcile_all(candidates: list[FieldCandidate], paths: list[str], low_quality_documents: set[str] | None = None) -> dict[str, ReconciledField]:
    grouped: dict[str, list[FieldCandidate]] = defaultdict(list)
    for candidate in candidates: grouped[candidate.field_path].append(candidate)
    _drop_driving_issue_dates_that_reuse_birth_evidence(grouped)
    _drop_issue_dates_that_are_not_before_expiry(grouped)
    return {path: reconcile_field(grouped[path], low_quality_documents) for path in paths}


def compare_names(name_a: str | None, name_b: str | None) -> tuple[float | None, bool | None]:
    similarity = name_similarity(name_a, name_b)
    if similarity is None: return None, None
    return similarity, similarity >= 0.86
