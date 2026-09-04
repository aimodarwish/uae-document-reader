from __future__ import annotations

import html
import json
import os
import re
import tempfile
from datetime import date, datetime
from pathlib import Path
from typing import Any

import gradio as gr

from car_rental_document_reader.normalize import display_date, normalize_date
from car_rental_document_reader.pipeline import (
    FIELD_PATHS,
    WRONG_CUSTOMER_TYPE,
    DocumentReader,
    ProcessingSession,
    write_processing_report,
)
from car_rental_document_reader.schemas import (
    CustomerType,
    FieldMetadata,
    FieldStatus,
)


APP_TITLE = "UAE Residence Document Reader"
APP_SUBTITLE = "Emirates ID, passport and UAE driving licence extraction"


FIELD_LABELS: dict[str, str] = {
    "personal_info.first_name": "First Name",
    "personal_info.last_name": "Last Name",
    "personal_info.gender": "Gender",
    "personal_info.date_of_birth": "Date of Birth",
    "personal_info.nationality_name": "Nationality",
    "emirates_id.number": "Emirates ID Number",
    "emirates_id.issue_date": "Issue Date",
    "emirates_id.expiry_date": "Expiry Date",
    "passport.number": "Passport Number",
    "passport.issued_by_name": "Issued By",
    "passport.issue_date": "Issue Date",
    "passport.expiry_date": "Expiry Date",
    "uae_driving_licence.number": "Licence Number",
    "uae_driving_licence.issued_by_name": "Issued By",
    "uae_driving_licence.issue_date": "Issue Date",
    "uae_driving_licence.expiry_date": "Expiry Date",
}

_UNKNOWN_PATHS = set(FIELD_LABELS) - set(FIELD_PATHS)
if _UNKNOWN_PATHS:
    raise RuntimeError(f"Unknown field paths: {sorted(_UNKNOWN_PATHS)}")


PERSONAL_PATHS = tuple(
    path for path in FIELD_LABELS if path.startswith("personal_info.")
)
EMIRATES_PATHS = tuple(
    path for path in FIELD_LABELS if path.startswith("emirates_id.")
)
PASSPORT_PATHS = tuple(
    path for path in FIELD_LABELS if path.startswith("passport.")
)
LICENCE_PATHS = tuple(
    path for path in FIELD_LABELS if path.startswith("uae_driving_licence.")
)

SECTIONS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("Personal Information", "person", PERSONAL_PATHS),
    ("Emirates ID", "id", EMIRATES_PATHS),
    ("Passport", "passport", PASSPORT_PATHS),
    ("UAE Driving Licence", "car", LICENCE_PATHS),
)

SECTION_ICONS = {
    "person": (
        "<svg viewBox='0 0 24 24' fill='none' stroke='currentColor' stroke-width='1.8' "
        "stroke-linecap='round' stroke-linejoin='round'>"
        "<path d='M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2'/>"
        "<circle cx='12' cy='7' r='4'/></svg>"
    ),
    "id": (
        "<svg viewBox='0 0 24 24' fill='none' stroke='currentColor' stroke-width='1.8' "
        "stroke-linecap='round' stroke-linejoin='round'>"
        "<rect x='2' y='5' width='20' height='14' rx='2'/>"
        "<circle cx='8.5' cy='11' r='2.2'/>"
        "<path d='M5 16.2c.8-1.4 2-2.1 3.5-2.1s2.7.7 3.5 2.1'/>"
        "<path d='M15 10h4M15 13.5h4'/></svg>"
    ),
    "passport": (
        "<svg viewBox='0 0 24 24' fill='none' stroke='currentColor' stroke-width='1.8' "
        "stroke-linecap='round' stroke-linejoin='round'>"
        "<path d='M5 3.8h11.2a2 2 0 0 1 2 2v12.4a2 2 0 0 1-2 2H5z'/>"
        "<path d='M5 3.8v16.4'/><circle cx='11.6' cy='10' r='2.6'/>"
        "<path d='M8.8 15.4h5.6'/></svg>"
    ),
    "car": (
        "<svg viewBox='0 0 24 24' fill='none' stroke='currentColor' stroke-width='1.8' "
        "stroke-linecap='round' stroke-linejoin='round'>"
        "<path d='M5 17h14M4 17v-4.2a2 2 0 0 1 .2-.9l1.9-3.8A2 2 0 0 1 7.9 7h8.2"
        "a2 2 0 0 1 1.8 1.1l1.9 3.8a2 2 0 0 1 .2.9V17'/>"
        "<circle cx='7.5' cy='17.5' r='1.8'/><circle cx='16.5' cy='17.5' r='1.8'/>"
        "<path d='M4.6 12.4h14.8'/></svg>"
    ),
}


DATE_PATHS = {
    path for path in FIELD_LABELS
    if path.endswith(("issue_date", "expiry_date", "date_of_birth"))
}
EXPIRY_PATHS = (
    "emirates_id.expiry_date",
    "passport.expiry_date",
    "uae_driving_licence.expiry_date",
)

PASSPORT_MERGE_PATHS = (
    "passport.number",
    "passport.issued_by_code",
    "passport.issued_by_name",
    "passport.issue_date",
    "passport.expiry_date",
    "passport.holder_id",
)

EXPORT_KEYS = {
    "personal_info.first_name": "first_name",
    "personal_info.last_name": "last_name",
    "personal_info.gender": "gender",
    "personal_info.date_of_birth": "date_of_birth",
    "personal_info.nationality_name": "nationality",
    "emirates_id.number": "emirates_id_number",
    "emirates_id.issue_date": "emirates_id_issue_date",
    "emirates_id.expiry_date": "emirates_id_expiry_date",
    "passport.number": "passport_number",
    "passport.issued_by_name": "passport_issued_by_country",
    "passport.issue_date": "passport_issue_date",
    "passport.expiry_date": "passport_expiry_date",
    "uae_driving_licence.number": "driving_licence_number",
    "uae_driving_licence.issued_by_name": "driving_licence_issued_by",
    "uae_driving_licence.issue_date": "driving_licence_issue_date",
    "uae_driving_licence.expiry_date": "driving_licence_expiry_date",
}


GENDER_CHOICES = [
    ("Select gender", ""),
    ("Male", "M"),
    ("Female", "F"),
    ("Other", "X"),
]

GENDER_DISPLAY = {"M": "Male", "F": "Female", "X": "Other"}


STATUS_STYLE: dict[str, tuple[str, str]] = {
    FieldStatus.VERIFIED.value: ("ok", "Verified"),
    FieldStatus.HIGH_CONFIDENCE.value: ("good", "High confidence"),
    FieldStatus.MANUALLY_EDITED.value: ("edited", "Edited"),
    FieldStatus.NEEDS_REVIEW.value: ("warn", "Needs review"),
    FieldStatus.CONFLICTING.value: ("bad", "Conflicting"),
    FieldStatus.MISSING.value: ("none", "Not found"),
}

REVIEW_STATUSES = {
    FieldStatus.NEEDS_REVIEW.value,
    FieldStatus.CONFLICTING.value,
    FieldStatus.MISSING.value,
}

PASSPORT_DOCUMENT_TYPE = "PASSPORT_BIODATA"
UNKNOWN_DOCUMENT_TYPE = "UNKNOWN"

_BUNDLE_PAGE = re.compile(r"document_bundle:(\d+):(\d+)")


def _escape(value: Any) -> str:
    return html.escape(str(value), quote=True)


def _bundle_index(text: str) -> int | None:
    match = _BUNDLE_PAGE.search(text or "")
    return int(match.group(1)) - 1 if match else None


def _status_of(result: Any, path: str) -> str:
    metadata = result.field_metadata.get(path) if result is not None else None
    return metadata.status.value if metadata else FieldStatus.MISSING.value


def _raw_value(result: Any, path: str) -> str | None:
    section, attribute = path.split(".", 1)
    return getattr(getattr(result, section), attribute)


def _display_value(result: Any, path: str) -> str:
    value = _raw_value(result, path)
    if path in DATE_PATHS:
        return display_date(value) or ""
    if path == "personal_info.gender":
        return GENDER_DISPLAY.get(value or "", value or "")
    return value or ""


def _form_value(result: Any, path: str) -> str:
    value = _raw_value(result, path)
    if path in DATE_PATHS:
        return display_date(value) or ""
    return value or ""


def _passport_presented(result: Any) -> bool:
    if result is None:
        return False
    if any(_raw_value(result, path) for path in PASSPORT_PATHS):
        return True
    return any(
        document.detected_type == PASSPORT_DOCUMENT_TYPE
        for document in result.documents
    )


def _reviewable_paths(result: Any) -> tuple[str, ...]:
    if _passport_presented(result):
        return tuple(FIELD_LABELS)
    return tuple(path for path in FIELD_LABELS if not path.startswith("passport."))


def _expiry_note(iso_value: str | None) -> tuple[str, str] | None:
    if not iso_value:
        return None
    try:
        expires = datetime.strptime(iso_value, "%Y-%m-%d").date()
    except (TypeError, ValueError):
        return None
    days = (expires - date.today()).days
    if days < 0:
        return ("bad", f"Expired {abs(days)} days ago")
    if days <= 30:
        return ("warn", f"Expires in {days} days")
    if days <= 90:
        return ("good", f"Valid · {days} days left")
    return ("ok", "Valid")


def _field_card(result: Any, path: str) -> str:
    status = _status_of(result, path)
    css_class, status_text = STATUS_STYLE.get(status, ("none", status.title()))
    value = _display_value(result, path)
    empty = "" if value else " is-empty"
    shown = _escape(value) if value else "—"
    extra = ""
    if path in EXPIRY_PATHS:
        note = _expiry_note(_raw_value(result, path))
        if note:
            extra = f"<span class='uae-note uae-{note[0]}'>{_escape(note[1])}</span>"
    return (
        f"<div class='uae-field{empty}'>"
        f"<div class='uae-field-label'>{_escape(FIELD_LABELS[path])}</div>"
        f"<div class='uae-field-value'>{shown}{extra}</div>"
        f"<div class='uae-chip uae-{css_class}'>{_escape(status_text)}</div>"
        f"</div>"
    )


def _headline(result: Any) -> str:
    first = (_raw_value(result, "personal_info.first_name") or "").strip()
    last = (_raw_value(result, "personal_info.last_name") or "").strip()
    name = " ".join(part for part in (first, last) if part)
    nationality = (_raw_value(result, "personal_info.nationality_name") or "").strip()
    if not name and not nationality:
        return ""
    detail = (
        f"<div class='uae-headline-meta'>{_escape(nationality)}</div>"
        if nationality else ""
    )
    return (
        "<div class='uae-headline'>"
        f"<div class='uae-headline-name'>{_escape(name or '—')}</div>"
        f"{detail}</div>"
    )


def result_cards_html(session: ProcessingSession | None) -> str:
    if session is None or session.result is None:
        return (
            "<div class='uae-empty'>"
            f"<div class='uae-empty-icon'>{SECTION_ICONS['id']}</div>"
            "<h3>No documents read yet</h3>"
            "<p>Upload the Emirates ID, the passport page and the UAE driving "
            "licence, then press <b>Read Documents</b>.</p></div>"
        )
    result = session.result
    passport_seen = _passport_presented(result)
    blocks = [_headline(result)]
    for title, icon, paths in SECTIONS:
        badge = ""
        if paths is PASSPORT_PATHS and not passport_seen:
            badge = "<span class='uae-card-badge'>Not provided</span>"
        fields = "".join(_field_card(result, path) for path in paths)
        blocks.append(
            "<section class='uae-card'>"
            "<header class='uae-card-head'>"
            f"<span class='uae-card-icon'>{SECTION_ICONS[icon]}</span>"
            f"<span class='uae-card-title'>{_escape(title)}</span>{badge}"
            "</header>"
            f"<div class='uae-card-body'>{fields}</div>"
            "</section>"
        )
    return f"<div class='uae-results'>{''.join(blocks)}</div>"


def status_banner_html(session: ProcessingSession | None) -> str:
    if session is None or session.result is None:
        return (
            "<div class='uae-banner uae-idle'><b>Ready</b>"
            "<span>Upload a UAE resident's documents to begin.</span></div>"
        )
    result = session.result
    if any(error.startswith("OCR_ENGINE_UNAVAILABLE:") for error in result.errors):
        return (
            "<div class='uae-banner uae-error'><b>OCR unavailable</b>"
            "<span>The engine did not start, so the empty fields are not a "
            "result. Check the processing details below.</span></div>"
        )
    if result.errors:
        detail = _escape("; ".join(result.errors[:2]))
        return (
            "<div class='uae-banner uae-error'><b>Completed with errors</b>"
            f"<span>{detail}</span></div>"
        )
    flagged = [
        FIELD_LABELS[path] for path in _reviewable_paths(result)
        if _status_of(result, path) in REVIEW_STATUSES
    ]
    if flagged:
        listed = _escape(", ".join(flagged[:4]))
        more = f" and {len(flagged) - 4} more" if len(flagged) > 4 else ""
        return (
            "<div class='uae-banner uae-review'><b>Manual review needed</b>"
            f"<span>{listed}{more}</span></div>"
        )
    seconds = result.processing.duration_seconds
    took = f" in {seconds:.1f}s" if isinstance(seconds, (int, float)) else ""
    return (
        "<div class='uae-banner uae-ready'><b>All fields read</b>"
        f"<span>Completed{took}. Review the values and confirm.</span></div>"
    )


def confirmed_card_html(result: Any) -> str:
    if result is None:
        return ""
    review = result.manual_review_required
    tone = "uae-final-review" if review else "uae-final-ok"
    heading = (
        "Confirmed with open review items"
        if review else "Confirmed and ready for the CRM"
    )
    rows = "".join(
        f"<li><span>{_escape(FIELD_LABELS[path])}</span>"
        f"<b>{_escape(_display_value(result, path) or '—')}</b></li>"
        for path in FIELD_LABELS
    )
    return (
        f"<div class='uae-final {tone}'><h3>{_escape(heading)}</h3>"
        f"<ul class='uae-final-list'>{rows}</ul></div>"
    )


def _quality_rows(session: ProcessingSession) -> list[list[Any]]:
    rows = []
    for artifact in session.artifacts:
        quality = artifact.preprocessed.quality
        rows.append([
            artifact.upload_id,
            artifact.detected_type.value,
            quality.orientation,
            f"{quality.width}x{quality.height}",
            quality.blur_score,
            "YES" if "BLUR_WARNING" in quality.warnings else "NO",
            "YES" if quality.glare_detected else "NO",
            "YES" if quality.crop_warning else "NO",
            "YES" if quality.unreadable else "NO",
            ", ".join(quality.warnings),
        ])
    return rows


def _summary(result: Any) -> dict[str, Any]:
    return {
        "workflow": "UAE_RESIDENT",
        "passport_presented": _passport_presented(result),
        "processing": result.processing.model_dump(mode="json"),
        "documents": [record.model_dump(mode="json") for record in result.documents],
        "cross_document_checks": result.cross_document_checks.model_dump(mode="json"),
        "warnings": result.warnings,
        "errors": result.errors,
        "manual_review_required": result.manual_review_required,
        "note": "The final JSON stays locked until Confirm is pressed.",
    }


def _passport_page_indices(result: Any) -> list[int]:
    indices: set[int] = set()
    for warning in result.warnings:
        if warning.startswith(f"{WRONG_CUSTOMER_TYPE}:"):
            index = _bundle_index(warning)
            if index is not None:
                indices.add(index)
    for document in result.documents:
        if document.detected_type in {UNKNOWN_DOCUMENT_TYPE, PASSPORT_DOCUMENT_TYPE}:
            index = _bundle_index(document.upload_id)
            if index is not None:
                indices.add(index)
    return sorted(indices)


def _release_mismatch_reports(result: Any, cleared: set[int]) -> None:
    released: set[str] = set()
    kept: list[str] = []
    for warning in result.warnings:
        if warning.startswith(f"{WRONG_CUSTOMER_TYPE}:"):
            index = _bundle_index(warning)
            if index in cleared:
                parts = warning.split(":")
                if len(parts) > 1:
                    released.add(parts[1])
                continue
        kept.append(warning)
    result.warnings = kept
    still_flagged = {
        warning.split(":")[1] for warning in kept
        if warning.startswith(f"{WRONG_CUSTOMER_TYPE}:") and ":" in warning
    }
    dropped = released - still_flagged
    if not dropped:
        return
    result.errors = [
        error for error in result.errors
        if not (
            error.startswith(f"{WRONG_CUSTOMER_TYPE}:")
            and error.split(":")[1] in dropped
        )
    ]


def _merge_passport(session: ProcessingSession, probe: ProcessingSession,
                    indices: list[int]) -> None:
    result = session.result
    source = probe.result
    if result is None or source is None:
        return
    confirmed: dict[int, str] = {}
    for document in source.documents:
        position = _bundle_index(document.upload_id)
        if (
            position is None
            or position >= len(indices)
            or document.detected_type != PASSPORT_DOCUMENT_TYPE
        ):
            continue
        confirmed[indices[position]] = document.upload_id
    if not confirmed:
        return
    if not any(getattr(source.passport, path.split(".", 1)[1]) for path in PASSPORT_MERGE_PATHS):
        return
    for path in PASSPORT_MERGE_PATHS:
        attribute = path.split(".", 1)[1]
        setattr(result.passport, attribute, getattr(source.passport, attribute))
        metadata = source.field_metadata.get(path)
        if metadata is not None:
            result.field_metadata[path] = metadata
    result.passport.mrz = source.passport.mrz
    passport_ids = set(confirmed.values())
    replaced = set(confirmed)
    result.documents = [
        document for document in result.documents
        if _bundle_index(document.upload_id) not in replaced
    ] + [
        document for document in source.documents
        if document.upload_id in passport_ids
    ]
    session.artifacts = [
        artifact for artifact in session.artifacts
        if _bundle_index(artifact.upload_id) not in replaced
    ] + [
        artifact for artifact in probe.artifacts
        if artifact.upload_id in passport_ids
    ]
    _release_mismatch_reports(result, replaced)
    for warning in source.warnings:
        index = _bundle_index(warning)
        if index is None or index >= len(indices):
            continue
        if indices[index] in replaced and warning not in result.warnings:
            result.warnings.append(warning)
    duration = result.processing.duration_seconds or 0.0
    probe_duration = source.processing.duration_seconds or 0.0
    result.processing.duration_seconds = round(duration + probe_duration, 3)
    for stage, seconds in source.processing.stage_seconds.items():
        result.processing.stage_seconds[stage] = round(
            result.processing.stage_seconds.get(stage, 0.0) + seconds, 3,
        )
    result.manual_review_required = bool(result.errors) or any(
        _status_of(result, path) in REVIEW_STATUSES
        for path in _reviewable_paths(result)
    )


def _read_passport(reader: DocumentReader, bundle: list[Any],
                   session: ProcessingSession) -> None:
    result = session.result
    if result is None:
        return
    indices = [index for index in _passport_page_indices(result) if index < len(bundle)]
    if not indices:
        return
    probe = reader.process(
        CustomerType.TOURIST,
        {
            "document_bundle": [bundle[index] for index in indices],
            "licence_country": None,
            "gcc_country": None,
        },
        ProcessingSession(),
    )
    _merge_passport(session, probe, indices)


def process_documents(document_bundle: Any, session: ProcessingSession | None,
                      reader: DocumentReader) -> tuple[Any, ...]:
    if not document_bundle:
        raise gr.Error("Upload the documents before reading.")
    bundle = list(document_bundle) if isinstance(document_bundle, list) else [document_bundle]
    processed = reader.process(
        CustomerType.UAE_RESIDENT,
        {"document_bundle": bundle, "licence_country": None, "gcc_country": None},
        session or ProcessingSession(),
    )
    _read_passport(reader, bundle, processed)
    result = processed.result
    assert result is not None
    evidence_choices = [(FIELD_LABELS[path], path) for path in FIELD_LABELS]
    form_values: list[Any] = []
    for path in FIELD_LABELS:
        if path == "personal_info.gender":
            form_values.append(_raw_value(result, path) or "")
        else:
            form_values.append(_form_value(result, path))
        form_values.append(_status_of(result, path))
    return (
        processed,
        status_banner_html(processed),
        result_cards_html(processed),
        [artifact.preview for artifact in processed.artifacts],
        _quality_rows(processed),
        _summary(result),
        gr.update(choices=evidence_choices, value=evidence_choices[0][1]),
        *form_values,
    )


def show_evidence(path: str, session: ProcessingSession | None) -> dict[str, Any]:
    if not path or session is None or session.result is None:
        return {}
    metadata = session.result.field_metadata.get(path)
    return metadata.model_dump(mode="json") if metadata else {}


def _manual_values(values: list[Any]) -> dict[str, str | None]:
    edits: dict[str, str | None] = {}
    for path, value in zip(FIELD_LABELS, values):
        text = value.strip() if isinstance(value, str) else value
        if path in DATE_PATHS and text:
            normalized = normalize_date(text, day_first_hint=True)
            edits[path] = normalized.value if normalized.value else text
        else:
            edits[path] = text or None
    return edits


def _apply_passport_edits(result: Any, edits: dict[str, str | None]) -> None:
    for path in PASSPORT_PATHS:
        if path not in edits:
            continue
        attribute = path.split(".", 1)[1]
        value = edits[path]
        if value == getattr(result.passport, attribute):
            continue
        setattr(result.passport, attribute, value or None)
        result.field_metadata[path] = FieldMetadata(
            status=FieldStatus.MANUALLY_EDITED,
            confidence=1.0,
            confidence_components={"human_confirmation": True},
            source_document="human_confirmation",
            source_method="manual_edit",
            validation_results=["CONFIRMED_BY_USER"],
            manually_edited=True,
        )


def _commit(session: ProcessingSession, values: list[Any]) -> None:
    edits = _manual_values(values)
    _apply_passport_edits(session.result, edits)
    session.confirm({
        path: value for path, value in edits.items()
        if not path.startswith("passport.")
    })


def _export_payload(result: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {"customer_type": CustomerType.UAE_RESIDENT.value}
    for path, key in EXPORT_KEYS.items():
        value = _raw_value(result, path)
        payload[key] = str(value) if value else None
    payload["passport_presented"] = _passport_presented(result)
    payload["field_status"] = {
        EXPORT_KEYS[path]: _status_of(result, path) for path in EXPORT_KEYS
    }
    payload["manual_review_required"] = result.manual_review_required
    payload["confirmed_by_user"] = result.confirmed_by_user
    return payload


def _temporary_output(session: ProcessingSession, suffix: str) -> Path:
    handle, raw = tempfile.mkstemp(
        prefix=f"uae_reader_{session.session_id[:8]}_", suffix=suffix,
    )
    os.close(handle)
    path = Path(raw)
    session.temporary_outputs.add(path)
    return path


def apply_edits(session: ProcessingSession | None, *values: Any) -> tuple[Any, ...]:
    if session is None or session.result is None:
        raise gr.Error("Read the documents before editing.")
    _commit(session, list(values))
    session.result.confirmed_by_user = False
    session.confirmed_json = None
    statuses = [_status_of(session.result, path) for path in FIELD_LABELS]
    return (
        session,
        status_banner_html(session),
        result_cards_html(session),
        *statuses,
    )


def confirm_result(session: ProcessingSession | None, *values: Any) -> tuple[Any, ...]:
    if session is None or session.result is None:
        raise gr.Error("Read the documents before confirming.")
    _commit(session, list(values))
    payload = json.dumps(_export_payload(session.result), indent=2, ensure_ascii=False)
    session.confirmed_json = payload
    json_path = _temporary_output(session, ".json")
    json_path.write_text(payload, encoding="utf-8")
    report_path = write_processing_report(
        session, _temporary_output(session, "_report.json"),
    )
    return (
        session,
        confirmed_card_html(session.result),
        payload,
        str(json_path),
        str(report_path),
        result_cards_html(session),
    )


def download_report(session: ProcessingSession | None) -> str | None:
    if session is None or session.result is None:
        return None
    return str(write_processing_report(
        session, _temporary_output(session, "_report.json"),
    ))


def reset_everything(session: ProcessingSession | None) -> tuple[Any, ...]:
    if session is not None:
        session.reset()
    fresh = ProcessingSession()
    blanks: list[Any] = []
    for _ in FIELD_LABELS:
        blanks.extend(["", ""])
    return (
        fresh,
        status_banner_html(None),
        result_cards_html(None),
        [],
        [],
        {},
        gr.update(choices=[], value=None),
        {},
        "",
        "",
        None,
        None,
        None,
        *blanks,
    )


CSS = """
:root {
  --uae-green-900: #052e1f;
  --uae-green-800: #064e3b;
  --uae-green-700: #047857;
  --uae-green-600: #059669;
  --uae-green-100: #d1fae5;
  --uae-green-50:  #ecfdf5;
  --uae-line: #bbf7d0;
}
.gradio-container {max-width: 1240px !important; margin: 0 auto !important;
  background: linear-gradient(180deg, #f6fdf9 0%, #ffffff 340px) !important;}
.gradio-container .wrap {overflow-wrap: anywhere}

.uae-masthead {
  background: linear-gradient(120deg, var(--uae-green-800) 0%, var(--uae-green-600) 62%, #34d399 100%);
  color: #fff; border-radius: 20px; padding: 26px 30px; margin-bottom: 6px;
  box-shadow: 0 18px 40px -24px rgba(5,80,60,.75);
}
.uae-masthead h1 {margin: 0; font-size: 1.72rem; font-weight: 750; letter-spacing: -.02em; color: #fff;}
.uae-masthead p {margin: 8px 0 0; color: #d8fbe9; font-size: .97rem;}
.uae-tags {margin-top: 16px; display: flex; flex-wrap: wrap; gap: 8px;}
.uae-tags span {
  background: rgba(255,255,255,.16); border: 1px solid rgba(255,255,255,.3);
  padding: 5px 12px; border-radius: 999px; font-size: .78rem;
}

.uae-privacy {
  border: 1px solid #f0c26b; background: #fffbf0; color: #7a4a05;
  border-radius: 14px; padding: 13px 16px; font-size: .87rem; line-height: 1.55;
}
.uae-banner {
  display: flex; gap: 14px; align-items: baseline; flex-wrap: wrap;
  border-radius: 14px; padding: 14px 18px; border: 1px solid var(--uae-line);
  background: var(--uae-green-50); color: var(--uae-green-900); font-size: .9rem;
}
.uae-banner b {font-size: .96rem; letter-spacing: -.01em;}
.uae-banner.uae-ready {border-color: #6ee7b7; background: #e8fdf3;}
.uae-banner.uae-review {border-color: #fcd34d; background: #fffbeb; color: #78350f;}
.uae-banner.uae-error {border-color: #fca5a5; background: #fef2f2; color: #7f1d1d;}
.uae-banner.uae-idle {border-color: #d1fae5; background: #f6fdf9; color: #14532d;}

.uae-results {display: flex; flex-direction: column; gap: 16px;}
.uae-headline {
  border-radius: 16px; padding: 16px 20px; background: #fff;
  border: 1px solid var(--uae-line); box-shadow: 0 10px 26px -22px rgba(6,78,59,.6);
}
.uae-headline-name {font-size: 1.28rem; font-weight: 720; color: var(--uae-green-900); letter-spacing: -.01em;}
.uae-headline-meta {margin-top: 4px; font-size: .92rem; color: var(--uae-green-700);}
.uae-card {
  background: #fff; border: 1px solid var(--uae-line); border-radius: 16px;
  overflow: hidden; box-shadow: 0 10px 26px -22px rgba(6,78,59,.6);
}
.uae-card-head {
  display: flex; align-items: center; gap: 10px; padding: 13px 18px;
  background: linear-gradient(90deg, var(--uae-green-800), var(--uae-green-600));
  color: #fff;
}
.uae-card-icon svg {width: 20px; height: 20px; display: block;}
.uae-card-title {font-weight: 680; letter-spacing: -.01em;}
.uae-card-badge {
  margin-inline-start: auto; font-size: .72rem; font-weight: 600;
  background: rgba(255,255,255,.18); border: 1px solid rgba(255,255,255,.32);
  padding: 3px 10px; border-radius: 999px;
}
.uae-card-body {display: grid; grid-template-columns: repeat(auto-fit, minmax(255px, 1fr));
  gap: 1px; background: var(--uae-green-100);}
.uae-field {background: #fff; padding: 13px 18px 14px; display: flex; flex-direction: column; gap: 5px;}
.uae-field.is-empty {background: #fbfefc;}
.uae-field-label {font-size: .74rem; text-transform: uppercase; letter-spacing: .06em; color: #4b7f68;}
.uae-field-value {font-size: 1.06rem; font-weight: 640; color: #06281c; word-break: break-word;}
.uae-field.is-empty .uae-field-value {color: #9db8ac; font-weight: 500;}
.uae-note {display: inline-block; margin-inline-start: 8px; font-size: .72rem; font-weight: 600;
  padding: 2px 8px; border-radius: 999px; vertical-align: middle;}
.uae-note.uae-ok {background: #d1fae5; color: #065f46;}
.uae-note.uae-good {background: #dcfce7; color: #166534;}
.uae-note.uae-warn {background: #fef3c7; color: #92400e;}
.uae-note.uae-bad {background: #fee2e2; color: #991b1b;}
.uae-chip {
  align-self: flex-start; display: inline-flex; align-items: center;
  font-size: .72rem; font-weight: 650; padding: 3px 10px; border-radius: 999px;
  border: 1px solid transparent;
}
.uae-chip.uae-ok {background: #059669; color: #fff;}
.uae-chip.uae-good {background: #d1fae5; color: #065f46; border-color: #6ee7b7;}
.uae-chip.uae-edited {background: #cffafe; color: #0e7490; border-color: #67e8f9;}
.uae-chip.uae-warn {background: #fef3c7; color: #92400e; border-color: #fcd34d;}
.uae-chip.uae-bad {background: #fee2e2; color: #991b1b; border-color: #fca5a5;}
.uae-chip.uae-none {background: #f1f5f4; color: #64748b; border-color: #dbe5e1;}

.uae-empty {
  border: 1.5px dashed #a7f3d0; border-radius: 18px; background: #f8fefb;
  padding: 40px 26px; text-align: center; color: #2f6b55;
}
.uae-empty-icon svg {width: 42px; height: 42px; color: var(--uae-green-600); margin: 0 auto 10px;}
.uae-empty h3 {margin: 0 0 6px; color: var(--uae-green-800); font-weight: 700;}
.uae-empty p {margin: 3px 0; font-size: .9rem;}

.uae-final {border-radius: 16px; padding: 18px 20px; border: 1px solid var(--uae-line); background: var(--uae-green-50);}
.uae-final.uae-final-review {border-color: #fcd34d; background: #fffbeb;}
.uae-final h3 {margin: 0 0 12px; font-size: 1.02rem; color: var(--uae-green-800);}
.uae-final.uae-final-review h3 {color: #92400e;}
.uae-final-list {list-style: none; margin: 0; padding: 0;
  display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 6px 20px;}
.uae-final-list li {display: flex; justify-content: space-between; gap: 12px;
  padding: 6px 0; border-bottom: 1px dashed rgba(6,78,59,.14); font-size: .89rem;}
.uae-final-list span {color: #45705f;}
.uae-final-list b {color: #06281c; text-align: right;}

button.primary, .gr-button-primary {
  background: linear-gradient(90deg, #047857, #10b981) !important;
  color: #fff !important; border: none !important;
  box-shadow: 0 10px 22px -14px rgba(4,120,87,.9) !important;
}
button.primary:hover, .gr-button-primary:hover {
  background: linear-gradient(90deg, #065f46, #059669) !important;
}
button.secondary {border-color: #a7f3d0 !important; color: #065f46 !important;}

.uae-status-box input {text-align: center; font-weight: 700; font-size: .78rem;}
.uae-edit-row {align-items: end;}
.uae-section-title {font-weight: 700; color: var(--uae-green-800); margin-top: 4px;}
footer {display: none !important;}
"""


def _theme():
    return gr.themes.Soft(
        primary_hue="emerald",
        secondary_hue="green",
        neutral_hue="slate",
    )


def build_uae_demo(reader: DocumentReader | None = None, *,
                   privacy_notice: str | None = None):
    reader = reader or DocumentReader(initialize_vlm=False)
    runtime_label = os.environ.get("DOCUMENT_READER_RUNTIME", "this local runtime")
    privacy_notice = privacy_notice or (
        f"{runtime_label} may be a cloud environment. Use only synthetic or "
        "fully anonymised documents for this proof of concept."
    )

    with gr.Blocks(title=APP_TITLE, theme=_theme(), css=CSS) as demo:
        session = gr.State(ProcessingSession())

        gr.HTML(
            "<div class='uae-masthead'>"
            f"<h1>{_escape(APP_TITLE)}</h1>"
            f"<p>{_escape(APP_SUBTITLE)}</p>"
            "<div class='uae-tags'><span>UAE Resident route only</span>"
            "<span>Emirates ID front and back</span>"
            "<span>Passport biodata page</span>"
            "<span>UAE Driving Licence</span>"
            "<span>Runs on 127.0.0.1, no public share</span></div></div>"
        )
        gr.HTML(
            "<div class='uae-privacy'><b>Privacy notice:</b> "
            f"{_escape(privacy_notice)} This tool is not identity verification, "
            "forgery detection, face matching, biometric or liveness checking, "
            "sanctions screening, or government-database verification.</div>"
        )

        banner = gr.HTML(status_banner_html(None))

        with gr.Row():
            with gr.Column(scale=3):
                bundle = gr.File(
                    label="Emirates ID, passport page and UAE driving licence",
                    file_count="multiple",
                    file_types=["image", ".pdf", ".heic"],
                )
            with gr.Column(scale=2):
                gr.Markdown(
                    "**Upload every page at once.**\n\n"
                    "The reader identifies each page itself, so file names and "
                    "upload order do not matter. The passport page is optional: "
                    "when one is present it is read and reported alongside the "
                    "Emirates ID and the licence.\n\n"
                    "Dates are shown as DD-MM-YYYY and stored as YYYY-MM-DD."
                )
        with gr.Row():
            read_button = gr.Button("Read Documents", variant="primary", scale=3)
            cancel_button = gr.Button("Cancel", variant="stop", scale=1)
            reset_button = gr.Button("Reset", scale=1)

        cards = gr.HTML(result_cards_html(None))

        with gr.Accordion("Corrections", open=False):
            gr.Markdown(
                "Edit any value below and press **Apply corrections**. An edited "
                "field is recorded as manually confirmed and its status becomes "
                "*Edited*. Gender is filled only from an explicit gender field or "
                "a valid MRZ; if the document does not state it, set it here."
            )
            editors: dict[str, tuple[Any, Any]] = {}
            for title, _icon, paths in SECTIONS:
                gr.Markdown(f"#### {title}", elem_classes=["uae-section-title"])
                for path in paths:
                    with gr.Row(elem_classes=["uae-edit-row"]):
                        if path == "personal_info.gender":
                            value = gr.Dropdown(
                                label=FIELD_LABELS[path], choices=GENDER_CHOICES,
                                value="", interactive=True, scale=4,
                            )
                        else:
                            value = gr.Textbox(
                                label=FIELD_LABELS[path], interactive=True, scale=4,
                            )
                        status = gr.Textbox(
                            label="Status", interactive=False, scale=1,
                            elem_classes=["uae-status-box"],
                        )
                    editors[path] = (value, status)
            apply_button = gr.Button("Apply corrections", variant="secondary")

        with gr.Accordion("Pages, quality and evidence", open=False):
            preview = gr.Gallery(
                label="Normalised page previews", columns=3, height=260,
                object_fit="contain",
            )
            quality = gr.Dataframe(
                headers=[
                    "Upload", "Detected", "Orientation", "Size", "Blur score",
                    "Blur", "Glare", "Crop", "Unreadable", "Warnings",
                ],
                interactive=False, wrap=True,
            )
            evidence_field = gr.Dropdown(choices=[], label="Inspect field evidence")
            evidence = gr.JSON(label="Evidence, alternates, validation and review reason")
            summary = gr.JSON(label="Processing summary")

        gr.Markdown("### Confirm and export")
        with gr.Row():
            confirm_button = gr.Button("Confirm and Continue", variant="primary", scale=3)
            report_button = gr.Button("Processing report", scale=1)
            delete_button = gr.Button("Delete session data", variant="stop", scale=1)

        final_card = gr.HTML("")
        final_json = gr.Code(label="Final JSON", language="json", interactive=False)
        with gr.Row():
            final_download = gr.File(label="Download final JSON", interactive=False)
            report_download = gr.File(label="Download processing report", interactive=False)

        value_inputs = [editors[path][0] for path in FIELD_LABELS]
        status_outputs = [editors[path][1] for path in FIELD_LABELS]
        paired_outputs: list[Any] = []
        for path in FIELD_LABELS:
            paired_outputs.extend(editors[path])

        read_event = read_button.click(
            lambda files, state: process_documents(files, state, reader),
            [bundle, session],
            [session, banner, cards, preview, quality, summary, evidence_field,
             *paired_outputs],
        )
        cancel_button.click(fn=None, cancels=[read_event])

        apply_button.click(
            apply_edits,
            [session, *value_inputs],
            [session, banner, cards, *status_outputs],
        )

        evidence_field.change(show_evidence, [evidence_field, session], evidence)

        confirm_button.click(
            confirm_result,
            [session, *value_inputs],
            [session, final_card, final_json, final_download, report_download, cards],
        )
        report_button.click(download_report, session, report_download)

        reset_targets = [
            session, banner, cards, preview, quality, summary, evidence_field,
            evidence, final_card, final_json, final_download, report_download,
            bundle, *paired_outputs,
        ]
        reset_button.click(reset_everything, session, reset_targets)
        delete_button.click(reset_everything, session, reset_targets)

    return demo
