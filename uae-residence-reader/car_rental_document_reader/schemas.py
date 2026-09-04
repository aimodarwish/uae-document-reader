from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class CustomerType(str, Enum):
    UAE_RESIDENT = "UAE_RESIDENT"
    GCC_NATIONAL = "GCC_NATIONAL"
    TOURIST = "TOURIST"


class DocumentType(str, Enum):
    EMIRATES_ID_FRONT = "EMIRATES_ID_FRONT"
    EMIRATES_ID_BACK = "EMIRATES_ID_BACK"
    UAE_DRIVING_LICENCE_FRONT = "UAE_DRIVING_LICENCE_FRONT"
    UAE_DRIVING_LICENCE_BACK = "UAE_DRIVING_LICENCE_BACK"
    GCC_IDENTITY_FRONT = "GCC_IDENTITY_FRONT"
    GCC_IDENTITY_BACK = "GCC_IDENTITY_BACK"
    GCC_DRIVING_LICENCE_FRONT = "GCC_DRIVING_LICENCE_FRONT"
    GCC_DRIVING_LICENCE_BACK = "GCC_DRIVING_LICENCE_BACK"
    PASSPORT_BIODATA = "PASSPORT_BIODATA"
    INTERNATIONAL_DRIVING_PERMIT = "INTERNATIONAL_DRIVING_PERMIT"
    NATIONAL_DRIVING_LICENCE_FRONT = "NATIONAL_DRIVING_LICENCE_FRONT"
    NATIONAL_DRIVING_LICENCE_BACK = "NATIONAL_DRIVING_LICENCE_BACK"
    UNKNOWN = "UNKNOWN"


class FieldStatus(str, Enum):
    VERIFIED = "VERIFIED"
    HIGH_CONFIDENCE = "HIGH_CONFIDENCE"
    NEEDS_REVIEW = "NEEDS_REVIEW"
    CONFLICTING = "CONFLICTING"
    MISSING = "MISSING"
    MANUALLY_EDITED = "MANUALLY_EDITED"


class FieldCandidate(BaseModel):
    field_path: str
    value: str | None = None
    normalized_value: str | None = None
    source_document: str
    source_method: str
    confidence: float = Field(ge=0, le=1)
    evidence_text: str | None = None
    bounding_box: list[list[float]] | None = None
    validation_passed: bool | None = None
    warnings: list[str] = Field(default_factory=list)


class FieldMetadata(BaseModel):
    status: FieldStatus
    confidence: float | None = Field(default=None, ge=0, le=1)
    confidence_components: dict[str, Any] = Field(default_factory=dict)
    source_document: str = ""
    source_method: str = ""
    evidence_text: str | None = None
    bounding_box: list[list[float]] | None = None
    alternate_candidates: list[dict[str, Any]] = Field(default_factory=list)
    validation_results: list[str] = Field(default_factory=list)
    reason_for_review: str | None = None
    manually_edited: bool = False


class QualityInfo(BaseModel):
    blur_score: float | None = None
    glare_detected: bool | None = None
    low_resolution: bool | None = None
    rotation_degrees: float | None = None
    width: int | None = None
    height: int | None = None
    orientation: str | None = None
    crop_warning: bool | None = None
    overexposed: bool | None = None
    underexposed: bool | None = None
    unreadable: bool | None = None
    warnings: list[str] = Field(default_factory=list)


class DocumentRecord(BaseModel):
    upload_id: str
    expected_type: str
    detected_type: str
    type_match: bool
    side: str = ""
    issuing_country_code: str | None = None
    # Language names reported by the visual router for difficult tourist
    # documents. Empty on workflows that do not need multilingual routing.
    detected_languages: list[str] = Field(default_factory=list)
    quality: QualityInfo = Field(default_factory=QualityInfo)


class PersonalInfo(BaseModel):
    first_name: str | None = None
    # Retained internally for backward-compatible parsing of passports, but it
    # is intentionally omitted from every exported JSON result and from the UI.
    middle_name: str | None = Field(default=None, exclude=True)
    last_name: str | None = None
    full_name: str | None = None
    full_name_arabic: str | None = None
    gender: str | None = None
    date_of_birth: str | None = None
    nationality_code: str | None = None
    nationality_name: str | None = None
    place_of_birth: str | None = None


class BasicDocument(BaseModel):
    number: str | None = None
    issued_by_code: str | None = None
    issued_by_name: str | None = None
    issue_date: str | None = None
    expiry_date: str | None = None


# The GCC workflow reports only the customer fields, so the licence carries no
# class or blood-group attribute for a value to land in.
GCCDrivingLicence = BasicDocument


class EmiratesId(BaseModel):
    number: str | None = None
    issue_date: str | None = None
    expiry_date: str | None = None


class MRZChecks(BaseModel):
    document_number: bool | None = None
    date_of_birth: bool | None = None
    expiry_date: bool | None = None
    optional_data: bool | None = None
    composite: bool | None = None


class MRZData(BaseModel):
    type: str | None = None
    raw_lines: list[str] = Field(default_factory=list)
    normalized_lines: list[str] = Field(default_factory=list)
    valid: bool | None = None
    checks: MRZChecks = Field(default_factory=MRZChecks)
    corrections: list[dict[str, Any]] = Field(default_factory=list)


class Passport(BasicDocument):
    # The lifelong citizen number, printed beside the document number and never
    # to be confused with it. Kept because it is the one identifier a passport
    # and a national driving licence share, which makes it the strongest
    # evidence that the two documents belong to the same person.
    holder_id: str | None = None
    mrz: MRZData = Field(default_factory=MRZData)


class NationalDrivingLicence(BasicDocument):
    # Designator 4d on the EU and Vienna models: the holder's national number,
    # not the licence number.
    holder_id: str | None = None


class CrossDocumentChecks(BaseModel):
    name_similarity: float | None = None
    date_of_birth_match: bool | None = None
    nationality_match: bool | None = None
    gender_match: bool | None = None
    holder_id_match: bool | None = None
    conflicts: list[dict[str, Any]] = Field(default_factory=list)


class ProcessingInfo(BaseModel):
    duration_seconds: float | None = None
    # Where those seconds went, by stage. A read that takes twice as long as
    # it should is otherwise a single number with no way to attribute it, and
    # the stages differ by orders of magnitude in what they cost: a network
    # OCR call is seconds of waiting, preprocessing is processor time, and the
    # extraction that follows both is milliseconds. Timings only; no page
    # content, identifier or filename appears here.
    stage_seconds: dict[str, float] = Field(default_factory=dict)
    device: str | None = None
    ocr_model: str | None = None
    vlm_model: str | None = None
    versions: dict[str, str | None] = Field(default_factory=dict)


class LicencePolicyDecision(BaseModel):
    country: str | None = None
    iso3: str | None = None
    requirement: str | None = None
    accepted_document: str | None = None
    template_family: str | None = None
    nationality_match_required: bool = False
    nationality_match: bool | None = None
    layout_sources: list[str] = Field(default_factory=list)
    # How the country was arrived at. A country the operator chose, one the
    # licence proved and one merely inferred from the passport carry very
    # different weight, and an operator reviewing the result has to be able to
    # tell them apart.
    detected_from: str | None = None
    detection_confidence: float | None = Field(default=None, ge=0, le=1)
    detection_evidence: str | None = None
    # Which UN model the presented permit conforms to, when one was presented.
    idp_convention: str | None = None


class GCCProfileDecision(BaseModel):
    country: str | None = None
    iso3: str | None = None
    identity_layout: str | None = None
    licence_layout: str | None = None
    source_urls: list[str] = Field(default_factory=list)


class ExtractionResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    customer_type: CustomerType
    documents: list[DocumentRecord] = Field(default_factory=list)
    personal_info: PersonalInfo = Field(default_factory=PersonalInfo)
    emirates_id: EmiratesId = Field(default_factory=EmiratesId)
    passport: Passport = Field(default_factory=Passport)
    uae_driving_licence: BasicDocument = Field(default_factory=BasicDocument)
    gcc_identity: BasicDocument = Field(default_factory=BasicDocument)
    gcc_driving_licence: GCCDrivingLicence = Field(default_factory=GCCDrivingLicence)
    international_driving_permit: BasicDocument = Field(default_factory=BasicDocument)
    national_driving_licence: NationalDrivingLicence = Field(default_factory=NationalDrivingLicence)
    licence_policy: LicencePolicyDecision = Field(default_factory=LicencePolicyDecision)
    gcc_profile: GCCProfileDecision = Field(default_factory=GCCProfileDecision)
    field_metadata: dict[str, FieldMetadata] = Field(default_factory=dict)
    cross_document_checks: CrossDocumentChecks = Field(default_factory=CrossDocumentChecks)
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    manual_review_required: bool = True
    confirmed_by_user: bool = False
    processing: ProcessingInfo = Field(default_factory=ProcessingInfo)


REQUIRED_SLOTS: dict[CustomerType, tuple[str, ...]] = {
    CustomerType.UAE_RESIDENT: ("emirates_id_front", "emirates_id_back", "uae_licence_front"),
    CustomerType.GCC_NATIONAL: ("gcc_identity_front", "gcc_licence_front"),
    CustomerType.TOURIST: ("passport", "idp_pages"),
}
