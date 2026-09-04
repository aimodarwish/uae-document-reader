from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from PIL import Image

from .config import AppConfig, ModelChoice, RuntimeInfo, clear_accelerators
from .gcc_profiles import GCC_COUNTRY_NAMES, normalize_gcc_number
from .normalize import (
    implausible_birth_date, normalize_country, normalize_date, normalize_emirates_id,
)
from .ocr import OCRLine


VLM_SYSTEM_INSTRUCTION = (
    "You extract identity-document fields only from visible document evidence.\n"
    "Never guess, complete, infer, or fabricate a value.\n"
    "Never repair an incomplete number unless a deterministic machine-readable checksum proves the correction.\n"
    "Return null for missing, obscured, uncertain, or unreadable values.\n"
    "Do not translate personal names into a different identity value.\n"
    "Return gender only when Sex/Gender/الجنس/النوع is explicitly printed or encoded in a valid MRZ; "
    "never infer it from a name, title, portrait, clothing, or document number.\n"
    "Return strict JSON only."
)


_ROUTABLE_DOCUMENT_TYPES = frozenset({
    "PASSPORT_BIODATA",
    "NATIONAL_DRIVING_LICENCE_FRONT",
    "NATIONAL_DRIVING_LICENCE_BACK",
    "INTERNATIONAL_DRIVING_PERMIT",
})


def _validated_routing(payload: Any) -> dict[str, Any]:
    """Keep only conservative document-routing metadata from a VLM answer."""
    if not isinstance(payload, dict):
        return {}
    routed: dict[str, Any] = {}
    document_type = str(payload.get("document_type") or "").strip().upper()
    if document_type in _ROUTABLE_DOCUMENT_TYPES:
        routed["document_type"] = document_type
    raw_country = str(
        payload.get("issuing_country_code")
        or payload.get("issuing_country")
        or ""
    ).strip()
    if raw_country:
        code, name, _ = normalize_country(raw_country)
        if code and name:
            routed["issuing_country_code"] = code
            routed["issuing_country_name"] = name
    raw_languages = payload.get("languages")
    if isinstance(raw_languages, str):
        raw_languages = [raw_languages]
    if isinstance(raw_languages, list):
        languages = []
        for value in raw_languages[:4]:
            language = " ".join(str(value).strip().split())[:40]
            if language and re.fullmatch(r"[A-Za-z][A-Za-z -]{0,39}", language):
                languages.append(language)
        if languages:
            routed["languages"] = list(dict.fromkeys(languages))
    return routed


def _parse_json_object(text: str) -> dict[str, Any]:
    """Parse the first complete JSON object from a model response.

    Multimodal instruction models occasionally wrap an otherwise valid answer
    in a Markdown fence or a one-line preamble.  Rejecting the whole extraction
    for that harmless formatting variation loses good document evidence, so we
    locate the first balanced object while still requiring strict JSON inside it.
    """
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].lstrip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()
    try:
        payload = json.loads(stripped)
        if not isinstance(payload, dict):
            raise ValueError("VLM JSON root must be an object")
        return payload
    except json.JSONDecodeError:
        pass

    start = stripped.find("{")
    if start < 0:
        raise ValueError("No JSON object found")
    depth = 0
    in_string = False
    escaped = False
    safe_commas: list[int] = []
    for index in range(start, len(stripped)):
        character = stripped[index]
        if in_string:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
            continue
        if character == '"':
            in_string = True
        elif character == "{":
            depth += 1
        elif character == "}":
            depth -= 1
            if depth == 0:
                payload = json.loads(stripped[start:index + 1])
                if not isinstance(payload, dict):
                    raise ValueError("VLM JSON root must be an object")
                return payload
        elif character == "," and depth > 0:
            safe_commas.append(index)
    # A generation limit can cut off the final field. Keep fully completed
    # pairs before the last safe comma, then close only the open JSON objects.
    for end in [len(stripped), *reversed(safe_commas)]:
        fragment = stripped[start:end].rstrip().rstrip(",")
        balance = 0
        fragment_in_string = False
        fragment_escaped = False
        for character in fragment:
            if fragment_in_string:
                if fragment_escaped:
                    fragment_escaped = False
                elif character == "\\":
                    fragment_escaped = True
                elif character == '"':
                    fragment_in_string = False
            elif character == '"':
                fragment_in_string = True
            elif character == "{":
                balance += 1
            elif character == "}":
                balance -= 1
        if fragment_in_string or balance <= 0:
            continue
        try:
            payload = json.loads(fragment + "}" * balance)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    raise ValueError("Incomplete JSON object")


@dataclass
class VLMState:
    model_id: str | None
    license: str = "Apache-2.0"
    loaded: bool = False
    local_path: str | None = None
    warnings: list[str] = field(default_factory=list)


class LocalVLM:
    def __init__(self, choice: ModelChoice, runtime: RuntimeInfo, config: AppConfig):
        self.choice, self.runtime, self.config = choice, runtime, config
        self.state = VLMState(choice.model_id)
        self.model: Any = None
        self.processor: Any = None

    def initialize(self, download: bool = True) -> VLMState:
        if self.state.loaded or not self.choice.model_id:
            return self.state
        try:
            from huggingface_hub import HfApi, snapshot_download
            from transformers import AutoProcessor, BitsAndBytesConfig
            try:
                # Qwen3-VL's current model card uses this general multimodal
                # auto-class. Retain the older alias for controlled environments.
                from transformers import AutoModelForMultimodalLM as AutoMultimodalModel
            except ImportError:
                from transformers import AutoModelForImageTextToText as AutoMultimodalModel
            if download:
                info = HfApi(token=False).model_info(self.choice.model_id)
                if info.id != self.choice.model_id:
                    raise RuntimeError("Model identity mismatch")
                local_path = snapshot_download(self.choice.model_id, token=False)
            else:
                local_path = snapshot_download(self.choice.model_id, local_files_only=True, token=False)
            kwargs: dict[str, Any] = {"device_map": "auto", "local_files_only": True, "low_cpu_mem_usage": True}
            if self.choice.quantization == "4-bit":
                kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True, bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype="bfloat16", bnb_4bit_use_double_quant=True,
                )
            self.processor = AutoProcessor.from_pretrained(local_path, local_files_only=True)
            self.model = AutoMultimodalModel.from_pretrained(local_path, **kwargs).eval()
            self.state.local_path, self.state.loaded = local_path, True
        except Exception as exc:
            detail = str(exc).replace("\n", " ").strip()[:500]
            self.state.warnings.append(f"VLM_LOAD_FAILED:{type(exc).__name__}:{detail}")
            self.model = self.processor = None
            clear_accelerators()
        return self.state

    def extract(self, image: Image.Image, expected_type: str, ocr_lines: list[OCRLine], schema: dict[str, Any]) -> dict[str, Any]:
        if not self.state.loaded or self.model is None or self.processor is None:
            return {"values": {}, "warning": "VLM_UNAVAILABLE"}
        # Keep the OCR transcript compact: the image conveys layout, while these
        # strings help the model preserve exact spellings in small print.
        evidence = []
        seen_evidence: set[tuple[str, str]] = set()
        for line in sorted(ocr_lines, key=lambda item: item.confidence, reverse=True):
            text = " ".join(line.text.split())
            key = (line.language, text.casefold())
            if not text or key in seen_evidence:
                continue
            seen_evidence.add(key)
            evidence.append(f"[{line.language}] {text}")
            if len(evidence) >= 80:
                break
        field_paths = list(schema)
        aliases = {str(index): path for index, path in enumerate(field_paths)}
        field_legend = "\n".join(f"{alias}={path}" for alias, path in aliases.items())
        request = (
            f"Expected document type: {expected_type}\n"
            f"OCR evidence: {json.dumps(evidence, ensure_ascii=False)}\n"
            f"Allowed field keys (short_key=meaning):\n{field_legend}\n"
            "For a Russian passport, use the printed Latin transliteration or valid MRZ for names. "
            "SURNAME/FAMILY NAME is last_name. Put the complete printed GIVEN NAMES value in first_name, "
            "preserving every word (for example, AMINA NOOR stays AMINA NOOR); do not shorten it to the first word. "
            "Do not use an expiry or issue date as date_of_birth. "
            "For an international driving permit booklet of any issuing country, personal identity fields come "
            "from the holder-data page, while permit number, place of issue, issue date, expiry date and class may "
            "come from a separate ISSUE OF PERMIT page shown in the same image. A cover or convention-title-only "
            "page has no holder fields. Values aligned with printed labels may be handwritten; read clearly visible "
            "handwriting from the image even when ordinary OCR did not transcribe it. Preserve personal-name "
            "spelling exactly. "
            "On any international driving permit, the permit number is the value printed beside the numero "
            "sign (No/N°) under the INTERNATIONAL DRIVING PERMIT title, usually in red ink, and it may contain "
            "letters and spaces, as in 01 EA 044761. Return it exactly as printed, including its letters and "
            "grouping, without the numero sign. The number at the foot of the same column labelled number of "
            "national driving licence is a different document's number: never return it as "
            "international_driving_permit.number. Do not use a convention year, a date, or a class code as "
            "the permit number. On a booklet whose ISSUE OF PERMIT page has no numero sign, an identifier written "
            "directly below CLASS OF LICENCE and above CERTIFICATE OF COMPETENCE is the permit/certificate number, "
            "not the class itself. "
            "For EU/EEA national driving licences, numbered field 1 is surname, 2 is given names, "
            "3 is date of birth, 4a is issue date, 4b is expiry date, 4c is issuing authority, "
            "and 5 is the driving-licence number. Do not confuse fields 9-12 (vehicle categories, "
            "validity by category, or restriction codes) with the licence number or personal dates. "
            "For any other national driving licence, use the visible local-language labels and layout; "
            "distinguish the licence/driver number from a card serial, civil ID, traffic number, or barcode serial. "
            "For a UAE driving licence, accept both Driving License and Driving Licence titles; extract the value "
            "immediately beside Licence No/License No as uae_driving_licence.number. It must be the visible "
            "numeric value only; never return Traffic No, AJTR/card/reference codes, or another nearby number. Read Issue Date, "
            "Expiry Date, and Place of Issue from their matching rows. An Emirates ID number begins with 784 "
            "and must never be returned as the UAE driving-licence number. "
            "For an Emirates ID, Issuing Date/Issue Date belongs to emirates_id.issue_date and "
            "Expiry Date belongs to emirates_id.expiry_date; never swap the two. "
            "For a GCC identity card, use only the value beside that country's National ID, Civil ID, "
            "Personal Number or CPR label as gcc_identity.number; never use card serial, traffic file, "
            "barcode serial or address number. For the supplied Saudi national-ID layout headed الهوية الوطنية, "
            "ID Number/رقم الهوية is a 10-digit number beginning with 1. Never use رقم النسخة, QR content, "
            "or barcode digits. Its Gregorian Date of Birth and Expiry Date are printed on the English left "
            "side; reject the adjacent Hijri dates. It has no printed issue date, so gcc_identity.issue_date "
            "must be null. "
            "For the supplied Saudi driving licence headed رخصة سياقة, ID Number/رقم الهوية is the licence "
            "identifier and belongs in gcc_driving_licence.number; this layout has no separate Licence No. "
            "Read Issue Date, Date of Birth, Nationality and Expiry Date from their matching rows. "
            "For other GCC layouts, use only an explicitly labelled licence number. "
            "GCC identity and driving-licence values are never interchangeable: a number, issue date or "
            "expiry read from the identity card belongs only to gcc_identity.*, and one read from the "
            "licence belongs only to gcc_driving_licence.*. Never copy a value from the other document. "
            "On both Saudi layouts the Latin name is SURNAME, GIVEN NAMES: last_name is the text before the "
            "comma and first_name is all printed text after the comma. Do not shorten it or output a middle name. "
            "For every bilingual GCC document, use only the printed Latin name for first_name, last_name, "
            "and full_name. Put an explicitly labelled Arabic holder name only in full_name_arabic; never "
            "transliterate it or compare it with Latin name components. Return gender only from an explicit "
            "Sex/Gender/الجنس/النوع field; never infer gender from the holder name, portrait, or ID number. "
            "Issue and expiry dates require separate visible labels. If only expiry is printed, return "
            "issue date as null. Prefer a printed Gregorian date and never convert or return a Hijri year "
            "as if it were Gregorian. "
            "A country name or nationality printed on a licence is not the holder's passport nationality. "
            "Return dates as YYYY-MM-DD and document numbers without surrounding labels.\n"
            "Also route the visible page without guessing. Return routing.document_type as one of "
            "PASSPORT_BIODATA, NATIONAL_DRIVING_LICENCE_FRONT, NATIONAL_DRIVING_LICENCE_BACK, or "
            "INTERNATIONAL_DRIVING_PERMIT; routing.issuing_country_code as an ISO-3 code only when "
            "the driving document itself visibly identifies its issuing country; and routing.languages "
            "as up to four language names visibly used on the page. A passport nationality does not "
            "prove the issuer of a separate driving document.\n"
            "Return exactly one compact object shaped as "
            "{\"values\":{\"short_key\":\"value\"},\"routing\":{\"document_type\":\"...\","
            "\"issuing_country_code\":\"...\","
            "\"languages\":[\"...\"]}}. "
            "Include only visibly supported non-null values. Use only the short keys above. No Markdown."
        )
        vlm_image = image.convert("RGB")
        if max(vlm_image.size) > 1600:
            vlm_image.thumbnail((1600, 1600), Image.Resampling.LANCZOS)
        messages = [
            {"role": "system", "content": [{"type": "text", "text": VLM_SYSTEM_INSTRUCTION}]},
            {"role": "user", "content": [{"type": "image", "image": vlm_image}, {"type": "text", "text": request}]},
        ]
        try:
            import torch
            prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self.processor(text=[prompt], images=[vlm_image], return_tensors="pt")
            device = next(self.model.parameters()).device
            inputs = {key: value.to(device) if hasattr(value, "to") else value for key, value in inputs.items()}
            with torch.inference_mode():
                generated = self.model.generate(**inputs, do_sample=False, temperature=None, top_p=None, max_new_tokens=self.config.max_new_tokens)
            trimmed = [output[len(input_ids):] for input_ids, output in zip(inputs["input_ids"], generated)]
            decoded = self.processor.batch_decode(trimmed, skip_special_tokens=True)[0].strip()
            payload = _parse_json_object(decoded)
            raw_values = payload.get("values")
            if not isinstance(raw_values, dict):
                # Smaller instruction models sometimes omit the requested
                # outer `values` key and return the short-key map directly.
                if payload and all(str(key) in aliases or str(key) in schema for key in payload):
                    raw_values = payload
                else:
                    raise ValueError("Invalid VLM JSON shape")
            values = {
                aliases.get(str(key), str(key)): value
                for key, value in raw_values.items()
                if aliases.get(str(key), str(key)) in schema
            }
            return {
                "values": values,
                "routing": _validated_routing(payload.get("routing")),
            }
        except (json.JSONDecodeError, ValueError, RuntimeError, TypeError) as exc:
            detail = re.sub(r"[^A-Za-z0-9_ -]", "_", str(exc)).strip().replace(" ", "_")[:80]
            suffix = f":{detail}" if detail else ""
            return {"values": {}, "warning": f"VLM_OUTPUT_REJECTED:{type(exc).__name__}{suffix}"}
        finally:
            clear_accelerators()


def vlm_candidates(
    payload: dict[str, Any], source_document: str,
    licence_country: str | None = None,
) -> list[dict[str, Any]]:
    accepted: list[dict[str, Any]] = []
    items = payload.get("candidates")
    if not isinstance(items, list):
        values = payload.get("values", {})
        items = [
            {"field_path": path, "value": value}
            for path, value in values.items()
        ] if isinstance(values, dict) else []
    for item in items:
        if not isinstance(item, dict) or not item.get("field_path") or item.get("value") in (None, ""):
            continue
        path, raw_value = str(item["field_path"]), str(item["value"]).strip()
        if licence_country == "Saudi Arabia" and path == "personal_info.middle_name":
            continue
        is_gcc = licence_country in GCC_COUNTRY_NAMES
        contains_arabic = re.search(r"[\u0600-\u06ff]", raw_value) is not None
        if is_gcc and path in {"personal_info.first_name", "personal_info.last_name"} and contains_arabic:
            # Keep Arabic names from competing with Latin CRM components.
            continue
        if is_gcc and path == "personal_info.full_name" and contains_arabic:
            path = "personal_info.full_name_arabic"
        normalized_value: str | None = raw_value
        validation: bool | None = None
        if path.endswith(("date_of_birth", "issue_date", "expiry_date")):
            normalized = normalize_date(raw_value, day_first_hint=True)
            normalized_value, validation = normalized.value, normalized.value is not None
            if path.startswith(("gcc_identity.", "gcc_driving_licence.")) and normalized_value:
                year = int(normalized_value[:4])
                if year < 1900 or year > 2100:
                    normalized_value, validation = None, False
            if path == "personal_info.date_of_birth" and implausible_birth_date(normalized_value):
                normalized_value, validation = None, False
        elif path.endswith("nationality_name") or path.endswith("issued_by_name"):
            code, name, _ = normalize_country(raw_value)
            if code and name:
                normalized_value, validation = name, True
        elif path.endswith("gender"):
            upper = raw_value.upper()
            normalized_value = {
                "MALE": "M", "FEMALE": "F", "М": "M", "Ж": "F",
                "ذكر": "M", "أنثى": "F", "انثى": "F",
            }.get(upper, upper)
            validation = normalized_value in {"M", "F", "X"}
        elif path == "gcc_identity.number":
            normalized_value = normalize_gcc_number(raw_value, licence_country, identity=True)
            if normalized_value is None:
                continue
            validation = True
        elif path == "gcc_driving_licence.number":
            normalized_value = normalize_gcc_number(raw_value, licence_country, identity=False)
            if normalized_value is None:
                continue
            validation = True
        elif path == "emirates_id.number":
            # An Emirates ID is fifteen digits beginning 784. Without this the
            # generic number rule below accepts any four-to-twenty characters,
            # so another state's national number reaches the field looking
            # fully validated.
            normalized = normalize_emirates_id(raw_value)
            if normalized.value is None:
                continue
            normalized_value, validation = normalized.value, True
        elif path == "uae_driving_licence.number":
            compact = re.sub(r"[\s-]", "", raw_value)
            if re.fullmatch(r"\d{4,15}", compact) is None:
                continue
            normalized_value, validation = compact, True
        elif path.endswith(".number"):
            normalized_value = re.sub(r"[^A-Z0-9/-]", "", raw_value.upper())
            compact = re.sub(r"[^A-Z0-9]", "", normalized_value)
            max_length = 15 if path == "passport.number" else 20
            minimum_digits = 4 if path == "passport.number" else 1
            validation = (
                4 <= len(compact) <= max_length
                and sum(character.isdigit() for character in compact) >= minimum_digits
            )
            if not validation:
                continue
        elif "name" in path or path == "personal_info.place_of_birth":
            normalized_value = " ".join(raw_value.split())
            if licence_country == "Saudi Arabia" and path == "personal_info.first_name":
                name_part = normalized_value.split(",", 1)[-1].strip()
                normalized_value = name_part
            elif licence_country == "Saudi Arabia" and path == "personal_info.last_name":
                normalized_value = normalized_value.split(",", 1)[0].strip().split()[0]
            if path == "personal_info.place_of_birth" and (
                not any(character.isalpha() for character in normalized_value)
                or any(character.isdigit() for character in normalized_value)
            ):
                continue
            validation = bool(normalized_value)
        if not normalized_value:
            continue
        accepted.append({
            "field_path": path, "value": raw_value,
            "normalized_value": normalized_value, "source_document": source_document,
            "source_method": "vlm", "confidence": 0.72,
            "evidence_text": item.get("evidence_text"), "bounding_box": item.get("bounding_box"),
            "validation_passed": validation, "warnings": ["VLM_ONLY_REQUIRES_REVIEW"],
        })
        if (
            licence_country == "Saudi Arabia"
            and path == "personal_info.full_name"
            and "," in normalized_value
        ):
            surname, given_names = (part.strip(" ,") for part in normalized_value.split(",", 1))
            derived = {
                "personal_info.first_name": given_names,
                "personal_info.last_name": surname,
            }
            for derived_path, derived_value in derived.items():
                if not derived_value:
                    continue
                accepted.append({
                    "field_path": derived_path, "value": derived_value,
                    "normalized_value": derived_value, "source_document": source_document,
                    "source_method": "vlm_name_layout", "confidence": 0.70,
                    "evidence_text": raw_value, "bounding_box": item.get("bounding_box"),
                    "validation_passed": True,
                    "warnings": ["DERIVED_FROM_VISIBLE_SAUDI_SURNAME_FIRST_NAME"],
                })
    return accepted
