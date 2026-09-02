from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any

from PIL import Image
from pydantic import BaseModel

from Life.Extraction import config
from Life.Extraction.mrz_utils import transliterate
from Life.Extraction.utils import append_jsonl, page_key, read_latest_jsonl, run_field_ocr, utc_now_iso

FIELD_SYSTEM_PROMPT = """You extract printed text fields from a cropped identity-document field zone. Return only the requested JSON fields. If a field is not visible or illegible, return null. Do not guess."""
FIELD_USER_PROMPT = """Read the cropped field zone and return JSON with exactly these keys: geburtsort, ausstellende_behoerde, surname, given_names. Use the printed text only. Return null for any illegible or missing value."""

LABEL_PATTERNS = {
    "geburtsort": [r"(?:GEBURTSORT|PLACE OF BIRTH)\s*[:\-]?\s*([^\n]+)"],
    "ausstellende_behoerde": [r"(?:AUSSTELLENDE\s+BEH[OÖ]RDE|ISSUING\s+AUTHORITY)\s*[:\-]?\s*([^\n]+)"],
    "printed_surname": [r"(?:NAME|FAMILIENNAME|SURNAME)\s*[:\-]?\s*([^\n]+)"],
    "printed_given_names": [r"(?:VORNAMEN?|GIVEN\s+NAMES?|FORENAMES?)\s*[:\-]?\s*([^\n]+)"],
}


class PrintedFieldResponse(BaseModel):
    geburtsort: str | None = None
    ausstellende_behoerde: str | None = None
    surname: str | None = None
    given_names: str | None = None


def crop_printed_field_zone(record: dict[str, Any]) -> Path:
    image = Image.open(record["resolved_image_path"]).convert("RGB")
    width, height = image.size
    rotation = int(record.get("rotation_degrees") or 0)
    if rotation:
        image = image.rotate(rotation, expand=True)
        width, height = image.size
    y1 = int(record.get("y0") or height * 0.75)
    if y1 <= height * 0.25:
        y1 = int(height * 0.75)
    crop = image.crop((0, 0, width, max(y1, 1)))
    out_path = config.FIELD_CROP_DIR / f"{record.get('image_sha256') or record['masterindex_id']}_{record['page_number']}_fields.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    crop.save(out_path)
    return out_path


def clean_field_value(value: str | None) -> str | None:
    if value is None:
        return None
    cleaned = re.sub(r"[\r\n\t]+", " ", str(value)).strip(" .,:;-")
    return cleaned or None


def extract_value(text: str, field: str) -> str | None:
    compact = re.sub(r"[ \t]+", " ", text.upper())
    for pattern in LABEL_PATTERNS[field]:
        match = re.search(pattern, compact)
        if match:
            value = re.sub(r"[^A-ZÄÖÜ0-9 .,/\-]", "", match.group(1)).strip(" .,:;-\t")
            if value:
                return value[:120]
    return None


# Compare names in MRZ spelling: the printed zone shows MÜLLER where the MRZ shows MUELLER.
def normalize_name(value: str | None) -> str | None:
    if not value:
        return None
    normalized = re.sub(r"[^A-Z0-9]", "", transliterate(value.replace("<", " ")))
    return normalized or None


def name_cross_check(record: dict[str, Any], printed: dict[str, str | None]) -> dict[str, Any]:
    checks = []
    for mrz_field, printed_field in [("surname", "printed_surname"), ("given_names", "printed_given_names")]:
        mrz_value = record.get(mrz_field)
        printed_value = printed.get(printed_field)
        mrz_norm = normalize_name(mrz_value)
        printed_norm = normalize_name(printed_value)
        if mrz_norm and printed_norm:
            status = "match" if mrz_norm == printed_norm else "mismatch"
        elif mrz_norm or printed_norm:
            status = "missing_counterpart"
        else:
            status = "not_available"
        checks.append({"field": mrz_field, "mrz_value": mrz_value, "printed_value": printed_value, "status": status})
    mismatch = any(item["status"] == "mismatch" for item in checks)
    return {"name_cross_check": checks, "name_cross_check_passed": not mismatch, "name_cross_check_mismatch": mismatch}


def extract_with_ocr(crop_path: Path) -> dict[str, Any]:
    text, backend = run_field_ocr(crop_path)
    return {
        "field_extraction_source": "field_ocr",
        "field_extraction_backend": "ocr",
        "field_ocr_backend": backend,
        "field_ocr_stage_version": config.FIELD_OCR_STAGE_VERSION,
        "field_ocr_text": text,
        "geburtsort": extract_value(text, "geburtsort"),
        "ausstellende_behoerde": extract_value(text, "ausstellende_behoerde"),
        "printed_surname": extract_value(text, "printed_surname"),
        "printed_given_names": extract_value(text, "printed_given_names"),
    }


def parse_llm_answer(answer: Any) -> PrintedFieldResponse:
    if isinstance(answer, PrintedFieldResponse):
        return answer
    if hasattr(answer, "model_dump"):
        return PrintedFieldResponse.model_validate(answer.model_dump())
    if isinstance(answer, str):
        try:
            return PrintedFieldResponse.model_validate_json(answer)
        except Exception:
            match = re.search(r"\{.*\}", answer, flags=re.DOTALL)
            if match:
                return PrintedFieldResponse.model_validate_json(match.group(0))
            raise
    return PrintedFieldResponse.model_validate(answer)


def call_securegpt_field_backend(crop_path: Path) -> dict[str, Any]:
    if str(config.SECUREGPT_WRAPPER_DIR) not in sys.path:
        sys.path.insert(0, str(config.SECUREGPT_WRAPPER_DIR))
    try:
        from securegpt_vision import MODEL_NAME, SEED, TEMPERATURE, create_securegpt_client, normalize_page_image
    except ModuleNotFoundError as error:
        raise RuntimeError("SecureGPT field backend requires the existing ausweiskopie_page_detection environment with axallm.securegpt.v2 installed. Use --backend ocr in this environment or run stage 05 from the SecureGPT-enabled environment.") from error

    client = create_securegpt_client()
    image_data_url = normalize_page_image(crop_path)
    call_modes = ("full", "no_detail", "raw_json")
    last_type_error: TypeError | None = None
    for mode in call_modes:
        try:
            if mode == "full":
                response = client.new_chat(system_prompt=FIELD_SYSTEM_PROMPT, user_prompt=FIELD_USER_PROMPT, user_image=image_data_url, image_detail="high", response_model=PrintedFieldResponse)
            elif mode == "no_detail":
                response = client.new_chat(system_prompt=FIELD_SYSTEM_PROMPT, user_prompt=FIELD_USER_PROMPT, user_image=image_data_url, response_model=PrintedFieldResponse)
            else:
                response = client.new_chat(system_prompt=FIELD_SYSTEM_PROMPT, user_prompt=FIELD_USER_PROMPT + " Return valid JSON only.", user_image=image_data_url)
        except TypeError as error:
            last_type_error = error
            continue
        answer = response.answer if hasattr(response, "answer") else response
        parsed = parse_llm_answer(answer)
        return {
            "field_extraction_source": "field_llm",
            "field_llm_model_id": MODEL_NAME,
            "field_llm_temperature": TEMPERATURE,
            "field_llm_seed": SEED,
            "geburtsort": clean_field_value(parsed.geburtsort),
            "ausstellende_behoerde": clean_field_value(parsed.ausstellende_behoerde),
            "printed_surname": clean_field_value(parsed.surname),
            "printed_given_names": clean_field_value(parsed.given_names),
        }
    raise TypeError(f"No supported SecureGPT new_chat signature found. Last TypeError: {last_type_error}")


def extract_printed_fields(crop_path: Path, backend: str) -> dict[str, Any]:
    if backend == "ocr":
        return extract_with_ocr(crop_path)
    if backend == "llm":
        return call_securegpt_field_backend(crop_path)
    raise ValueError(f"Unsupported field extraction backend: {backend}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=config.AUSWEISTYP_JSONL)
    parser.add_argument("--output", type=Path, default=config.FIELD_EXTRACTION_JSONL)
    parser.add_argument("--backend", choices=["llm", "ocr"], default=config.FIELD_EXTRACTION_BACKEND)
    args = parser.parse_args()
    records = read_latest_jsonl(args.input, page_key)
    done = read_latest_jsonl(args.output, page_key)
    for key, record in sorted(records.items()):
        previous = done.get(key)
        if (
            previous
            and previous.get("status") == "success"
            and previous.get("field_extraction_backend") == args.backend
            and previous.get("field_ocr_stage_version") == config.FIELD_OCR_STAGE_VERSION
            and previous.get("document_number") == record.get("document_number")
            and previous.get("mrz_crop_path") == record.get("mrz_crop_path")
            and previous.get("detector_version") == record.get("detector_version")
        ):
            continue
        if record.get("status") != "success":
            append_jsonl(args.output, {**record, "status": "failed", "needs_human_review": True, "error": record.get("error") or "Ausweistyp mapping failed", "processed_at_utc": utc_now_iso()})
            continue
        try:
            crop_path = crop_printed_field_zone(record)
            extracted = extract_printed_fields(crop_path, args.backend)
            cross_check = name_cross_check(record, extracted)
            # Geburtsort and ausstellende Behoerde sit on opposite sides of a Personalausweis,
            # so one page yielding only one of them is normal. Review only when neither is found.
            found = [name for name in ("geburtsort", "ausstellende_behoerde") if extracted.get(name)]
            needs_review = not found or cross_check["name_cross_check_mismatch"]
            errors = []
            if not found:
                errors.append("Printed field extraction returned neither geburtsort nor ausstellende_behoerde")
            elif len(found) == 1:
                errors.append(f"Only {found[0]} found on this page; the other field is likely on the reverse side")
            if cross_check["name_cross_check_mismatch"]:
                errors.append("Printed names disagree with checksum-valid MRZ names")
            output = {**record, **extracted, **cross_check, "field_crop_path": str(crop_path), "status": "success", "needs_human_review": needs_review, "error": "; ".join(errors),
                      "processed_at_utc": utc_now_iso()}
        except Exception as error:
            output = {**record, "field_extraction_source": f"field_{args.backend}", "field_extraction_backend": args.backend, "field_ocr_stage_version": config.FIELD_OCR_STAGE_VERSION, "geburtsort": None, "ausstellende_behoerde": None,
                      "printed_surname": None, "printed_given_names": None, "status": "failed", "needs_human_review": True, "error": str(error), "processed_at_utc": utc_now_iso()}
        append_jsonl(args.output, output)
    print(f"Wrote printed-field extraction cache: {args.output}")


if __name__ == "__main__":
    main()
