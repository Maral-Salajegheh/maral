from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from Life.Extraction import config
from Life.Extraction.utils import page_key, read_latest_jsonl, utc_now_iso


FULL_MRZ_FIELDS = [
    "document_code",
    "issuing_state",
    "surname",
    "given_names",
    "document_number",
    "nationality",
    "date_of_birth",
    "sex",
    "expiry_date",
    "optional_data",
]


CSV_FIELDS = [
    "masterindex_id",
    "page_number",
    "doc_type",
    "ausweisnummer",
    "ausweistyp",
    "nationalitaet",
    "gueltigkeitsdatum",
    "surname",
    "given_names",
    "date_of_birth",
    "sex",
    "geburtsort",
    "ausstellende_behoerde",
    "mrz_checksum_passed",
    "source_geburtsort",
    "needs_human_review",
    "status",
    "failure_stage",
    "error",
]


# The last stage that produced a usable result, so a blank row says which step to look at.
def failure_stage(record: dict[str, Any]) -> str:
    if record.get("status") == "success":
        return ""
    if not record.get("mrz_crop_path"):
        return "01_detect"
    if not str(record.get("mrz_ocr_text") or "").strip():
        return "02_ocr"
    if not record.get("checks_valid"):
        return "03_parse"
    if not record.get("ausweistyp"):
        return "04_ausweistyp"
    return "05_fields"


def mrz_field(record: dict[str, Any], field: str, checksum_valid: bool) -> dict[str, Any]:
    return {
        "value": record.get(field) if checksum_valid else None,
        "source": "mrz",
        "confidence_class": "checksum_validated" if checksum_valid else "rejected_checksum_failed",
        "checksum_valid": checksum_valid,
    }


def printed_field(record: dict[str, Any], field: str) -> dict[str, Any]:
    return {
        "value": record.get(field),
        "source": record.get("field_extraction_source") or "field_unknown",
        "confidence_class": "not_checksum_validated",
        "checksum_valid": None,
    }


def build_payload(record: dict[str, Any]) -> dict[str, Any]:
    checksum_valid = bool(record.get("checks_valid"))
    name_mismatch = bool(record.get("name_cross_check_mismatch"))
    needs_review = bool(record.get("needs_human_review")) or not checksum_valid or name_mismatch
    fields = {
        "ausweisnummer": mrz_field(record, "document_number", checksum_valid),
        "ausweistyp": {
            "value": record.get("ausweistyp") if checksum_valid else None,
            "source": "mrz_document_code",
            "confidence_class": "checksum_validated" if checksum_valid else "rejected_checksum_failed",
            "checksum_valid": checksum_valid,
        },
        "nationalitaet": mrz_field(record, "nationality", checksum_valid),
        "gueltigkeitsdatum": mrz_field(record, "expiry_date", checksum_valid),
        "geburtsort": printed_field(record, "geburtsort"),
        "ausstellende_behoerde": printed_field(record, "ausstellende_behoerde"),
    }
    return {
        "schema_version": "ausweiskopie_gwg_v1",
        "created_at_utc": utc_now_iso(),
        "masterindex_id": record.get("masterindex_id"),
        "page_number": record.get("page_number"),
        "pdf_path_in_zip": record.get("pdf_path_in_zip"),
        "image_path": record.get("image_path"),
        "resolved_image_path": record.get("resolved_image_path"),
        "mrz_format": record.get("format") if checksum_valid else None,
        "mrz_lines": record.get("mrz_lines") if checksum_valid else None,
        "mrz_fields": {field: record.get(field) for field in FULL_MRZ_FIELDS} if checksum_valid else None,
        "mrz_check_digit_results": record.get("check_digit_results") or record.get("checks"),
        "repair_applied": bool(record.get("repair_applied")),
        "printed_fields": {
            "source": record.get("field_extraction_source") or "field_unknown",
            "geburtsort": record.get("geburtsort"),
            "ausstellende_behoerde": record.get("ausstellende_behoerde"),
            "surname": record.get("printed_surname"),
            "given_names": record.get("printed_given_names"),
        },
        "cross_checks": {
            "name_cross_check": record.get("name_cross_check"),
            "name_cross_check_passed": record.get("name_cross_check_passed"),
            "name_cross_check_mismatch": name_mismatch,
        },
        "fields": fields,
        "needs_human_review": needs_review,
        "status": record.get("status") or "failed",
        "failure_stage": failure_stage(record),
        "mrz_ocr_text": record.get("mrz_ocr_text") or "",
        "mrz_crop_path": record.get("mrz_crop_path"),
        "mrz_overlay_path": record.get("mrz_overlay_path"),
        "error": record.get("error") or "",
    }


def field_value(payload: dict[str, Any], name: str) -> Any:
    return payload["fields"][name]["value"]


def build_csv_row(payload: dict[str, Any]) -> dict[str, Any]:
    mrz_fields = payload.get("mrz_fields") or {}
    checksum_passed = bool(payload["fields"]["ausweisnummer"]["checksum_valid"])
    return {
        "masterindex_id": payload.get("masterindex_id"),
        "page_number": payload.get("page_number"),
        "doc_type": payload.get("mrz_format"),
        "ausweisnummer": field_value(payload, "ausweisnummer"),
        "ausweistyp": field_value(payload, "ausweistyp"),
        "nationalitaet": field_value(payload, "nationalitaet"),
        "gueltigkeitsdatum": field_value(payload, "gueltigkeitsdatum"),
        "surname": mrz_fields.get("surname"),
        "given_names": mrz_fields.get("given_names"),
        "date_of_birth": mrz_fields.get("date_of_birth"),
        "sex": mrz_fields.get("sex"),
        "geburtsort": field_value(payload, "geburtsort"),
        "ausstellende_behoerde": field_value(payload, "ausstellende_behoerde"),
        "mrz_checksum_passed": checksum_passed,
        "source_geburtsort": payload["fields"]["geburtsort"]["source"],
        "needs_human_review": payload.get("needs_human_review"),
        "status": payload.get("status"),
        "failure_stage": payload.get("failure_stage"),
        "error": payload.get("error"),
    }


def write_flat_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in CSV_FIELDS})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=config.FIELD_EXTRACTION_JSONL)
    parser.add_argument("--output-dir", type=Path, default=config.FINAL_JSON_DIR)
    parser.add_argument("--csv-output", type=Path, default=config.FINAL_CSV_PATH)
    args = parser.parse_args()
    records = read_latest_jsonl(args.input, page_key)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_rows = []
    for key, record in sorted(records.items()):
        payload = build_payload(record)
        mid = payload.get("masterindex_id") or key[0]
        page_number = payload.get("page_number") or key[2]
        suffix = str(record.get("image_sha256") or "")[:8]
        out_path = args.output_dir / (f"{mid}_page_{page_number}_{suffix}.json" if suffix else f"{mid}_page_{page_number}.json")
        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
        csv_rows.append(build_csv_row(payload))
    write_flat_csv(args.csv_output, csv_rows)
    print(f"Wrote document JSON files: {args.output_dir}")
    print(f"Wrote flat document CSV: {args.csv_output}")


if __name__ == "__main__":
    main()
