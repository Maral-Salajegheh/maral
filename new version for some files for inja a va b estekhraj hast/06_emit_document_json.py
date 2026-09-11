from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from Life.Extraction import config
from Life.Extraction.utils import document_key, page_key, read_latest_jsonl, utc_now_iso


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
    "source_ausstellende_behoerde",
    "name_cross_check_state",
    "repair_ambiguous",
    "field_conflicts",
    "needs_human_review",
    "status",
    "failure_stage",
    "mrz_ocr_text",
    "error",
]


# A page with no MRZ band is not a failure: the front of a Personalausweis carries none.
# Only call it a failure if no page of the same document produced a valid MRZ either.
def resolve_page_roles(payloads: list[dict[str, Any]]) -> None:
    by_document: dict[Any, list[dict[str, Any]]] = {}
    for payload in payloads:
        by_document.setdefault(document_key(payload), []).append(payload)
    for pages in by_document.values():
        carrier = next((page for page in pages if page["fields"]["ausweisnummer"]["checksum_valid"]), None)
        for page in pages:
            if page is carrier:
                continue
            if carrier is not None and page.get("failure_stage") == "01_detect":
                page.update({"status": "no_mrz_on_page", "failure_stage": "", "needs_human_review": False,
                             "error": f"No MRZ on this page; document MRZ read from page {carrier.get('page_number')}"})
        merge_printed_fields(pages, carrier)
    for payload in payloads:
        apply_review_flags(payload)


# Geburtsort is printed on the card front and ausstellende Behoerde on the back, so the
# two required fields arrive on different pages. Fill each from whichever page found it.
def merge_printed_fields(pages: list[dict[str, Any]], carrier: dict[str, Any] | None) -> None:
    if carrier is None:
        return
    conflicts = list(carrier.get("field_conflicts") or [])
    for name in ("geburtsort", "ausstellende_behoerde"):
        target = carrier["fields"][name]
        for donor in pages:
            if donor is carrier or not donor["fields"][name]["value"]:
                continue
            value = donor["fields"][name]["value"]
            if target["value"] and target["value"] != value:
                # Two pages disagree on the same field. Record it rather than overwrite.
                conflicts.append({"field": name, "kept": target["value"], "other": value, "other_page": donor.get("page_number")})
                continue
            if not target["value"]:
                # Each field keeps its own source and donor page, not the other field's.
                target.update({"value": value, "source": donor["fields"][name]["source"], "donor_page": donor.get("page_number")})
    carrier["field_conflicts"] = conflicts


# Review flags are recomputed after merging, so a field filled from the companion page no
# longer counts as missing, while genuine extraction failures still surface.
def apply_review_flags(payload: dict[str, Any]) -> None:
    checksum_valid = bool(payload["fields"]["ausweisnummer"]["checksum_valid"])
    printed = [name for name in ("geburtsort", "ausstellende_behoerde") if payload["fields"][name]["value"]]
    reasons = []
    if payload.get("status") == "no_mrz_on_page":
        payload["needs_human_review"] = False
        return
    if not checksum_valid:
        reasons.append("MRZ check digits did not pass")
    if payload["cross_checks"].get("name_cross_check_state") == "mismatched":
        reasons.append("Printed names disagree with MRZ names")
    if payload.get("repair_ambiguous"):
        reasons.append("Several different MRZ repairs satisfy the check digits")
    if payload.get("mrz_filler_repaired"):
        reasons.append("Optional-data zone was rewritten as filler")
    if payload.get("field_conflicts"):
        reasons.append("Pages disagree on a printed field")
    if not printed:
        reasons.append("Neither geburtsort nor ausstellende_behoerde was extracted")
    payload["needs_human_review"] = bool(reasons)
    payload["review_reasons"] = reasons


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
        "donor_page": None,
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
        "repair_ambiguous": bool(record.get("repair_ambiguous")),
        "repair_alternatives": record.get("repair_alternatives"),
        "repair_changes": record.get("repair_changes") or [],
        "mrz_lines_raw": record.get("mrz_lines_raw"),
        "mrz_filler_repaired": bool(record.get("mrz_filler_repaired")),
        "mrz_parser_version": record.get("mrz_parser_version"),
        "pdf_path_in_zip": record.get("pdf_path_in_zip"),
        "field_conflicts": [],
        "printed_fields": {
            "source": record.get("field_extraction_source") or "field_unknown",
            "geburtsort": record.get("geburtsort"),
            "ausstellende_behoerde": record.get("ausstellende_behoerde"),
            "surname": record.get("printed_surname"),
            "given_names": record.get("printed_given_names"),
        },
        "cross_checks": {
            "name_cross_check": record.get("name_cross_check"),
            "name_cross_check_state": record.get("name_cross_check_state") or "not_checked",
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


# Source plus donor page, so provenance survives into the flat file too.
def source_label(payload: dict[str, Any], name: str) -> str:
    entry = payload["fields"][name]
    donor = entry.get("donor_page")
    return f"{entry['source']}_page_{donor}" if donor else str(entry["source"])


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
        "source_geburtsort": source_label(payload, "geburtsort"),
        "source_ausstellende_behoerde": source_label(payload, "ausstellende_behoerde"),
        "name_cross_check_state": payload["cross_checks"].get("name_cross_check_state"),
        "repair_ambiguous": payload.get("repair_ambiguous"),
        "field_conflicts": "; ".join(f"{item['field']}:{item['other']}@p{item['other_page']}" for item in payload.get("field_conflicts") or []),
        "needs_human_review": payload.get("needs_human_review"),
        "status": payload.get("status"),
        "failure_stage": payload.get("failure_stage"),
        "mrz_ocr_text": " | ".join(str(payload.get("mrz_ocr_text") or "").splitlines()),
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
    # Resolve page roles and merge fields first, then serialise. Writing JSON inside the
    # loop meant the two outputs were built from different states of the same records.
    payloads = []
    for key, record in sorted(records.items()):
        payload = build_payload(record)
        payload["_suffix"] = str(record.get("image_sha256") or "")[:8]
        payload["_key"] = key
        payloads.append(payload)
    resolve_page_roles(payloads)
    for payload in payloads:
        mid = payload.get("masterindex_id") or payload["_key"][0]
        page_number = payload.get("page_number") or payload["_key"][2]
        suffix = payload.pop("_suffix")
        payload.pop("_key")
        out_path = args.output_dir / (f"{mid}_page_{page_number}_{suffix}.json" if suffix else f"{mid}_page_{page_number}.json")
        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    write_flat_csv(args.csv_output, [build_csv_row(payload) for payload in payloads])
    print(f"Wrote document JSON files: {args.output_dir}")
    print(f"Wrote flat document CSV: {args.csv_output}")


if __name__ == "__main__":
    main()
