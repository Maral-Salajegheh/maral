from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from pipeline_utils import (
    append_jsonl,
    build_ausweis_segments,
    page_record_key,
    read_latest_jsonl,
    utc_now_iso,
)
from securegpt_vision import (
    MODEL_NAME,
    MODEL_VERSION,
    SEED,
    TEMPERATURE,
    create_securegpt_client,
    screen_image,
    validate_configuration,
)


PAGE_INVENTORY_PATH = Path("outputs/page_inventory.csv")
PAGE_RESULTS_PATH = Path("outputs/ausweis_page_screening.jsonl")
MASTERINDEX_RESULTS_PATH = Path(
    "outputs/ausweis_screening_predictions.jsonl"
)

ALLOWED_LABELS = {
    "ausweiskopie",
    "not_ausweiskopie",
    "unclear",
}

ALLOWED_EVIDENCE_CODES = {
    "id_card_layout",
    "passport_layout",
    "portrait_and_id_layout",
    "mrz_like_area",
    "multiple_id_sides",
    "no_id_features",
    "unreadable_or_ambiguous",
    "technical_render_failure",
    "technical_quality_issue",
}

PROMPT_VERSION = "ausweiskopie_page_screening_v3"

PAGE_LIMIT: int | None = None


def validate_screening_result(
    model_result: dict[str, Any],
) -> dict[str, str]:
    """Validate the model label and PII-safe evidence code."""
    label = model_result.get("label")
    evidence_code = model_result.get("evidence_code")

    if label not in ALLOWED_LABELS:
        raise ValueError(
            f"Invalid screening label returned: {label!r}"
        )

    if evidence_code not in ALLOWED_EVIDENCE_CODES:
        raise ValueError(
            f"Invalid evidence code returned: {evidence_code!r}"
        )

    return {
        "label": str(label),
        "evidence": str(evidence_code),
    }


def inventory_page_records(
    page_inventory: pd.DataFrame,
) -> list[dict[str, Any]]:
    """Convert page rows with known page numbers to dictionaries."""
    records: list[dict[str, Any]] = []

    for _, row in page_inventory.iterrows():
        if pd.isna(row.get("page_number")):
            continue

        records.append({
            "masterindex_id": str(row["masterindex_id"]),
            "pdf_path_in_zip": str(row["pdf_path_in_zip"]),
            "pdf_order_in_masterindex": int(
                row["pdf_order_in_masterindex"]
            ),
            "page_number": int(row["page_number"]),
            "source_page_number": int(row["source_page_number"]),
            "image_path": (
                None
                if pd.isna(row.get("image_path"))
                else str(row["image_path"])
            ),
            "image_sha256": (
                None
                if pd.isna(row.get("image_sha256"))
                else str(row["image_sha256"])
            ),
            "render_status": str(row["render_status"]),
            "quality_status": str(row["quality_status"]),
        })

    return records


def technical_result(
    page: dict[str, Any],
    *,
    status: str,
    evidence: str,
    error: str | None,
) -> dict[str, Any]:
    """Create an unclear result without a SecureGPT call."""
    return {
        **page,
        "label": "unclear",
        "evidence": evidence,
        "prompt_version": PROMPT_VERSION,
        "model_name": MODEL_NAME,
        "model_version": MODEL_VERSION,
        "temperature": TEMPERATURE,
        "seed": SEED,
        "processed_at_utc": utc_now_iso(),
        "status": status,
        "error": error,
    }


def write_masterindex_results(
    page_inventory: pd.DataFrame,
    current_results: dict[tuple[Any, ...], dict[str, Any]],
    output_path: Path,
) -> None:
    """Write nested results with explicit completeness metadata."""
    inventory_records = inventory_page_records(page_inventory)

    by_masterindex: dict[str, list[dict[str, Any]]] = {}
    for page in inventory_records:
        by_masterindex.setdefault(
            page["masterindex_id"],
            [],
        ).append(page)

    failed_pdf_counts = (
        page_inventory[
            page_inventory["page_number"].isna()
        ]
        .groupby("masterindex_id")
        .size()
        .to_dict()
    )

    masterindex_ids = sorted(
        set(page_inventory["masterindex_id"].astype(str))
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as output_file:
        for masterindex_id in masterindex_ids:
            expected_pages = sorted(
                by_masterindex.get(masterindex_id, []),
                key=lambda item: item["page_number"],
            )

            detailed_pages: list[dict[str, Any]] = []
            successful_count = 0
            technical_failure_count = 0

            for page in expected_pages:
                result = current_results.get(
                    page_record_key(page)
                )
                if result is None:
                    continue

                detailed_pages.append({
                    "page": page["page_number"],
                    "label": result["label"],
                    "pdf_path_in_zip": page["pdf_path_in_zip"],
                })

                if result.get("status") == "success":
                    successful_count += 1
                else:
                    technical_failure_count += 1

            expected_count = len(expected_pages)
            processed_count = len(detailed_pages)
            failed_pdf_count = int(
                failed_pdf_counts.get(masterindex_id, 0)
            )

            nested_result = {
                "masterindex_id": masterindex_id,
                "expected_page_count": expected_count,
                "processed_page_count": processed_count,
                "successful_screening_count": successful_count,
                "technical_failure_count": technical_failure_count,
                "failed_pdf_count": failed_pdf_count,
                "is_complete": (
                    processed_count == expected_count
                    and failed_pdf_count == 0
                ),
                "pages": [
                    {
                        "page": page["page"],
                        "label": page["label"],
                    }
                    for page in detailed_pages
                ],
                "ausweis_segments": build_ausweis_segments(
                    detailed_pages
                ),
            }

            output_file.write(
                json.dumps(nested_result, ensure_ascii=False)
                + "\n"
            )


def main() -> None:
    validate_configuration()

    if not PAGE_INVENTORY_PATH.exists():
        raise FileNotFoundError(
            f"Page inventory not found: {PAGE_INVENTORY_PATH}. "
            "Run 02_render_pdf_pages.py first."
        )

    page_inventory = pd.read_csv(
        PAGE_INVENTORY_PATH,
        dtype=object,
    )
    pages = inventory_page_records(page_inventory)

    latest_results = read_latest_jsonl(
        PAGE_RESULTS_PATH,
        page_record_key,
    )

    client = create_securegpt_client()

    attempted = 0
    successful = 0
    technical_unclear = 0
    failed_calls = 0

    for page in sorted(
        pages,
        key=lambda item: (
            item["masterindex_id"],
            item["page_number"],
        ),
    ):
        key = page_record_key(page)
        existing = latest_results.get(key)

        terminal_statuses = {
            "success",
            "render_failed",
            "quality_rejected",
        }
        if existing and existing.get("status") in terminal_statuses:
            continue

        if PAGE_LIMIT is not None and attempted >= PAGE_LIMIT:
            break

        if page["render_status"] != "success":
            record = technical_result(
                page,
                status="render_failed",
                evidence="technical_render_failure",
                error="Page rendering failed.",
            )
            technical_unclear += 1

        elif page["quality_status"] != "usable":
            record = technical_result(
                page,
                status="quality_rejected",
                evidence="technical_quality_issue",
                error=(
                    "Image quality status: "
                    f"{page['quality_status']}"
                ),
            )
            technical_unclear += 1

        else:
            try:
                model_result = screen_image(
                    client=client,
                    image_path=Path(str(page["image_path"])),
                )
                validated = validate_screening_result(
                    model_result
                )

                record = {
                    **page,
                    **validated,
                    "prompt_version": PROMPT_VERSION,
                    "model_name": MODEL_NAME,
                    "model_version": MODEL_VERSION,
                    "temperature": TEMPERATURE,
                    "seed": SEED,
                    "processed_at_utc": utc_now_iso(),
                    "status": "success",
                    "error": None,
                }
                successful += 1

            except Exception as error:
                record = technical_result(
                    page,
                    status="failed",
                    evidence="unreadable_or_ambiguous",
                    error=str(error),
                )
                failed_calls += 1

        append_jsonl(PAGE_RESULTS_PATH, record)
        latest_results[key] = record
        attempted += 1

        print(
            f"Screened {record['masterindex_id']} "
            f"page {record['page_number']}: "
            f"{record['label']} ({record['status']})"
        )

    current_results = {
        page_record_key(page): latest_results[
            page_record_key(page)
        ]
        for page in pages
        if page_record_key(page) in latest_results
    }

    write_masterindex_results(
        page_inventory=page_inventory,
        current_results=current_results,
        output_path=MASTERINDEX_RESULTS_PATH,
    )

    print(f"New page attempts: {attempted}")
    print(f"Successful SecureGPT calls: {successful}")
    print(f"Technical pages marked unclear: {technical_unclear}")
    print(f"Failed SecureGPT calls marked unclear: {failed_calls}")
    print(f"Page checkpoint: {PAGE_RESULTS_PATH}")
    print(f"Nested predictions: {MASTERINDEX_RESULTS_PATH}")


if __name__ == "__main__":
    main()
