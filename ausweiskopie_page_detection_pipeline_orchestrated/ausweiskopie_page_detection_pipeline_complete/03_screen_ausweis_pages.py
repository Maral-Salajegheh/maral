from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from pipeline_utils import (
    append_jsonl,
    page_record_key,
    read_latest_jsonl,
    utc_now_iso,
)
from securegpt_vision import (
    MODEL_NAME,
    SEED,
    TEMPERATURE,
    SecureGPTScreeningError,
    create_securegpt_client,
    screen_image,
)


PAGE_INVENTORY_PATH = Path("outputs/page_inventory.csv")

PAGE_RESULTS_PATH = Path("outputs/ausweis_page_screening.jsonl")

HUMAN_REVIEW_THRESHOLD = 0.60


def inventory_page_records(
    page_inventory: pd.DataFrame,
) -> list[dict[str, Any]]:
    """
    Convert usable inventory rows to page records.

    source_page_number and image_sha256 must be carried along:
    page_record_key builds the resume/dedup key from them, and
    04_prepare_detection_review.py matches results on the same key.
    """
    records: list[dict[str, Any]] = []

    for _, row in page_inventory.iterrows():
        if pd.isna(row.get("page_number")):
            continue

        records.append({
            "masterindex_id": str(row["masterindex_id"]),
            "pdf_path_in_zip": str(row["pdf_path_in_zip"]),
            "page_number": int(float(row["page_number"])),
            "source_page_number": int(float(row["source_page_number"])),
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


def needs_human_review(
    label: str,
    confidence: float | None,
    status: str,
) -> bool:
    """Route only failed or strongly uncertain pages to a human."""
    if status != "success":
        return True

    if label == "unclear":
        return True

    if confidence is None:
        return True

    return confidence < HUMAN_REVIEW_THRESHOLD


def main() -> None:
    if not PAGE_INVENTORY_PATH.exists():
        raise FileNotFoundError(
            f"Page inventory not found: {PAGE_INVENTORY_PATH}"
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

    for page in sorted(
        pages,
        key=lambda item: (
            item["masterindex_id"],
            item["page_number"],
        ),
    ):
        key = page_record_key(page)
        existing = latest_results.get(key)

        # Successful results are not sent again.
        if existing and existing.get("status") == "success":
            continue

        if page["render_status"] != "success":
            record = {
                **page,
                "label": "unclear",
                "evidence": "technical_render_failure",
                "confidence": None,
                "needs_human_review": True,
                "model_id": MODEL_NAME,
                "temperature": TEMPERATURE,
                "seed": SEED,
                "attempt_count": 0,
                "retry_count": 0,
                "processed_at_utc": utc_now_iso(),
                "status": "render_failed",
                "error": "Page rendering failed.",
            }

        elif page["quality_status"] != "usable":
            record = {
                **page,
                "label": "unclear",
                "evidence": "technical_quality_issue",
                "confidence": None,
                "needs_human_review": True,
                "model_id": MODEL_NAME,
                "temperature": TEMPERATURE,
                "seed": SEED,
                "attempt_count": 0,
                "retry_count": 0,
                "processed_at_utc": utc_now_iso(),
                "status": "quality_rejected",
                "error": (
                    "Image quality status: "
                    f"{page['quality_status']}"
                ),
            }

        else:
            try:
                result = screen_image(
                    client=client,
                    image_path=Path(str(page["image_path"])),
                )

                label = str(result["label"])
                confidence = float(result["confidence"])

                record = {
                    **page,
                    "label": label,
                    "evidence": str(result["evidence_code"]),
                    "confidence": confidence,
                    "needs_human_review": needs_human_review(
                        label=label,
                        confidence=confidence,
                        status="success",
                    ),
                    "model_id": MODEL_NAME,
                    "temperature": TEMPERATURE,
                    "seed": SEED,
                    "attempt_count": int(result["attempt_count"]),
                    "retry_count": int(result["retry_count"]),
                    "processed_at_utc": utc_now_iso(),
                    "status": "success",
                    "error": None,
                }

            except SecureGPTScreeningError as error:
                record = {
                    **page,
                    "label": "unclear",
                    "evidence": "technical_securegpt_failure",
                    "confidence": None,
                    "needs_human_review": True,
                    "model_id": MODEL_NAME,
                    "temperature": TEMPERATURE,
                    "seed": SEED,
                    "attempt_count": error.attempts,
                    "retry_count": max(error.attempts - 1, 0),
                    "processed_at_utc": utc_now_iso(),
                    "status": "failed",
                    "error": str(error),
                }

            except Exception as error:
                record = {
                    **page,
                    "label": "unclear",
                    "evidence": "unexpected_screening_failure",
                    "confidence": None,
                    "needs_human_review": True,
                    "model_id": MODEL_NAME,
                    "temperature": TEMPERATURE,
                    "seed": SEED,
                    "attempt_count": 1,
                    "retry_count": 0,
                    "processed_at_utc": utc_now_iso(),
                    "status": "failed",
                    "error": str(error),
                }

        append_jsonl(PAGE_RESULTS_PATH, record)

        latest_results[key] = record

        print(
            f"Screened {record['masterindex_id']} "
            f"page {record['page_number']}: "
            f"{record['label']} "
            f"confidence={record['confidence']} "
            f"status={record['status']}"
        )

    print(f"Results written to: {PAGE_RESULTS_PATH}")


if __name__ == "__main__":
    main()