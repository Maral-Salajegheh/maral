from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from pipeline_utils import build_ausweis_segments, utc_now_iso


REVIEW_PATH = Path("outputs/ausweis_detection_review.xlsx")
PAGE_GT_PATH = Path(
    "outputs/ausweis_classification_ground_truth_pages.csv"
)
NESTED_GT_PATH = Path(
    "outputs/ausweis_classification_ground_truth.jsonl"
)
AUDIT_PATH = Path(
    "outputs/ausweis_classification_excluded_rows.csv"
)

ALLOWED_LABELS = {
    "ausweiskopie",
    "not_ausweiskopie",
    "unclear",
}


def clean_text(value: Any) -> str:
    """Return a stripped string or an empty string."""
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def determine_ground_truth_label(
    row: pd.Series,
) -> tuple[str | None, str | None]:
    """Return the reviewed label or an exclusion reason."""
    review_status = clean_text(row.get("review_status")).lower()
    predicted_label = clean_text(
        row.get("predicted_label")
    ).lower()
    reviewed_label = clean_text(
        row.get("reviewed_label")
    ).lower()

    if review_status == "accepted":
        if predicted_label not in ALLOWED_LABELS:
            return None, "invalid_predicted_label"
        return predicted_label, None

    if review_status == "corrected":
        if reviewed_label not in ALLOWED_LABELS:
            return None, "missing_or_invalid_reviewed_label"
        return reviewed_label, None

    if review_status in {"unreadable", "out_of_scope"}:
        return None, review_status

    return None, "incomplete_review"


def write_nested_ground_truth(
    page_ground_truth: pd.DataFrame,
    output_path: Path,
) -> None:
    """Group reviewed labels without crossing PDF boundaries."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as output_file:
        for masterindex_id, group in page_ground_truth.groupby(
            "masterindex_id",
            sort=True,
        ):
            detailed_pages = [
                {
                    "page": int(row["page_number"]),
                    "label": row["is_ausweiskopie_gt"],
                    "pdf_path_in_zip": row["pdf_path_in_zip"],
                }
                for _, row in group.sort_values(
                    "page_number"
                ).iterrows()
            ]

            result = {
                "masterindex_id": str(masterindex_id),
                "ground_truth_scope": "reviewed_pages_only",
                "reviewed_page_count": len(detailed_pages),
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
                json.dumps(result, ensure_ascii=False) + "\n"
            )


def main() -> None:
    if not REVIEW_PATH.exists():
        raise FileNotFoundError(
            f"Review workbook not found: {REVIEW_PATH}. "
            "Run 04_prepare_detection_review.py first."
        )

    review = pd.read_excel(
        REVIEW_PATH,
        sheet_name="review",
        dtype=object,
    )

    processed_at = utc_now_iso()
    ground_truth_rows: list[dict[str, Any]] = []
    excluded_rows: list[dict[str, Any]] = []

    for _, row in review.iterrows():
        gt_label, exclusion_reason = determine_ground_truth_label(row)

        if gt_label is None:
            excluded_row = row.to_dict()
            excluded_row["exclusion_reason"] = exclusion_reason
            excluded_row["review_processed_at_utc"] = processed_at
            excluded_rows.append(excluded_row)
            continue

        ground_truth_rows.append({
            "masterindex_id": row.get("masterindex_id"),
            "pdf_path_in_zip": row.get("pdf_path_in_zip"),
            "pdf_order_in_masterindex": row.get(
                "pdf_order_in_masterindex"
            ),
            "page_number": int(row["page_number"]),
            "source_page_number": int(
                row["source_page_number"]
            ),
            "image_path": row.get("image_path"),
            "image_sha256": row.get("image_sha256"),
            "predicted_label": row.get("predicted_label"),
            "is_ausweiskopie_gt": gt_label,
            "review_reason": row.get("review_reason"),
            "sampling_probability": row.get(
                "sampling_probability"
            ),
            "sampling_weight": row.get("sampling_weight"),
            "review_status": clean_text(
                row.get("review_status")
            ).lower(),
            "reviewer": row.get("reviewer"),
            "review_comment": row.get("review_comment"),
            "review_processed_at_utc": processed_at,
        })

    ground_truth_columns = [
        "masterindex_id",
        "pdf_path_in_zip",
        "pdf_order_in_masterindex",
        "page_number",
        "source_page_number",
        "image_path",
        "image_sha256",
        "predicted_label",
        "is_ausweiskopie_gt",
        "review_reason",
        "sampling_probability",
        "sampling_weight",
        "review_status",
        "reviewer",
        "review_comment",
        "review_processed_at_utc",
    ]
    excluded_columns = [
        *review.columns.tolist(),
        "exclusion_reason",
        "review_processed_at_utc",
    ]

    page_ground_truth = pd.DataFrame(
        ground_truth_rows,
        columns=ground_truth_columns,
    )
    excluded = pd.DataFrame(
        excluded_rows,
        columns=excluded_columns,
    )

    PAGE_GT_PATH.parent.mkdir(parents=True, exist_ok=True)
    page_ground_truth.to_csv(PAGE_GT_PATH, index=False)
    excluded.to_csv(AUDIT_PATH, index=False)

    if not page_ground_truth.empty:
        write_nested_ground_truth(
            page_ground_truth=page_ground_truth,
            output_path=NESTED_GT_PATH,
        )
    else:
        NESTED_GT_PATH.write_text("", encoding="utf-8")

    print(f"Reviewed Ground Truth pages: {len(page_ground_truth)}")
    print(f"Excluded or incomplete rows: {len(excluded)}")
    print(f"Page Ground Truth written to: {PAGE_GT_PATH}")
    print(f"Nested Ground Truth written to: {NESTED_GT_PATH}")
    print(f"Excluded rows written to: {AUDIT_PATH}")


if __name__ == "__main__":
    main()
