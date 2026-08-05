from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from pipeline_utils import build_ausweis_segments, utc_now_iso


REVIEW_PATH = Path(
    "outputs/ausweis_detection_review.xlsx"
)

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
    """Return a stripped lowercase string."""
    if value is None or pd.isna(value):
        return ""

    return str(value).strip().lower()


def determine_ground_truth_label(
    row: pd.Series,
) -> tuple[str | None, str | None]:
    """
    Return the human-verified Ground Truth label.

    accepted:
        The reviewer confirmed the model prediction.

    corrected:
        The reviewer replaced the model prediction.

    unreadable / out_of_scope:
        The row is excluded from classification Ground Truth.
    """
    review_status = clean_text(
        row.get("review_status")
    )

    predicted_label = clean_text(
        row.get("predicted_label")
    )

    reviewed_label = clean_text(
        row.get("reviewed_label")
    )

    if review_status == "accepted":
        if predicted_label not in ALLOWED_LABELS:
            return None, "invalid_predicted_label"

        return predicted_label, None

    if review_status == "corrected":
        if reviewed_label not in ALLOWED_LABELS:
            return (
                None,
                "missing_or_invalid_reviewed_label",
            )

        return reviewed_label, None

    if review_status in {
        "unreadable",
        "out_of_scope",
    }:
        return None, review_status

    return None, "incomplete_review"


def optional_value(
    row: pd.Series,
    column_name: str,
) -> Any:
    """Return a value when the column exists."""
    if column_name not in row.index:
        return None

    value = row.get(column_name)

    if pd.isna(value):
        return None

    return value


def optional_integer(
    row: pd.Series,
    column_name: str,
) -> int | None:
    """Return an optional integer value."""
    value = optional_value(
        row,
        column_name,
    )

    if value is None:
        return None

    return int(value)


def write_nested_ground_truth(
    page_ground_truth: pd.DataFrame,
    output_path: Path,
) -> None:
    """Write reviewed page labels grouped by MasterIndexID."""
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as output_file:

        for masterindex_id, group in (
            page_ground_truth.groupby(
                "masterindex_id",
                sort=True,
            )
        ):
            detailed_pages = [
                {
                    "page": int(
                        row["page_number"]
                    ),
                    "label": row[
                        "is_ausweiskopie_gt"
                    ],
                    "pdf_path_in_zip": row[
                        "pdf_path_in_zip"
                    ],
                }
                for _, row in group.sort_values(
                    [
                        "pdf_path_in_zip",
                        "page_number",
                    ]
                ).iterrows()
            ]

            result = {
                "masterindex_id": str(
                    masterindex_id
                ),
                "ground_truth_scope": (
                    "human_reviewed_pages_only"
                ),
                "reviewed_page_count": len(
                    detailed_pages
                ),
                "pages": [
                    {
                        "page": page["page"],
                        "label": page["label"],
                        "pdf_path_in_zip": page[
                            "pdf_path_in_zip"
                        ],
                    }
                    for page in detailed_pages
                ],
                "ausweis_segments": (
                    build_ausweis_segments(
                        detailed_pages
                    )
                ),
            }

            output_file.write(
                json.dumps(
                    result,
                    ensure_ascii=False,
                )
                + "\n"
            )


def main() -> None:
    if not REVIEW_PATH.exists():
        raise FileNotFoundError(
            f"Review workbook not found: "
            f"{REVIEW_PATH}. "
            "Run "
            "04_prepare_detection_review.py "
            "first."
        )

    review = pd.read_excel(
        REVIEW_PATH,
        sheet_name="review",
        dtype=object,
    )

    required_columns = {
        "masterindex_id",
        "pdf_path_in_zip",
        "page_number",
        "predicted_label",
        "review_status",
        "reviewed_label",
    }

    missing_columns = (
        required_columns
        - set(review.columns)
    )

    if missing_columns:
        raise ValueError(
            "Missing required review columns: "
            + ", ".join(
                sorted(missing_columns)
            )
        )

    processed_at = utc_now_iso()

    ground_truth_rows: list[
        dict[str, Any]
    ] = []

    excluded_rows: list[
        dict[str, Any]
    ] = []

    for _, row in review.iterrows():
        (
            ground_truth_label,
            exclusion_reason,
        ) = determine_ground_truth_label(row)

        if ground_truth_label is None:
            excluded_row = row.to_dict()

            excluded_row[
                "exclusion_reason"
            ] = exclusion_reason

            excluded_row[
                "review_processed_at_utc"
            ] = processed_at

            excluded_rows.append(
                excluded_row
            )

            continue

        ground_truth_rows.append({
            "masterindex_id": str(
                row["masterindex_id"]
            ),
            "pdf_path_in_zip": str(
                row["pdf_path_in_zip"]
            ),
            "pdf_order_in_masterindex": (
                optional_integer(
                    row,
                    "pdf_order_in_masterindex",
                )
            ),
            "page_number": int(
                row["page_number"]
            ),
            "source_page_number": (
                optional_integer(
                    row,
                    "source_page_number",
                )
            ),
            "image_path": optional_value(
                row,
                "image_path",
            ),
            "image_sha256": optional_value(
                row,
                "image_sha256",
            ),
            "predicted_label": clean_text(
                row.get("predicted_label")
            ),
            "prediction_confidence": (
                optional_value(
                    row,
                    "prediction_confidence",
                )
                if "prediction_confidence"
                in review.columns
                else optional_value(
                    row,
                    "confidence",
                )
            ),
            # The review workbook column is named "status".
            "screening_status": optional_value(
                row,
                "status",
            ),
            "is_ausweiskopie_gt": (
                ground_truth_label
            ),
            "review_reason": optional_value(
                row,
                "review_reason",
            ),
            "review_status": clean_text(
                row.get("review_status")
            ),
            "reviewer": optional_value(
                row,
                "reviewer",
            ),
            "review_comment": optional_value(
                row,
                "review_comment",
            ),
            "review_processed_at_utc": (
                processed_at
            ),
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
        "prediction_confidence",
        "screening_status",
        "is_ausweiskopie_gt",
        "review_reason",
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

    PAGE_GT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    page_ground_truth.to_csv(
        PAGE_GT_PATH,
        index=False,
    )

    excluded.to_csv(
        AUDIT_PATH,
        index=False,
    )

    if page_ground_truth.empty:
        NESTED_GT_PATH.write_text(
            "",
            encoding="utf-8",
        )
    else:
        write_nested_ground_truth(
            page_ground_truth=(
                page_ground_truth
            ),
            output_path=NESTED_GT_PATH,
        )

    print(
        "Human-reviewed Ground Truth pages: "
        f"{len(page_ground_truth)}"
    )

    print(
        "Excluded or incomplete review rows: "
        f"{len(excluded)}"
    )

    print(
        f"Page Ground Truth written to: "
        f"{PAGE_GT_PATH}"
    )

    print(
        f"Nested Ground Truth written to: "
        f"{NESTED_GT_PATH}"
    )

    print(
        f"Excluded rows written to: "
        f"{AUDIT_PATH}"
    )


if __name__ == "__main__":
    main()