from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from pipeline_utils import page_record_key, read_latest_jsonl


PAGE_INVENTORY_PATH = Path("outputs/page_inventory.csv")
PAGE_RESULTS_PATH = Path("outputs/ausweis_page_screening.jsonl")
OUTPUT_PATH = Path("outputs/ausweis_detection_review.xlsx")

NEGATIVE_AUDIT_FRACTION = 0.05
NEGATIVE_AUDIT_MIN_PAGES = 50
RANDOM_SEED = 42


def read_current_screening_results() -> pd.DataFrame:
    """Return only records matching the current rendered page images."""
    page_inventory = pd.read_csv(
        PAGE_INVENTORY_PATH,
        dtype=object,
    )
    latest_results = read_latest_jsonl(
        PAGE_RESULTS_PATH,
        page_record_key,
    )

    records: list[dict[str, Any]] = []

    for _, row in page_inventory.iterrows():
        if pd.isna(row.get("page_number")):
            continue

        page = {
            "masterindex_id": str(row["masterindex_id"]),
            "pdf_path_in_zip": str(row["pdf_path_in_zip"]),
            "source_page_number": int(row["source_page_number"]),
            "image_sha256": (
                None
                if pd.isna(row.get("image_sha256"))
                else str(row["image_sha256"])
            ),
        }

        result = latest_results.get(page_record_key(page))
        if result is not None:
            records.append(result)

    return pd.DataFrame(records)


def choose_negative_audit_sample(
    negative_pages: pd.DataFrame,
) -> pd.DataFrame:
    """
    Draw a global random page sample.

    The target metric is the corpus-level page miss rate, so a simple global
    page sample is more appropriate than an unweighted per-folder sample.
    """
    if negative_pages.empty:
        return negative_pages.copy()

    fraction_size = round(
        len(negative_pages) * NEGATIVE_AUDIT_FRACTION
    )
    sample_size = max(
        NEGATIVE_AUDIT_MIN_PAGES,
        fraction_size,
    )
    sample_size = min(sample_size, len(negative_pages))

    return negative_pages.sample(
        n=sample_size,
        random_state=RANDOM_SEED,
    )


def main() -> None:
    if not PAGE_INVENTORY_PATH.exists():
        raise FileNotFoundError(
            f"Page inventory not found: {PAGE_INVENTORY_PATH}."
        )

    if not PAGE_RESULTS_PATH.exists():
        raise FileNotFoundError(
            f"Screening results not found: {PAGE_RESULTS_PATH}. "
            "Run 03_screen_ausweis_pages.py first."
        )

    results = read_current_screening_results()

    if results.empty:
        raise ValueError("No current page screening results were found.")

    positives = results[
        results["label"] == "ausweiskopie"
    ].copy()
    positives["review_reason"] = "predicted_ausweiskopie"
    positives["sampling_probability"] = 1.0

    unclear = results[
        results["label"] == "unclear"
    ].copy()
    unclear["review_reason"] = "unclear"
    unclear["sampling_probability"] = 1.0

    negatives = results[
        results["label"] == "not_ausweiskopie"
    ].copy()
    negative_audit = choose_negative_audit_sample(negatives)
    negative_audit["review_reason"] = "negative_audit"

    negative_probability = (
        len(negative_audit) / len(negatives)
        if len(negatives)
        else 0.0
    )
    negative_audit["sampling_probability"] = (
        negative_probability
    )

    review = pd.concat(
        [positives, unclear, negative_audit],
        ignore_index=True,
    )

    review["sampling_weight"] = review[
        "sampling_probability"
    ].apply(
        lambda value: 1.0 / value if value else None
    )

    review = review.sort_values(
        ["masterindex_id", "page_number"]
    ).reset_index(drop=True)

    review = review.rename(
        columns={"label": "predicted_label"}
    )

    review["reviewed_label"] = None
    review["review_status"] = None
    review["reviewer"] = None
    review["review_comment"] = None

    preferred_columns = [
        "masterindex_id",
        "pdf_path_in_zip",
        "pdf_order_in_masterindex",
        "page_number",
        "source_page_number",
        "image_path",
        "image_sha256",
        "predicted_label",
        "evidence",
        "status",
        "error",
        "review_reason",
        "sampling_probability",
        "sampling_weight",
        "prompt_version",
        "model_name",
        "model_version",
        "temperature",
        "seed",
        "reviewed_label",
        "review_status",
        "reviewer",
        "review_comment",
    ]

    review = review.reindex(columns=preferred_columns)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    review.to_excel(
        OUTPUT_PATH,
        index=False,
        sheet_name="review",
    )

    audited_masterindexes = (
        negative_audit["masterindex_id"].nunique()
        if not negative_audit.empty
        else 0
    )
    negative_masterindexes = (
        negatives["masterindex_id"].nunique()
        if not negatives.empty
        else 0
    )

    print(f"Predicted positive pages: {len(positives)}")
    print(f"Unclear pages: {len(unclear)}")
    print(f"Available predicted negatives: {len(negatives)}")
    print(f"Negative audit sample: {len(negative_audit)}")
    print(
        "Negative MasterIndex audit coverage: "
        f"{audited_masterindexes}/{negative_masterindexes}"
    )
    print(f"Total review rows: {len(review)}")
    print(f"Review workbook written to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
