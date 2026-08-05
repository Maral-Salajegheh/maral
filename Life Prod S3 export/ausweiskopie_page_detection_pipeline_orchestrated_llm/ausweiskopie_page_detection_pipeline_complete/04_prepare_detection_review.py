from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from pipeline_utils import page_record_key, read_latest_jsonl


PAGE_INVENTORY_PATH = Path(
    "outputs/page_inventory.csv"
)

PAGE_RESULTS_PATH = Path(
    "outputs/ausweis_page_screening.jsonl"
)

OUTPUT_PATH = Path(
    "outputs/ausweis_detection_review.xlsx"
)


# Small audit samples are required so Ground Truth does not contain
# only difficult or uncertain pages.
#
# The audit sample is a percentage of auto-accepted pages, but with
# an absolute floor and an absolute cap. Statistical confidence
# depends on the absolute number of audited pages, not on the
# fraction: about 400 pages per label bounds the precision estimate
# to roughly +/- 2 percentage points regardless of corpus size, so
# sampling more than that at 200k+ files would multiply human
# effort without improving the estimate.
AUDIT_FRACTION = 0.05
AUDIT_MIN_PAGES_PER_LABEL = 20
AUDIT_MAX_PAGES_PER_LABEL = 400
RANDOM_SEED = 42


def read_current_screening_results() -> pd.DataFrame:
    """
    Return screening results that match the currently rendered pages.
    """
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
            "masterindex_id": str(
                row["masterindex_id"]
            ),
            "pdf_path_in_zip": str(
                row["pdf_path_in_zip"]
            ),
            "source_page_number": int(
                float(row["source_page_number"])
            ),
            "image_sha256": (
                None
                if pd.isna(
                    row.get("image_sha256")
                )
                else str(
                    row["image_sha256"]
                )
            ),
        }

        result = latest_results.get(
            page_record_key(page)
        )

        if result is not None:
            records.append(result)

    return pd.DataFrame(records)


def to_boolean(value: Any) -> bool:
    """
    Convert common boolean representations to True or False.
    """
    if isinstance(value, bool):
        return value

    if value is None or pd.isna(value):
        return False

    return str(value).strip().lower() in {
        "true",
        "1",
        "yes",
    }


def choose_audit_sample(
    pages: pd.DataFrame,
) -> pd.DataFrame:
    """
    Select a small random sample from auto-accepted pages.
    """
    if pages.empty:
        return pages.copy()

    fraction_size = round(
        len(pages) * AUDIT_FRACTION
    )

    sample_size = max(
        AUDIT_MIN_PAGES_PER_LABEL,
        fraction_size,
    )

    sample_size = min(
        sample_size,
        AUDIT_MAX_PAGES_PER_LABEL,
        len(pages),
    )

    return pages.sample(
        n=sample_size,
        random_state=RANDOM_SEED,
    ).copy()


def add_sampling_information(
    sample: pd.DataFrame,
    pool_size: int,
) -> pd.DataFrame:
    """
    Add the probability that a page was selected for audit.
    """
    sample = sample.copy()

    sampling_probability = (
        len(sample) / pool_size
        if pool_size
        else 0.0
    )

    sample["sampling_probability"] = (
        sampling_probability
    )

    sample["sampling_weight"] = (
        1.0 / sampling_probability
        if sampling_probability
        else None
    )

    return sample


def main() -> None:
    if not PAGE_INVENTORY_PATH.exists():
        raise FileNotFoundError(
            f"Page inventory not found: "
            f"{PAGE_INVENTORY_PATH}."
        )

    if not PAGE_RESULTS_PATH.exists():
        raise FileNotFoundError(
            f"Screening results not found: "
            f"{PAGE_RESULTS_PATH}. "
            "Run 03_screen_ausweis_pages.py first."
        )

    results = read_current_screening_results()

    if results.empty:
        raise ValueError(
            "No current page screening results were found."
        )

    required_columns = {
        "masterindex_id",
        "pdf_path_in_zip",
        "page_number",
        "label",
        "status",
    }

    missing_columns = (
        required_columns
        - set(results.columns)
    )

    if missing_columns:
        raise ValueError(
            "Missing required screening columns: "
            + ", ".join(
                sorted(missing_columns)
            )
        )

    results["label"] = (
        results["label"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    results["status"] = (
        results["status"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    if "needs_human_review" in results.columns:
        needs_review = results[
            "needs_human_review"
        ].apply(to_boolean)
    else:
        needs_review = (
            (results["status"] != "success")
            | (results["label"] == "unclear")
        )

    # These pages must be reviewed because the model was uncertain
    # or the screening call failed.
    mandatory_review = results[
        needs_review
    ].copy()

    mandatory_review[
        "review_reason"
    ] = mandatory_review.apply(
        lambda row: (
            "screening_failed"
            if row["status"] != "success"
            else (
                "unclear"
                if row["label"] == "unclear"
                else "low_confidence"
            )
        ),
        axis=1,
    )

    mandatory_review[
        "sampling_probability"
    ] = 1.0

    mandatory_review[
        "sampling_weight"
    ] = 1.0

    # Pages not requiring mandatory review were auto-accepted.
    auto_accepted = results[
        ~needs_review
    ].copy()

    auto_positive = auto_accepted[
        auto_accepted["label"]
        == "ausweiskopie"
    ].copy()

    auto_negative = auto_accepted[
        auto_accepted["label"]
        == "not_ausweiskopie"
    ].copy()

    positive_audit = choose_audit_sample(
        auto_positive
    )

    positive_audit[
        "review_reason"
    ] = "positive_audit"

    positive_audit = add_sampling_information(
        positive_audit,
        pool_size=len(auto_positive),
    )

    negative_audit = choose_audit_sample(
        auto_negative
    )

    negative_audit[
        "review_reason"
    ] = "negative_audit"

    negative_audit = add_sampling_information(
        negative_audit,
        pool_size=len(auto_negative),
    )

    review = pd.concat(
        [
            mandatory_review,
            positive_audit,
            negative_audit,
        ],
        ignore_index=True,
    )

    review = review.rename(
        columns={
            "label": "predicted_label",
        }
    )

    review["reviewed_label"] = None
    review["review_status"] = None
    review["reviewer"] = None
    review["review_comment"] = None

    # Only columns that actually exist in the screening records.
    preferred_columns = [
        "masterindex_id",
        "pdf_path_in_zip",
        "page_number",
        "source_page_number",
        "image_path",
        "image_sha256",
        "predicted_label",
        "confidence",
        "evidence",
        "status",
        "error",
        "needs_human_review",
        "attempt_count",
        "retry_count",
        "review_reason",
        "sampling_probability",
        "sampling_weight",
        "model_id",
        "temperature",
        "seed",
        "reviewed_label",
        "review_status",
        "reviewer",
        "review_comment",
    ]

    review = review.reindex(
        columns=preferred_columns
    )

    sort_columns = [
        column
        for column in [
            "masterindex_id",
            "pdf_path_in_zip",
            "page_number",
        ]
        if column in review.columns
    ]

    review = review.sort_values(
        sort_columns,
        na_position="last",
    ).reset_index(drop=True)

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    review.to_excel(
        OUTPUT_PATH,
        index=False,
        sheet_name="review",
    )

    print(
        "Mandatory review pages: "
        f"{len(mandatory_review)}"
    )

    print(
        "Auto-accepted positive pages: "
        f"{len(auto_positive)}"
    )

    print(
        "Positive audit sample: "
        f"{len(positive_audit)}"
    )

    print(
        "Auto-accepted negative pages: "
        f"{len(auto_negative)}"
    )

    print(
        "Negative audit sample: "
        f"{len(negative_audit)}"
    )

    print(
        "Total human review rows: "
        f"{len(review)}"
    )

    print(
        f"Review workbook written to: "
        f"{OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()