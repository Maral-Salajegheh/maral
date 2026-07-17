from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pandas as pd
from pandas.errors import EmptyDataError

from pipeline_utils import page_record_key, read_latest_jsonl


PAGE_INVENTORY_PATH = Path("outputs/page_inventory.csv")
PAGE_RESULTS_PATH = Path("outputs/ausweis_page_screening.jsonl")
GROUND_TRUTH_PATH = Path(
    "outputs/ausweis_classification_ground_truth_pages.csv"
)
EXCLUDED_PATH = Path(
    "outputs/ausweis_classification_excluded_rows.csv"
)
OUTPUT_PATH = Path(
    "outputs/ausweis_detection_audit_summary.csv"
)
ERRORS_PATH = Path(
    "outputs/ausweis_detection_review_errors.csv"
)


def safe_divide(
    numerator: float,
    denominator: float,
) -> float:
    """Avoid division by zero."""
    return numerator / denominator if denominator else 0.0


def wilson_interval(
    positive_count: int,
    sample_size: int,
    z_value: float = 1.96,
) -> tuple[float, float]:
    """Return a 95% Wilson interval for a proportion."""
    if sample_size == 0:
        return 0.0, 0.0

    proportion = positive_count / sample_size
    z_squared = z_value ** 2
    denominator = 1 + z_squared / sample_size

    centre = (
        proportion
        + z_squared / (2 * sample_size)
    ) / denominator

    margin = (
        z_value
        * math.sqrt(
            (
                proportion * (1 - proportion)
                + z_squared / (4 * sample_size)
            )
            / sample_size
        )
        / denominator
    )

    return max(0.0, centre - margin), min(1.0, centre + margin)


def current_screening_records() -> pd.DataFrame:
    """Return only results matching current rendered images."""
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


def metric(
    name: str,
    value: float,
    numerator: float | None,
    denominator: float | None,
    scope: str,
    note: str,
) -> dict[str, Any]:
    """Build one explicit summary metric row."""
    return {
        "metric": name,
        "value": value,
        "numerator": numerator,
        "denominator": denominator,
        "scope": scope,
        "note": note,
    }


def main() -> None:
    for path in [
        PAGE_INVENTORY_PATH,
        PAGE_RESULTS_PATH,
        GROUND_TRUTH_PATH,
    ]:
        if not path.exists():
            raise FileNotFoundError(
                f"Required file not found: {path}"
            )

    screening = current_screening_records()
    ground_truth = pd.read_csv(
        GROUND_TRUTH_PATH,
        dtype=object,
        keep_default_na=False,
    )

    if screening.empty:
        raise ValueError("No current screening results were found.")

    if ground_truth.empty:
        raise ValueError("Ground Truth contains no reviewed rows.")

    predicted = ground_truth["predicted_label"].astype(str)
    actual = ground_truth["is_ausweiskopie_gt"].astype(str)
    review_reason = ground_truth["review_reason"].astype(str)

    reviewed_count = len(ground_truth)
    agreement_count = int((predicted == actual).sum())

    total_positive = int(
        (screening["label"] == "ausweiskopie").sum()
    )
    total_negative = int(
        (screening["label"] == "not_ausweiskopie").sum()
    )
    total_unclear = int(
        (screening["label"] == "unclear").sum()
    )

    positive_review = ground_truth[
        review_reason == "predicted_ausweiskopie"
    ]
    reviewed_positive_count = len(positive_review)
    confirmed_positive_count = int(
        (
            positive_review["is_ausweiskopie_gt"]
            == "ausweiskopie"
        ).sum()
    )
    positive_rate = safe_divide(
        confirmed_positive_count,
        reviewed_positive_count,
    )
    positive_low, positive_high = wilson_interval(
        confirmed_positive_count,
        reviewed_positive_count,
    )

    negative_audit = ground_truth[
        review_reason == "negative_audit"
    ]
    audit_count = len(negative_audit)
    missed_positive_count = int(
        (
            negative_audit["is_ausweiskopie_gt"]
            == "ausweiskopie"
        ).sum()
    )
    miss_rate = safe_divide(
        missed_positive_count,
        audit_count,
    )
    miss_low, miss_high = wilson_interval(
        missed_positive_count,
        audit_count,
    )

    estimated_true_detected = positive_rate * total_positive
    estimated_true_detected_low = positive_low * total_positive
    estimated_true_detected_high = positive_high * total_positive

    estimated_missed = miss_rate * total_negative
    estimated_missed_low = miss_low * total_negative
    estimated_missed_high = miss_high * total_negative

    estimated_recall = safe_divide(
        estimated_true_detected,
        estimated_true_detected + estimated_missed,
    )
    recall_low = safe_divide(
        estimated_true_detected_low,
        estimated_true_detected_low + estimated_missed_high,
    )
    recall_high = safe_divide(
        estimated_true_detected_high,
        estimated_true_detected_high + estimated_missed_low,
    )

    audited_masterindexes = (
        negative_audit["masterindex_id"].nunique()
        if not negative_audit.empty
        else 0
    )
    negative_masterindexes = (
        screening[
            screening["label"] == "not_ausweiskopie"
        ]["masterindex_id"].nunique()
    )
    masterindexes_with_miss = (
        negative_audit[
            negative_audit["is_ausweiskopie_gt"]
            == "ausweiskopie"
        ]["masterindex_id"].nunique()
        if not negative_audit.empty
        else 0
    )

    excluded_positive_count = 0
    if EXCLUDED_PATH.exists():
        try:
            excluded = pd.read_csv(
                EXCLUDED_PATH,
                dtype=object,
                keep_default_na=False,
            )
        except EmptyDataError:
            excluded = pd.DataFrame()

        if (
            not excluded.empty
            and "review_reason" in excluded.columns
        ):
            excluded_positive_count = int(
                (
                    excluded["review_reason"]
                    == "predicted_ausweiskopie"
                ).sum()
            )

    rows = [
        metric(
            "agreement_rate_within_reviewed_rows",
            safe_divide(agreement_count, reviewed_count),
            agreement_count,
            reviewed_count,
            "stratified reviewed rows only",
            "Not corpus-wide accuracy.",
        ),
        metric(
            "total_predicted_positive_pages",
            total_positive,
            total_positive,
            total_positive,
            "all current screened pages",
            "",
        ),
        metric(
            "reviewed_predicted_positive_pages",
            reviewed_positive_count,
            reviewed_positive_count,
            total_positive,
            "predicted-positive review",
            "",
        ),
        metric(
            "excluded_predicted_positive_pages",
            excluded_positive_count,
            excluded_positive_count,
            total_positive,
            "predicted-positive review",
            "",
        ),
        metric(
            "predicted_positive_confirmation_rate",
            positive_rate,
            confirmed_positive_count,
            reviewed_positive_count,
            "reviewed predicted positives",
            "Headline precision-like metric.",
        ),
        metric(
            "predicted_positive_confirmation_rate_ci95_low",
            positive_low,
            None,
            None,
            "reviewed predicted positives",
            "Wilson interval.",
        ),
        metric(
            "predicted_positive_confirmation_rate_ci95_high",
            positive_high,
            None,
            None,
            "reviewed predicted positives",
            "Wilson interval.",
        ),
        metric(
            "total_predicted_negative_pages",
            total_negative,
            total_negative,
            total_negative,
            "all current screened pages",
            "",
        ),
        metric(
            "negative_audit_page_count",
            audit_count,
            audit_count,
            total_negative,
            "global random negative-page audit",
            "",
        ),
        metric(
            "negative_audit_missed_positive_rate",
            miss_rate,
            missed_positive_count,
            audit_count,
            "global random negative-page audit",
            "Estimated page-level false-negative rate.",
        ),
        metric(
            "negative_audit_miss_rate_ci95_low",
            miss_low,
            None,
            None,
            "global random negative-page audit",
            "Wilson interval.",
        ),
        metric(
            "negative_audit_miss_rate_ci95_high",
            miss_high,
            None,
            None,
            "global random negative-page audit",
            "Wilson interval.",
        ),
        metric(
            "estimated_missed_positive_pages",
            estimated_missed,
            None,
            total_negative,
            "full predicted-negative page pool",
            "Audit miss rate multiplied by negative pool size.",
        ),
        metric(
            "estimated_missed_positive_pages_ci95_low",
            estimated_missed_low,
            None,
            total_negative,
            "full predicted-negative page pool",
            "Derived from audit-rate lower bound.",
        ),
        metric(
            "estimated_missed_positive_pages_ci95_high",
            estimated_missed_high,
            None,
            total_negative,
            "full predicted-negative page pool",
            "Derived from audit-rate upper bound.",
        ),
        metric(
            "approximate_estimated_detection_recall",
            estimated_recall,
            None,
            None,
            "page level",
            "Uses positive review and negative audit estimates.",
        ),
        metric(
            "approximate_estimated_detection_recall_ci95_low",
            recall_low,
            None,
            None,
            "page level",
            "Approximate conservative bound.",
        ),
        metric(
            "approximate_estimated_detection_recall_ci95_high",
            recall_high,
            None,
            None,
            "page level",
            "Approximate optimistic bound.",
        ),
        metric(
            "unclear_page_rate",
            safe_divide(total_unclear, len(screening)),
            total_unclear,
            len(screening),
            "all current screened pages",
            "",
        ),
        metric(
            "negative_audit_masterindex_coverage",
            safe_divide(
                audited_masterindexes,
                negative_masterindexes,
            ),
            audited_masterindexes,
            negative_masterindexes,
            "MasterIndex coverage of page-level audit",
            "Coverage only; the audit estimates a page-level rate.",
        ),
        metric(
            "audited_masterindexes_with_missed_positive",
            masterindexes_with_miss,
            masterindexes_with_miss,
            audited_masterindexes,
            "audited MasterIndex folders",
            "",
        ),
    ]

    summary = pd.DataFrame(rows)
    errors = ground_truth[
        predicted != actual
    ].copy()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUTPUT_PATH, index=False)
    errors.to_csv(ERRORS_PATH, index=False)

    print(f"Reviewed pages evaluated: {reviewed_count}")
    print(
        "Predicted-positive confirmation rate: "
        f"{positive_rate:.4f}"
    )
    print(
        "Negative-audit missed-positive rate: "
        f"{miss_rate:.4f}"
    )
    print(
        "Approximate estimated page-level recall: "
        f"{estimated_recall:.4f}"
    )
    print(f"Audit summary written to: {OUTPUT_PATH}")
    print(f"Review errors written to: {ERRORS_PATH}")


if __name__ == "__main__":
    main()
