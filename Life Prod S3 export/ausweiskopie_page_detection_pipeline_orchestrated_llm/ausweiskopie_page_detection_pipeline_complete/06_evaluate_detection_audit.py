from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


GROUND_TRUTH_PATH = Path(
    "outputs/"
    "ausweis_classification_ground_truth_pages.csv"
)

OUTPUT_PATH = Path(
    "outputs/"
    "ausweis_detection_audit_summary.csv"
)

ERRORS_PATH = Path(
    "outputs/"
    "ausweis_detection_review_errors.csv"
)


POSITIVE_LABEL = "ausweiskopie"
NEGATIVE_LABEL = "not_ausweiskopie"


def safe_divide(
    numerator: int,
    denominator: int,
) -> float:
    """Return zero when the denominator is zero."""
    if denominator == 0:
        return 0.0

    return numerator / denominator


def metric(
    name: str,
    value: float | int,
    numerator: int | None = None,
    denominator: int | None = None,
    scope: str = "human-reviewed pages",
) -> dict[str, Any]:
    """Build one metric output row."""
    return {
        "metric": name,
        "value": value,
        "numerator": numerator,
        "denominator": denominator,
        "scope": scope,
    }


def main() -> None:
    if not GROUND_TRUTH_PATH.exists():
        raise FileNotFoundError(
            f"Ground Truth file not found: "
            f"{GROUND_TRUTH_PATH}. "
            "Run "
            "05_build_ausweis_classification_ground_truth.py "
            "first."
        )

    ground_truth = pd.read_csv(
        GROUND_TRUTH_PATH,
        dtype=object,
        keep_default_na=False,
    )

    if ground_truth.empty:
        raise ValueError(
            "Ground Truth contains no "
            "human-reviewed pages."
        )

    required_columns = {
        "predicted_label",
        "is_ausweiskopie_gt",
    }

    missing_columns = (
        required_columns
        - set(ground_truth.columns)
    )

    if missing_columns:
        raise ValueError(
            "Missing required Ground Truth columns: "
            + ", ".join(
                sorted(missing_columns)
            )
        )

    ground_truth[
        "predicted_label"
    ] = (
        ground_truth["predicted_label"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    ground_truth[
        "is_ausweiskopie_gt"
    ] = (
        ground_truth["is_ausweiskopie_gt"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    # "unclear" is useful for routing, but it is not
    # a positive or negative classification result.
    evaluable = ground_truth[
        ground_truth[
            "is_ausweiskopie_gt"
        ].isin({
            POSITIVE_LABEL,
            NEGATIVE_LABEL,
        })
    ].copy()

    if evaluable.empty:
        raise ValueError(
            "Ground Truth contains no evaluable "
            "positive or negative pages."
        )

    predicted = evaluable[
        "predicted_label"
    ]

    actual = evaluable[
        "is_ausweiskopie_gt"
    ]

    true_positive = int(
        (
            (predicted == POSITIVE_LABEL)
            & (actual == POSITIVE_LABEL)
        ).sum()
    )

    false_positive = int(
        (
            (predicted == POSITIVE_LABEL)
            & (actual == NEGATIVE_LABEL)
        ).sum()
    )

    false_negative = int(
        (
            (predicted != POSITIVE_LABEL)
            & (actual == POSITIVE_LABEL)
        ).sum()
    )

    true_negative = int(
        (
            (predicted == NEGATIVE_LABEL)
            & (actual == NEGATIVE_LABEL)
        ).sum()
    )

    unclear_predictions = int(
        (
            predicted == "unclear"
        ).sum()
    )

    reviewed_count = len(
        ground_truth
    )

    evaluated_count = len(
        evaluable
    )

    correct_count = int(
        (
            predicted == actual
        ).sum()
    )

    precision = safe_divide(
        true_positive,
        true_positive + false_positive,
    )

    recall = safe_divide(
        true_positive,
        true_positive + false_negative,
    )

    f1 = safe_divide(
        2 * precision * recall,
        precision + recall,
    )

    accuracy = safe_divide(
        correct_count,
        evaluated_count,
    )

    specificity = safe_divide(
        true_negative,
        true_negative + false_positive,
    )

    rows = [
        metric(
            name="reviewed_page_count",
            value=reviewed_count,
            numerator=reviewed_count,
            denominator=reviewed_count,
        ),
        metric(
            name="evaluated_page_count",
            value=evaluated_count,
            numerator=evaluated_count,
            denominator=reviewed_count,
        ),
        metric(
            name="true_positive",
            value=true_positive,
            numerator=true_positive,
            denominator=evaluated_count,
        ),
        metric(
            name="false_positive",
            value=false_positive,
            numerator=false_positive,
            denominator=evaluated_count,
        ),
        metric(
            name="false_negative",
            value=false_negative,
            numerator=false_negative,
            denominator=evaluated_count,
        ),
        metric(
            name="true_negative",
            value=true_negative,
            numerator=true_negative,
            denominator=evaluated_count,
        ),
        metric(
            name="precision",
            value=precision,
            numerator=true_positive,
            denominator=(
                true_positive
                + false_positive
            ),
        ),
        metric(
            name="recall",
            value=recall,
            numerator=true_positive,
            denominator=(
                true_positive
                + false_negative
            ),
        ),
        metric(
            name="f1_score",
            value=f1,
        ),
        metric(
            name="accuracy",
            value=accuracy,
            numerator=correct_count,
            denominator=evaluated_count,
        ),
        metric(
            name="specificity",
            value=specificity,
            numerator=true_negative,
            denominator=(
                true_negative
                + false_positive
            ),
        ),
        metric(
            name="unclear_prediction_count",
            value=unclear_predictions,
            numerator=unclear_predictions,
            denominator=evaluated_count,
        ),
        metric(
            name="unclear_prediction_rate",
            value=safe_divide(
                unclear_predictions,
                evaluated_count,
            ),
            numerator=unclear_predictions,
            denominator=evaluated_count,
        ),
    ]

    summary = pd.DataFrame(rows)

    errors = evaluable[
        predicted != actual
    ].copy()

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    errors.to_csv(
        ERRORS_PATH,
        index=False,
    )

    print(
        f"Human-reviewed pages: "
        f"{reviewed_count}"
    )

    print(
        f"Evaluated positive/negative pages: "
        f"{evaluated_count}"
    )

    print(
        f"Precision: {precision:.4f}"
    )

    print(
        f"Recall: {recall:.4f}"
    )

    print(
        f"F1 score: {f1:.4f}"
    )

    print(
        f"Accuracy: {accuracy:.4f}"
    )

    print(
        f"Audit summary written to: "
        f"{OUTPUT_PATH}"
    )

    print(
        f"Review errors written to: "
        f"{ERRORS_PATH}"
    )


if __name__ == "__main__":
    main()