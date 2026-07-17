from __future__ import annotations

from pathlib import Path

import pandas as pd


INPUT_PATH = Path(
    "outputs/ausweis_page_screening.jsonl"
)

POSITIVE_OUTPUT_PATH = Path(
    "outputs/ausweis_masterindexes.csv"
)

HUMAN_REVIEW_OUTPUT_PATH = Path(
    "outputs/ausweis_human_review.csv"
)


def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Screening output not found: "
            f"{INPUT_PATH}"
        )

    results = pd.read_json(
        INPUT_PATH,
        lines=True,
    )

    # Keep only the latest result for each page.
    results = results.drop_duplicates(
        subset=[
            "masterindex_id",
            "pdf_path_in_zip",
            "page_number",
        ],
        keep="last",
    )

    positive_pages = results[
        (results["status"] == "success")
        & (results["label"] == "ausweiskopie")
        & (results["needs_human_review"] == False)
    ].copy()

    if positive_pages.empty:
        positive_output = pd.DataFrame(
            columns=[
                "masterindex_id",
                "ausweis_pages",
            ]
        )
    else:
        positive_pages["page_number"] = (
            positive_pages["page_number"]
            .astype(int)
        )

        positive_output = (
            positive_pages
            .groupby(
                "masterindex_id",
                as_index=False,
            )
            .agg(
                ausweis_pages=(
                    "page_number",
                    lambda pages: ", ".join(
                        str(page)
                        for page in sorted(
                            set(pages)
                        )
                    ),
                )
            )
            .sort_values(
                "masterindex_id"
            )
        )

    positive_output.to_csv(
        POSITIVE_OUTPUT_PATH,
        index=False,
    )

    human_review = results[
        results["needs_human_review"] == True
    ][
        [
            "masterindex_id",
            "pdf_path_in_zip",
            "page_number",
            "image_path",
            "label",
            "confidence",
            "status",
            "error",
        ]
    ].sort_values(
        [
            "masterindex_id",
            "page_number",
        ]
    )

    human_review.to_csv(
        HUMAN_REVIEW_OUTPUT_PATH,
        index=False,
    )

    print(
        "Confirmed high-confidence "
        f"Ausweiskopie MasterIndexIDs: "
        f"{len(positive_output)}"
    )

    print(
        "Pages routed to human review: "
        f"{len(human_review)}"
    )

    print(
        f"Positive output: "
        f"{POSITIVE_OUTPUT_PATH}"
    )

    print(
        f"Human-review output: "
        f"{HUMAN_REVIEW_OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()