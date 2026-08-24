"""
Build the AB1 pseudo-document index.

Joins the page screening results to the AB1 mapping file and writes one
row per page with the derived pseudo-document columns: segment_role,
pseudo_doc_id, training_page_sst, and the first/last page flags used as
PnC boundary targets.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from pipeline_utils import page_record_key, read_latest_jsonl

# AB1 mapping file: whitespace-delimited, with a header row.
MAPPING_PATH = Path("inputs/ab1_mapping.txt")

PAGE_RESULTS_PATH = Path("outputs/ausweis_page_screening.jsonl")

PSEUDO_DOC_PATH = Path("outputs/ab1_pseudo_documents.csv")

MAPPING_MID_COLUMN = "masterindex_id"
MAPPING_STACK_ID_COLUMN = "stack_id"
MAPPING_DOC_ID_COLUMN = "doc_id"
MAPPING_SST_COLUMN = "sst"

# Identity columns, written first.
ID_COLUMNS = [
    MAPPING_MID_COLUMN,
    MAPPING_STACK_ID_COLUMN,
    MAPPING_DOC_ID_COLUMN,
    "pseudo_doc_id",
]


# Page labels produced by 03_screen_ausweis_pages.py.
LABEL_ID = "ausweiskopie"
LABEL_NOT_ID = "not_ausweiskopie"

# Resolved roles used for segmentation.
ROLE_ID = "id"
ROLE_NOT_ID = "not_id"

# Quality status that makes an unresolved page fall back to ROLE_ID.
ID_FALLBACK_QUALITY = "low_contrast"

# SST assigned to pages inside an ID segment.
ID_SEGMENT_SST = "G07"

SEGMENT_ID_TEMPLATE = "{stack_id}_{doc_id}_seg_{index}"


def labelled_role(label: str) -> str | None:
    """Map a content label to a role. Anything else stays unresolved."""
    if label == LABEL_ID:
        return ROLE_ID
    if label == LABEL_NOT_ID:
        return ROLE_NOT_ID
    return None


def fallback_role(quality_status: str) -> str:
    """Role for an unresolved page when its neighbours do not agree."""
    if quality_status == ID_FALLBACK_QUALITY:
        return ROLE_ID
    return ROLE_NOT_ID


def previous_known(roles: list[str | None], index: int) -> str | None:
    """Nearest resolved role before index."""
    for position in range(index - 1, -1, -1):
        if roles[position] is not None:
            return roles[position]
    return None


def next_known(roles: list[str | None], index: int) -> str | None:
    """Nearest resolved role after index."""
    for position in range(index + 1, len(roles)):
        if roles[position] is not None:
            return roles[position]
    return None


def bridge_roles(
    roles: list[str | None],
    fallbacks: list[str],
) -> list[str]:
    """
    Fill unresolved roles.

    An unresolved page takes the surrounding role when the nearest
    resolved page on both sides agree; otherwise it takes its fallback.
    Bridging reads the original roles, so the result does not depend on
    the order pages are filled.
    """
    filled: list[str] = []

    for index, role in enumerate(roles):
        if role is not None:
            filled.append(role)
            continue

        before = previous_known(roles, index)
        after = next_known(roles, index)

        if before is not None and before == after:
            filled.append(before)
        else:
            filled.append(fallbacks[index])

    return filled


def has_id_page(roles: list[str]) -> bool:
    """A document only needs pseudo-document structure if it encloses an ID page."""
    return ROLE_ID in roles


def assign_pseudo_doc_ids(
    stack_id: str,
    doc_id: str,
    roles: list[str],
) -> list[str | None]:
    """
    One segment per contiguous run of identical roles.

    Documents without an ID page have no internal structure, so they get
    no pseudo-document ID.
    """
    if not has_id_page(roles):
        return [None] * len(roles)

    ids: list[str | None] = []
    segment = 0

    for index, role in enumerate(roles):
        if index == 0 or role != roles[index - 1]:
            segment += 1
        ids.append(
            SEGMENT_ID_TEMPLATE.format(
                stack_id=stack_id,
                doc_id=doc_id,
                index=segment,
            )
        )

    return ids


def training_page_sst(role: str, parent_sst: str) -> str:
    """ID pages carry the ID segment SST, all other pages the parent SST."""
    if role == ROLE_ID:
        return ID_SEGMENT_SST
    return parent_sst


def derive_boundaries(roles: list[str]) -> tuple[list[int], list[int]]:
    """
    First page flags mark each role change; last page flags are the first
    page flags shifted by one, with the final page always closing.
    """
    first = [
        1 if index == 0 or roles[index] != roles[index - 1] else 0
        for index in range(len(roles))
    ]
    last = first[1:] + [1]

    return first, last


def is_masked(label: str) -> bool:
    """A page is masked for training when its role was not directly labelled."""
    return labelled_role(label) is None


def build_document(
    pages: list[dict[str, Any]],
    stack_id: str,
    doc_id: str,
    parent_sst: str,
) -> list[dict[str, Any]]:
    """Add the derived columns to one document's pages, in page order."""
    labels = [str(page["label"]) for page in pages]
    qualities = [str(page["quality_status"]) for page in pages]

    roles = bridge_roles(
        roles=[labelled_role(label) for label in labels],
        fallbacks=[fallback_role(quality) for quality in qualities],
    )

    pseudo_doc_ids = assign_pseudo_doc_ids(stack_id, doc_id, roles)
    first, last = derive_boundaries(roles)

    return [
        {
            **page,
            "stack_id": stack_id,
            "doc_id": doc_id,
            "parent_sst": parent_sst,
            "segment_role": roles[index],
            "pseudo_doc_id": pseudo_doc_ids[index],
            "training_page_sst": training_page_sst(
                roles[index], parent_sst
            ),
            "is_first_page": first[index],
            "is_last_page": last[index],
            "masked_for_training": is_masked(labels[index]),
        }
        for index, page in enumerate(pages)
    ]


def read_mapping(path: Path) -> pd.DataFrame:
    """Read the whitespace-delimited mapping file and keep the needed columns."""
    if not path.exists():
        raise FileNotFoundError(f"Mapping file not found: {path}")

    mapping = pd.read_csv(path, sep=r"\s+", dtype=object)

    required = [
        MAPPING_MID_COLUMN,
        MAPPING_STACK_ID_COLUMN,
        MAPPING_DOC_ID_COLUMN,
        MAPPING_SST_COLUMN,
    ]
    missing = [name for name in required if name not in mapping.columns]

    if missing:
        raise ValueError(
            f"Mapping file {path} is missing columns {missing}. "
            f"Found: {list(mapping.columns)}"
        )

    return mapping[required].drop_duplicates()


def read_pages(path: Path) -> list[dict[str, Any]]:
    """Latest screening record per page."""
    if not path.exists():
        raise FileNotFoundError(f"Screening results not found: {path}")

    return list(read_latest_jsonl(path, page_record_key).values())


def group_pages(
    pages: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """Group pages by MasterIndex ID, sorted by page number."""
    grouped: dict[str, list[dict[str, Any]]] = {}

    for page in pages:
        grouped.setdefault(str(page["masterindex_id"]), []).append(page)

    for masterindex_id, document_pages in grouped.items():
        document_pages.sort(key=lambda item: int(item["page_number"]))

    return grouped


def check_page_numbers(
    masterindex_id: str,
    pages: list[dict[str, Any]],
) -> None:
    """Page numbers must be a contiguous run, or runs cannot be trusted."""
    numbers = [int(page["page_number"]) for page in pages]
    expected = list(range(numbers[0], numbers[0] + len(numbers)))

    if numbers != expected:
        raise ValueError(
            f"{masterindex_id}: page numbers are not contiguous. "
            f"Got {numbers}"
        )


def build_rows(
    grouped: dict[str, list[dict[str, Any]]],
    mapping: pd.DataFrame,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Derive every mapped document; return the rows and the unmapped IDs."""
    lookup = mapping.set_index(MAPPING_MID_COLUMN).to_dict("index")

    rows: list[dict[str, Any]] = []
    unmapped: list[str] = []

    for masterindex_id, pages in grouped.items():
        entry = lookup.get(masterindex_id)

        if entry is None:
            unmapped.append(masterindex_id)
            continue

        check_page_numbers(masterindex_id, pages)

        rows.extend(
            build_document(
                pages=pages,
                stack_id=str(entry[MAPPING_STACK_ID_COLUMN]),
                doc_id=str(entry[MAPPING_DOC_ID_COLUMN]),
                parent_sst=str(entry[MAPPING_SST_COLUMN]),
            )
        )

    return rows, unmapped


def order_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Put the identity columns first, keep everything else in place."""
    rest = [name for name in frame.columns if name not in ID_COLUMNS]
    return frame[ID_COLUMNS + rest]


def report(rows: list[dict[str, Any]], unmapped: list[str]) -> None:
    """Print the counts worth seeing before training."""
    frame = pd.DataFrame(rows)

    internal_first = frame[
        (frame["is_first_page"] == 1) & (frame["page_number"].astype(int) > 1)
    ]

    nested = frame[frame["pseudo_doc_id"].notna()]

    print(f"Documents:                  {frame['doc_id'].nunique()}")
    print(f"Pages:                      {len(frame)}")
    print(f"Documents with an ID page:  {nested['doc_id'].nunique()}")
    print(f"Segments:                   {nested['pseudo_doc_id'].nunique()}")
    print(f"Internal first pages:       {len(internal_first)}")
    print(
        "Documents with an internal first page: "
        f"{internal_first['doc_id'].nunique()}"
    )
    print(f"Masked pages:               {int(frame['masked_for_training'].sum())}")
    print(f"Unmapped MasterIndex IDs:   {len(unmapped)}")

    print("\nPages per segment role:")
    print(frame["segment_role"].value_counts().to_string())


def main() -> None:
    mapping = read_mapping(MAPPING_PATH)
    grouped = group_pages(read_pages(PAGE_RESULTS_PATH))

    rows, unmapped = build_rows(grouped, mapping)

    PSEUDO_DOC_PATH.parent.mkdir(parents=True, exist_ok=True)
    order_columns(pd.DataFrame(rows)).to_csv(PSEUDO_DOC_PATH, index=False)

    report(rows, unmapped)
    print(f"\nWritten to: {PSEUDO_DOC_PATH}")


if __name__ == "__main__":
    main()
