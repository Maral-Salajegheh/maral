#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Map the delivered MasterIndex file to final Analyse-DB page labels.

TB0 is not detected with OCR. The final page table already carries the
verified document class in sfdoc_class. A page is labelled TB0 when its
sfdoc_class contains "Technikblatt"; otherwise its existing SST is retained.

The PDF page of each row inside its stack PDF is computed with the rule
verified on real documents: a document starts right after all pages of the
documents delivered before it in the stack, and seqno gives the position
inside the document. Documents are ordered by where their first row appears in
the Snowflake result, so the rule runs before any sorting.

Run from life-docai/Antrag/Datasets/Mapping:

    pixi run -e dev python 00_map_mid_to_adb.py --schema D131_D2D

Outputs are written to the existing Mapping/output directory:

    life_mid_documents.csv
    life_mid_documents.parquet
    life_mid_pages.csv
    life_mid_pages.parquet
    life_mid_summary.csv
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Optional

import pandas as pd
import sqlalchemy

from life_docai.utils.snowflake_utils import get_engine


DEFAULT_SCHEMA = "D131_D2D"
MAPPING_FILE = Path(
    "/home/shared_folders/life_ai/mid_mapping/sample_A00_G07_MAD_6000.txt"
)
OUTPUT_DIR = Path(__file__).resolve().parent / "output"

SOURCE_PAGE_TABLE = "PROC_LIFE_FINAL_PAGE_LABELS"
SNOWFLAKE_OUTPUT_TABLE = "PROC_LIFE_MID_ADB"

DOCUMENTS_CSV = OUTPUT_DIR / "life_mid_documents.csv"
DOCUMENTS_PARQUET = OUTPUT_DIR / "life_mid_documents.parquet"
PAGES_CSV = OUTPUT_DIR / "life_mid_pages.csv"
PAGES_PARQUET = OUTPUT_DIR / "life_mid_pages.parquet"
SUMMARY_CSV = OUTPUT_DIR / "life_mid_summary.csv"

OK = "OK"
TB0 = "TB0"


# ---------------------------------------------------------------------------
# General helpers
# ---------------------------------------------------------------------------

def validate_identifier(value: str) -> str:
    """Allow only safe unquoted Snowflake identifiers."""
    if not re.fullmatch(r"[A-Za-z0-9_]+", value):
        raise ValueError(f"Unsafe Snowflake identifier: {value!r}")
    return value.upper()


def clean(value) -> Optional[str]:
    """Return stripped text, or None for an empty value."""
    if pd.isna(value):
        return None
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return None
    return text


def stack_key(value) -> Optional[str]:
    text = clean(value)
    return None if text is None else text.lower()


def id_key(value) -> Optional[str]:
    """Normalize zero-padded document identifiers for joining only."""
    text = clean(value)
    if text is None:
        return None
    return str(int(text)) if text.isdigit() else text.lower()


def upper_or_none(value) -> Optional[str]:
    text = clean(value)
    return None if text is None else text.upper()


def first_non_null(values: pd.Series):
    values = values.dropna()
    return values.iloc[0] if not values.empty else None


def join_text(values: pd.Series) -> str:
    return ",".join(str(value) for value in values.dropna())


# ---------------------------------------------------------------------------
# MasterIndex input file
# ---------------------------------------------------------------------------

COLUMN_ALIASES = {
    "stackid": "stack_id",
    "stack_id": "stack_id",
    "docid": "doc_id",
    "doc_id": "doc_id",
    "subdocid": "subdoc_idx",
    "subdoc_idx": "subdoc_idx",
    "sst": "mid_sst",
    "masterindexid": "masterindex_id",
    "masterindex_id": "masterindex_id",
}
REQUIRED_MAPPING_COLUMNS = [
    "stack_id",
    "doc_id",
    "subdoc_idx",
    "mid_sst",
    "masterindex_id",
]
SEPARATORS = [",", ";", "\t", "|", ":"]
ENCODINGS = ["utf-8-sig", "latin-1"]


def rename_mapping_columns(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    data.columns = [str(column).strip().lower() for column in data.columns]
    return data.rename(columns=COLUMN_ALIASES)


def read_with(
    path: Path,
    separator: str,
    encoding: str,
    nrows: Optional[int] = None,
) -> pd.DataFrame:
    return pd.read_csv(
        path,
        sep=separator,
        dtype=str,
        encoding=encoding,
        skipinitialspace=True,
        low_memory=False,
        nrows=nrows,
    )


def detect_text_format(path: Path) -> tuple[str, str]:
    """Find the separator and encoding that recover the expected header."""
    best_score = -1
    best_separator = None
    best_encoding = None

    for encoding in ENCODINGS:
        for separator in SEPARATORS:
            try:
                sample = rename_mapping_columns(
                    read_with(path, separator, encoding, nrows=5)
                )
                score = sum(
                    column in sample.columns
                    for column in REQUIRED_MAPPING_COLUMNS
                )
            except Exception:
                score = -1

            if score > best_score:
                best_score = score
                best_separator = separator
                best_encoding = encoding

    if best_score <= 0 or best_separator is None or best_encoding is None:
        raise ValueError(
            f"Could not parse {path.name}. Expected columns: "
            f"{REQUIRED_MAPPING_COLUMNS}"
        )
    return best_separator, best_encoding


def load_mapping(path: Path) -> pd.DataFrame:
    """Load the delivered MasterIndex file and create normalized join keys."""
    if not path.is_file():
        raise FileNotFoundError(f"MasterIndex mapping file not found: {path}")

    if path.suffix.lower() in {".xlsx", ".xls"}:
        data = rename_mapping_columns(pd.read_excel(path, dtype=str))
    else:
        separator, encoding = detect_text_format(path)
        print(f"MasterIndex separator {separator!r}, encoding {encoding}.")
        data = rename_mapping_columns(read_with(path, separator, encoding))

    missing = set(REQUIRED_MAPPING_COLUMNS) - set(data.columns)
    if missing:
        raise ValueError(f"MasterIndex file is missing columns: {sorted(missing)}")

    data = data[REQUIRED_MAPPING_COLUMNS].copy()
    for column in REQUIRED_MAPPING_COLUMNS:
        data[column] = data[column].map(clean)

    data["stack_id_key"] = data["stack_id"].map(stack_key)
    data["doc_id_key"] = data["doc_id"].map(id_key)
    data["subdoc_idx_key"] = data["subdoc_idx"].map(id_key)
    data["mid_sst"] = data["mid_sst"].map(upper_or_none)
    return data.drop_duplicates().reset_index(drop=True)


# ---------------------------------------------------------------------------
# Final Analyse-DB page labels
# ---------------------------------------------------------------------------

DOCUMENT_KEYS = [
    "stack_id_key",
    "process_id",
    "doc_id_key",
    "subdoc_idx_key",
]
MAPPING_JOIN_KEYS = ["stack_id_key", "doc_id_key", "subdoc_idx_key"]


def load_page_labels(
    engine: sqlalchemy.Engine,
    schema: str,
    stack_ids: list[str],
) -> pd.DataFrame:
    """Read final page rows only for stacks present in the input file.

    delivery_row records the order in which Snowflake returned the rows. The
    PDF-page rule depends on that order, so it is captured before any sort.
    """
    if not stack_ids:
        raise ValueError("The MasterIndex mapping contains no usable stack_id.")

    query = sqlalchemy.text(
        f"""
        SELECT
            stack_id,
            process_id,
            doc_id,
            subdoc_idx,
            image_id,
            seqno,
            sst,
            sfdoc_class,
            label_tier,
            training_label_quality
        FROM {schema}.{SOURCE_PAGE_TABLE}
        WHERE stack_id IS NOT NULL
          AND image_id IS NOT NULL
          AND LOWER(stack_id) IN :stack_ids
        """
    ).bindparams(sqlalchemy.bindparam("stack_ids", expanding=True))
    data = pd.read_sql_query(
        query,
        engine,
        params={"stack_ids": sorted(set(stack_ids))},
    )
    data.columns = [str(column).strip().lower() for column in data.columns]
    data["delivery_row"] = range(len(data))
    return prepare_page_labels(data)


def prepare_page_labels(data: pd.DataFrame) -> pd.DataFrame:
    """Clean identifiers while preserving their original displayed values."""
    required = {
        "stack_id",
        "process_id",
        "doc_id",
        "subdoc_idx",
        "image_id",
        "seqno",
        "sst",
        "sfdoc_class",
        "label_tier",
        "training_label_quality",
        "delivery_row",
    }
    missing = required - set(data.columns)
    if missing:
        raise ValueError(
            f"{SOURCE_PAGE_TABLE} is missing columns: {sorted(missing)}"
        )

    data = data.copy()
    for column in ["stack_id", "process_id", "doc_id", "subdoc_idx", "image_id"]:
        data[column] = data[column].map(clean)

    data["stack_id_key"] = data["stack_id"].map(stack_key)
    data["doc_id_key"] = data["doc_id"].map(id_key)
    data["subdoc_idx_key"] = data["subdoc_idx"].map(id_key)
    data["sst"] = data["sst"].map(upper_or_none)
    data["sfdoc_class"] = data["sfdoc_class"].map(clean)
    data["image_id_num"] = pd.to_numeric(data["image_id"], errors="coerce")
    data["seqno_num"] = pd.to_numeric(data["seqno"], errors="coerce")
    return data.reset_index(drop=True)


# ---------------------------------------------------------------------------
# PDF page inside the stack PDF
# ---------------------------------------------------------------------------

def document_offsets(pages: pd.DataFrame) -> pd.DataFrame:
    """Pages before each document in its stack.

    Documents are ordered by where their first row appears in the delivery.
    Each document starts after the full page count of the documents before it,
    even when a document's rows are split in the delivery.
    """
    documents = (
        pages.groupby(DOCUMENT_KEYS, dropna=False)
        .agg(first_row=("delivery_row", "min"), n_doc_pages=("image_id", "size"))
        .reset_index()
        .sort_values(["stack_id_key", "first_row"])
    )
    documents["pages_before"] = (
        documents.groupby("stack_id_key")["n_doc_pages"].cumsum()
        - documents["n_doc_pages"]
    )
    return documents[DOCUMENT_KEYS + ["pages_before"]]


def add_stack_pdf_page(pages: pd.DataFrame) -> pd.DataFrame:
    """PDF page = pages of all earlier documents in the stack + seqno."""
    pages = pages.merge(document_offsets(pages), on=DOCUMENT_KEYS, how="left")
    pages["stack_pdf_page"] = (pages["pages_before"] + pages["seqno_num"]).astype("Int64")
    pages["stack_pdf_page_duplicate"] = pages.duplicated(
        ["stack_id_key", "stack_pdf_page"],
        keep=False,
    )
    return pages.drop(columns=["pages_before"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Document summary
# ---------------------------------------------------------------------------

def order_pages(data: pd.DataFrame) -> pd.DataFrame:
    """Order pages inside each Analyse-DB document by seqno."""
    return data.sort_values(
        DOCUMENT_KEYS + ["seqno_num", "image_id_num", "image_id"],
        na_position="last",
    )


def summarize_documents(pages: pd.DataFrame) -> pd.DataFrame:
    """Create one validation row per final Analyse-DB document."""
    ordered = order_pages(pages)
    grouped = ordered.groupby(DOCUMENT_KEYS, dropna=False)

    documents = grouped.agg(
        stack_id=("stack_id", "first"),
        doc_id=("doc_id", "first"),
        subdoc_idx=("subdoc_idx", "first"),
        sst=("sst", first_non_null),
        sfdoc_class=("sfdoc_class", first_non_null),
        label_tier=("label_tier", first_non_null),
        training_label_quality=("training_label_quality", first_non_null),
        n_pages=("image_id", "size"),
        n_distinct_image_ids=("image_id", "nunique"),
        n_distinct_sst=("sst", "nunique"),
        n_distinct_sfdoc_class=("sfdoc_class", "nunique"),
        image_ids=("image_id", join_text),
        seqnos=("seqno", join_text),
        first_delivery_row=("delivery_row", "min"),
    ).reset_index()
    return documents


# ---------------------------------------------------------------------------
# Mapping validation
# ---------------------------------------------------------------------------

def attach_process_count(documents: pd.DataFrame) -> pd.DataFrame:
    """Count final process IDs for each mapping-file document key."""
    counts = (
        documents.groupby(MAPPING_JOIN_KEYS, dropna=False)["process_id"]
        .nunique()
        .rename("n_process_ids_for_key")
        .reset_index()
    )
    return documents.merge(counts, on=MAPPING_JOIN_KEYS, how="left")


def join_mapping(
    mapping: pd.DataFrame,
    documents: pd.DataFrame,
) -> pd.DataFrame:
    """Attach Analyse-DB documents to the delivered MasterIndex rows."""
    documents = attach_process_count(documents)
    return mapping.merge(
        documents,
        on=MAPPING_JOIN_KEYS,
        how="left",
        suffixes=("_mid", ""),
        validate="one_to_many",
    )


def classify_documents(data: pd.DataFrame) -> pd.DataFrame:
    """Validate the mapping without assuming image_id is a PDF page number."""
    data = data.copy()
    data["sst_matches_mid"] = (
        data["sst"].notna()
        & data["mid_sst"].notna()
        & data["sst"].eq(data["mid_sst"])
    )
    data["mapping_status"] = OK

    rules = [
        (data["process_id"].isna(), "NO_ANALYSE_DB_PAGES"),
        (data["sst"].isna(), "MISSING_SST"),
        (data["n_distinct_sst"].gt(1), "DOCUMENT_MULTIPLE_SST"),
        (data["sst_matches_mid"].ne(True), "SST_MISMATCH"),
        (data["n_process_ids_for_key"].gt(1), "AMBIGUOUS_PROCESS_ID"),
        (
            data["n_pages"].ne(data["n_distinct_image_ids"]),
            "DUPLICATE_IMAGE_ID",
        ),
    ]
    for condition, status in rules:
        applies = condition.fillna(False) & data["mapping_status"].eq(OK)
        data.loc[applies, "mapping_status"] = status

    data["is_usable"] = data["mapping_status"].eq(OK)
    data["is_training_eligible"] = (
        data["is_usable"]
        & data["training_label_quality"].isin(["GOLD", "SILVER"])
    )
    return data


# ---------------------------------------------------------------------------
# Page-level output and TB0 label
# ---------------------------------------------------------------------------

def make_page_class(data: pd.DataFrame) -> pd.DataFrame:
    """Use sfdoc_class for TB0; retain the final SST for every other page."""
    data = data.copy()
    is_technikblatt = (
        data["sfdoc_class"]
        .fillna("")
        .str.contains("Technikblatt", case=False, regex=False)
    )
    data["page_class"] = data["sst"]
    data["page_class_source"] = "SST"
    data.loc[is_technikblatt, "page_class"] = TB0
    data.loc[is_technikblatt, "page_class_source"] = "SFDOC_CLASS"
    return data


def build_pages(
    page_labels: pd.DataFrame,
    documents: pd.DataFrame,
) -> pd.DataFrame:
    """Return exact Analyse-DB page rows for usable mapped documents."""
    usable = documents[documents["is_usable"]].copy()
    if usable.empty:
        return pd.DataFrame()

    carry = DOCUMENT_KEYS + [
        "masterindex_id",
        "mid_sst",
        "mapping_status",
        "is_usable",
        "is_training_eligible",
    ]
    usable = usable[carry].drop_duplicates(DOCUMENT_KEYS)
    pages = page_labels.merge(
        usable,
        on=DOCUMENT_KEYS,
        how="inner",
        validate="many_to_one",
    )
    pages = order_pages(pages).copy()
    pages["page_number"] = (
        pages.groupby(DOCUMENT_KEYS, dropna=False).cumcount() + 1
    )
    pages["n_pages"] = pages.groupby(DOCUMENT_KEYS, dropna=False)[
        "image_id"
    ].transform("size")
    pages["is_first_page"] = pages["page_number"].eq(1)
    pages["is_last_page"] = pages["page_number"].eq(pages["n_pages"])
    pages = make_page_class(pages)
    # page_number needed seqno order above; the output follows Snowflake's order.
    return pages.sort_values(["stack_id_key", "delivery_row"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

DOCUMENT_COLUMNS = [
    "masterindex_id",
    "stack_id",
    "process_id",
    "doc_id",
    "subdoc_idx",
    "sst",
    "mid_sst",
    "sst_matches_mid",
    "sfdoc_class",
    "label_tier",
    "training_label_quality",
    "n_pages",
    "image_ids",
    "seqnos",
    "first_delivery_row",
    "n_process_ids_for_key",
    "mapping_status",
    "is_usable",
    "is_training_eligible",
]

PAGE_COLUMNS = [
    "masterindex_id",
    "stack_id",
    "process_id",
    "doc_id",
    "subdoc_idx",
    "image_id",
    "seqno",
    "delivery_row",
    "stack_pdf_page",
    "stack_pdf_page_duplicate",
    "page_number",
    "n_pages",
    "sst",
    "mid_sst",
    "sfdoc_class",
    "page_class",
    "page_class_source",
    "label_tier",
    "training_label_quality",
    "is_first_page",
    "is_last_page",
    "mapping_status",
    "is_usable",
    "is_training_eligible",
]


def status_summary(documents: pd.DataFrame) -> pd.DataFrame:
    return (
        documents.groupby("mapping_status", dropna=False)
        .agg(
            n_documents=("mapping_status", "size"),
            n_masterindex_ids=("masterindex_id", "nunique"),
            n_stacks=("stack_id_key", "nunique"),
            n_pages=("n_pages", "sum"),
        )
        .reset_index()
        .sort_values("n_documents", ascending=False)
    )


def write_frame(data: pd.DataFrame, csv_path: Path, parquet_path: Path) -> None:
    data.to_csv(csv_path, index=False, encoding="utf-8-sig")
    try:
        data.to_parquet(parquet_path, index=False)
    except ImportError:
        print(f"WARNING: Parquet skipped for {parquet_path.name}; pyarrow missing.")


def write_outputs(documents: pd.DataFrame, pages: pd.DataFrame) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    documents = documents.sort_values(
        ["stack_id_key", "first_delivery_row"],
        na_position="last",
    )
    document_columns = [column for column in DOCUMENT_COLUMNS if column in documents]
    page_columns = [column for column in PAGE_COLUMNS if column in pages]
    write_frame(
        documents[document_columns],
        DOCUMENTS_CSV,
        DOCUMENTS_PARQUET,
    )
    write_frame(pages[page_columns], PAGES_CSV, PAGES_PARQUET)
    status_summary(documents).to_csv(
        SUMMARY_CSV,
        index=False,
        encoding="utf-8-sig",
    )


def print_pdf_page_report(pages: pd.DataFrame) -> None:
    """How many stacks the PDF-page rule placed without a collision."""
    by_stack = pages.groupby("stack_id")["stack_pdf_page_duplicate"].any()
    n_bad = int(by_stack.sum())
    print(f"\nStacks with a duplicate stack_pdf_page: {n_bad:,} of {len(by_stack):,}")


def print_report(documents: pd.DataFrame, pages: pd.DataFrame) -> None:
    print("\nMapping status:")
    print(status_summary(documents).to_string(index=False))

    print(f"\nMapped documents: {len(documents):,}")
    print(f"Usable documents: {int(documents['is_usable'].sum()):,}")
    print(f"Output pages: {len(pages):,}")

    if pages.empty:
        print("No usable page rows were produced.")
        return

    print("\nFinal page classes:")
    print(pages["page_class"].value_counts(dropna=False).to_string())
    print(f"\nTB0 pages from sfdoc_class: {int(pages['page_class'].eq(TB0).sum()):,}")
    print(
        "TB0 documents: "
        f"{pages.loc[pages['page_class'].eq(TB0), DOCUMENT_KEYS].drop_duplicates().shape[0]:,}"
    )
    print_pdf_page_report(pages)


# ---------------------------------------------------------------------------
# Optional publication
# ---------------------------------------------------------------------------

def load_existing_documents() -> pd.DataFrame:
    if not DOCUMENTS_CSV.is_file():
        raise FileNotFoundError(f"Not found: {DOCUMENTS_CSV}\nBuild the mapping first.")
    data = pd.read_csv(DOCUMENTS_CSV, dtype=str, low_memory=False)
    data["is_usable"] = data["is_usable"].astype(str).str.lower().eq("true")
    return data


def publish_to_snowflake(
    documents: pd.DataFrame,
    engine: sqlalchemy.Engine,
    schema: str,
) -> None:
    from snowflake.connector.pandas_tools import pd_writer

    usable = documents[documents["is_usable"]].copy()
    if usable.empty:
        raise RuntimeError("No usable documents to publish.")

    usable.columns = [column.upper() for column in usable.columns]
    usable.to_sql(
        SNOWFLAKE_OUTPUT_TABLE,
        con=engine,
        schema=schema,
        if_exists="replace",
        index=False,
        method=pd_writer,
    )
    print(
        f"Published {len(usable):,} rows to "
        f"{schema}.{SNOWFLAKE_OUTPUT_TABLE}."
    )


def confirm_publish(schema: str) -> bool:
    answer = input(f"Replace {schema}.{SNOWFLAKE_OUTPUT_TABLE}? (y/n): ")
    return answer.strip().lower() == "y"


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schema", default=DEFAULT_SCHEMA)
    parser.add_argument("--mapping-file", type=Path, default=MAPPING_FILE)
    parser.add_argument("--publish-snowflake", action="store_true")
    parser.add_argument("--yes", action="store_true")
    return parser.parse_args()


def build(args: argparse.Namespace, engine: sqlalchemy.Engine, schema: str) -> None:
    mapping = load_mapping(args.mapping_file)
    print(f"MasterIndex rows: {len(mapping):,}")

    selected_stack_ids = mapping["stack_id_key"].dropna().unique().tolist()
    page_labels = load_page_labels(engine, schema, selected_stack_ids)
    page_labels = add_stack_pdf_page(page_labels)
    print(f"Final Analyse-DB page rows: {len(page_labels):,}")

    adb_documents = summarize_documents(page_labels)
    documents = classify_documents(join_mapping(mapping, adb_documents))
    pages = build_pages(page_labels, documents)

    write_outputs(documents, pages)
    print_report(documents, pages)
    print(f"\nDocuments: {DOCUMENTS_CSV}")
    print(f"Pages:     {PAGES_CSV}")
    print(f"Summary:   {SUMMARY_CSV}")


def main() -> int:
    args = parse_args()
    schema = validate_identifier(args.schema)
    engine = get_engine(schema=schema)

    if args.publish_snowflake:
        documents = load_existing_documents()
        if args.yes or confirm_publish(schema):
            publish_to_snowflake(documents, engine, schema)
    else:
        build(args, engine, schema)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())