#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Turn the delivered MasterIndex mapping into document- and page-level tables.

The mapping file has one row per document, with image_ids and pages as two
positionally paired lists: page k of that document's PDF is image_ids[k]. The
per-document SST is not in the file, so it is joined from Analyse-DB.

    python 00_map_mid_to_adb.py --schema D131_D2D
    python 00_map_mid_to_adb.py --schema D131_D2D --publish-snowflake
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
    "/home/shared_folders/life_ai/mid_mapping/mapping_result_process_level_AI.xlsx"
)
OUTPUT_DIR = Path(__file__).resolve().parent / "output"

SOURCE_DOCUMENT_TABLE = "PROC_LIFE_FINAL_DOCUMENT_LABELS"
SNOWFLAKE_OUTPUT_TABLE = "PROC_LIFE_MID_ADB"

DOCUMENTS_CSV = OUTPUT_DIR / "life_mid_documents.csv"
DOCUMENTS_PARQUET = OUTPUT_DIR / "life_mid_documents.parquet"
PAGES_CSV = OUTPUT_DIR / "life_mid_pages.csv"
PAGES_PARQUET = OUTPUT_DIR / "life_mid_pages.parquet"
SUMMARY_CSV = OUTPUT_DIR / "life_mid_summary.csv"

DOCUMENT_KEYS = ["stack_id_key", "process_id", "doc_id", "subdoc_idx"]
OK = "OK"


# --- helpers ---------------------------------------------------------------

def validate_identifier(value: str) -> str:
    """Allow only unquoted Snowflake identifiers."""
    if not re.fullmatch(r"[A-Za-z0-9_]+", value):
        raise ValueError(f"Unsafe Snowflake identifier: {value!r}")
    return value.upper()


def clean(value) -> Optional[str]:
    """Stripped text, or None if empty."""
    if pd.isna(value):
        return None
    text = str(value).strip()
    return None if text == "" or text.lower() in {"nan", "none", "null"} else text


def stack_key(value) -> Optional[str]:
    """Case-insensitive stack_id join key."""
    text = clean(value)
    return None if text is None else text.lower()


def parse_int_list(value) -> list[int]:
    """Parse a comma-separated list of integers; unparseable entries are dropped."""
    text = clean(value)
    if text is None:
        return []
    return [int(part) for part in text.split(",") if part.strip().isdigit()]


# --- mapping file ----------------------------------------------------------

REQUIRED_COLUMNS = [
    "masterindex_id", "stack_id", "process_id",
    "doc_id", "subdoc_idx", "image_ids", "pages",
]


def read_mapping_file(path: Path) -> pd.DataFrame:
    """Read the delivered mapping file (.xlsx or .csv) as text."""
    if not path.is_file():
        raise FileNotFoundError(f"Mapping file not found: {path}")

    if path.suffix.lower() in {".xlsx", ".xls"}:
        data = pd.read_excel(path, dtype=str)
    else:
        data = pd.read_csv(path, dtype=str, encoding="utf-8-sig",
                           skipinitialspace=True, low_memory=False)

    data.columns = [str(c).strip().lower() for c in data.columns]
    missing = [c for c in REQUIRED_COLUMNS if c not in data.columns]
    if missing:
        raise ValueError(f"Mapping file is missing columns: {missing}")
    return data


def prepare_mapping(data: pd.DataFrame) -> pd.DataFrame:
    """Clean the key columns and parse the two list columns."""
    data = data.copy()
    for column in ["masterindex_id", "stack_id", "process_id", "doc_id",
                   "subdoc_idx", "stack_ssts"]:
        if column in data.columns:
            data[column] = data[column].map(clean)

    data["stack_id_key"] = data["stack_id"].map(stack_key)
    data["image_id_list"] = data["image_ids"].map(parse_int_list)
    data["page_list"] = data["pages"].map(parse_int_list)
    data["n_pages"] = data["page_list"].map(len)
    return data.reset_index(drop=True)


def load_mapping(path: Path) -> pd.DataFrame:
    """Read and prepare the mapping file."""
    return prepare_mapping(read_mapping_file(path))


# --- document checks -------------------------------------------------------

def is_one_to_n(values: list[int]) -> bool:
    """True when the values are exactly 1..len(values), each once."""
    return sorted(values) == list(range(1, len(values) + 1))


def max_gap(values: list[int]) -> int:
    """Largest jump between consecutive image_ids; 1 means contiguous."""
    if len(values) < 2:
        return 1
    ordered = sorted(values)
    return max(b - a for a, b in zip(ordered, ordered[1:]))


def add_document_checks(data: pd.DataFrame) -> pd.DataFrame:
    """Flag list-pairing problems and image_id discontinuity per document."""
    data = data.copy()
    data["lists_aligned"] = (
        data["image_id_list"].map(len) == data["page_list"].map(len)
    )
    data["pages_are_1_to_n"] = data["page_list"].map(is_one_to_n)
    data["max_image_id_gap"] = data["image_id_list"].map(max_gap)
    # A large gap means the document's pages are far apart in the original scan,
    # which is the shape of an insert that was never split off. OCR triage only.
    data["image_ids_contiguous"] = data["max_image_id_gap"] == 1
    return data


def add_stack_checks(data: pd.DataFrame) -> pd.DataFrame:
    """Check that each stack's image_ids form a complete 1..N partition."""
    per_stack = (data.groupby("stack_id_key")["image_id_list"]
                 .apply(lambda lists: [i for sub in lists for i in sub]))

    stats = pd.DataFrame({
        "stack_id_key": per_stack.index,
        "n_stack_image_ids": per_stack.map(len).values,
        "n_stack_distinct": per_stack.map(lambda v: len(set(v))).values,
        "stack_partition_complete": per_stack.map(is_one_to_n).values,
    })
    stats["n_documents_in_stack"] = (
        data.groupby("stack_id_key").size().reindex(stats["stack_id_key"]).values
    )
    return data.merge(stats, on="stack_id_key", how="left")


# --- Analyse-DB ------------------------------------------------------------

def load_final_documents(engine: sqlalchemy.Engine, schema: str) -> pd.DataFrame:
    """Read per-document SST and label quality from Analyse-DB."""
    query = sqlalchemy.text(f"""
        SELECT stack_id, process_id, doc_id, subdoc_idx, sst,
               label_tier, training_label_quality
        FROM {schema}.{SOURCE_DOCUMENT_TABLE}
        WHERE stack_id IS NOT NULL AND doc_id IS NOT NULL
    """)
    data = pd.read_sql_query(query, engine)
    data.columns = [str(c).strip().lower() for c in data.columns]

    for column in ["stack_id", "process_id", "doc_id", "subdoc_idx", "sst"]:
        data[column] = data[column].map(clean)
    data["stack_id_key"] = data["stack_id"].map(stack_key)

    has_sst = data["sst"].notna()
    data.loc[has_sst, "sst"] = data.loc[has_sst, "sst"].astype(str).str.upper()
    return data.drop(columns=["stack_id"]).reset_index(drop=True)


def attach_sst(mapping: pd.DataFrame, documents: pd.DataFrame) -> pd.DataFrame:
    """Join the per-document SST onto the mapping rows."""
    return mapping.merge(documents, on=DOCUMENT_KEYS, how="left")


# --- status ----------------------------------------------------------------

def rules(data: pd.DataFrame) -> list[tuple[pd.Series, str]]:
    """Disqualifying conditions, most fundamental first."""
    return [
        (data["sst"].isna() & data["label_tier"].isna(), "NO_FINAL_DOCUMENT"),
        (data["sst"].isna(), "MISSING_SST"),
        (~data["lists_aligned"], "LIST_LENGTH_MISMATCH"),
        (~data["pages_are_1_to_n"], "PAGES_NOT_1_TO_N"),
        (~data["stack_partition_complete"], "STACK_PARTITION_BROKEN"),
    ]


def classify(data: pd.DataFrame) -> pd.DataFrame:
    """Assign the first failing status so the counts form a partition."""
    data = data.copy()
    data["mapping_status"] = OK

    for condition, status in rules(data):
        data.loc[condition & data["mapping_status"].eq(OK), "mapping_status"] = status

    data["is_usable"] = data["mapping_status"].eq(OK)
    data["is_training_eligible"] = (
        data["is_usable"]
        & data["training_label_quality"].isin(["GOLD", "SILVER"])
    )
    return data


def build_documents(mapping: pd.DataFrame, documents: pd.DataFrame) -> pd.DataFrame:
    """Run every check and attach the SST, one row per document."""
    data = add_stack_checks(add_document_checks(mapping))
    return classify(attach_sst(data, documents))


# --- page grain ------------------------------------------------------------

PAGE_COLUMNS = [
    "masterindex_id", "stack_id", "process_id", "doc_id", "subdoc_idx",
    "sst", "training_label_quality", "n_pages", "image_ids_contiguous",
]


def explode_pages(data: pd.DataFrame) -> pd.DataFrame:
    """One row per PDF page, pairing page_number with its image_id."""
    usable = data[data["is_usable"]].copy()
    if usable.empty:
        return pd.DataFrame(columns=PAGE_COLUMNS + ["page_number", "image_id"])

    usable["page_pairs"] = usable.apply(
        lambda row: list(zip(row["page_list"], row["image_id_list"])), axis=1
    )
    pages = usable[PAGE_COLUMNS + ["page_pairs"]].explode("page_pairs")
    pages["page_number"] = pages["page_pairs"].map(lambda pair: pair[0])
    pages["image_id"] = pages["page_pairs"].map(lambda pair: pair[1])
    return pages.drop(columns=["page_pairs"]).reset_index(drop=True)


def add_page_boundaries(pages: pd.DataFrame) -> pd.DataFrame:
    """Mark the first and last page of each document."""
    pages = pages.copy()
    pages["is_first_page"] = pages["page_number"] == 1
    pages["is_last_page"] = pages["page_number"] == pages["n_pages"]
    return pages.sort_values(
        ["stack_id", "masterindex_id", "doc_id", "page_number"]
    ).reset_index(drop=True)


def add_pdf_page_number(pages: pd.DataFrame) -> pd.DataFrame:
    """Position of each page inside its MasterIndex PDF.

    page_number restarts at 1 per document, so when one MID covers several
    documents they have to be laid end to end. Document order is assumed to
    follow the lowest image_id, which is UNVERIFIED — check a multi-document
    MID against its PDF before relying on pdf_page_number.
    """
    pages = pages.copy()
    order = (pages.groupby(["masterindex_id", "doc_id"])["image_id"]
             .min().rename("doc_start").reset_index())
    pages = pages.merge(order, on=["masterindex_id", "doc_id"], how="left")

    pages = pages.sort_values(["masterindex_id", "doc_start", "page_number"])
    pages["pdf_page_number"] = pages.groupby("masterindex_id").cumcount() + 1

    n_docs = (pages.groupby("masterindex_id")["doc_id"]
              .nunique().rename("n_documents_in_mid"))
    pages = pages.merge(n_docs, on="masterindex_id", how="left")
    return pages.drop(columns=["doc_start"]).reset_index(drop=True)


def build_pages(documents: pd.DataFrame) -> pd.DataFrame:
    """Page-level table for usable documents."""
    return add_pdf_page_number(add_page_boundaries(explode_pages(documents)))


# --- output ----------------------------------------------------------------

def status_summary(data: pd.DataFrame) -> pd.DataFrame:
    """Documents, MIDs, stacks and pages per status."""
    return (data.groupby("mapping_status", dropna=False)
            .agg(n_documents=("masterindex_id", "size"),
                 n_masterindex_ids=("masterindex_id", "nunique"),
                 n_stacks=("stack_id_key", "nunique"),
                 n_pages=("n_pages", "sum"))
            .reset_index().sort_values("n_documents", ascending=False))


def report(documents: pd.DataFrame, pages: pd.DataFrame) -> None:
    """Print the status breakdown and the OCR triage counts."""
    print("\nStatus:")
    print(status_summary(documents).to_string(index=False))

    usable = documents[documents["is_usable"]]
    if usable.empty:
        print("\nNo usable documents.")
        return

    n_split = int((~usable["image_ids_contiguous"]).sum())
    print(f"\nUsable documents: {len(usable):,}  pages: {len(pages):,}")
    print(f"Non-contiguous image_ids: {n_split:,} ({n_split / len(usable):.2%}) "
          "— candidates for OCR review.")

    print("\nDocuments per stack:")
    print(usable.drop_duplicates("stack_id_key")["n_documents_in_stack"]
          .value_counts().sort_index().to_string())

    per_mid = pages.drop_duplicates("masterindex_id")["n_documents_in_mid"]
    n_multi = int((per_mid > 1).sum())
    print("\nDocuments per MasterIndex ID:")
    print(per_mid.value_counts().sort_index().to_string())
    if n_multi:
        print(f"{n_multi:,} MIDs ({n_multi / len(per_mid):.2%}) hold several "
              "documents; pdf_page_number assumes they are laid end to end in "
              "image_id order. Verify one against its PDF.")


def write_frame(data: pd.DataFrame, csv_path: Path, parquet_path: Path) -> None:
    """Write one frame as CSV, and as Parquet when the engine is available."""
    data.to_csv(csv_path, index=False, encoding="utf-8-sig")
    try:
        data.to_parquet(parquet_path, index=False)
    except ImportError:
        print(f"WARNING: Parquet skipped for {parquet_path.name}; pyarrow missing.")


def write_outputs(documents: pd.DataFrame, pages: pd.DataFrame) -> None:
    """Write the document table, the page table, and the status summary."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # The list columns do not survive CSV round-trips as lists; keep the raw text.
    write_frame(documents.drop(columns=["image_id_list", "page_list"]),
                DOCUMENTS_CSV, DOCUMENTS_PARQUET)
    write_frame(pages, PAGES_CSV, PAGES_PARQUET)
    status_summary(documents).to_csv(SUMMARY_CSV, index=False, encoding="utf-8-sig")


# --- Snowflake -------------------------------------------------------------

PUBLISH_COLUMNS = [
    "masterindex_id", "stack_id", "process_id", "doc_id", "subdoc_idx",
    "sst", "stack_ssts", "label_tier", "training_label_quality",
    "n_pages", "image_ids", "pages", "image_ids_contiguous", "max_image_id_gap",
]


def load_existing_documents() -> pd.DataFrame:
    """Read the document table that was already built."""
    if not DOCUMENTS_CSV.is_file():
        raise FileNotFoundError(
            f"Not found: {DOCUMENTS_CSV}\nBuild the mapping first."
        )
    data = pd.read_csv(DOCUMENTS_CSV, dtype=str, low_memory=False)
    data["is_usable"] = data["is_usable"].astype(str).str.lower().eq("true")
    return data


def publish_to_snowflake(documents: pd.DataFrame, engine: sqlalchemy.Engine,
                         schema: str) -> None:
    """Replace <schema>.PROC_LIFE_MID_ADB with the usable documents."""
    from snowflake.connector.pandas_tools import pd_writer

    usable = documents[documents["is_usable"]]
    if usable.empty:
        raise RuntimeError("No usable documents to publish.")

    usable = usable[[c for c in PUBLISH_COLUMNS if c in usable.columns]].copy()
    usable.columns = [c.upper() for c in usable.columns]
    usable.to_sql(SNOWFLAKE_OUTPUT_TABLE, con=engine, schema=schema,
                  if_exists="replace", index=False, method=pd_writer)

    with engine.connect() as connection:
        published = connection.execute(sqlalchemy.text(
            f"SELECT COUNT(*) FROM {schema}.{SNOWFLAKE_OUTPUT_TABLE}"
        )).scalar_one()

    if int(published) != len(usable):
        raise RuntimeError(
            f"Row-count mismatch: local {len(usable):,}, published {int(published):,}."
        )
    print(f"Published {len(usable):,} rows to {schema}.{SNOWFLAKE_OUTPUT_TABLE}.")


# --- entry point -----------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Life MID document and page tables.")
    parser.add_argument("--schema", default=DEFAULT_SCHEMA)
    parser.add_argument("--mapping-file", type=Path, default=MAPPING_FILE)
    parser.add_argument("--publish-snowflake", action="store_true")
    parser.add_argument("--yes", action="store_true", help="Skip the publish prompt.")
    return parser.parse_args()


def confirm_publish(schema: str) -> bool:
    """Ask before replacing the Snowflake table."""
    answer = input(f"Replace {schema}.{SNOWFLAKE_OUTPUT_TABLE}? (y/n): ")
    return answer.strip().lower() == "y"


def build(args: argparse.Namespace, engine: sqlalchemy.Engine, schema: str) -> None:
    """Build both tables from the mapping file and write them."""
    mapping = load_mapping(args.mapping_file)
    print(f"Mapping rows: {len(mapping):,}")

    adb_documents = load_final_documents(engine, schema)
    print(f"Analyse-DB documents: {len(adb_documents):,}")

    documents = build_documents(mapping, adb_documents)
    pages = build_pages(documents)

    report(documents, pages)
    write_outputs(documents, pages)
    print(f"\nDocuments: {DOCUMENTS_CSV}")
    print(f"Pages:     {PAGES_CSV}")
    print(f"Summary:   {SUMMARY_CSV}")


def publish(args: argparse.Namespace, engine: sqlalchemy.Engine, schema: str) -> None:
    """Publish the document table that was already built."""
    documents = load_existing_documents()
    if args.yes or confirm_publish(schema):
        publish_to_snowflake(documents, engine, schema)


def main() -> int:
    args = parse_args()
    schema = validate_identifier(args.schema)
    engine = get_engine(schema=schema)

    if args.publish_snowflake:
        publish(args, engine, schema)
    else:
        build(args, engine, schema)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
