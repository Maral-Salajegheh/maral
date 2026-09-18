#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Join the MasterIndex file to Analyse-DB page labels.

The MasterIndex file has one row per document: stackID, DocID, SubDocId, SST,
MasterindexID. It carries no pages. Analyse-DB knows which image_ids belong to
each document and in which order (seqno), so joining the two gives, per
document, the image_ids in document order and a page number 1..N.

That is the mapping_result_process_level table, and the page-level rows under
it are the training-label seed.

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


def id_key(value) -> Optional[str]:
    """Join key for doc_id / subdoc_idx, which may be zero-padded on one side."""
    text = clean(value)
    if text is None:
        return None
    return str(int(text)) if text.isdigit() else text.lower()


def upper_or_none(value) -> Optional[str]:
    """Upper-case code, or None."""
    text = clean(value)
    return None if text is None else text.upper()


# --- MasterIndex file ------------------------------------------------------

# The delivered header uses its own spelling; map it onto our names.
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
REQUIRED_COLUMNS = ["stack_id", "doc_id", "subdoc_idx", "mid_sst", "masterindex_id"]

SEPARATORS = [",", ";", "\t", "|", ":"]
ENCODINGS = ["utf-8-sig", "latin-1"]


def rename_columns(data: pd.DataFrame) -> pd.DataFrame:
    """Lower-case the headers and map the delivered names onto ours."""
    data.columns = [str(c).strip().lower() for c in data.columns]
    return data.rename(columns=COLUMN_ALIASES)


def read_with(path: Path, separator: str, encoding: str,
              nrows: Optional[int] = None) -> pd.DataFrame:
    """Read the file with one separator/encoding pair."""
    return pd.read_csv(path, sep=separator, dtype=str, encoding=encoding,
                       skipinitialspace=True, low_memory=False, nrows=nrows)


def score_separator(path: Path, separator: str, encoding: str) -> int:
    """How many required columns a separator recovers; -1 when it cannot read."""
    try:
        sample = rename_columns(read_with(path, separator, encoding, nrows=5))
    except Exception:
        return -1
    return sum(column in sample.columns for column in REQUIRED_COLUMNS)


def detect_format(path: Path) -> tuple[str, str]:
    """Pick the separator and encoding that recover the most required columns."""
    best = (-1, None, None)
    for encoding in ENCODINGS:
        for separator in SEPARATORS:
            score = score_separator(path, separator, encoding)
            if score > best[0]:
                best = (score, separator, encoding)

    score, separator, encoding = best
    if score <= 0:
        raise ValueError(
            f"Could not parse {path.name}: no separator in {SEPARATORS} produced "
            f"the expected columns {REQUIRED_COLUMNS}. Check the header row."
        )
    return separator, encoding


def read_mapping_file(path: Path) -> pd.DataFrame:
    """Read the MasterIndex file; the separator is detected for text files."""
    if not path.is_file():
        raise FileNotFoundError(f"MasterIndex file not found: {path}")

    if path.suffix.lower() in {".xlsx", ".xls"}:
        data = rename_columns(pd.read_excel(path, dtype=str))
    else:
        separator, encoding = detect_format(path)
        print(f"MasterIndex separator {separator!r}, encoding {encoding}.")
        data = rename_columns(read_with(path, separator, encoding))

    missing = [c for c in REQUIRED_COLUMNS if c not in data.columns]
    if missing:
        raise ValueError(
            f"MasterIndex file is missing columns: {missing}\n"
            f"Found: {list(data.columns)}"
        )
    return data


def prepare_mapping(data: pd.DataFrame) -> pd.DataFrame:
    """Clean the keys and build the join keys."""
    data = data[REQUIRED_COLUMNS].copy()
    for column in REQUIRED_COLUMNS:
        data[column] = data[column].map(clean)

    data["stack_id_key"] = data["stack_id"].map(stack_key)
    data["doc_id_key"] = data["doc_id"].map(id_key)
    data["subdoc_idx_key"] = data["subdoc_idx"].map(id_key)
    data["mid_sst"] = data["mid_sst"].map(upper_or_none)
    return data.drop_duplicates().reset_index(drop=True)


def load_mapping(path: Path) -> pd.DataFrame:
    """Read and prepare the MasterIndex file."""
    return prepare_mapping(read_mapping_file(path))


# --- Analyse-DB page labels ------------------------------------------------

def load_page_labels(engine: sqlalchemy.Engine, schema: str) -> pd.DataFrame:
    """Read every final export page with its document key and order."""
    query = sqlalchemy.text(f"""
        SELECT stack_id, process_id, doc_id, subdoc_idx,
               image_id, seqno, sst, label_tier, training_label_quality
        FROM {schema}.{SOURCE_PAGE_TABLE}
        WHERE stack_id IS NOT NULL AND image_id IS NOT NULL
    """)
    data = pd.read_sql_query(query, engine)
    data.columns = [str(c).strip().lower() for c in data.columns]
    return prepare_page_labels(data)


def prepare_page_labels(data: pd.DataFrame) -> pd.DataFrame:
    """Clean the keys and make image_id and seqno numeric where possible."""
    data = data.copy()
    for column in ["stack_id", "process_id", "doc_id", "subdoc_idx"]:
        data[column] = data[column].map(clean)

    data["stack_id_key"] = data["stack_id"].map(stack_key)
    data["doc_id_key"] = data["doc_id"].map(id_key)
    data["subdoc_idx_key"] = data["subdoc_idx"].map(id_key)
    data["sst"] = data["sst"].map(upper_or_none)

    # image_id is VARCHAR by design; keep the original and add a numeric view.
    data["image_id_num"] = pd.to_numeric(data["image_id"], errors="coerce")
    data["seqno_num"] = pd.to_numeric(data["seqno"], errors="coerce")
    return data.reset_index(drop=True)


DOCUMENT_KEYS = ["stack_id_key", "process_id", "doc_id_key", "subdoc_idx_key"]


def order_pages(data: pd.DataFrame) -> pd.DataFrame:
    """Sort pages into document order: seqno first, image_id as fallback."""
    return data.sort_values(
        DOCUMENT_KEYS + ["seqno_num", "image_id_num", "image_id"],
        na_position="last",
    )


def add_page_number(data: pd.DataFrame) -> pd.DataFrame:
    """Number each document's pages 1..N in document order."""
    data = order_pages(data).copy()
    data["page_number"] = data.groupby(DOCUMENT_KEYS, dropna=False).cumcount() + 1
    data["n_pages"] = data.groupby(DOCUMENT_KEYS, dropna=False)["page_number"].transform("max")
    return data.reset_index(drop=True)


def aggregate_documents(pages: pd.DataFrame) -> pd.DataFrame:
    """Collapse pages to one row per document, keeping the ordered lists."""
    grouped = pages.groupby(DOCUMENT_KEYS, dropna=False)
    documents = grouped.agg(
        stack_id=("stack_id", "first"),
        doc_id=("doc_id", "first"),
        subdoc_idx=("subdoc_idx", "first"),
        sst=("sst", "first"),
        label_tier=("label_tier", "first"),
        training_label_quality=("training_label_quality", "first"),
        n_pages=("page_number", "max"),
        n_distinct_sst=("sst", "nunique"),
        image_id_list=("image_id", list),
        image_id_num_list=("image_id_num", list),
    ).reset_index()

    documents["image_ids"] = documents["image_id_list"].map(
        lambda values: ",".join(str(v) for v in values)
    )
    documents["pages"] = documents["n_pages"].map(
        lambda n: ",".join(str(i) for i in range(1, int(n) + 1))
    )
    return documents


def build_documents_from_pages(pages: pd.DataFrame) -> pd.DataFrame:
    """Document rows with ordered image_ids, built from the page table."""
    return aggregate_documents(add_page_number(pages))


# --- stack-level context ---------------------------------------------------

def add_stack_ssts(documents: pd.DataFrame) -> pd.DataFrame:
    """Attach the set of SSTs present in each stack."""
    per_stack = (documents[documents["sst"].notna()]
                 .groupby("stack_id_key")["sst"]
                 .apply(lambda values: ", ".join(sorted(set(values))))
                 .rename("stack_ssts"))
    return documents.merge(per_stack, on="stack_id_key", how="left")


def flatten(lists) -> list:
    """One flat list from a column of lists."""
    return [item for sub in lists for item in sub]


def is_one_to_n(values: list) -> bool:
    """True when the values are exactly 1..len(values), each once."""
    numbers = [v for v in values if pd.notna(v)]
    return sorted(numbers) == list(range(1, len(numbers) + 1))


def add_stack_partition(documents: pd.DataFrame) -> pd.DataFrame:
    """Check that each stack's image_ids form a complete 1..N partition."""
    per_stack = (documents.groupby("stack_id_key")["image_id_num_list"]
                 .apply(flatten))
    stats = pd.DataFrame({
        "stack_id_key": per_stack.index,
        "n_stack_pages": per_stack.map(len).values,
        "stack_partition_complete": per_stack.map(is_one_to_n).values,
    })
    stats["n_documents_in_stack"] = (
        documents.groupby("stack_id_key").size().reindex(stats["stack_id_key"]).values
    )
    return documents.merge(stats, on="stack_id_key", how="left")


# --- document checks -------------------------------------------------------

def max_gap(values: list) -> float:
    """Largest jump between consecutive image_ids; 1 means contiguous."""
    numbers = sorted(v for v in values if pd.notna(v))
    if len(numbers) < 2:
        return 1
    return max(b - a for a, b in zip(numbers, numbers[1:]))


def add_document_checks(documents: pd.DataFrame) -> pd.DataFrame:
    """Flag image_id discontinuity per document."""
    documents = documents.copy()
    documents["max_image_id_gap"] = documents["image_id_num_list"].map(max_gap)
    # A large gap means the document's pages sit far apart in the original scan,
    # which is the shape of an insert that was never split off. OCR triage only.
    documents["image_ids_contiguous"] = documents["max_image_id_gap"] == 1
    return documents


def add_sst_comparison(data: pd.DataFrame) -> pd.DataFrame:
    """Compare the MasterIndex SST with the Analyse-DB SST; needs the join."""
    data = data.copy()
    data["sst_matches_mid"] = (
        data["sst"].notna()
        & data["mid_sst"].notna()
        & (data["sst"] == data["mid_sst"])
    )
    return data


# --- join and status -------------------------------------------------------

JOIN_KEYS = ["stack_id_key", "doc_id_key", "subdoc_idx_key"]


def join_mapping(mapping: pd.DataFrame, documents: pd.DataFrame) -> pd.DataFrame:
    """Attach the MasterIndex ID to each Analyse-DB document.

    The MasterIndex file has no process_id, so a stack carrying more than one
    would fan the join out; n_process_ids_for_key records that.
    """
    fan_out = (documents.groupby(JOIN_KEYS)["process_id"]
               .nunique().rename("n_process_ids_for_key").reset_index())
    documents = documents.merge(fan_out, on=JOIN_KEYS, how="left")
    return mapping.merge(documents, on=JOIN_KEYS, how="outer",
                         suffixes=("_mid", ""), indicator="join_side")


def rules(data: pd.DataFrame) -> list[tuple[pd.Series, str]]:
    """Disqualifying conditions, most fundamental first.

    The outer join leaves NaN in the columns from the missing side, so the
    boolean checks use .ne(True) rather than ~, which cannot invert NaN.
    """
    return [
        (data["join_side"] == "left_only", "NO_ANALYSE_DB_PAGES"),
        (data["join_side"] == "right_only", "NO_MASTERINDEX_ID"),
        (data["sst"].isna(), "MISSING_SST"),
        (data["n_distinct_sst"] > 1, "DOCUMENT_MULTIPLE_SST"),
        (data["sst_matches_mid"].ne(True), "SST_MISMATCH"),
        (data["n_process_ids_for_key"] > 1, "AMBIGUOUS_PROCESS_ID"),
        (data["stack_partition_complete"].ne(True), "STACK_PARTITION_BROKEN"),
    ]


def classify(data: pd.DataFrame) -> pd.DataFrame:
    """Assign the first failing status so the counts form a partition."""
    data = data.copy()
    data["mapping_status"] = OK

    for condition, status in rules(data):
        data.loc[condition.fillna(False) & data["mapping_status"].eq(OK),
                 "mapping_status"] = status

    data["is_usable"] = data["mapping_status"].eq(OK)
    data["is_training_eligible"] = (
        data["is_usable"]
        & data["training_label_quality"].isin(["GOLD", "SILVER"])
    )
    return data


def build_mapping(mapping: pd.DataFrame, page_labels: pd.DataFrame) -> pd.DataFrame:
    """Full document table: MasterIndex ID, ordered pages, checks, status."""
    documents = build_documents_from_pages(page_labels)
    documents = add_document_checks(add_stack_partition(add_stack_ssts(documents)))
    return classify(add_sst_comparison(join_mapping(mapping, documents)))


# --- page grain ------------------------------------------------------------

PAGE_CARRY = [
    "masterindex_id", "stack_id", "process_id", "doc_id", "subdoc_idx",
    "sst", "training_label_quality", "n_pages", "image_ids_contiguous",
]


def explode_pages(documents: pd.DataFrame) -> pd.DataFrame:
    """One row per page of every usable document."""
    usable = documents[documents["is_usable"]].copy()
    if usable.empty:
        return pd.DataFrame(columns=PAGE_CARRY + ["page_number", "image_id"])

    usable["page_pairs"] = usable["image_id_list"].map(
        lambda values: list(enumerate(values, start=1))
    )
    pages = usable[PAGE_CARRY + ["page_pairs"]].explode("page_pairs")
    pages["page_number"] = pages["page_pairs"].map(lambda pair: pair[0])
    pages["image_id"] = pages["page_pairs"].map(lambda pair: pair[1])
    return pages.drop(columns=["page_pairs"]).reset_index(drop=True)


def add_page_boundaries(pages: pd.DataFrame) -> pd.DataFrame:
    """Mark the first and last page of each document."""
    pages = pages.copy()
    pages["is_first_page"] = pages["page_number"] == 1
    pages["is_last_page"] = pages["page_number"] == pages["n_pages"]
    return pages


def add_pdf_page_number(pages: pd.DataFrame) -> pd.DataFrame:
    """Position of each page inside its MasterIndex PDF.

    page_number restarts at 1 per document, so when one MID covers several
    documents they have to be laid end to end. Document order is assumed to
    follow the first page's position, which is UNVERIFIED — check a
    multi-document MID against its PDF before relying on pdf_page_number.
    """
    pages = pages.sort_values(["masterindex_id", "doc_id", "page_number"]).copy()
    pages["pdf_page_number"] = pages.groupby("masterindex_id").cumcount() + 1
    n_docs = (pages.groupby("masterindex_id")["doc_id"]
              .nunique().rename("n_documents_in_mid"))
    return pages.merge(n_docs, on="masterindex_id", how="left").reset_index(drop=True)


def build_pages(documents: pd.DataFrame) -> pd.DataFrame:
    """Page-level table for usable documents."""
    return add_pdf_page_number(add_page_boundaries(explode_pages(documents)))


# --- output ----------------------------------------------------------------

def status_summary(data: pd.DataFrame) -> pd.DataFrame:
    """Documents, MIDs, stacks and pages per status."""
    return (data.groupby("mapping_status", dropna=False)
            .agg(n_documents=("mapping_status", "size"),
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

    n_split = int(usable["image_ids_contiguous"].ne(True).sum())
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
              "documents; pdf_page_number assumes they are laid end to end. "
              "Verify one against its PDF.")


DOCUMENT_COLUMNS = [
    "masterindex_id", "stack_id", "process_id", "doc_id", "subdoc_idx",
    "stack_ssts", "sst", "mid_sst", "sst_matches_mid",
    "label_tier", "training_label_quality",
    "n_pages", "image_ids", "pages",
    "image_ids_contiguous", "max_image_id_gap",
    "n_documents_in_stack", "stack_partition_complete",
    "mapping_status", "is_usable", "is_training_eligible",
]


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
    columns = [c for c in DOCUMENT_COLUMNS if c in documents.columns]
    write_frame(documents[columns], DOCUMENTS_CSV, DOCUMENTS_PARQUET)
    write_frame(pages, PAGES_CSV, PAGES_PARQUET)
    status_summary(documents).to_csv(SUMMARY_CSV, index=False, encoding="utf-8-sig")


# --- Snowflake -------------------------------------------------------------

def load_existing_documents() -> pd.DataFrame:
    """Read the document table that was already built."""
    if not DOCUMENTS_CSV.is_file():
        raise FileNotFoundError(f"Not found: {DOCUMENTS_CSV}\nBuild the mapping first.")
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

    usable = usable[[c for c in DOCUMENT_COLUMNS if c in usable.columns]].copy()
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
    parser = argparse.ArgumentParser(
        description="Join the MasterIndex file to Analyse-DB page labels."
    )
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
    """Build both tables and write them."""
    mapping = load_mapping(args.mapping_file)
    print(f"MasterIndex rows: {len(mapping):,}")

    page_labels = load_page_labels(engine, schema)
    print(f"Analyse-DB pages: {len(page_labels):,}")

    documents = build_mapping(mapping, page_labels)
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