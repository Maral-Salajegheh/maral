#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Map Life MasterIndex IDs to final Analyse-DB documents.

The MasterIndex file gives masterindex_id -> stack_id. The join is on stack_id
only. A row is SAFE_DIRECT when the MID has one stack, the stack has one MID,
and the stack has one document with one SST. Ambiguous rows are kept in the
audit output, never guessed.

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
MASTERINDEX_FILE = Path(
    "/home/shared_folders/life_ai/mid_mapping/VPSR.JOBW90T.SRMIH300.G0009V00.txt"
)
OUTPUT_DIR = Path(__file__).resolve().parent / "output"

SOURCE_DOCUMENT_TABLE = "PROC_LIFE_FINAL_DOCUMENT_LABELS"
SNOWFLAKE_OUTPUT_TABLE = "PROC_LIFE_MID_ADB"

ALL_CSV = OUTPUT_DIR / "life_masterindex_mapping_all.csv"
ALL_PARQUET = OUTPUT_DIR / "life_masterindex_mapping_all.parquet"
TRAINING_CSV = OUTPUT_DIR / "life_masterindex_training_mapping.csv"
TRAINING_PARQUET = OUTPUT_DIR / "life_masterindex_training_mapping.parquet"
SUMMARY_CSV = OUTPUT_DIR / "life_masterindex_mapping_summary.csv"

SAFE = "SAFE_DIRECT"


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


def parse_time(series: pd.Series) -> pd.Series:
    """Parse mixed timestamp formats without raising."""
    try:
        return pd.to_datetime(series, errors="coerce", format="mixed")
    except TypeError:
        return pd.to_datetime(series, errors="coerce")


# --- MasterIndex -----------------------------------------------------------

def read_raw(path: Path) -> pd.DataFrame:
    """Read the headerless ';' file into columns c1..cN."""
    if not path.is_file():
        raise FileNotFoundError(f"MasterIndex file not found: {path}")
    try:
        raw = pd.read_csv(path, sep=";", header=None, dtype=str,
                          encoding="utf-8-sig", low_memory=False)
    except UnicodeDecodeError:
        raw = pd.read_csv(path, sep=";", header=None, dtype=str,
                          encoding="latin-1", low_memory=False)
    if raw.shape[1] < 6:
        raise ValueError("MasterIndex file needs at least 6 columns (c1,c2,c3,c6).")
    raw.columns = [f"c{i + 1}" for i in range(raw.shape[1])]
    return raw


def name_columns(raw: pd.DataFrame) -> pd.DataFrame:
    """c1=MID, c2=stack_id, c3=timestamp, c6=page count (diagnostic only)."""
    data = pd.DataFrame({
        "masterindex_id": raw["c1"].map(clean),
        "stack_id": raw["c2"].map(clean),
        "masterindex_time_raw": raw["c3"].map(clean),
        "c6_raw": raw["c6"].map(clean),
    })
    data["stack_id_key"] = data["stack_id"].map(stack_key)
    data["masterindex_time"] = parse_time(data["masterindex_time_raw"])
    # C6 is zero-padded, e.g. 000112.
    data["c6_page_count"] = pd.to_numeric(data["c6_raw"], errors="coerce").astype("Int64")
    return data[data["masterindex_id"].notna() & data["stack_id_key"].notna()]


def dedupe_pairs(data: pd.DataFrame) -> pd.DataFrame:
    """One row per (MID, stack), keeping the latest valid timestamp."""
    counts = (data.groupby(["masterindex_id", "stack_id_key"]).size()
              .rename("n_raw_rows_for_pair").reset_index())
    # NaT sorts first, so keep="last" prefers a real timestamp when one exists.
    return (data.sort_values("masterindex_time", na_position="first")
            .drop_duplicates(["masterindex_id", "stack_id_key"], keep="last")
            .merge(counts, on=["masterindex_id", "stack_id_key"], how="left")
            .reset_index(drop=True))


def load_masterindex(path: Path) -> pd.DataFrame:
    """Read and deduplicate the MasterIndex file."""
    return dedupe_pairs(name_columns(read_raw(path)))


# --- Analyse-DB ------------------------------------------------------------

def load_final_documents(engine: sqlalchemy.Engine, schema: str) -> pd.DataFrame:
    """Read all final documents; label quality is applied later."""
    query = sqlalchemy.text(f"""
        SELECT stack_id, process_id, doc_id, subdoc_idx, sst, sfdoc_class,
               verified_by, label_tier, training_label_quality
        FROM {schema}.{SOURCE_DOCUMENT_TABLE}
        WHERE stack_id IS NOT NULL AND doc_id IS NOT NULL
    """)
    data = pd.read_sql_query(query, engine)
    data.columns = [str(c).strip().lower() for c in data.columns]
    return add_document_key(data)


def add_document_key(data: pd.DataFrame) -> pd.DataFrame:
    """Clean the keys and build a globally unique document_key.

    stack_id_key is part of the key because doc_id restarts inside a stack.
    """
    data = data.copy()
    for column in ["stack_id", "process_id", "doc_id", "subdoc_idx", "sst"]:
        data[column] = data[column].map(clean)
    data["stack_id_key"] = data["stack_id"].map(stack_key)

    has_sst = data["sst"].notna()
    data.loc[has_sst, "sst"] = data.loc[has_sst, "sst"].astype(str).str.upper()

    parts = ["stack_id_key", "process_id", "doc_id", "subdoc_idx"]
    data["document_key"] = (
        data[parts].fillna("<NULL>").agg("||".join, axis=1)
    )
    return data.reset_index(drop=True)


# --- mapping ---------------------------------------------------------------

def add_counts(masterindex: pd.DataFrame, documents: pd.DataFrame) -> pd.DataFrame:
    """Join on stack_id and attach the counts the rules need."""
    mids_per_stack = (masterindex.groupby("stack_id_key")["masterindex_id"]
                      .nunique().rename("n_masterindex_ids_for_stack"))
    stacks_per_mid = (masterindex.groupby("masterindex_id")["stack_id_key"]
                      .nunique().rename("n_stacks_for_masterindex_id"))
    docs_per_stack = (documents.groupby("stack_id_key")["document_key"]
                      .nunique().rename("n_final_documents_for_stack"))

    masterindex = (masterindex
                   .merge(mids_per_stack, on="stack_id_key", how="left")
                   .merge(stacks_per_mid, on="masterindex_id", how="left"))
    documents = documents.merge(docs_per_stack, on="stack_id_key", how="left")

    mapping = masterindex.merge(documents, on="stack_id_key", how="left",
                                suffixes=("_masterindex", "_analysedb"))

    sst_per_mid = (mapping[mapping["sst"].notna()].groupby("masterindex_id")["sst"]
                   .nunique().rename("n_sst_for_masterindex_id"))
    mapping = mapping.merge(sst_per_mid, on="masterindex_id", how="left")
    mapping["n_sst_for_masterindex_id"] = (
        mapping["n_sst_for_masterindex_id"].fillna(0).astype(int)
    )
    return mapping


def rules(mapping: pd.DataFrame) -> list[tuple[pd.Series, str, str]]:
    """Disqualifying reasons, most fundamental first."""
    return [
        (mapping["doc_id"].isna(),
         "NO_FINAL_DOCUMENT",
         "No final Analyse-DB document exists for this stack_id."),
        (mapping["n_stacks_for_masterindex_id"] > 1,
         "AMBIGUOUS_MID_MULTIPLE_STACKS",
         "The same masterindex_id occurs with more than one stack_id."),
        (mapping["n_masterindex_ids_for_stack"] > 1,
         "AMBIGUOUS_STACK_MULTIPLE_MIDS",
         "The stack contains more than one MasterIndex ID."),
        (mapping["n_final_documents_for_stack"].fillna(0) > 1,
         "AMBIGUOUS_STACK_MULTIPLE_DOCUMENTS",
         "The stack contains more than one final document."),
        # Cannot fire after the rules above; kept as a guard if they change.
        (mapping["n_sst_for_masterindex_id"] > 1,
         "AMBIGUOUS_MID_MULTIPLE_SST",
         "The MasterIndex ID would inherit more than one SST."),
        (mapping["doc_id"].notna() & mapping["sst"].isna(),
         "MISSING_SST",
         "The resolved final document has no SST."),
    ]


def classify(mapping: pd.DataFrame) -> pd.DataFrame:
    """Assign the first matching status so the counts form a partition."""
    mapping = mapping.copy()
    mapping["mapping_status"] = SAFE
    mapping["mapping_reason"] = "One MID, one stack, one document, one SST."

    for condition, status, reason in rules(mapping):
        target = condition & mapping["mapping_status"].eq(SAFE)
        mapping.loc[target, "mapping_status"] = status
        mapping.loc[target, "mapping_reason"] = reason

    mapping["is_structurally_safe"] = mapping["mapping_status"].eq(SAFE)
    mapping["is_training_eligible"] = (
        mapping["is_structurally_safe"]
        & mapping["training_label_quality"].isin(["GOLD", "SILVER"])
    )
    return mapping


def build_mapping(masterindex: pd.DataFrame, documents: pd.DataFrame) -> pd.DataFrame:
    """Join MasterIndex to documents and classify each row."""
    return classify(add_counts(masterindex, documents))


# --- validation and output -------------------------------------------------

def validate_training_mapping(mapping: pd.DataFrame) -> None:
    """Require the training subset to be a clean 1:1 MID-to-document edge."""
    if mapping.empty:
        raise RuntimeError("Mapping result is empty.")

    training = mapping[mapping["is_training_eligible"]]
    if training.empty:
        print("WARNING: No structurally safe Gold/Silver mappings found.")
        return

    for left, right in [("masterindex_id", "sst"),
                        ("masterindex_id", "document_key"),
                        ("document_key", "masterindex_id")]:
        bad = training.groupby(left)[right].nunique().gt(1)
        if bad.any():
            raise RuntimeError(
                f"Validation failed: {int(bad.sum())} {left} values map to "
                f"multiple {right} values."
            )
    print("Training mapping validation passed.")


def status_summary(mapping: pd.DataFrame) -> pd.DataFrame:
    """Rows, MIDs and stacks per mapping status."""
    return (mapping.groupby("mapping_status", dropna=False)
            .agg(n_rows=("masterindex_id", "size"),
                 n_masterindex_ids=("masterindex_id", "nunique"),
                 n_stacks=("stack_id_key", "nunique"))
            .reset_index().sort_values("n_rows", ascending=False))


def write_frame(data: pd.DataFrame, csv_path: Path, parquet_path: Path) -> None:
    """Write one frame as CSV, and as Parquet when the engine is available."""
    data.to_csv(csv_path, index=False, encoding="utf-8-sig")
    try:
        data.to_parquet(parquet_path, index=False)
    except ImportError:
        print(f"WARNING: Parquet skipped for {parquet_path.name}; pyarrow missing.")


def write_outputs(mapping: pd.DataFrame) -> None:
    """Write the audit output, then validate, then the training subset."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    write_frame(mapping, ALL_CSV, ALL_PARQUET)
    status_summary(mapping).to_csv(SUMMARY_CSV, index=False, encoding="utf-8-sig")

    validate_training_mapping(mapping)

    training = mapping[mapping["is_training_eligible"]]
    if not training.empty:
        write_frame(training, TRAINING_CSV, TRAINING_PARQUET)


def load_existing_mapping() -> pd.DataFrame:
    """Read the mapping that was already built and diagnosed."""
    if not ALL_CSV.is_file():
        raise FileNotFoundError(
            f"Mapping not found: {ALL_CSV}\nRun the mapping and diagnostics first."
        )

    mapping = pd.read_csv(ALL_CSV, low_memory=False)
    mapping["is_structurally_safe"] = (
        mapping["is_structurally_safe"].astype(str).str.lower().eq("true")
    )
    return mapping


PUBLISH_COLUMNS = [
    "masterindex_id", "stack_id_masterindex", "process_id", "doc_id", "subdoc_idx",
    "sst", "sfdoc_class", "verified_by", "label_tier", "training_label_quality",
    "c6_raw", "c6_page_count", "masterindex_time_raw", "masterindex_time",
]


def publish_to_snowflake(mapping: pd.DataFrame, engine: sqlalchemy.Engine,
                         schema: str) -> None:
    """Replace <schema>.PROC_LIFE_MID_ADB with the structurally safe rows."""
    from snowflake.connector.pandas_tools import pd_writer

    safe = mapping[mapping["is_structurally_safe"]]
    if safe.empty:
        raise RuntimeError("No structurally safe rows to publish.")

    safe = safe[[c for c in PUBLISH_COLUMNS if c in safe.columns]].copy()
    safe.columns = [c.upper() for c in safe.columns]
    safe.to_sql(SNOWFLAKE_OUTPUT_TABLE, con=engine, schema=schema,
                if_exists="replace", index=False, method=pd_writer)

    with engine.connect() as connection:
        published = connection.execute(sqlalchemy.text(
            f"SELECT COUNT(*) FROM {schema}.{SNOWFLAKE_OUTPUT_TABLE}"
        )).scalar_one()

    if int(published) != len(safe):
        raise RuntimeError(
            f"Row-count mismatch: local {len(safe):,}, published {int(published):,}."
        )
    print(f"Published {len(safe):,} rows to {schema}.{SNOWFLAKE_OUTPUT_TABLE}.")


# --- entry point -----------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Map Life MasterIndex IDs to documents.")
    parser.add_argument("--schema", default=DEFAULT_SCHEMA)
    parser.add_argument("--masterindex-file", type=Path, default=MASTERINDEX_FILE)
    parser.add_argument("--publish-snowflake", action="store_true")
    parser.add_argument("--yes", action="store_true",
                        help="Skip the publication prompt.")
    return parser.parse_args()


def confirm_publish(schema: str) -> bool:
    """Ask before replacing the Snowflake table."""
    answer = input(f"Replace {schema}.{SNOWFLAKE_OUTPUT_TABLE}? (y/n): ")
    return answer.strip().lower() == "y"


def main() -> int:
    args = parse_args()
    schema = validate_identifier(args.schema)
    engine = get_engine(schema=schema)

    if args.publish_snowflake:
        mapping = load_existing_mapping()
        if args.yes or confirm_publish(schema):
            publish_to_snowflake(mapping, engine, schema)
        return 0

    masterindex = load_masterindex(args.masterindex_file)
    print(f"MID-stack pairs: {len(masterindex):,}")

    documents = load_final_documents(engine, schema)
    print(f"Final documents: {len(documents):,}")

    mapping = build_mapping(masterindex, documents)
    print("\nMapping status:")
    print(mapping["mapping_status"].value_counts(dropna=False).to_string())

    write_outputs(mapping)
    print(f"\nAudit:    {ALL_CSV}")
    if TRAINING_CSV.is_file():
        print(f"Training: {TRAINING_CSV}")
    print(f"Summary:  {SUMMARY_CSV}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())