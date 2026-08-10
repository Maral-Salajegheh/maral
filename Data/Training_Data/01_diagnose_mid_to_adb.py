#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Diagnostics for the MasterIndex mapping. Reports only, changes nothing.

Reports MIDs per stack, documents per stack, and C6 against the final
Analyse-DB page count on SAFE_DIRECT rows.

    python 01_diagnose_mid_to_adb.py --schema D131_D2D
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd
import sqlalchemy

from life_docai.utils.snowflake_utils import get_engine


DEFAULT_SCHEMA = "D131_D2D"
MAPPING_CSV = Path(
    "/home/shared_folders/life_ai/mid_mapping/output/life_masterindex_mapping_all.csv"
)
SOURCE_PAGE_TABLE = "PROC_LIFE_FINAL_PAGE_LABELS"


def validate_identifier(value: str) -> str:
    """Allow only unquoted Snowflake identifiers."""
    if not re.fullmatch(r"[A-Za-z0-9_]+", value):
        raise ValueError(f"Unsafe Snowflake identifier: {value!r}")
    return value.upper()


def load_mapping(path: Path) -> pd.DataFrame:
    """Read the audit output written by 00_map_mid_to_adb.py."""
    if not path.is_file():
        raise FileNotFoundError(f"Not found: {path}\nRun 00_map_mid_to_adb.py first.")
    return pd.read_csv(path, dtype=str, encoding="utf-8-sig", low_memory=False)


def load_page_counts(engine: sqlalchemy.Engine, schema: str) -> pd.DataFrame:
    """Distinct final export pages per stack, from the unfiltered page table."""
    query = sqlalchemy.text(f"""
        SELECT stack_id, COUNT(DISTINCT image_id) AS analysedb_page_count
        FROM {schema}.{SOURCE_PAGE_TABLE}
        WHERE stack_id IS NOT NULL AND image_id IS NOT NULL
        GROUP BY stack_id
    """)
    data = pd.read_sql_query(query, engine)
    data.columns = [str(c).strip().lower() for c in data.columns]
    data["stack_id_key"] = data["stack_id"].astype(str).str.strip().str.lower()
    data["analysedb_page_count"] = pd.to_numeric(
        data["analysedb_page_count"], errors="coerce"
    )
    return data[["stack_id_key", "analysedb_page_count"]]


def report_counts(counts: pd.Series, title: str, unit: str) -> None:
    """Print how many stacks hold more than one of something, plus the shape."""
    print(f"\n--- {title} ---")
    if counts.empty:
        print("No rows.")
        return
    n_multi = int((counts > 1).sum())
    print(f"Stacks:      {len(counts):,}")
    print(f"With >1 {unit}: {n_multi:,} ({n_multi / len(counts):.2%})")
    print(counts.value_counts().sort_index().to_string())


def report_mids_per_stack(mapping: pd.DataFrame) -> None:
    """Distinct MasterIndex IDs per stack."""
    pairs = mapping[["stack_id_key", "masterindex_id"]].dropna().drop_duplicates()
    counts = pairs.groupby("stack_id_key")["masterindex_id"].nunique()
    report_counts(counts, "MasterIndex IDs per stack", "MID")


def report_documents_per_stack(mapping: pd.DataFrame) -> None:
    """Final Analyse-DB documents per mapped stack."""
    resolved = mapping[mapping["doc_id"].notna() & mapping["stack_id_key"].notna()]
    per_stack = resolved.drop_duplicates("stack_id_key")
    counts = pd.to_numeric(
        per_stack.set_index("stack_id_key")["n_final_documents_for_stack"],
        errors="coerce",
    ).dropna()
    report_counts(counts, "Final documents per stack", "document")


def safe_rows_with_page_counts(mapping: pd.DataFrame,
                               page_counts: pd.DataFrame) -> pd.DataFrame:
    """SAFE_DIRECT rows joined to their stack page count, with the C6 delta."""
    safe = mapping[
        mapping["is_structurally_safe"].astype(str).str.lower().eq("true")
    ]
    data = safe.merge(page_counts, on="stack_id_key", how="left")
    data["c6"] = pd.to_numeric(data["c6_page_count"], errors="coerce")
    data = data[data["c6"].notna() & data["analysedb_page_count"].notna()].copy()
    data["delta"] = data["c6"] - data["analysedb_page_count"]
    return data


def report_c6_vs_page_count(mapping: pd.DataFrame, page_counts: pd.DataFrame) -> None:
    """C6 against the realised page count on safe rows."""
    print("\n--- C6 vs final Analyse-DB page count ---")
    data = safe_rows_with_page_counts(mapping, page_counts)
    if data.empty:
        print("No comparable SAFE_DIRECT rows.")
        return

    n_exact = int((data["delta"] == 0).sum())
    print(f"Comparable rows: {len(data):,}")
    print(f"Exact matches:   {n_exact:,} ({n_exact / len(data):.2%})")
    print("\nDelta (C6 - page count):")
    print(data["delta"].value_counts().sort_index().to_string())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose the MasterIndex mapping.")
    parser.add_argument("--schema", default=DEFAULT_SCHEMA)
    parser.add_argument("--mapping-csv", type=Path, default=MAPPING_CSV)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    schema = validate_identifier(args.schema)

    mapping = load_mapping(args.mapping_csv)
    print(f"Mapping rows: {len(mapping):,}")

    report_mids_per_stack(mapping)
    report_documents_per_stack(mapping)
    report_c6_vs_page_count(mapping, load_page_counts(get_engine(schema=schema), schema))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
