#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Read the mapping outputs and print the numbers that decide the next step.

    python check_outputs.py
"""

from pathlib import Path

import pandas as pd


OUTPUT_DIR = Path(__file__).resolve().parent / "output"
DOCUMENTS_CSV = OUTPUT_DIR / "life_mid_documents.csv"
PAGES_CSV = OUTPUT_DIR / "life_mid_pages.csv"


def load(path: Path) -> pd.DataFrame:
    """Read one output file."""
    if not path.is_file():
        raise FileNotFoundError(f"Not found: {path}")
    return pd.read_csv(path, low_memory=False)


def report_status(docs: pd.DataFrame) -> None:
    """Status counts. NO_MASTERINDEX_ID is the rest of Analyse-DB, not an error."""
    print("\n--- mapping_status ---")
    print(docs["mapping_status"].value_counts().to_string())


def report_coverage(docs: pd.DataFrame, pages: pd.DataFrame) -> None:
    """How much of the MasterIndex sample came through."""
    ok = docs[docs["is_usable"]]
    print("\n--- coverage ---")
    print(f"usable documents: {len(ok):,}")
    print(f"distinct MIDs:    {ok['masterindex_id'].nunique():,}")
    print(f"page rows:        {len(pages):,}")
    print("\nSST of usable documents:")
    print(ok["sst"].value_counts().to_string())


def report_ocr_scope(docs: pd.DataFrame) -> None:
    """Size of the two OCR problems."""
    ok = docs[docs["is_usable"]]

    print("\n--- Technikblatt scope: A00 documents per stack ---")
    a00 = ok[ok["sst"] == "A00"]
    per_stack = a00.groupby("stack_id")["doc_id"].nunique()
    print(per_stack.value_counts().sort_index().to_string())
    n_multi = int((per_stack > 1).sum())
    print(f"stacks with >1 A00 document: {n_multi:,} of {len(per_stack):,}")

    print("\n--- MAD-inside-A00 scope: non-contiguous image_ids ---")
    print(ok["image_ids_contiguous"].value_counts().to_string())


def main() -> int:
    docs = load(DOCUMENTS_CSV)
    pages = load(PAGES_CSV)

    report_status(docs)
    report_coverage(docs, pages)
    report_ocr_scope(docs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())