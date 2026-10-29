#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Locate rendered pages that belong to another MID of the same stack.

A MasterIndex PDF can contain a page that Analyse-DB assigns to a different
MID of the same stack, because the page was never physically detached. That
page has no label row in its host MID, so the host has more rendered pages
than label rows and every label after the insertion point is mapped to the
wrong image.

The same scan also exists as a page of its own MID's PDF. Stage 03 writes
image_sha256 as the hash of the rendered image bytes, so if both renderings
are identical the two pages share a hash. This script tests that: for every
MID with more rendered pages than label rows, it looks for pages whose hash
also appears under another MID of the same stack. A MID is RESOLVED when
removing those pages makes the counts balance exactly.

Read-only. Writes one report CSV next to the mapping outputs and changes
nothing else. Lives in datasets/mapping/ because what it tests is the
mapping's page allocation.

    python check_undetached_pages.py
    python check_undetached_pages.py --batch-id <other_batch>
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


MAPPING_DIR = Path(__file__).resolve().parent
INGESTION_DIR = MAPPING_DIR.parent / "aws_download_ingestion"

DEFAULT_BATCH_ID = "antrag_life_A00_G07_MAD"
DEFAULT_PAGES = MAPPING_DIR / "output" / "life_mid_pages.csv"
DEFAULT_REPORT = MAPPING_DIR / "output" / "undetached_page_report.csv"


def default_inventory(batch_id: str) -> Path:
    """Page inventory of one render batch."""
    return INGESTION_DIR / "output" / batch_id / "page_inventory.parquet"

# Written by 03_render_pdf_pages.py (PAGE_SCHEMA).
INVENTORY_COLUMNS = [
    "batch_id", "masterindex_id", "source_page_number", "image_path",
    "image_sha256", "quality_status", "status", "render_dpi",
    "render_config_version", "renderer_version", "corpus_document_page_count",
]
# A blank or near-blank page renders to the same pixels in any PDF, so a hash
# match on one is coincidence, not evidence of a shared scan.
UNRELIABLE_QUALITY = {"likely_blank", "too_small"}


# --- loading ---------------------------------------------------------------

def load_inventory(path: Path) -> pd.DataFrame:
    """Successfully rendered pages, one row per physical page."""
    if not path.is_file():
        raise FileNotFoundError(f"Page inventory not found: {path}")

    data = pd.read_parquet(path)
    data.columns = [str(c).strip().lower() for c in data.columns]

    missing = [c for c in ["masterindex_id", "source_page_number", "image_sha256"]
               if c not in data.columns]
    if missing:
        raise ValueError(
            f"page_inventory is missing columns: {missing}\n"
            f"Available: {sorted(data.columns)}"
        )

    data = data[[c for c in INVENTORY_COLUMNS if c in data.columns]].copy()
    data["masterindex_id"] = data["masterindex_id"].astype(str).str.strip()
    data["source_page_number"] = pd.to_numeric(
        data["source_page_number"], errors="coerce"
    )
    return data


def rendered_only(inventory: pd.DataFrame) -> pd.DataFrame:
    """Drop failed renders; they carry no image and no hash."""
    total = len(inventory)
    data = inventory
    if "status" in data.columns:
        data = data[data["status"] == "success"]
    data = data[data["source_page_number"].notna() & data["image_sha256"].notna()]

    dropped = total - len(data)
    if dropped:
        print(f"Dropped {dropped:,} inventory rows that are not successful renders.")
    return data.reset_index(drop=True)


def load_pages(path: Path) -> pd.DataFrame:
    """Analyse-DB label rows from the mapping output."""
    if not path.is_file():
        raise FileNotFoundError(f"Mapping pages file not found: {path}")

    wanted = ["masterindex_id", "stack_id", "doc_id", "subdoc_idx",
              "pdf_page_number", "sst", "sfdoc_class", "page_class"]
    data = pd.read_csv(path, dtype=str, low_memory=False)
    data.columns = [str(c).strip().lower() for c in data.columns]

    missing = [c for c in ["masterindex_id", "stack_id", "pdf_page_number"]
               if c not in data.columns]
    if missing:
        raise ValueError(f"life_mid_pages.csv is missing columns: {missing}")

    data = data[[c for c in wanted if c in data.columns]].copy()
    data["masterindex_id"] = data["masterindex_id"].astype(str).str.strip()
    data["stack_id_key"] = data["stack_id"].astype(str).str.strip().str.lower()
    data["pdf_page_number"] = pd.to_numeric(data["pdf_page_number"], errors="coerce")
    return data


# --- render consistency ----------------------------------------------------

def check_render_consistency(inventory: pd.DataFrame) -> None:
    """Hash equality only holds when both pages were rendered the same way."""
    print("\n--- render settings ---")
    for column in ["render_config_version", "renderer_version", "render_dpi"]:
        if column not in inventory.columns:
            continue
        values = inventory[column].dropna().unique()
        print(f"{column}: {len(values)} distinct")
        if 0 < len(values) <= 6:
            print(f"  {sorted(map(str, values))}")

    for column in ["render_config_version", "renderer_version"]:
        if column in inventory.columns and inventory[column].nunique(dropna=True) > 1:
            print(f"WARNING: {column} is not uniform. Pages rendered with "
                  "different settings will not share a hash even when the scan "
                  "is the same, so the resolution rate below is a lower bound.")


def check_batches(inventory: pd.DataFrame) -> None:
    """A MID must belong to one batch only."""
    if "batch_id" not in inventory.columns:
        return
    print("\n--- batches ---")
    print(inventory.groupby("batch_id")["masterindex_id"].nunique().to_string())

    per_mid = inventory.groupby("masterindex_id")["batch_id"].nunique()
    crossing = int((per_mid > 1).sum())
    if crossing:
        print(f"WARNING: {crossing:,} MIDs appear in more than one batch.")


# --- counts ----------------------------------------------------------------

def mid_counts(pages: pd.DataFrame, inventory: pd.DataFrame) -> pd.DataFrame:
    """Rendered pages against label rows, per rendered MID."""
    rendered = (inventory.groupby("masterindex_id")
                .size().rename("rendered_pages").reset_index())
    labelled = (pages.groupby("masterindex_id")
                .agg(label_rows=("pdf_page_number", "size"),
                     stack_id_key=("stack_id_key", "first"))
                .reset_index())

    counts = rendered.merge(labelled, on="masterindex_id", how="left")
    counts["label_rows"] = counts["label_rows"].fillna(0).astype(int)
    counts["extra_pages"] = counts["rendered_pages"] - counts["label_rows"]
    return counts


def attach_stack(inventory: pd.DataFrame, pages: pd.DataFrame) -> pd.DataFrame:
    """Give every rendered page the stack of its MID."""
    mid_to_stack = (pages[["masterindex_id", "stack_id_key"]]
                    .drop_duplicates("masterindex_id"))
    return inventory.merge(mid_to_stack, on="masterindex_id", how="left")


# --- foreign-page detection ------------------------------------------------

def reliable_pages(owners: pd.DataFrame) -> pd.DataFrame:
    """Pages whose hash can carry evidence: mapped MID, real content."""
    data = owners[owners["stack_id_key"].notna()]
    if "quality_status" in data.columns:
        before = len(data)
        data = data[~data["quality_status"].isin(UNRELIABLE_QUALITY)]
        skipped = before - len(data)
        if skipped:
            print(f"\nExcluded {skipped:,} blank/too-small pages from hash matching.")
    return data


def find_foreign_pages(owners: pd.DataFrame, suspects: set[str]) -> pd.DataFrame:
    """Pages of a suspect MID whose hash also appears under a sibling MID.

    Only cross-MID matches inside the same stack count. Two identical pages
    within one MID are not evidence of a foreign page.
    """
    scope = reliable_pages(owners)
    pairs = scope.merge(
        scope[["stack_id_key", "image_sha256", "masterindex_id",
               "source_page_number"]],
        on=["stack_id_key", "image_sha256"],
        suffixes=("", "_other"),
    )
    pairs = pairs[pairs["masterindex_id"] != pairs["masterindex_id_other"]]
    pairs = pairs[pairs["masterindex_id"].isin(suspects)]

    return (pairs.sort_values(["masterindex_id", "source_page_number"])
            .drop_duplicates(["masterindex_id", "source_page_number"])
            .reset_index(drop=True))


def resolve_mids(counts: pd.DataFrame, foreign: pd.DataFrame) -> pd.DataFrame:
    """Decide, per suspect MID, whether removing foreign pages balances."""
    found = (foreign.groupby("masterindex_id")
             .agg(n_foreign=("source_page_number", "size"),
                  foreign_pages=("source_page_number",
                                 lambda v: ",".join(str(int(x)) for x in sorted(v))),
                  foreign_owners=("masterindex_id_other",
                                  lambda v: ",".join(sorted(set(v)))))
             .reset_index())

    result = counts.merge(found, on="masterindex_id", how="left")
    result["n_foreign"] = result["n_foreign"].fillna(0).astype(int)
    result["balance"] = (
        result["rendered_pages"] - result["n_foreign"] - result["label_rows"]
    )

    rules = [
        (result["label_rows"].eq(0), "NOT_IN_MAPPING"),
        (result["extra_pages"].lt(0), "MISSING_RENDERED_PAGES"),
        (result["extra_pages"].eq(0), "OK"),
        (result["balance"].eq(0) & result["n_foreign"].gt(0), "RESOLVED"),
        (result["n_foreign"].eq(0), "UNRESOLVED_NO_HASH_MATCH"),
    ]
    result["status"] = "UNRESOLVED_COUNT_MISMATCH"
    assigned = pd.Series(False, index=result.index)
    for condition, status in rules:
        target = condition & ~assigned
        result.loc[target, "status"] = status
        assigned |= target
    return result


# --- reporting -------------------------------------------------------------

def print_population(counts: pd.DataFrame) -> None:
    """Shape of the rendered population."""
    print("\n--- population ---")
    print(f"Rendered MIDs:              {len(counts):,}")
    print(f"Rendered pages:             {int(counts['rendered_pages'].sum()):,}")
    print(f"Label rows in those MIDs:   {int(counts['label_rows'].sum()):,}")


def print_resolution(result: pd.DataFrame) -> None:
    """Status breakdown and the resolution rate."""
    print("\n--- status ---")
    print(result["status"].value_counts().to_string())

    suspects = result[result["extra_pages"] > 0]
    if suspects.empty:
        print("\nNo MID has more rendered pages than label rows.")
        return

    resolved = int(suspects["status"].eq("RESOLVED").sum())
    print(f"\nSuspect MIDs (rendered > rows): {len(suspects):,} "
          f"({len(suspects) / len(result):.2%} of rendered MIDs)")
    print(f"Extra pages in them:            {int(suspects['extra_pages'].sum()):,}")
    print(f"Resolved by hash match:         {resolved:,} "
          f"({resolved / len(suspects):.1%})")
    print("\nExtra pages per suspect MID:")
    print(suspects["extra_pages"].value_counts().sort_index().to_string())


def print_examples(result: pd.DataFrame, pages: pd.DataFrame, n: int = 5) -> None:
    """Resolved MIDs with the position the foreign page sits at."""
    resolved = result[result["status"] == "RESOLVED"].head(n)
    if resolved.empty:
        return

    print("\n--- examples to check against the real PDFs ---")
    for row in resolved.itertuples():
        own = pages[pages["masterindex_id"] == row.masterindex_id]
        classes = (own["page_class"].value_counts().to_dict()
                   if "page_class" in own.columns else {})
        print(f"\n{row.masterindex_id}  stack={row.stack_id_key}")
        print(f"  rendered {row.rendered_pages}, label rows {row.label_rows}")
        print(f"  foreign page(s) {row.foreign_pages} owned by {row.foreign_owners}")
        print(f"  own page_class counts: {classes}")


REPORT_COLUMNS = ["masterindex_id", "stack_id_key", "rendered_pages", "label_rows",
                  "extra_pages", "n_foreign", "balance", "foreign_pages",
                  "foreign_owners", "status"]


def write_report(result: pd.DataFrame, path: Path) -> None:
    """One row per rendered MID."""
    path.parent.mkdir(parents=True, exist_ok=True)
    result[[c for c in REPORT_COLUMNS if c in result.columns]].to_csv(
        path, index=False, encoding="utf-8-sig"
    )
    print(f"\nReport: {path}")


# --- entry point -----------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-id", default=DEFAULT_BATCH_ID)
    parser.add_argument("--pages", type=Path, default=DEFAULT_PAGES)
    parser.add_argument("--inventory", type=Path, default=None,
                        help="Defaults to the batch's page_inventory.parquet.")
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    inventory_path = args.inventory or default_inventory(args.batch_id)
    print(f"Pages:     {args.pages}")
    print(f"Inventory: {inventory_path}")

    inventory = rendered_only(load_inventory(inventory_path))
    pages = load_pages(args.pages)

    check_render_consistency(inventory)
    check_batches(inventory)

    counts = mid_counts(pages, inventory)
    owners = attach_stack(inventory, pages)
    suspects = set(counts.loc[counts["extra_pages"] > 0, "masterindex_id"])
    foreign = find_foreign_pages(owners, suspects)
    result = resolve_mids(counts, foreign)

    print_population(counts)
    print_resolution(result)
    print_examples(result, pages)
    write_report(result, args.report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
