#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Detect Technikblatt pages inside A00 documents and label them TB0.

Run from life-docai/Antrag/Datasets/Mapping:
    pixi run -e dev python 01_detect_technikblatt.py

Short test on complete MasterIndex IDs:
    pixi run -e dev python 01_detect_technikblatt.py --limit 20

Rendered-image locations and physical PDF page numbers come exclusively from
the renderer's page_inventory.parquet; this script does not guess paths.
"""

from __future__ import annotations

import argparse
import re
import tempfile
from difflib import SequenceMatcher
from pathlib import Path
from typing import Optional

import pandas as pd
from PIL import Image, ImageOps


MAPPING_DIR = Path(__file__).resolve().parent
DATASETS_DIR = MAPPING_DIR.parent
BATCH_ID = "life_20260918_A00_G07_MAD_6000_sample_MID"

PAGES_CSV = MAPPING_DIR / "output" / "life_mid_pages.csv"
PAGE_INVENTORY_PARQUET = (
    DATASETS_DIR
    / "AWS_download_ingestion"
    / "output"
    / BATCH_ID
    / "page_inventory.parquet"
)
RENDER_ROOT = DATASETS_DIR / "AWS_download_ingestion" / "RenderedPages"
OUTPUT_DIR = MAPPING_DIR / "output"
PAGE_LABELS_CSV = OUTPUT_DIR / "life_page_labels.csv"
OCR_CACHE_CSV = OUTPUT_DIR / "life_header_ocr_cache.csv"
DOCUMENT_REPORT_CSV = OUTPUT_DIR / "life_tb0_documents.csv"
DEBUG_CROP_DIR = OUTPUT_DIR / "technikblatt_header_crops"

TARGET_SST = "A00"
OCR_ENGINE = "RapidOCR-header-v3-page-inventory"
HEADER_RATIO = 0.35
UPSCALE_FACTOR = 2
KEYWORD = "technikblatt"
FUZZY_THRESHOLD = 0.85
CHECKPOINT_EVERY = 100
KEYWORD_PATTERN = re.compile(r"\btechnik[\s_-]*blatt\b", re.IGNORECASE)

PAGE_KEYS = ["masterindex_id", "source_page_number"]
CACHE_KEYS = [
    "masterindex_id",
    "source_page_number",
    "image_path",
    "image_sha256",
    "header_ratio",
    "ocr_engine",
]
CACHE_COLUMNS = CACHE_KEYS + [
    "header_text",
    "header_crop_path",
    "ocr_status",
    "ocr_error",
]
INVENTORY_METADATA = [
    "batch_id",
    "corpus_document_id",
    "corpus_page_id",
    "corpus_document_page_count",
    "pdf_path_in_zip",
    "pdf_sha256",
    "image_path",
    "image_sha256",
    "render_format",
    "render_config_version",
    "quality_status",
    "status",
]


def normalize_id(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip()


def load_pages(path: Path) -> pd.DataFrame:
    """Load the Analyse-DB page-to-document mapping."""
    if not path.is_file():
        raise FileNotFoundError(f"Page mapping CSV not found: {path}")

    pages = pd.read_csv(path, low_memory=False)
    required = {
        "masterindex_id", "sst", "stack_id", "process_id", "doc_id",
        "subdoc_idx", "image_id", "page_number",
    }
    missing = required - set(pages.columns)
    if missing:
        raise ValueError(f"Missing columns in {path}: {sorted(missing)}")

    pages["masterindex_id"] = normalize_id(pages["masterindex_id"])
    # In Analyse-DB, image_id is the page's physical position in the stack PDF.
    pages["source_page_number"] = pd.to_numeric(
        pages["image_id"], errors="coerce"
    ).astype("Int64")
    invalid = pages["source_page_number"].isna()
    if invalid.any():
        raise ValueError(
            f"{int(invalid.sum()):,} mapping rows have an invalid image_id."
        )
    return pages


def load_page_inventory(path: Path) -> pd.DataFrame:
    """Load the renderer manifest containing exact paths and SHA-256 values."""
    if not path.is_file():
        raise FileNotFoundError(f"Page inventory parquet not found: {path}")

    inventory = pd.read_parquet(path)
    required = {
        "masterindex_id", "source_page_number", "image_path", "image_sha256"
    }
    missing = required - set(inventory.columns)
    if missing:
        raise ValueError(f"Missing columns in {path}: {sorted(missing)}")

    inventory["masterindex_id"] = normalize_id(inventory["masterindex_id"])
    inventory["source_page_number"] = pd.to_numeric(
        inventory["source_page_number"], errors="coerce"
    ).astype("Int64")
    if "batch_id" in inventory:
        inventory = inventory[
            normalize_id(inventory["batch_id"]).eq(BATCH_ID)
        ].copy()
    if "status" in inventory:
        inventory = inventory[
            inventory["status"].astype("string").str.casefold().eq("success")
        ].copy()

    invalid = inventory["source_page_number"].isna()
    if invalid.any():
        raise ValueError(
            f"{int(invalid.sum()):,} inventory rows have an invalid "
            "source_page_number."
        )
    duplicate = inventory.duplicated(PAGE_KEYS, keep=False)
    if duplicate.any():
        examples = inventory.loc[duplicate, PAGE_KEYS].head(10)
        raise ValueError(
            "Duplicate physical page keys in page_inventory.parquet:\n"
            + examples.to_string(index=False)
        )
    return inventory


def select_mids(pages: pd.DataFrame, limit: Optional[int]) -> pd.DataFrame:
    """Select complete MIDs, never unrelated individual rows."""
    if limit is None:
        return pages.copy()
    mids = pages["masterindex_id"].drop_duplicates().head(limit)
    return pages[pages["masterindex_id"].isin(mids)].copy()


def resolve_image_path(row) -> str:
    """Use the inventory path, rebasing only a stale absolute prefix."""
    inventory_path = Path(str(row.image_path))
    if inventory_path.is_file():
        return str(inventory_path)

    version = getattr(row, "render_config_version", None)
    if pd.isna(version) or not str(version).strip():
        version = inventory_path.parent.name

    candidate = (
        RENDER_ROOT
        / BATCH_ID
        / str(row.masterindex_id)
        / str(version)
        / inventory_path.name
    )
    return str(candidate) if candidate.is_file() else str(inventory_path)


def prepare_target_pages(
    pages: pd.DataFrame,
    inventory: pd.DataFrame,
    ratio: float,
    limit: Optional[int],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Select A00 rows from this batch and attach exact inventory paths."""
    a00 = pages[pages["sst"].astype("string").eq(TARGET_SST)].copy()

    # life_mid_pages.csv covers far more data than this 6,000-MID render batch.
    # Restrict to inventory MIDs before applying --limit.
    inventory_mids = set(inventory["masterindex_id"].dropna())
    selected = a00[a00["masterindex_id"].isin(inventory_mids)].copy()
    selected = select_mids(selected, limit)
    selected["selected_for_ocr"] = True

    metadata = [column for column in INVENTORY_METADATA if column in inventory]
    selected = selected.merge(
        inventory[PAGE_KEYS + metadata],
        on=PAGE_KEYS,
        how="left",
        validate="many_to_one",
    )

    # image_path can contain an absolute prefix from the machine/location where
    # rendering ran. Keep it for audit, then rebase only that stale prefix onto
    # the current RenderedPages root; batch/MID/version/filename stay unchanged.
    selected["inventory_image_path"] = selected["image_path"]
    selected["image_path"] = [
        resolve_image_path(row) for row in selected.itertuples()
    ]

    missing_row = selected["image_path"].isna()
    if missing_row.any():
        print(
            f"WARNING: {int(missing_row.sum()):,} selected A00 rows have no "
            "matching page_inventory row."
        )

    selected["image_file_exists"] = selected["image_path"].map(
        lambda value: isinstance(value, str) and Path(value).is_file()
    )
    missing_file = selected["image_path"].notna() & ~selected["image_file_exists"]
    if missing_file.any():
        print(
            f"WARNING: {int(missing_file.sum()):,} inventory image paths do not "
            "exist on this machine."
        )

    available = selected[selected["image_file_exists"]].copy()
    available["header_ratio"] = ratio
    available["ocr_engine"] = OCR_ENGINE
    return selected, available


def crop_header(image_path: Path, ratio: float) -> Image.Image:
    """Crop and enhance the page header sent to OCR."""
    with Image.open(image_path) as source:
        image = ImageOps.exif_transpose(source).convert("L")
        width, height = image.size
        header = image.crop((0, 0, width, max(1, int(height * ratio))))
    header = ImageOps.autocontrast(header)
    return header.resize(
        (header.width * UPSCALE_FACTOR, header.height * UPSCALE_FACTOR),
        Image.Resampling.LANCZOS,
    )


def debug_crop_path(row) -> Path:
    safe_mid = re.sub(r"[^A-Za-z0-9_.-]", "_", str(row.masterindex_id))
    filename = f"page_{int(row.source_page_number):04d}_header.png"
    return DEBUG_CROP_DIR / safe_mid / filename


def create_ocr():
    """Create one local RapidOCR engine from the Pixi dev environment."""
    try:
        from rapidocr_onnxruntime import RapidOCR
    except ImportError as error:
        raise RuntimeError(
            "RapidOCR is unavailable. Run through the Pixi dev environment."
        ) from error
    print("OCR engine: RapidOCR")
    return RapidOCR()


def run_rapidocr(image: Image.Image, engine) -> str:
    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / "header.png"
        image.save(path)
        result, _ = engine(str(path))
    if not result:
        return ""
    return " ".join(str(item[1]) for item in result if len(item) > 1 and item[1])


def ocr_header(
    image_path: Path,
    ratio: float,
    engine,
    crop_path: Optional[Path],
) -> tuple[str, str, Optional[str]]:
    try:
        header = crop_header(image_path, ratio)
        if crop_path is not None:
            crop_path.parent.mkdir(parents=True, exist_ok=True)
            header.save(crop_path)
        text = run_rapidocr(header, engine)
    except Exception as error:
        return "", "failed", str(error)
    text = re.sub(r"\s+", " ", text).strip()
    return text, "success" if text else "empty", None


def empty_cache() -> pd.DataFrame:
    return pd.DataFrame(columns=CACHE_COLUMNS)


def load_cache(path: Path) -> pd.DataFrame:
    """Load only cache files that use the current inventory-based key."""
    if not path.is_file():
        return empty_cache()
    cache = pd.read_csv(path, low_memory=False)
    missing = set(CACHE_COLUMNS) - set(cache.columns)
    if missing:
        print(
            "OCR cache: incompatible old cache ignored. Missing columns: "
            + ", ".join(sorted(missing))
        )
        return empty_cache()
    cache = cache[CACHE_COLUMNS].copy()
    cache["masterindex_id"] = normalize_id(cache["masterindex_id"])
    cache["source_page_number"] = pd.to_numeric(
        cache["source_page_number"], errors="coerce"
    ).astype("Int64")
    return cache


def cache_key(row) -> tuple:
    return tuple(getattr(row, column) for column in CACHE_KEYS)


def pending_rows(
    pages: pd.DataFrame,
    cache: pd.DataFrame,
    save_debug_crops: bool,
) -> list:
    completed = cache[cache["ocr_status"].isin(["success", "empty"])]
    if save_debug_crops:
        crop_exists = completed["header_crop_path"].fillna("").map(
            lambda value: bool(value) and Path(value).is_file()
        ).astype(bool)
        completed = completed.loc[crop_exists]
    done = set(completed[CACHE_KEYS].itertuples(index=False, name=None))
    return [row for row in pages.itertuples() if cache_key(row) not in done]


def cache_record(row, engine, save_debug_crops: bool) -> dict:
    crop_path = debug_crop_path(row) if save_debug_crops else None
    text, status, error = ocr_header(
        Path(row.image_path), float(row.header_ratio), engine, crop_path
    )
    record = {column: getattr(row, column) for column in CACHE_KEYS}
    record.update(
        header_text=text,
        header_crop_path=str(crop_path) if crop_path else "",
        ocr_status=status,
        ocr_error=error,
    )
    return record


def save_cache(cache: pd.DataFrame, records: list[dict]) -> pd.DataFrame:
    if not records:
        return cache
    merged = pd.concat([cache, pd.DataFrame(records)], ignore_index=True)
    merged = merged[CACHE_COLUMNS].drop_duplicates(CACHE_KEYS, keep="last")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    merged.to_csv(OCR_CACHE_CSV, index=False, encoding="utf-8-sig")
    return merged


def run_ocr(
    pages: pd.DataFrame,
    cache: pd.DataFrame,
    checkpoint_every: int,
    save_debug_crops: bool,
) -> pd.DataFrame:
    if pages.empty:
        print("No selected A00 page has an existing rendered image. OCR not started.")
        return cache

    pending = pending_rows(pages, cache, save_debug_crops)
    if not pending:
        print("All available A00 pages are already cached.")
        return cache

    engine = create_ocr()
    print(f"Pages pending OCR: {len(pending):,}")
    records: list[dict] = []
    try:
        for number, row in enumerate(pending, start=1):
            records.append(cache_record(row, engine, save_debug_crops))
            if number % checkpoint_every == 0:
                cache = save_cache(cache, records)
                records.clear()
                print(f"  {number:,}/{len(pending):,}")
    except BaseException:
        save_cache(cache, records)
        raise
    return save_cache(cache, records)


def keyword_match(text) -> tuple[Optional[str], float]:
    if not isinstance(text, str):
        return None, 0.0
    exact = KEYWORD_PATTERN.search(text)
    if exact:
        return exact.group(0), 1.0

    words = re.findall(r"[a-z]+", text.casefold())
    candidates = words + [
        words[index] + words[index + 1] for index in range(len(words) - 1)
    ]
    if not candidates:
        return None, 0.0
    candidate = max(
        candidates,
        key=lambda value: SequenceMatcher(None, value, KEYWORD).ratio(),
    )
    score = SequenceMatcher(None, candidate, KEYWORD).ratio()
    return (candidate, score) if score >= FUZZY_THRESHOLD else (None, score)


def label_target_pages(available: pd.DataFrame, cache: pd.DataFrame) -> pd.DataFrame:
    result_columns = [
        "header_text", "header_crop_path", "ocr_status", "ocr_error"
    ]
    labelled = available.merge(
        cache[CACHE_KEYS + result_columns],
        on=CACHE_KEYS,
        how="left",
        validate="many_to_one",
    )
    matches = labelled["header_text"].map(keyword_match)
    labelled[["regex_match", "match_score"]] = pd.DataFrame(
        matches.tolist(), index=labelled.index
    )
    labelled["page_class"] = TARGET_SST
    labelled["page_class_source"] = "MAPPING"
    matched = labelled["regex_match"].notna()
    labelled.loc[matched, "page_class"] = "TB0"
    labelled.loc[
        matched & labelled["match_score"].eq(1.0), "page_class_source"
    ] = "OCR_EXACT"
    labelled.loc[
        matched & labelled["match_score"].lt(1.0), "page_class_source"
    ] = "OCR_FUZZY"
    return labelled


def merge_page_results(
    pages: pd.DataFrame,
    selected: pd.DataFrame,
    labelled: pd.DataFrame,
) -> pd.DataFrame:
    metadata = [
        column
        for column in INVENTORY_METADATA + [
            "inventory_image_path", "selected_for_ocr", "image_file_exists"
        ]
        if column in selected
    ]
    selected_info = selected[PAGE_KEYS + metadata].drop_duplicates(PAGE_KEYS)
    result_columns = [
        "page_class", "page_class_source", "regex_match", "match_score",
        "header_text", "header_crop_path", "ocr_status", "ocr_error",
    ]
    results = labelled[PAGE_KEYS + result_columns]

    merged = pages.merge(
        selected_info, on=PAGE_KEYS, how="left", validate="many_to_one"
    )
    merged = merged.merge(
        results, on=PAGE_KEYS, how="left", validate="many_to_one"
    )
    merged["page_class"] = merged["page_class"].fillna(merged["sst"])
    merged["page_class_source"] = merged["page_class_source"].fillna("MAPPING")

    is_target = merged["sst"].astype("string").eq(TARGET_SST)
    is_selected = merged["selected_for_ocr"].eq(True)
    has_image = merged["image_file_exists"].eq(True)
    merged.loc[~is_target, "ocr_status"] = "not_scanned_non_target"
    merged.loc[is_target & ~is_selected, "ocr_status"] = "not_scanned"
    merged.loc[is_target & is_selected & ~has_image, "ocr_status"] = "image_missing"
    merged.loc[
        is_target & is_selected & has_image & merged["ocr_status"].isna(),
        "ocr_status",
    ] = "not_scanned"
    return merged


def join_values(values: pd.Series) -> str:
    return ",".join(str(value) for value in values.dropna())


def build_document_report(labels: pd.DataFrame) -> pd.DataFrame:
    keys = ["stack_id", "process_id", "doc_id", "subdoc_idx", "masterindex_id"]
    scanned = labels[
        labels["ocr_status"].isin(["success", "empty", "failed"])
    ].copy()
    if scanned.empty:
        return pd.DataFrame(columns=keys + ["n_scanned_pages", "n_tb0_pages"])

    scanned["is_tb0"] = scanned["page_class"].eq("TB0")
    scanned["tb0_image_id"] = scanned["image_id"].where(scanned["is_tb0"])
    scanned["tb0_document_page_number"] = scanned["page_number"].where(
        scanned["is_tb0"]
    )
    scanned["tb0_source_page_number"] = scanned["source_page_number"].where(
        scanned["is_tb0"]
    )
    aggregations = {
        "n_scanned_pages": ("image_id", "size"),
        "n_tb0_pages": ("is_tb0", "sum"),
        "tb0_image_ids": ("tb0_image_id", join_values),
        "tb0_document_page_numbers": ("tb0_document_page_number", join_values),
        "tb0_source_page_numbers": ("tb0_source_page_number", join_values),
    }
    if "corpus_document_page_count" in scanned:
        aggregations["rendered_pdf_page_count"] = (
            "corpus_document_page_count", "max"
        )
    report = scanned.groupby(keys, dropna=False).agg(**aggregations).reset_index()
    document_totals = (
        labels.groupby(keys, dropna=False).size()
        .rename("n_document_input_pages").reset_index()
    )
    mid_totals = (
        labels.groupby("masterindex_id", dropna=False).size()
        .rename("n_mid_input_pages").reset_index()
    )
    return (
        report.merge(document_totals, on=keys, how="left")
        .merge(mid_totals, on="masterindex_id", how="left")
    )


OUTPUT_COLUMNS = [
    "masterindex_id", "stack_id", "process_id", "doc_id", "subdoc_idx",
    "image_id", "page_number", "source_page_number", "corpus_document_id",
    "corpus_page_id", "corpus_document_page_count", "pdf_path_in_zip",
    "pdf_sha256", "sst", "page_class", "page_class_source", "regex_match",
    "match_score", "header_text", "header_crop_path", "inventory_image_path",
    "image_path",
    "image_sha256", "quality_status", "ocr_status", "ocr_error",
]


def write_outputs(labels: pd.DataFrame, documents: pd.DataFrame) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    columns = [column for column in OUTPUT_COLUMNS if column in labels]
    labels[columns].to_csv(PAGE_LABELS_CSV, index=False, encoding="utf-8-sig")
    documents.to_csv(DOCUMENT_REPORT_CSV, index=False, encoding="utf-8-sig")


def print_report(labels: pd.DataFrame, documents: pd.DataFrame) -> None:
    print("\nPage classes:")
    print(labels["page_class"].value_counts(dropna=False).to_string())
    print("\nOCR status:")
    print(labels["ocr_status"].fillna("unknown").value_counts().to_string())
    print(f"\nTB0 pages: {int(labels['page_class'].eq('TB0').sum()):,}")
    print(f"Scanned document groups: {len(documents):,}")
    count = int(documents["n_tb0_pages"].gt(0).sum()) if not documents.empty else 0
    print(f"Document groups containing TB0: {count:,}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--header-ratio", type=float, default=HEADER_RATIO)
    parser.add_argument("--checkpoint-every", type=int, default=CHECKPOINT_EVERY)
    parser.add_argument(
        "--limit", type=int, default=None,
        help="OCR all A00 rows for the first N MIDs present in this render batch.",
    )
    parser.add_argument("--save-debug-crops", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not 0 < args.header_ratio <= 1:
        raise ValueError("--header-ratio must be greater than 0 and at most 1.")
    if args.checkpoint_every < 1:
        raise ValueError("--checkpoint-every must be positive.")
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be positive.")

    pages = load_pages(PAGES_CSV)
    inventory = load_page_inventory(PAGE_INVENTORY_PARQUET)
    print(f"Page mapping: {PAGES_CSV}")
    print(f"Page inventory: {PAGE_INVENTORY_PARQUET}")
    print(f"Current rendered-pages root: {RENDER_ROOT}")
    print(f"Mapping rows: {len(pages):,}")
    print(f"Rendered inventory rows: {len(inventory):,}")
    print(f"MIDs in rendered inventory: {inventory['masterindex_id'].nunique():,}")

    selected, available = prepare_target_pages(
        pages, inventory, args.header_ratio, args.limit
    )
    selected_mids = set(selected["masterindex_id"])
    rendered_selected = inventory[inventory["masterindex_id"].isin(selected_mids)]
    print(f"MasterIndex IDs selected: {len(selected_mids):,}")
    print(f"Rendered PDF pages for selected MIDs: {len(rendered_selected):,}")
    print(f"A00 mapping pages selected: {len(selected):,}")
    print(f"A00 pages with an existing rendered image: {len(available):,}")

    cache = load_cache(OCR_CACHE_CSV)
    save_debug_crops = args.save_debug_crops or args.limit is not None
    cache = run_ocr(
        available, cache, args.checkpoint_every, save_debug_crops
    )
    labelled = label_target_pages(available, cache)
    labels = merge_page_results(pages, selected, labelled)
    documents = build_document_report(labels)
    write_outputs(labels, documents)
    print_report(labels, documents)
    print(f"\nPage labels: {PAGE_LABELS_CSV}")
    print(f"TB0 documents: {DOCUMENT_REPORT_CSV}")
    print(f"OCR cache: {OCR_CACHE_CSV}")
    if save_debug_crops:
        print(f"Header crops: {DEBUG_CROP_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
