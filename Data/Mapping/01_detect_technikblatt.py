#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Detect Technikblatt pages inside A00 documents and label them TB0.

Run from life-docai/Antrag/Datasets/Mapping:

    pixi run -e dev python 01_detect_technikblatt.py

For a short test:

    pixi run -e dev python 01_detect_technikblatt.py --limit 20
"""

from __future__ import annotations

import argparse
import re
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Optional

import pandas as pd
from PIL import Image, ImageOps


MAPPING_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = MAPPING_DIR.parents[2]

PAGES_CSV = MAPPING_DIR / "output" / "life_mid_pages.csv"
RENDER_ROOT = PROJECT_ROOT / "AWS_download_ingestion" / "RenderedPages"
OUTPUT_DIR = MAPPING_DIR / "output"

PAGE_LABELS_CSV = OUTPUT_DIR / "life_page_labels.csv"
OCR_CACHE_CSV = OUTPUT_DIR / "life_header_ocr_cache.csv"
DOCUMENT_REPORT_CSV = OUTPUT_DIR / "life_tb0_documents.csv"

TARGET_SST = "A00"
OCR_ENGINE = "RapidOCR"
HEADER_RATIO = 0.25
CHECKPOINT_EVERY = 100
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
KEYWORD_PATTERN = re.compile(r"\btechnik[\s_-]*blatt\b", re.IGNORECASE)

CACHE_KEY_COLUMNS = [
    "masterindex_id",
    "pdf_page_number",
    "image_path",
    "image_mtime_ns",
    "header_ratio",
    "ocr_engine",
]
CACHE_COLUMNS = CACHE_KEY_COLUMNS + [
    "header_text",
    "ocr_status",
    "ocr_error",
]


def load_pages(path: Path) -> pd.DataFrame:
    """Load the page table produced by the mapping pipeline."""
    if not path.is_file():
        raise FileNotFoundError(f"Page table not found: {path}")

    pages = pd.read_csv(path, low_memory=False)
    required = {
        "masterindex_id",
        "pdf_page_number",
        "sst",
        "stack_id",
        "process_id",
        "doc_id",
        "subdoc_idx",
        "image_id",
        "page_number",
    }
    missing = required - set(pages.columns)
    if missing:
        raise ValueError(f"Missing columns in {path}: {sorted(missing)}")

    pages["masterindex_id"] = pages["masterindex_id"].astype(str)
    pages["pdf_page_number"] = pd.to_numeric(
        pages["pdf_page_number"], errors="coerce"
    ).astype("Int64")
    return pages


def page_number_from_name(path: Path) -> Optional[int]:
    """Read the final number from names such as page_0001.png."""
    numbers = re.findall(r"\d+", path.stem)
    return int(numbers[-1]) if numbers else None


def index_pages(mid_directory: Path) -> dict[int, Path]:
    """Index the rendered pages below one MasterIndex directory, including v1."""
    images = sorted(
        path
        for path in mid_directory.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    page_numbers = [page_number_from_name(path) for path in images]

    if any(number is None for number in page_numbers):
        return {number: path for number, path in enumerate(images, start=1)}

    if len(set(page_numbers)) != len(page_numbers):
        return {number: path for number, path in enumerate(images, start=1)}

    return dict(zip(page_numbers, images))


def find_mid_directories(
    render_root: Path,
    masterindex_ids: set[str],
) -> dict[str, Path]:
    """Find every requested MasterIndex directory in one filesystem scan."""
    if not render_root.is_dir():
        raise FileNotFoundError(f"Rendered pages not found: {render_root}")

    matches: dict[str, list[Path]] = defaultdict(list)
    for path in render_root.rglob("*"):
        if path.is_dir() and path.name in masterindex_ids:
            matches[path.name].append(path)

    ambiguous = {key: paths for key, paths in matches.items() if len(paths) > 1}
    if ambiguous:
        details = "\n".join(
            f"{key}: {', '.join(str(path) for path in paths)}"
            for key, paths in sorted(ambiguous.items())
        )
        raise RuntimeError(f"Several render directories found for the same MID:\n{details}")

    return {key: paths[0] for key, paths in matches.items()}


def build_image_index(
    render_root: Path,
    masterindex_ids: set[str],
) -> dict[str, dict[int, Path]]:
    """Map each requested MasterIndex ID to its rendered page images."""
    directories = find_mid_directories(render_root, masterindex_ids)
    return {
        masterindex_id: index_pages(directory)
        for masterindex_id, directory in directories.items()
    }


def attach_image_paths(
    pages: pd.DataFrame,
    image_index: dict[str, dict[int, Path]],
) -> pd.DataFrame:
    """Attach one rendered image path to every selected page row."""
    pages = pages.copy()
    paths = []

    for row in pages.itertuples():
        if pd.isna(row.pdf_page_number):
            paths.append("")
            continue

        path = image_index.get(row.masterindex_id, {}).get(int(row.pdf_page_number))
        paths.append(str(path) if path else "")

    pages["image_path"] = paths
    return pages


def crop_header(image_path: Path, ratio: float) -> Image.Image:
    """Crop the top part of a rendered page."""
    with Image.open(image_path) as source:
        image = ImageOps.exif_transpose(source).convert("L")
        width, height = image.size
        header = image.crop((0, 0, width, max(1, int(height * ratio))))
    return ImageOps.autocontrast(header)


def create_ocr():
    """Create one local RapidOCR engine."""
    try:
        from rapidocr_onnxruntime import RapidOCR
    except ImportError as error:
        raise RuntimeError(
            "RapidOCR is unavailable. Run this script through the Pixi dev environment."
        ) from error

    print("OCR engine: RapidOCR")
    return RapidOCR()


def run_rapidocr(image: Image.Image, engine) -> str:
    """Extract text from one cropped header."""
    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / "header.png"
        image.save(path)
        result, _ = engine(str(path))

    if not result:
        return ""

    texts = [
        str(item[1])
        for item in result
        if len(item) > 1 and item[1]
    ]
    return " ".join(texts)


def ocr_header(image_path: Path, ratio: float, engine) -> tuple[str, str, Optional[str]]:
    """Return normalized text, status, and an optional error."""
    try:
        text = run_rapidocr(crop_header(image_path, ratio), engine)
    except Exception as error:
        return "", "failed", str(error)

    text = re.sub(r"\s+", " ", text).strip()
    status = "success" if text else "empty"
    return text, status, None


def prepare_target_pages(
    pages: pd.DataFrame,
    ratio: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Keep A00 pages that have a rendered image and add cache metadata."""
    target = pages[pages["sst"].astype(str).eq(TARGET_SST)].copy()
    wanted_ids = set(target["masterindex_id"])
    image_index = build_image_index(RENDER_ROOT, wanted_ids)
    target = attach_image_paths(target, image_index)

    missing = target["image_path"].eq("")
    if missing.any():
        print(
            f"WARNING: {int(missing.sum()):,} A00 pages have no rendered image "
            f"for {target.loc[missing, 'masterindex_id'].nunique():,} MasterIndex IDs."
        )

    available = target[~missing].copy()
    available["image_mtime_ns"] = [
        Path(path).stat().st_mtime_ns for path in available["image_path"]
    ]
    available["header_ratio"] = ratio
    available["ocr_engine"] = OCR_ENGINE
    return target, available


def load_cache(path: Path) -> pd.DataFrame:
    """Load completed OCR work from earlier runs."""
    if not path.is_file():
        return pd.DataFrame(columns=CACHE_COLUMNS)

    cache = pd.read_csv(path, low_memory=False)
    for column in CACHE_COLUMNS:
        if column not in cache.columns:
            cache[column] = pd.NA
    return cache[CACHE_COLUMNS]


def cache_key(row) -> tuple:
    return tuple(getattr(row, column) for column in CACHE_KEY_COLUMNS)


def pending_rows(pages: pd.DataFrame, cache: pd.DataFrame) -> list:
    """Return pages that were not successfully completed before."""
    completed = cache[cache["ocr_status"].isin(["success", "empty"])]
    done = set(completed[CACHE_KEY_COLUMNS].itertuples(index=False, name=None))
    return [row for row in pages.itertuples() if cache_key(row) not in done]


def cache_record(row, engine) -> dict:
    """OCR one page and return one cache record."""
    text, status, error = ocr_header(
        Path(row.image_path),
        float(row.header_ratio),
        engine,
    )
    record = {column: getattr(row, column) for column in CACHE_KEY_COLUMNS}
    record.update(
        header_text=text,
        ocr_status=status,
        ocr_error=error,
    )
    return record


def save_cache(cache: pd.DataFrame, records: list[dict]) -> pd.DataFrame:
    """Merge new OCR records into the cache file."""
    if not records:
        return cache

    merged = pd.concat([cache, pd.DataFrame(records)], ignore_index=True)
    merged = merged.drop_duplicates(CACHE_KEY_COLUMNS, keep="last")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    merged.to_csv(OCR_CACHE_CSV, index=False, encoding="utf-8-sig")
    return merged


def run_ocr(
    pages: pd.DataFrame,
    cache: pd.DataFrame,
    checkpoint_every: int,
) -> pd.DataFrame:
    """OCR pending pages and checkpoint progress for safe resume."""
    pending = pending_rows(pages, cache)
    if not pending:
        print("All available A00 pages are already cached.")
        return cache

    engine = create_ocr()
    print(f"Pages pending OCR: {len(pending):,}")
    records: list[dict] = []

    try:
        for number, row in enumerate(pending, start=1):
            records.append(cache_record(row, engine))
            if number % checkpoint_every == 0:
                cache = save_cache(cache, records)
                records.clear()
                print(f"  {number:,}/{len(pending):,}")
    except BaseException:
        save_cache(cache, records)
        raise

    return save_cache(cache, records)


def keyword_match(text) -> Optional[str]:
    """Return the matched Technikblatt spelling, if present."""
    if not isinstance(text, str):
        return None
    match = KEYWORD_PATTERN.search(text)
    return match.group(0) if match else None


def label_target_pages(
    available: pd.DataFrame,
    cache: pd.DataFrame,
) -> pd.DataFrame:
    """Assign TB0 when RapidOCR text contains Technikblatt."""
    values = CACHE_KEY_COLUMNS + ["header_text", "ocr_status", "ocr_error"]
    labelled = available.merge(
        cache[values],
        on=CACHE_KEY_COLUMNS,
        how="left",
        validate="many_to_one",
    )
    labelled["regex_match"] = labelled["header_text"].map(keyword_match)
    labelled["page_class"] = TARGET_SST
    labelled["page_class_source"] = "MAPPING"

    matched = labelled["regex_match"].notna()
    labelled.loc[matched, "page_class"] = "TB0"
    labelled.loc[matched, "page_class_source"] = "OCR_REGEX"
    return labelled


def merge_page_results(
    pages: pd.DataFrame,
    all_target: pd.DataFrame,
    labelled: pd.DataFrame,
) -> pd.DataFrame:
    """Return every input page, including unscanned G07 and MAD pages."""
    keys = ["masterindex_id", "pdf_page_number"]
    image_paths = all_target[keys + ["image_path"]].drop_duplicates(keys)
    result_columns = [
        "page_class",
        "page_class_source",
        "regex_match",
        "header_text",
        "ocr_status",
        "ocr_error",
    ]

    results = labelled[keys + result_columns]
    merged = pages.merge(image_paths, on=keys, how="left", validate="many_to_one")
    merged = merged.merge(results, on=keys, how="left", validate="many_to_one")
    merged["page_class"] = merged["page_class"].fillna(merged["sst"])
    merged["page_class_source"] = merged["page_class_source"].fillna("MAPPING")

    is_target = merged["sst"].astype(str).eq(TARGET_SST)
    image_missing = is_target & merged["image_path"].fillna("").eq("")
    merged.loc[image_missing, "ocr_status"] = "image_missing"
    merged.loc[~is_target, "ocr_status"] = "not_scanned_non_target"
    merged.loc[
        is_target & ~image_missing & merged["ocr_status"].isna(),
        "ocr_status",
    ] = "not_scanned"
    return merged


def build_document_report(labels: pd.DataFrame) -> pd.DataFrame:
    """Summarize documents containing at least one TB0 page."""
    tb0 = labels[labels["page_class"].eq("TB0")]
    keys = ["stack_id", "process_id", "doc_id", "subdoc_idx", "masterindex_id"]

    if tb0.empty:
        return pd.DataFrame(columns=keys + ["n_tb0_pages"])

    report = tb0.groupby(keys, dropna=False).agg(
        n_tb0_pages=("image_id", "size"),
        tb0_image_ids=("image_id", lambda values: ",".join(map(str, values))),
        tb0_page_numbers=("page_number", lambda values: ",".join(map(str, values))),
    ).reset_index()

    totals = (
        labels.groupby(keys, dropna=False)
        .size()
        .rename("n_document_pages")
        .reset_index()
    )
    report = report.merge(totals, on=keys, how="left")
    report["whole_document_is_tb0"] = (
        report["n_tb0_pages"] == report["n_document_pages"]
    )
    return report


OUTPUT_COLUMNS = [
    "masterindex_id",
    "stack_id",
    "process_id",
    "doc_id",
    "subdoc_idx",
    "image_id",
    "page_number",
    "pdf_page_number",
    "sst",
    "page_class",
    "page_class_source",
    "regex_match",
    "header_text",
    "image_path",
    "ocr_status",
    "ocr_error",
]


def write_outputs(labels: pd.DataFrame, documents: pd.DataFrame) -> None:
    """Write page labels and the TB0 document report."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    columns = [column for column in OUTPUT_COLUMNS if column in labels.columns]
    labels[columns].to_csv(PAGE_LABELS_CSV, index=False, encoding="utf-8-sig")
    documents.to_csv(DOCUMENT_REPORT_CSV, index=False, encoding="utf-8-sig")


def print_report(labels: pd.DataFrame, documents: pd.DataFrame) -> None:
    """Print a short run summary."""
    print("\nPage classes:")
    print(labels["page_class"].value_counts(dropna=False).to_string())
    print("\nOCR status:")
    print(labels["ocr_status"].fillna("unknown").value_counts().to_string())
    print(f"\nTB0 pages: {int(labels['page_class'].eq('TB0').sum()):,}")
    print(f"Documents containing TB0: {len(documents):,}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--header-ratio",
        type=float,
        default=HEADER_RATIO,
        help="Fraction of the page height sent to OCR.",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=CHECKPOINT_EVERY,
        help="Save the OCR cache after this many pages.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="OCR only the first N available A00 pages for a test run.",
    )
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
    print(f"Page rows: {len(pages):,}")
    print(f"Rendered pages root: {RENDER_ROOT}")

    all_target, available = prepare_target_pages(pages, args.header_ratio)
    if args.limit is not None:
        available = available.head(args.limit)
    print(f"A00 pages available for OCR: {len(available):,}")

    cache = load_cache(OCR_CACHE_CSV)
    cache = run_ocr(available, cache, args.checkpoint_every)
    labelled = label_target_pages(available, cache)
    labels = merge_page_results(pages, all_target, labelled)
    documents = build_document_report(labels)

    write_outputs(labels, documents)
    print_report(labels, documents)
    print(f"\nPage labels: {PAGE_LABELS_CSV}")
    print(f"TB0 documents: {DOCUMENT_REPORT_CSV}")
    print(f"OCR cache: {OCR_CACHE_CSV}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
