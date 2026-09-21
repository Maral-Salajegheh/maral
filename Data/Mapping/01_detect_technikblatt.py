#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Find Technikblatt pages inside A00 documents and label them TB0.

Reads the page table from 00_map_mid_to_adb.py, keeps the pages of A00
documents, OCRs the header strip of each rendered page, and looks for the
Technikblatt keyword. A matching page keeps its Analyse-DB image_id and doc_id
and gets page_class TB0; every other A00 page keeps A00.

G07 and MAD documents are not scanned: they already carry their own
MasterIndex ID and doc_id in Analyse-DB.

Run this command from the Mapping directory:

    pixi run -e dev python 02_detect_technikblatt.py
"""

from __future__ import annotations

import argparse
import json
import re
import tempfile
from importlib.metadata import version
from pathlib import Path
from typing import Optional

import pandas as pd
from PIL import Image, ImageOps


MAPPING_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = MAPPING_DIR.parent

PAGES_CSV = MAPPING_DIR / "output" / "life_mid_pages.csv"
RENDER_ROOT = PROJECT_ROOT / "AWS_download_ingestion" / "RenderedPages"
OUTPUT_DIR = MAPPING_DIR / "output"

PAGE_LABELS_CSV = OUTPUT_DIR / "life_page_labels.csv"
OCR_CACHE_CSV = OUTPUT_DIR / "life_header_ocr_cache.csv"
DOCUMENT_REPORT_CSV = OUTPUT_DIR / "life_tb0_documents.csv"

TARGET_SST = "A00"
KEYWORD_PATTERN = re.compile(r"\btechnik[\s_-]*blatt\b", re.IGNORECASE)
HEADER_RATIO = 0.25
PADDLE_LANG = "german"
PADDLE_DEVICE = "cpu"
CHECKPOINT_EVERY = 100
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}

CACHE_KEY_COLUMNS = [
    "masterindex_id", "pdf_page_number", "image_path", "image_mtime_ns",
    "header_ratio", "ocr_engine", "ocr_lang", "ocr_device",
]
CACHE_COLUMNS = CACHE_KEY_COLUMNS + ["header_text", "ocr_status", "ocr_error"]


# --- page images -----------------------------------------------------------

def page_number_from_name(path: Path) -> Optional[int]:
    """Last run of digits in the file name, which the renderer uses as the page."""
    matches = re.findall(r"\d+", path.stem)
    return int(matches[-1]) if matches else None


def index_rendered_pages(mid_dir: Path) -> dict[int, Path]:
    """Map page number to image path for one MasterIndex folder.

    Falls back to sorted order when the file names carry no page number.
    """
    images = sorted(p for p in mid_dir.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES)
    numbers = [page_number_from_name(path) for path in images]
    if any(number is None for number in numbers) or len(set(numbers)) != len(numbers):
        return {index: path for index, path in enumerate(images, start=1)}
    return dict(zip(numbers, images))


def find_mid_directory(render_root: Path, masterindex_id: str) -> Optional[Path]:
    """Find one exact MasterIndex directory below RenderedPages."""
    matches = [
        path for path in render_root.rglob("*")
        if path.is_dir() and path.name == masterindex_id
    ]
    if len(matches) > 1:
        locations = "\n".join(str(path) for path in matches)
        raise RuntimeError(
            f"Several render directories found for {masterindex_id}:\n{locations}"
        )
    return matches[0] if matches else None


def build_image_index(render_root: Path,
                      masterindex_ids) -> dict[str, dict[int, Path]]:
    """Map requested MasterIndex IDs to page images, at any nesting depth."""
    if not render_root.is_dir():
        raise FileNotFoundError(f"Rendered pages not found: {render_root}")
    index = {}
    for masterindex_id in sorted({str(value) for value in masterindex_ids}):
        directory = find_mid_directory(render_root, masterindex_id)
        if directory:
            index[masterindex_id] = index_rendered_pages(directory)
    return index


# --- OCR -------------------------------------------------------------------

def crop_header(image_path: Path, ratio: float) -> Image.Image:
    """Top strip of a page, where the form header sits."""
    with Image.open(image_path) as source:
        image = ImageOps.exif_transpose(source).convert("L")
        width, height = image.size
        header = image.crop((0, 0, width, max(1, int(height * ratio))))
    return ImageOps.autocontrast(header)


def create_ocr(lang: str, device: str):
    """Create one PaddleOCR instance and reuse it for every page."""
    try:
        from paddleocr import PaddleOCR
    except ImportError as error:
        raise RuntimeError(
            "PaddleOCR is unavailable. Run this script through the Pixi dev environment."
        ) from error

    try:
        engine = PaddleOCR(
            lang=lang,
            device=device,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
        )
    except TypeError:
        # Compatibility with PaddleOCR 2.x.
        engine = PaddleOCR(
            lang=lang,
            use_angle_cls=False,
            use_gpu=device.lower().startswith("gpu"),
        )

    engine_version = version("paddleocr")
    print(f"OCR engine: PaddleOCR {engine_version}; language={lang}; device={device}")
    return engine, f"PaddleOCR-{engine_version}"


def texts_from_prediction(prediction) -> list[str]:
    """Extract recognized text from a PaddleOCR 3.x prediction."""
    texts: list[str] = []
    for item in prediction or []:
        data = getattr(item, "json", item)
        data = data() if callable(data) else data
        if isinstance(data, str):
            data = json.loads(data)
        if not isinstance(data, dict):
            continue
        result = data.get("res", data)
        texts.extend(str(text) for text in result.get("rec_texts", []) if text)
    return texts


def texts_from_legacy_result(result) -> list[str]:
    """Extract recognized text from a PaddleOCR 2.x result."""
    texts: list[str] = []
    for page in result or []:
        for line in page or []:
            if len(line) > 1 and isinstance(line[1], (list, tuple)) and line[1]:
                texts.append(str(line[1][0]))
    return texts


def run_paddle_ocr(image: Image.Image, engine) -> str:
    """OCR one cropped header with PaddleOCR."""
    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / "header.png"
        image.save(path)
        if hasattr(engine, "predict"):
            texts = texts_from_prediction(engine.predict(str(path)))
        else:
            texts = texts_from_legacy_result(engine.ocr(str(path), cls=False))
    return " ".join(texts)


def ocr_header(image_path: Path, ratio: float = HEADER_RATIO,
               engine=None) -> tuple[str, str, Optional[str]]:
    """Return collapsed header text, OCR status, and any error."""
    try:
        text = run_paddle_ocr(crop_header(image_path, ratio), engine)
    except Exception as error:
        return "", "failed", str(error)
    text = re.sub(r"\s+", " ", text).strip()
    return text, "success" if text else "empty", None


def ocr_one(task: tuple, engine) -> dict:
    """OCR one prepared task and return a cache row.

    ratio and lang travel in the task because default arguments bind at import
    time and would not pick up the command-line values in a worker process.
    """
    (masterindex_id, pdf_page_number, path, mtime_ns, ratio,
     engine_name, lang, device) = task
    text, status, error = ocr_header(Path(path), ratio, engine)
    return {
        "masterindex_id": masterindex_id,
        "pdf_page_number": pdf_page_number,
        "image_path": path,
        "image_mtime_ns": mtime_ns,
        "header_ratio": ratio,
        "ocr_lang": lang,
        "ocr_engine": engine_name,
        "ocr_device": device,
        "header_text": text,
        "ocr_status": status,
        "ocr_error": error,
    }


# --- selection -------------------------------------------------------------

def load_pages(path: Path) -> pd.DataFrame:
    """Read the page table built by 00_map_mid_to_adb.py."""
    if not path.is_file():
        raise FileNotFoundError(f"Page table not found: {path}\nRun 00_map_mid_to_adb.py first.")
    return pd.read_csv(path, low_memory=False)


def select_target_pages(pages: pd.DataFrame) -> pd.DataFrame:
    """Only the pages of A00 documents get scanned."""
    return pages[pages["sst"] == TARGET_SST].copy()


def attach_image_paths(pages: pd.DataFrame,
                       index: dict[str, dict[int, Path]]) -> pd.DataFrame:
    """Resolve each page row to its rendered image."""
    pages = pages.copy()
    pages["image_path"] = [
        str(index.get(str(mid), {}).get(page, ""))
        for mid, page in zip(pages["masterindex_id"], pages["pdf_page_number"])
    ]
    return pages


def attach_ocr_settings(pages: pd.DataFrame, ratio: float, engine_name: str,
                        lang: str, device: str) -> pd.DataFrame:
    """Add the image/configuration signature used by the OCR cache."""
    pages = pages.copy()
    pages["image_mtime_ns"] = [
        Path(path).stat().st_mtime_ns for path in pages["image_path"]
    ]
    pages["header_ratio"] = ratio
    pages["ocr_engine"] = engine_name
    pages["ocr_lang"] = lang
    pages["ocr_device"] = device
    return pages


def report_missing_images(pages: pd.DataFrame) -> pd.DataFrame:
    """Drop rows with no rendered image and say how many were lost."""
    missing = pages["image_path"] == ""
    if missing.any():
        print(f"WARNING: {int(missing.sum()):,} A00 pages have no rendered image "
              f"({pages.loc[missing, 'masterindex_id'].nunique():,} MasterIndex IDs). "
              "They keep page_class A00.")
    return pages[~missing]


# --- cache -----------------------------------------------------------------

def load_cache(path: Path) -> pd.DataFrame:
    """Header text already OCR'd in an earlier run."""
    if not path.is_file():
        return pd.DataFrame(columns=CACHE_COLUMNS)
    cache = pd.read_csv(path, low_memory=False)
    for column in CACHE_COLUMNS:
        if column not in cache:
            cache[column] = pd.NA
    return cache[CACHE_COLUMNS]


def pending_tasks(pages: pd.DataFrame, cache: pd.DataFrame) -> list[tuple]:
    """Pages without successful OCR for the same image and configuration."""
    successful = cache[cache["ocr_status"] == "success"]
    done = set(successful[CACHE_KEY_COLUMNS].itertuples(index=False, name=None))
    return [
        tuple(getattr(row, column) for column in CACHE_KEY_COLUMNS)
        for row in pages.itertuples()
        if tuple(getattr(row, column) for column in CACHE_KEY_COLUMNS) not in done
    ]


def run_ocr(tasks: list, engine, cache: pd.DataFrame, cache_path: Path,
            checkpoint_every: int) -> pd.DataFrame:
    """OCR every pending page, writing the cache as it goes."""
    if not tasks:
        print("All pages already OCR'd; using the cache.")
        return cache

    print(f"PaddleOCR on {len(tasks):,} pages...")
    results: list[dict] = []
    try:
        for done, task in enumerate(tasks, start=1):
            results.append(ocr_one(task, engine))
            if done % checkpoint_every == 0:
                print(f"  {done:,}/{len(tasks):,}")
                cache = save_cache(cache, results, cache_path)
                results.clear()
    except BaseException:
        if results:
            save_cache(cache, results, cache_path)
        raise

    return save_cache(cache, results, cache_path) if results else cache


def save_cache(cache: pd.DataFrame, results: list[dict], path: Path) -> pd.DataFrame:
    """Merge new OCR results into the cache file."""
    merged = pd.concat([cache, pd.DataFrame(results)], ignore_index=True)
    merged = merged.drop_duplicates(CACHE_KEY_COLUMNS, keep="last")
    path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(path, index=False, encoding="utf-8-sig")
    return merged


# --- labelling -------------------------------------------------------------

def find_keyword(text) -> Optional[str]:
    """The matched keyword, or None."""
    if not isinstance(text, str):
        return None
    match = KEYWORD_PATTERN.search(text)
    return match.group(0) if match else None


def apply_labels(pages: pd.DataFrame, cache: pd.DataFrame) -> pd.DataFrame:
    """Assign TB0 to matching pages; every other page keeps its mapping SST."""
    values = CACHE_KEY_COLUMNS + ["header_text", "ocr_status", "ocr_error"]
    labelled = pages.merge(
        cache[values], on=CACHE_KEY_COLUMNS, how="left", validate="many_to_one"
    )
    labelled["regex_match"] = labelled["header_text"].map(find_keyword)

    is_tb0 = labelled["regex_match"].notna()
    labelled["page_class"] = labelled["sst"]
    labelled.loc[is_tb0, "page_class"] = "TB0"
    labelled["page_class_source"] = "MAPPING"
    labelled.loc[is_tb0, "page_class_source"] = "REGEX"
    return labelled


def merge_all_pages(pages: pd.DataFrame, labelled: pd.DataFrame,
                    target_pages: pd.DataFrame) -> pd.DataFrame:
    """Every page row, with the scanned ones carrying their new class."""
    keys = ["masterindex_id", "pdf_page_number"]
    target = target_pages[keys + ["image_path"]].drop_duplicates(keys)
    new_columns = [
        "page_class", "page_class_source", "regex_match", "header_text",
        "ocr_status", "ocr_error",
    ]
    new = labelled[keys + new_columns]
    merged = pages.merge(target, on=keys, how="left", validate="many_to_one")
    merged = merged.merge(new, on=keys, how="left", validate="many_to_one")
    merged["page_class"] = merged["page_class"].fillna(merged["sst"])
    merged["page_class_source"] = merged["page_class_source"].fillna("MAPPING")
    is_target = merged["sst"] == TARGET_SST
    missing_image = is_target & merged["image_path"].fillna("").eq("")
    merged.loc[missing_image, "ocr_status"] = "image_missing"
    merged.loc[~is_target, "ocr_status"] = "not_scanned_non_target"
    merged.loc[
        is_target & ~missing_image & merged["ocr_status"].isna(), "ocr_status"
    ] = "not_scanned"
    return merged


# --- output ----------------------------------------------------------------

OUTPUT_COLUMNS = [
    "masterindex_id", "stack_id", "process_id", "doc_id", "subdoc_idx",
    "image_id", "page_number", "pdf_page_number",
    "sst", "page_class", "page_class_source", "regex_match", "header_text",
    "image_path", "ocr_status", "ocr_error",
]


def document_report(labels: pd.DataFrame) -> pd.DataFrame:
    """Per document: how many of its pages matched, with their image_ids."""
    tb0 = labels[labels["page_class"] == "TB0"]
    if tb0.empty:
        return pd.DataFrame(columns=["stack_id", "doc_id", "n_tb0_pages"])

    keys = ["stack_id", "process_id", "doc_id", "subdoc_idx", "masterindex_id"]
    report = tb0.groupby(keys, dropna=False).agg(
        n_tb0_pages=("image_id", "size"),
        tb0_image_ids=("image_id", lambda v: ",".join(str(x) for x in v)),
        tb0_page_numbers=("page_number", lambda v: ",".join(str(x) for x in v)),
    ).reset_index()

    total = labels.groupby(keys, dropna=False).size().rename("n_document_pages").reset_index()
    report = report.merge(total, on=keys, how="left")
    report["whole_document_is_tb0"] = report["n_tb0_pages"] == report["n_document_pages"]
    return report


def report(labels: pd.DataFrame, documents: pd.DataFrame) -> None:
    """Print what the regex found."""
    scanned = labels[labels["ocr_status"] == "success"]
    n_tb0 = int((labels["page_class"] == "TB0").sum())

    print("\npage_class:")
    print(labels["page_class"].value_counts().to_string())
    print("\nOCR status:")
    print(labels["ocr_status"].fillna("unknown").value_counts().to_string())
    print(f"\nTB0 pages: {n_tb0:,} of {len(scanned):,} successfully OCR'd "
          f"({n_tb0 / max(len(scanned), 1):.2%})")
    print(f"Documents with at least one TB0 page: {len(documents):,}")
    if not documents.empty:
        n_whole = int(documents["whole_document_is_tb0"].sum())
        print(f"  of which entirely TB0: {n_whole:,}")
        print("\nTB0 pages per document:")
        print(documents["n_tb0_pages"].value_counts().sort_index().to_string())


def report_empty_ocr(labels: pd.DataFrame) -> None:
    """Empty header text on most pages means OCR is not working, not that
    the pages are blank."""
    attempted = labels[labels["ocr_status"].isin(["success", "empty"])]
    if attempted.empty:
        return
    empty = attempted["header_text"].fillna("").str.strip().eq("")
    share = empty.mean()
    print(f"\nHeader text empty on {int(empty.sum()):,} of {len(attempted):,} attempted pages "
          f"({share:.2%}).")
    if share > 0.5:
        print("WARNING: more than half the headers are empty. Check the render "
              "quality, the crop ratio, and the PaddleOCR settings before "
              "trusting the TB0 count.")


def write_outputs(labels: pd.DataFrame, documents: pd.DataFrame) -> None:
    """Write the page labels and the per-document report."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    columns = [c for c in OUTPUT_COLUMNS if c in labels.columns]
    labels[columns].to_csv(PAGE_LABELS_CSV, index=False, encoding="utf-8-sig")
    documents.to_csv(DOCUMENT_REPORT_CSV, index=False, encoding="utf-8-sig")


# --- entry point -----------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Label Technikblatt pages inside A00 documents as TB0."
    )
    parser.add_argument("--render-root", type=Path, default=RENDER_ROOT)
    parser.add_argument("--pages-csv", type=Path, default=PAGES_CSV)
    parser.add_argument("--lang", default=PADDLE_LANG)
    parser.add_argument("--device", default=PADDLE_DEVICE,
                        help="PaddleOCR device, for example cpu or gpu:0.")
    parser.add_argument("--header-ratio", type=float, default=HEADER_RATIO,
                        help="Fraction of page height treated as the header.")
    parser.add_argument(
        "--checkpoint-every", type=int, default=CHECKPOINT_EVERY,
        help="Save completed OCR rows after this many pages.",
    )
    parser.add_argument("--limit", type=int, default=None,
                        help="Scan only the first N pages, for a trial run.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not 0 < args.header_ratio <= 1:
        raise ValueError("--header-ratio must be greater than 0 and at most 1.")
    if args.checkpoint_every < 1:
        raise ValueError("--checkpoint-every must be positive.")

    engine, engine_name = create_ocr(args.lang, args.device)

    pages = load_pages(args.pages_csv)
    print(f"Page rows: {len(pages):,}")

    target_rows = select_target_pages(pages)
    index = build_image_index(args.render_root, target_rows["masterindex_id"])
    print(f"Rendered MasterIndex folders: {len(index):,}")

    target_pages = attach_image_paths(target_rows, index)
    target = report_missing_images(target_pages)
    if args.limit:
        target = target.head(args.limit)
    target = attach_ocr_settings(
        target, args.header_ratio, engine_name, args.lang, args.device
    )
    print(f"{TARGET_SST} pages to scan: {len(target):,}")

    cache = load_cache(OCR_CACHE_CSV)
    tasks = pending_tasks(target, cache)
    cache = run_ocr(
        tasks, engine, cache, OCR_CACHE_CSV, args.checkpoint_every
    )

    labels = merge_all_pages(
        pages, apply_labels(target, cache), target_pages
    )
    documents = document_report(labels)

    report_empty_ocr(labels)
    report(labels, documents)
    write_outputs(labels, documents)
    print(f"\nPage labels:   {PAGE_LABELS_CSV}")
    print(f"TB0 documents: {DOCUMENT_REPORT_CSV}")
    print(f"OCR cache:     {OCR_CACHE_CSV}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
