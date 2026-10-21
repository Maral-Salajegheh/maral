"""Stage 03: render PDF pages to images (local) and build page_inventory.parquet.

Reads each PDF from the ZIP in memory (no extraction), renders pages with
adaptive zoom, saves images under RENDER_ROOT, and writes the page inventory
locally. Documents are rendered in parallel worker processes (--workers); each
worker opens the ZIP itself and reads one PDF at a time, so memory stays small.
Resumes documents already fully rendered.
"""
from __future__ import annotations

import argparse
import hashlib
import os
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path
from zipfile import ZipFile

import fitz
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

from common import stable_sha256, utc_now, validate_batch_id, write_parquet_records
from config import (
    BLANK_WHITE_PIXEL_RATIO,
    DEFAULT_RENDER_DPI,
    DEFAULT_RENDER_FORMAT,
    DEFAULT_RENDER_VERSION,
    LOCAL_OUTPUT_ROOT,
    LOCAL_TEMP_ROOT,
    LOW_CONTRAST_STDDEV,
    MIN_IMAGE_HEIGHT,
    MIN_IMAGE_WIDTH,
    RENDER_ROOT,
    TARGET_MAX_LONG_SIDE_PX,
    TARGET_MIN_SHORT_SIDE_PX,
)

PAGE_SCHEMA = pa.schema(
    [
        ("batch_id", pa.string()),
        ("masterindex_id", pa.string()),
        ("corpus_document_id", pa.string()),
        ("corpus_page_id", pa.string()),
        ("source_page_number", pa.int32()),
        ("corpus_document_page_count", pa.int32()),
        ("pdf_path_in_zip", pa.string()),
        ("pdf_sha256", pa.string()),
        ("page_width_pt", pa.float32()),
        ("page_height_pt", pa.float32()),
        ("image_path", pa.string()),
        ("image_sha256", pa.string()),
        ("image_width_px", pa.int32()),
        ("image_height_px", pa.int32()),
        ("image_size_bytes", pa.int64()),
        ("quality_status", pa.string()),
        ("render_dpi", pa.float32()),
        ("render_format", pa.string()),
        ("render_config_version", pa.string()),
        ("renderer_name", pa.string()),
        ("renderer_version", pa.string()),
        ("status", pa.string()),
        ("error_code", pa.string()),
        ("error_message", pa.string()),
        ("rendered_at_utc", pa.timestamp("us", tz="UTC")),
    ]
)

PROGRESS_EVERY = 200


def build_parser() -> argparse.ArgumentParser:
    """Command-line arguments for this stage."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-id", required=True)
    parser.add_argument("--zip-path", default=None, help="Defaults to stage 01 output.")
    parser.add_argument("--document-inventory", default=None)
    parser.add_argument("--dpi", type=int, default=DEFAULT_RENDER_DPI)
    parser.add_argument("--format", choices=["png", "jpeg"], default=DEFAULT_RENDER_FORMAT)
    parser.add_argument("--render-version", default=DEFAULT_RENDER_VERSION)
    parser.add_argument("--no-resume", action="store_true", help="Re-render everything.")
    parser.add_argument(
        "--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1),
        help="Parallel render processes (default: CPU count - 1).",
    )
    return parser


def quality_from_gray(width: int, height: int, gray: np.ndarray) -> str:
    """Flag a rendered page as ok / too_small / likely_blank / low_contrast."""
    if width < MIN_IMAGE_WIDTH or height < MIN_IMAGE_HEIGHT:
        return "too_small"
    if (gray >= 250).mean() >= BLANK_WHITE_PIXEL_RATIO:
        return "likely_blank"
    if gray.std() < LOW_CONTRAST_STDDEV:
        return "low_contrast"
    return "ok"


def check_image_quality(pixmap) -> str:
    """Quality flag for a freshly rendered pixmap."""
    if pixmap.width < MIN_IMAGE_WIDTH or pixmap.height < MIN_IMAGE_HEIGHT:
        return "too_small"
    image = Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)
    return quality_from_gray(pixmap.width, pixmap.height, np.asarray(image.convert("L")))


def compute_page_zoom(page_width_pt: float, page_height_pt: float, base_dpi: int) -> float:
    """Zoom for one page: short side upscaled to the minimum, long side capped."""
    base_zoom = base_dpi / 72.0
    if page_width_pt <= 0 or page_height_pt <= 0:
        return base_zoom
    zoom = base_zoom
    short_side = min(page_width_pt, page_height_pt) * zoom
    long_side = max(page_width_pt, page_height_pt) * zoom
    if short_side < TARGET_MIN_SHORT_SIDE_PX:
        zoom *= TARGET_MIN_SHORT_SIDE_PX / short_side
        long_side = max(page_width_pt, page_height_pt) * zoom
    if long_side > TARGET_MAX_LONG_SIDE_PX:
        zoom *= TARGET_MAX_LONG_SIDE_PX / long_side
    return zoom


def page_dir_for(document: dict, batch_id: str, args: argparse.Namespace) -> Path:
    return RENDER_ROOT / batch_id / document["masterindex_id"] / args.render_version


def image_extension(args: argparse.Namespace) -> str:
    return "jpg" if args.format == "jpeg" else "png"


def already_rendered(document: dict, batch_id: str, args: argparse.Namespace) -> bool:
    """True if every page of this document already exists on disk for this
    render version. Lets a re-run skip work even when a previous run was
    interrupted before writing the inventory."""
    page_dir = page_dir_for(document, batch_id, args)
    if not page_dir.is_dir():
        return False
    page_count = document["corpus_page_count"]
    if not page_count:
        return False
    extension = image_extension(args)
    return all(
        (page_dir / f"page_{page_number:04d}.{extension}").is_file()
        for page_number in range(1, page_count + 1)
    )


def new_page_record(document: dict, batch_id: str, args: argparse.Namespace) -> dict:
    """Page row with defaults; per-page fields filled during rendering."""
    return {
        "batch_id": batch_id,
        "masterindex_id": document["masterindex_id"],
        "corpus_document_id": document["corpus_document_id"],
        "corpus_page_id": None,
        "source_page_number": None,
        "corpus_document_page_count": document["corpus_page_count"],
        "pdf_path_in_zip": document["pdf_path_in_zip"],
        "pdf_sha256": document["pdf_sha256"],
        "page_width_pt": None,
        "page_height_pt": None,
        "image_path": None,
        "image_sha256": None,
        "image_width_px": None,
        "image_height_px": None,
        "image_size_bytes": None,
        "quality_status": None,
        "render_dpi": None,
        "render_format": args.format,
        "render_config_version": args.render_version,
        "renderer_name": "PyMuPDF",
        "renderer_version": fitz.VersionBind,
        "status": None,
        "error_code": None,
        "error_message": None,
        "rendered_at_utc": utc_now(),
    }


def rebuild_rows_from_disk(document: dict, batch_id: str, args: argparse.Namespace) -> list[dict]:
    """Reconstruct page rows for a document whose images are already on disk,
    so the inventory is complete without re-rendering. quality_status is
    recomputed from the image; page size in points and render DPI need the
    PDF and stay empty."""
    page_dir = page_dir_for(document, batch_id, args)
    extension = image_extension(args)
    records: list[dict] = []
    for page_number in range(1, document["corpus_page_count"] + 1):
        image_path = page_dir / f"page_{page_number:04d}.{extension}"
        image_bytes = image_path.read_bytes()
        with Image.open(image_path) as img:
            width, height = img.size
            gray = np.asarray(img.convert("L"))
        record = new_page_record(document, batch_id, args)
        record["corpus_page_id"] = stable_sha256(document["corpus_document_id"], page_number)
        record["source_page_number"] = page_number
        record["image_path"] = str(image_path)
        record["image_sha256"] = hashlib.sha256(image_bytes).hexdigest()
        record["image_width_px"] = width
        record["image_height_px"] = height
        record["image_size_bytes"] = len(image_bytes)
        record["quality_status"] = quality_from_gray(width, height, gray)
        record["status"] = "success"
        records.append(record)
    return records


def render_page(pdf, page_index: int, record: dict, image_path: Path, args: argparse.Namespace) -> None:
    """Render one page with adaptive zoom, save it, fill the record."""
    page = pdf.load_page(page_index)
    rect = page.rect
    zoom = compute_page_zoom(rect.width, rect.height, args.dpi)
    pixmap = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)

    image_path.parent.mkdir(parents=True, exist_ok=True)
    image_bytes = pixmap.tobytes("jpeg", jpg_quality=90) if args.format == "jpeg" else pixmap.tobytes("png")
    image_path.write_bytes(image_bytes)

    record.update(
        {
            "page_width_pt": round(rect.width, 2),
            "page_height_pt": round(rect.height, 2),
            "render_dpi": round(zoom * 72.0, 1),
            "image_path": str(image_path),
            "image_sha256": hashlib.sha256(image_bytes).hexdigest(),
            "image_width_px": pixmap.width,
            "image_height_px": pixmap.height,
            "image_size_bytes": len(image_bytes),
            "quality_status": check_image_quality(pixmap),
            "status": "success",
        }
    )


def render_document(document: dict, pdf_bytes: bytes, batch_id: str, args: argparse.Namespace) -> list[dict]:
    """Render all pages of one document; one row per page, errors included."""
    extension = image_extension(args)
    records: list[dict] = []
    pdf = None
    try:
        pdf = fitz.open(stream=pdf_bytes, filetype="pdf")
        if pdf.needs_pass:
            raise RuntimeError("PDF requires a password and cannot be rendered.")
        for page_index in range(pdf.page_count):
            page_number = page_index + 1
            record = new_page_record(document, batch_id, args)
            record["corpus_page_id"] = stable_sha256(document["corpus_document_id"], page_number)
            record["source_page_number"] = page_number
            record["corpus_document_page_count"] = pdf.page_count
            image_path = page_dir_for(document, batch_id, args) / f"page_{page_number:04d}.{extension}"
            try:
                render_page(pdf, page_index, record, image_path, args)
            except Exception as error:
                record["status"] = "error"
                record["error_code"] = error.__class__.__name__
                record["error_message"] = str(error)
            records.append(record)
    except Exception as error:
        record = new_page_record(document, batch_id, args)
        record["status"] = "error"
        record["error_code"] = error.__class__.__name__
        record["error_message"] = str(error)
        records.append(record)
    finally:
        if pdf is not None:
            pdf.close()
    return records


# ---------------------------------------------------------------------------
# Worker process: one open ZIP per process, one document per task
# ---------------------------------------------------------------------------

_WORKER_ZIP: ZipFile | None = None
_WORKER_ARGS: argparse.Namespace | None = None
_WORKER_BATCH_ID: str | None = None


def _init_worker(zip_path: str, args: argparse.Namespace, batch_id: str) -> None:
    global _WORKER_ZIP, _WORKER_ARGS, _WORKER_BATCH_ID
    _WORKER_ZIP = ZipFile(zip_path, "r")
    _WORKER_ARGS = args
    _WORKER_BATCH_ID = batch_id


def _process_document(document: dict) -> tuple[list[dict], bool]:
    """Rows for one document and whether it was resumed from disk."""
    args, batch_id = _WORKER_ARGS, _WORKER_BATCH_ID
    if not args.no_resume and already_rendered(document, batch_id, args):
        return rebuild_rows_from_disk(document, batch_id, args), True
    try:
        pdf_bytes = _WORKER_ZIP.read(document["pdf_path_in_zip"])
    except KeyError:
        record = new_page_record(document, batch_id, args)
        record["status"] = "error"
        record["error_code"] = "pdf_not_found_in_zip"
        record["error_message"] = f"PDF not found: {document['pdf_path_in_zip']}"
        return [record], False
    return render_document(document, pdf_bytes, batch_id, args), False


def main() -> None:
    """Run stage 03 (render, local output)."""
    args = build_parser().parse_args()
    batch_id = validate_batch_id(args.batch_id)
    if args.dpi <= 0 or args.dpi > 1200:
        raise ValueError("dpi must be between 1 and 1200.")
    if args.workers < 1:
        raise ValueError("--workers must be at least 1.")

    batch_root = LOCAL_TEMP_ROOT / batch_id
    zip_path = Path(args.zip_path) if args.zip_path else batch_root / "input" / "source.zip"
    if not zip_path.is_file():
        raise FileNotFoundError(f"ZIP does not exist: {zip_path}")

    inventory_path = (
        Path(args.document_inventory)
        if args.document_inventory
        else LOCAL_OUTPUT_ROOT / batch_id / "document_inventory.parquet"
    )
    if not inventory_path.is_file():
        raise FileNotFoundError(f"Document inventory not found: {inventory_path}")

    documents = [
        r
        for r in pq.read_table(inventory_path).to_pylist()
        if r["status"] == "success" and r["pdf_readable"] and not r["pdf_encrypted"]
    ]
    if not documents:
        raise RuntimeError("Document inventory contains no renderable documents.")

    output_path = LOCAL_OUTPUT_ROOT / batch_id / "page_inventory.parquet"
    print(f"Rendering {len(documents)} documents with {args.workers} workers")

    records: list[dict] = []
    skipped = 0
    with ProcessPoolExecutor(
        max_workers=args.workers,
        mp_context=get_context("spawn"),
        initializer=_init_worker,
        initargs=(str(zip_path), args, batch_id),
    ) as executor:
        # map keeps the inventory order of documents, so the output is deterministic.
        for index, (rows, resumed) in enumerate(executor.map(_process_document, documents, chunksize=1), start=1):
            records.extend(rows)
            skipped += resumed
            # Write the inventory periodically so progress survives an interruption.
            if index % PROGRESS_EVERY == 0:
                write_parquet_records(output_path, records, PAGE_SCHEMA)
                print(f"  ...{index}/{len(documents)} documents processed")

    if skipped:
        print(f"Resumed: {skipped} documents already rendered on disk were skipped.")

    write_parquet_records(output_path, records, PAGE_SCHEMA)
    print(f"Page inventory: {output_path}")
    print(f"Rendered pages: {sum(r['status'] == 'success' for r in records)}")
    print(f"Errors: {sum(r['status'] == 'error' for r in records)}")
    print(f"Images under: {RENDER_ROOT / batch_id}")


if __name__ == "__main__":
    main()