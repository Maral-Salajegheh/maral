"""Stage 03: render PDF pages to images (local) and build page_inventory.parquet.

Reads each PDF from the ZIP in memory (no extraction), renders pages with
adaptive zoom, saves images under RENDER_ROOT, and writes the page inventory
locally. Resumes documents already fully rendered.
"""
from __future__ import annotations

import argparse
import hashlib
from io import BytesIO
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
    return parser


def check_image_quality(pixmap) -> str:
    """Flag a rendered page as ok / too_small / likely_blank / low_contrast."""
    if pixmap.width < MIN_IMAGE_WIDTH or pixmap.height < MIN_IMAGE_HEIGHT:
        return "too_small"
    image = Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)
    gray = np.asarray(image.convert("L"))
    if (gray >= 250).mean() >= BLANK_WHITE_PIXEL_RATIO:
        return "likely_blank"
    if gray.std() < LOW_CONTRAST_STDDEV:
        return "low_contrast"
    return "ok"


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


def load_pdf_bytes(zip_path: Path) -> dict[str, bytes]:
    """Map every PDF's member path (as stored in the ZIP) to its bytes."""
    result: dict[str, bytes] = {}
    with ZipFile(zip_path, "r") as archive:
        for info in archive.infolist():
            if not info.is_dir() and info.filename.lower().endswith(".pdf"):
                result[info.filename] = archive.read(info)
    return result


def load_resumable_rows(previous: Path, image_format: str, render_version: str) -> dict[str, list[dict]]:
    """Previous rows per document, only for fully successful, same-config documents."""
    rows = pq.read_table(previous).to_pylist()
    by_document: dict[str, list[dict]] = {}
    for row in rows:
        by_document.setdefault(row["corpus_document_id"], []).append(row)
    resumable: dict[str, list[dict]] = {}
    for document_id, doc_rows in by_document.items():
        complete = all(
            r["status"] == "success"
            and r["render_format"] == image_format
            and r["render_config_version"] == render_version
            for r in doc_rows
        ) and len(doc_rows) == doc_rows[0]["corpus_document_page_count"]
        if complete:
            resumable[document_id] = doc_rows
    return resumable


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
    extension = "jpg" if args.format == "jpeg" else "png"
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
            image_path = (
                RENDER_ROOT / batch_id / document["masterindex_id"]
                / args.render_version / f"page_{page_number:04d}.{extension}"
            )
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


def main() -> None:
    """Run stage 03 (render, local output)."""
    args = build_parser().parse_args()
    batch_id = validate_batch_id(args.batch_id)
    if args.dpi <= 0 or args.dpi > 1200:
        raise ValueError("dpi must be between 1 and 1200.")

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
    resumable: dict[str, list[dict]] = {}
    if output_path.is_file() and not args.no_resume:
        resumable = load_resumable_rows(output_path, args.format, args.render_version)
        current_hash = {d["corpus_document_id"]: d["pdf_sha256"] for d in documents}
        resumable = {
            doc_id: rows
            for doc_id, rows in resumable.items()
            if doc_id in current_hash and rows[0]["pdf_sha256"] == current_hash[doc_id]
        }
        if resumable:
            print(f"Resuming: skipping {len(resumable)} completed documents.")

    pdf_map = load_pdf_bytes(zip_path)
    records: list[dict] = []
    for document in documents:
        if document["corpus_document_id"] in resumable:
            records.extend(resumable[document["corpus_document_id"]])
            continue
        pdf_bytes = pdf_map.get(document["pdf_path_in_zip"])
        if pdf_bytes is None:
            record = new_page_record(document, batch_id, args)
            record["status"] = "error"
            record["error_code"] = "pdf_not_found_in_zip"
            record["error_message"] = f"PDF not found: {document['pdf_path_in_zip']}"
            records.append(record)
            continue
        records.extend(render_document(document, pdf_bytes, batch_id, args))

    write_parquet_records(output_path, records, PAGE_SCHEMA)
    print(f"Page inventory: {output_path}")
    print(f"Rendered pages: {sum(r['status'] == 'success' for r in records)}")
    print(f"Errors: {sum(r['status'] == 'error' for r in records)}")
    print(f"Images under: {RENDER_ROOT / batch_id}")


if __name__ == "__main__":
    main()