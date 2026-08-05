"""Stage 04: render readable PDFs to page images with adaptive zoom, upload
them to S3, and write page_inventory.parquet. Resumes completed documents from
a previous run. Requires the staged PDFs from stage 02."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import boto3
import fitz
import pyarrow as pa
import pyarrow.parquet as pq

from common import stable_sha256, utc_now, validate_batch_id, write_parquet_records
from config import (
    AWS_REGION,
    CORPUS_ROOT_PREFIX,
    DEFAULT_RENDER_DPI,
    DEFAULT_RENDER_FORMAT,
    DEFAULT_RENDER_VERSION,
    LOCAL_TEMP_ROOT,
    PROJECT_BUCKET,
    TARGET_MAX_LONG_SIDE_PX,
    TARGET_MIN_SHORT_SIDE_PX,
)

PAGE_SCHEMA = pa.schema(
    [
        ("batch_id", pa.string()),
        ("masterindex_id", pa.string()),
        ("document_id", pa.string()),
        ("page_id", pa.string()),
        ("source_page_number", pa.int32()),
        ("document_page_count", pa.int32()),
        ("pdf_s3_key", pa.string()),
        ("pdf_sha256", pa.string()),
        ("page_width_pt", pa.float32()),
        ("page_height_pt", pa.float32()),
        ("image_s3_key", pa.string()),
        ("image_sha256", pa.string()),
        ("image_width_px", pa.int32()),
        ("image_height_px", pa.int32()),
        ("image_size_bytes", pa.int64()),
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
    parser.add_argument("--document-inventory", default=None)
    parser.add_argument("--dpi", type=int, default=DEFAULT_RENDER_DPI)
    parser.add_argument("--format", choices=["png", "jpeg"], default=DEFAULT_RENDER_FORMAT)
    parser.add_argument("--render-version", default=DEFAULT_RENDER_VERSION)
    parser.add_argument("--no-resume", action="store_true", help="Re-render everything.")
    return parser


def compute_page_zoom(page_width_pt: float, page_height_pt: float, base_dpi: int) -> float:
    """Zoom for one page: base DPI, short side upscaled to the minimum target,
    long side capped at the maximum target. Deterministic per page geometry."""
    base_zoom = base_dpi / 72.0
    if page_width_pt <= 0 or page_height_pt <= 0:
        return base_zoom
    zoom = base_zoom
    short_side_px = min(page_width_pt, page_height_pt) * zoom
    long_side_px = max(page_width_pt, page_height_pt) * zoom
    if short_side_px < TARGET_MIN_SHORT_SIDE_PX:
        zoom *= TARGET_MIN_SHORT_SIDE_PX / short_side_px
        long_side_px = max(page_width_pt, page_height_pt) * zoom
    if long_side_px > TARGET_MAX_LONG_SIDE_PX:
        zoom *= TARGET_MAX_LONG_SIDE_PX / long_side_px
    return zoom


def staged_pdf_path(batch_root: Path, pdf_s3_key: str, batch_id: str) -> Path:
    """Stage-02 staging path of a PDF, derived from its S3 key."""
    prefix = f"{CORPUS_ROOT_PREFIX}/unpacked/{batch_id}/"
    if not pdf_s3_key.startswith(prefix):
        raise ValueError(f"Unexpected PDF S3 key layout: {pdf_s3_key}")
    return batch_root / "extracted" / pdf_s3_key[len(prefix):]


def load_resumable_rows(
    previous_inventory: Path, image_format: str, render_version: str
) -> dict[str, list[dict]]:
    """Previous rows per document_id, only for fully successful documents with
    the same render configuration (version + format; DPI is per-page)."""
    rows = pq.read_table(previous_inventory).to_pylist()
    by_document: dict[str, list[dict]] = {}
    for row in rows:
        by_document.setdefault(row["document_id"], []).append(row)
    resumable: dict[str, list[dict]] = {}
    for document_id, document_rows in by_document.items():
        complete = all(
            r["status"] == "success"
            and r["render_format"] == image_format
            and r["render_config_version"] == render_version
            for r in document_rows
        ) and len(document_rows) == document_rows[0]["document_page_count"]
        if complete:
            resumable[document_id] = document_rows
    return resumable


def filter_unchanged_pdfs(
    resumable: dict[str, list[dict]], documents: list[dict]
) -> dict[str, list[dict]]:
    """Keep only resumable documents whose PDF hash matches the inventory."""
    current_hash = {d["document_id"]: d["pdf_sha256"] for d in documents}
    return {
        document_id: rows
        for document_id, rows in resumable.items()
        if document_id in current_hash
        and rows[0]["pdf_sha256"] == current_hash[document_id]
    }


def new_page_record(document: dict, batch_id: str, args: argparse.Namespace) -> dict:
    """Page row with defaults; per-page fields are filled during rendering."""
    return {
        "batch_id": batch_id,
        "masterindex_id": document["masterindex_id"],
        "document_id": document["document_id"],
        "page_id": None,
        "source_page_number": None,
        "document_page_count": document["page_count"],
        "pdf_s3_key": document["pdf_s3_key"],
        "pdf_sha256": document["pdf_sha256"],
        "page_width_pt": None,
        "page_height_pt": None,
        "image_s3_key": None,
        "image_sha256": None,
        "image_width_px": None,
        "image_height_px": None,
        "image_size_bytes": None,
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


def render_page(
    s3_client,
    pdf,
    page_index: int,
    record: dict,
    image_key: str,
    args: argparse.Namespace,
) -> None:
    """Render one page with adaptive zoom, upload the image, fill the record."""
    page = pdf.load_page(page_index)
    page_rect = page.rect
    zoom = compute_page_zoom(page_rect.width, page_rect.height, args.dpi)
    pixmap = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)

    if args.format == "jpeg":
        image_bytes = pixmap.tobytes("jpeg", jpg_quality=90)
        content_type = "image/jpeg"
    else:
        image_bytes = pixmap.tobytes("png")
        content_type = "image/png"

    image_sha256 = hashlib.sha256(image_bytes).hexdigest()
    s3_client.put_object(
        Bucket=PROJECT_BUCKET,
        Key=image_key,
        Body=image_bytes,
        ContentType=content_type,
        Metadata={
            "sha256": image_sha256,
            "pdf-sha256": record["pdf_sha256"],
            "render-version": args.render_version,
            "document-id": record["document_id"],
            "page-number": str(record["source_page_number"]),
        },
    )
    record.update(
        {
            "page_width_pt": round(page_rect.width, 2),
            "page_height_pt": round(page_rect.height, 2),
            "render_dpi": round(zoom * 72.0, 1),
            "image_sha256": image_sha256,
            "image_width_px": pixmap.width,
            "image_height_px": pixmap.height,
            "image_size_bytes": len(image_bytes),
            "status": "success",
        }
    )


def render_document(
    s3_client,
    document: dict,
    batch_root: Path,
    batch_id: str,
    args: argparse.Namespace,
) -> list[dict]:
    """Render all pages of one document; one row per page, errors included."""
    extension = "jpg" if args.format == "jpeg" else "png"
    records: list[dict] = []
    pdf = None
    try:
        local_pdf_path = staged_pdf_path(batch_root, document["pdf_s3_key"], batch_id)
        if not local_pdf_path.is_file():
            raise FileNotFoundError(
                f"Staged PDF missing: {local_pdf_path}. Re-run "
                "02_unpack_and_upload_documents.py for this batch."
            )
        pdf = fitz.open(local_pdf_path)
        if pdf.needs_pass:
            raise RuntimeError("PDF requires a password and cannot be rendered.")

        for page_index in range(pdf.page_count):
            page_number = page_index + 1
            record = new_page_record(document, batch_id, args)
            record["page_id"] = stable_sha256(document["document_id"], page_number)
            record["source_page_number"] = page_number
            record["document_page_count"] = pdf.page_count
            image_key = (
                f"{CORPUS_ROOT_PREFIX}/pages/{batch_id}/{document['document_id']}/"
                f"{args.render_version}/page_{page_number:04d}.{extension}"
            )
            record["image_s3_key"] = image_key
            try:
                render_page(s3_client, pdf, page_index, record, image_key, args)
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
    """Run stage 04 end to end."""
    args = build_parser().parse_args()
    batch_id = validate_batch_id(args.batch_id)
    if args.dpi <= 0 or args.dpi > 1200:
        raise ValueError("dpi must be between 1 and 1200.")

    batch_root = LOCAL_TEMP_ROOT / batch_id
    inventory_path = (
        Path(args.document_inventory)
        if args.document_inventory
        else batch_root / "manifests" / "document_inventory.parquet"
    )
    output_path = batch_root / "manifests" / "page_inventory.parquet"

    if not inventory_path.is_file():
        raise FileNotFoundError(f"Document inventory not found: {inventory_path}")

    documents = [
        r
        for r in pq.read_table(inventory_path).to_pylist()
        if r["status"] == "success" and r["pdf_readable"] and not r["pdf_encrypted"]
    ]
    if not documents:
        raise RuntimeError(
            "Document inventory contains no renderable documents. "
            "Check 03_build_document_inventory.py output first."
        )

    resumable: dict[str, list[dict]] = {}
    if output_path.is_file() and not args.no_resume:
        resumable = load_resumable_rows(output_path, args.format, args.render_version)
        resumable = filter_unchanged_pdfs(resumable, documents)
        if resumable:
            print(f"Resuming: skipping {len(resumable)} completed documents.")

    s3_client = boto3.client("s3", region_name=AWS_REGION)
    records: list[dict] = []
    for document in documents:
        if document["document_id"] in resumable:
            records.extend(resumable[document["document_id"]])
        else:
            records.extend(render_document(s3_client, document, batch_root, batch_id, args))

    write_parquet_records(output_path, records, PAGE_SCHEMA)
    page_inventory_key = f"{CORPUS_ROOT_PREFIX}/manifests/{batch_id}/page_inventory.parquet"
    s3_client.upload_file(
        str(output_path),
        PROJECT_BUCKET,
        page_inventory_key,
        ExtraArgs={"ContentType": "application/octet-stream"},
    )

    print(f"Page inventory: {output_path}")
    print(f"Successful pages: {sum(r['status'] == 'success' for r in records)}")
    print(f"Errors: {sum(r['status'] == 'error' for r in records)}")
    print(f"S3 page inventory: s3://{PROJECT_BUCKET}/{page_inventory_key}")


if __name__ == "__main__":
    main()
