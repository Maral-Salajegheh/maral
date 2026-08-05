"""Stage 03: build document_inventory.parquet with one row per PDF, including
the stable document_id, page count, readability, and duplicate flags.
Requires the staged files from stage 02 (re-run 02 if /tmp was cleaned)."""
from __future__ import annotations

import argparse
from pathlib import Path

import boto3
import pyarrow as pa
import pyarrow.parquet as pq
from pypdf import PdfReader

from common import (
    read_json,
    stable_sha256,
    utc_now,
    validate_batch_id,
    write_parquet_records,
)
from config import AWS_REGION, CORPUS_ROOT_PREFIX, LOCAL_TEMP_ROOT, PROJECT_BUCKET

DOCUMENT_SCHEMA = pa.schema(
    [
        ("batch_id", pa.string()),
        ("masterindex_id", pa.string()),
        ("document_id", pa.string()),
        ("source_zip_sha256", pa.string()),
        ("relative_path_in_zip", pa.string()),
        ("pdf_filename", pa.string()),
        ("pdf_s3_key", pa.string()),
        ("pdf_sha256", pa.string()),
        ("pdf_size_bytes", pa.int64()),
        ("pdf_order_in_masterindex", pa.int32()),
        ("pdf_count_in_masterindex", pa.int32()),
        ("page_count", pa.int32()),
        ("pdf_readable", pa.bool_()),
        ("pdf_encrypted", pa.bool_()),
        ("associated_csv_count", pa.int32()),
        ("is_duplicate_content", pa.bool_()),
        ("duplicate_of_document_id", pa.string()),
        ("source_system", pa.string()),
        ("status", pa.string()),
        ("error_code", pa.string()),
        ("error_message", pa.string()),
        ("created_at_utc", pa.timestamp("us", tz="UTC")),
    ]
)


def build_parser() -> argparse.ArgumentParser:
    """Command-line arguments for this stage."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-id", required=True)
    parser.add_argument("--unpack-manifest", default=None)
    return parser


def load_successful_rows(unpack_manifest_path: Path) -> tuple[list[dict], list[dict]]:
    """Read the unpack manifest; return (pdf_rows, csv_rows) in stable order."""
    rows = pq.read_table(unpack_manifest_path).to_pylist()
    successful = [r for r in rows if r["status"] == "success"]
    pdf_rows = [r for r in successful if r["file_type"] == "pdf"]
    csv_rows = [r for r in successful if r["file_type"] == "csv"]
    # Deterministic order: never depends on filesystem iteration.
    pdf_rows.sort(key=lambda r: (r["masterindex_id"] or "", r["zip_member_path"] or ""))
    return pdf_rows, csv_rows


def count_by_masterindex(rows: list[dict]) -> dict[str, int]:
    """Number of rows per MasterIndex ID."""
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["masterindex_id"]] = counts.get(row["masterindex_id"], 0) + 1
    return counts


def inspect_pdf(path: Path) -> tuple[int | None, bool, bool]:
    """Return (page_count, readable, encrypted) for one staged PDF."""
    reader = PdfReader(str(path), strict=False)
    encrypted = bool(reader.is_encrypted)
    if encrypted and not reader.decrypt(""):
        return None, False, True
    return len(reader.pages), True, encrypted


def build_document_record(
    row: dict,
    batch_id: str,
    document_id: str,
    order: int,
    pdf_count: int,
    csv_count: int,
    duplicate_of: str | None,
    source_system: str,
) -> dict:
    """Inventory row for one PDF, including its inspection result."""
    status, error_code, error_message = "success", None, None
    page_count, pdf_readable, pdf_encrypted = None, False, False
    local_path = Path(row["local_staging_path"])
    try:
        if not local_path.is_file():
            raise FileNotFoundError(
                f"Staged PDF missing: {local_path}. Re-run "
                "02_unpack_and_upload_documents.py for this batch."
            )
        page_count, pdf_readable, pdf_encrypted = inspect_pdf(local_path)
        if not pdf_readable and pdf_encrypted:
            status = "encrypted_pdf"
    except Exception as error:
        status = "invalid_pdf"
        error_code = error.__class__.__name__
        error_message = str(error)

    return {
        "batch_id": batch_id,
        "masterindex_id": row["masterindex_id"],
        "document_id": document_id,
        "source_zip_sha256": row["source_zip_sha256"],
        "relative_path_in_zip": row["zip_member_path"],
        "pdf_filename": row["filename"],
        "pdf_s3_key": row["target_s3_key"],
        "pdf_sha256": row["file_sha256"],
        "pdf_size_bytes": row["file_size_bytes"],
        "pdf_order_in_masterindex": order,
        "pdf_count_in_masterindex": pdf_count,
        "page_count": page_count,
        "pdf_readable": pdf_readable,
        "pdf_encrypted": pdf_encrypted,
        "associated_csv_count": csv_count,
        "is_duplicate_content": duplicate_of is not None,
        "duplicate_of_document_id": duplicate_of,
        "source_system": source_system,
        "status": status,
        "error_code": error_code,
        "error_message": error_message,
        "created_at_utc": utc_now(),
    }


def build_inventory(
    pdf_rows: list[dict], csv_rows: list[dict], batch_id: str, source_system: str
) -> list[dict]:
    """All inventory rows for one batch, with stable IDs and duplicate flags."""
    pdf_counts = count_by_masterindex(pdf_rows)
    csv_counts = count_by_masterindex(csv_rows)
    next_order: dict[str, int] = {}
    first_document_by_hash: dict[str, str] = {}
    records: list[dict] = []

    for row in pdf_rows:
        masterindex_id = row["masterindex_id"]
        document_id = stable_sha256(
            masterindex_id, row["relative_path_under_masterindex"], row["file_sha256"]
        )
        next_order[masterindex_id] = next_order.get(masterindex_id, 0) + 1
        duplicate_of = first_document_by_hash.get(row["file_sha256"])
        first_document_by_hash.setdefault(row["file_sha256"], document_id)
        records.append(
            build_document_record(
                row,
                batch_id,
                document_id,
                next_order[masterindex_id],
                pdf_counts[masterindex_id],
                csv_counts.get(masterindex_id, 0),
                duplicate_of,
                source_system,
            )
        )
    return records


def main() -> None:
    """Run stage 03 end to end."""
    args = build_parser().parse_args()
    batch_id = validate_batch_id(args.batch_id)
    batch_root = LOCAL_TEMP_ROOT / batch_id

    unpack_manifest_path = (
        Path(args.unpack_manifest)
        if args.unpack_manifest
        else batch_root / "manifests" / "unpack_manifest.parquet"
    )
    output_path = batch_root / "manifests" / "document_inventory.parquet"
    ingestion_manifest_path = batch_root / "manifests" / "ingestion_manifest.json"

    if not unpack_manifest_path.is_file():
        raise FileNotFoundError(f"Unpack manifest not found: {unpack_manifest_path}")

    source_system = "unknown"
    if ingestion_manifest_path.exists():
        source_system = read_json(ingestion_manifest_path).get("source_system", "unknown")

    pdf_rows, csv_rows = load_successful_rows(unpack_manifest_path)
    records = build_inventory(pdf_rows, csv_rows, batch_id, source_system)
    write_parquet_records(output_path, records, DOCUMENT_SCHEMA)

    if records and not any(r["status"] == "success" for r in records):
        raise RuntimeError(
            "No readable documents in this batch. If staged files were cleaned "
            "from /tmp, re-run 02_unpack_and_upload_documents.py. "
            f"Local inventory written for inspection: {output_path}"
        )

    s3_client = boto3.client("s3", region_name=AWS_REGION)
    inventory_key = f"{CORPUS_ROOT_PREFIX}/manifests/{batch_id}/document_inventory.parquet"
    s3_client.upload_file(
        str(output_path),
        PROJECT_BUCKET,
        inventory_key,
        ExtraArgs={"ContentType": "application/octet-stream"},
    )

    valid_count = sum(r["status"] == "success" for r in records)
    print(f"Document inventory: {output_path}")
    print(f"PDF documents: {len(records)}")
    print(f"Readable documents: {valid_count}")
    print(f"Non-readable/encrypted documents: {len(records) - valid_count}")
    print(f"S3 inventory: s3://{PROJECT_BUCKET}/{inventory_key}")


if __name__ == "__main__":
    main()
