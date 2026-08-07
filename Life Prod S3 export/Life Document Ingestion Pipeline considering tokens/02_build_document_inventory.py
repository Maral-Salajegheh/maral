"""Stage 02: scan the ZIP and build document_inventory.parquet (local).

One row per MasterIndex. Reads the ZIP without extracting. Each MasterIndex
folder holds one PDF and one CSV; both are recorded, and the PDF is inspected
for page count and readability. Output stays on the local server.
"""
from __future__ import annotations

import argparse
import hashlib
from io import BytesIO
from pathlib import Path, PurePosixPath
from zipfile import ZipFile

import pyarrow as pa
from pypdf import PdfReader

from common import read_json, sha256_file, stable_sha256, utc_now, validate_batch_id, write_parquet_records
from config import LOCAL_OUTPUT_ROOT, LOCAL_TEMP_ROOT, MAX_ZIP_MEMBERS

DOCUMENT_SCHEMA = pa.schema(
    [
        ("batch_id", pa.string()),
        ("masterindex_id", pa.string()),
        ("corpus_document_id", pa.string()),
        ("source_zip_sha256", pa.string()),
        ("wrapper_depth", pa.int16()),
        ("pdf_filename", pa.string()),
        ("pdf_path_in_zip", pa.string()),
        ("pdf_sha256", pa.string()),
        ("pdf_size_bytes", pa.int64()),
        ("corpus_page_count", pa.int32()),
        ("pdf_readable", pa.bool_()),
        ("pdf_encrypted", pa.bool_()),
        ("csv_filename", pa.string()),
        ("csv_path_in_zip", pa.string()),
        ("csv_sha256", pa.string()),
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
    parser.add_argument("--zip-path", default=None, help="Defaults to stage 01 output.")
    parser.add_argument("--wrapper-depth", default="auto", help="'auto' or an integer.")
    return parser


def clean_parts(name: str) -> tuple[str, ...]:
    """Return the path parts of a ZIP member, normalized."""
    normalized = name.replace("\\", "/")
    return tuple(p for p in PurePosixPath(normalized).parts if p not in {"", "."})


def detect_wrapper_depth(file_parts: list[tuple[str, ...]]) -> int:
    """Depth of a common wrapper folder above the MasterIndex folders."""
    candidates = [p for p in file_parts if len(p) >= 2]
    if not candidates:
        return 0
    max_possible = min(len(p) - 2 for p in candidates)
    depth = 0
    while depth < max_possible:
        expected = candidates[0][depth]
        if not all(p[depth] == expected for p in candidates):
            break
        depth += 1
    if depth == 0:
        return 0
    ids = {p[depth] for p in candidates}
    return depth if len(ids) >= 2 else 0


def inspect_pdf(pdf_bytes: bytes) -> tuple[int | None, bool, bool]:
    """Return (page_count, readable, encrypted) for a PDF held in memory."""
    reader = PdfReader(BytesIO(pdf_bytes), strict=False)
    encrypted = bool(reader.is_encrypted)
    if encrypted and not reader.decrypt(""):
        return None, False, True
    return len(reader.pages), True, encrypted


def group_by_masterindex(archive: ZipFile, wrapper_depth: int) -> dict[str, dict]:
    """Group ZIP members by MasterIndex; keep the first PDF and first CSV of each."""
    groups: dict[str, dict] = {}
    for info in archive.infolist():
        if info.is_dir():
            continue
        parts = clean_parts(info.filename)[wrapper_depth:]
        if len(parts) < 2:
            continue
        masterindex_id = parts[0]
        extension = Path(parts[-1]).suffix.lower()
        group = groups.setdefault(masterindex_id, {"pdf": None, "csv": None})
        if extension == ".pdf" and group["pdf"] is None:
            group["pdf"] = info
        elif extension == ".csv" and group["csv"] is None:
            group["csv"] = info
    return groups


def build_record(
    masterindex_id: str, group: dict, archive: ZipFile, batch_id: str, source_zip_sha256: str, wrapper_depth: int
) -> dict:
    """One inventory row for one MasterIndex (its PDF and CSV)."""
    record = {
        "batch_id": batch_id,
        "masterindex_id": masterindex_id,
        "corpus_document_id": None,
        "source_zip_sha256": source_zip_sha256,
        "wrapper_depth": wrapper_depth,
        "pdf_filename": None,
        "pdf_path_in_zip": None,
        "pdf_sha256": None,
        "pdf_size_bytes": None,
        "corpus_page_count": None,
        "pdf_readable": False,
        "pdf_encrypted": False,
        "csv_filename": None,
        "csv_path_in_zip": None,
        "csv_sha256": None,
        "status": "success",
        "error_code": None,
        "error_message": None,
        "created_at_utc": utc_now(),
    }

    csv_info = group["csv"]
    if csv_info is not None:
        csv_bytes = archive.read(csv_info)
        record["csv_filename"] = Path(csv_info.filename).name
        record["csv_path_in_zip"] = csv_info.filename
        record["csv_sha256"] = hashlib.sha256(csv_bytes).hexdigest()

    pdf_info = group["pdf"]
    if pdf_info is None:
        record["status"] = "missing_pdf"
        return record

    pdf_bytes = archive.read(pdf_info)
    pdf_sha256 = hashlib.sha256(pdf_bytes).hexdigest()
    record["pdf_filename"] = Path(pdf_info.filename).name
    record["pdf_path_in_zip"] = pdf_info.filename
    record["pdf_sha256"] = pdf_sha256
    record["pdf_size_bytes"] = len(pdf_bytes)
    record["corpus_document_id"] = stable_sha256(masterindex_id, pdf_sha256)

    try:
        page_count, readable, encrypted = inspect_pdf(pdf_bytes)
        record["corpus_page_count"] = page_count
        record["pdf_readable"] = readable
        record["pdf_encrypted"] = encrypted
        if not readable and encrypted:
            record["status"] = "encrypted_pdf"
    except Exception as error:
        record["status"] = "invalid_pdf"
        record["error_code"] = error.__class__.__name__
        record["error_message"] = str(error)

    return record


def main() -> None:
    """Run stage 02 (scan only, local output)."""
    args = build_parser().parse_args()
    batch_id = validate_batch_id(args.batch_id)

    batch_root = LOCAL_TEMP_ROOT / batch_id
    zip_path = Path(args.zip_path) if args.zip_path else batch_root / "input" / "source.zip"
    if not zip_path.is_file():
        raise FileNotFoundError(f"ZIP does not exist: {zip_path}")

    source_zip_sha256 = sha256_file(zip_path)
    ingestion_manifest = batch_root / "manifests" / "ingestion_manifest.json"
    if ingestion_manifest.exists():
        expected = read_json(ingestion_manifest).get("source_sha256")
        if expected and expected != source_zip_sha256:
            raise RuntimeError("ZIP SHA-256 does not match ingestion_manifest.json.")

    with ZipFile(zip_path, "r") as archive:
        if len(archive.infolist()) > MAX_ZIP_MEMBERS:
            raise RuntimeError(f"ZIP exceeds MAX_ZIP_MEMBERS={MAX_ZIP_MEMBERS:,}.")

        if args.wrapper_depth == "auto":
            file_parts = [clean_parts(i.filename) for i in archive.infolist() if not i.is_dir()]
            wrapper_depth = detect_wrapper_depth(file_parts)
        else:
            wrapper_depth = int(args.wrapper_depth)

        groups = group_by_masterindex(archive, wrapper_depth)
        records = [
            build_record(mid, groups[mid], archive, batch_id, source_zip_sha256, wrapper_depth)
            for mid in sorted(groups)
        ]

    output_path = LOCAL_OUTPUT_ROOT / batch_id / "document_inventory.parquet"
    write_parquet_records(output_path, records, DOCUMENT_SCHEMA)

    if records and not any(r["status"] == "success" for r in records):
        raise RuntimeError("No readable PDFs found in this batch.")

    valid = sum(r["status"] == "success" for r in records)
    print(f"Document inventory: {output_path}")
    print(f"MasterIndex folders: {len(records)}")
    print(f"Readable PDFs: {valid}")
    print(f"Problems (missing/invalid/encrypted): {len(records) - valid}")


if __name__ == "__main__":
    main()