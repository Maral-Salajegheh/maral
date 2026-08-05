"""Stage 02: safely unpack the batch ZIP (including nested ZIPs), upload PDFs
and CSVs to S3, and write unpack_manifest.parquet with one row per member."""
from __future__ import annotations

import argparse
import hashlib
import os
import re
import stat
from pathlib import Path, PurePosixPath
from zipfile import BadZipFile, ZipFile, ZipInfo

import boto3
import pyarrow as pa

from common import (
    ensure_free_space,
    read_json,
    sha256_file,
    utc_now,
    validate_batch_id,
    write_parquet_records,
)
from config import (
    ALLOWED_EXTENSIONS,
    AWS_REGION,
    CORPUS_ROOT_PREFIX,
    LOCAL_TEMP_ROOT,
    MAX_SINGLE_FILE_BYTES,
    MAX_UNCOMPRESSED_BYTES,
    MAX_ZIP_MEMBERS,
    MAX_ZIP_NESTING,
    PROJECT_BUCKET,
)

UNPACK_SCHEMA = pa.schema(
    [
        ("batch_id", pa.string()),
        ("source_zip_sha256", pa.string()),
        ("zip_member_path", pa.string()),
        ("nesting_level", pa.int16()),
        ("container_zip_path", pa.string()),
        ("wrapper_depth", pa.int16()),
        ("masterindex_id", pa.string()),
        ("relative_path_under_masterindex", pa.string()),
        ("filename", pa.string()),
        ("file_extension", pa.string()),
        ("file_type", pa.string()),
        ("file_size_bytes", pa.int64()),
        ("file_sha256", pa.string()),
        ("local_staging_path", pa.string()),
        ("target_s3_key", pa.string()),
        ("is_duplicate_content", pa.bool_()),
        ("duplicate_of_s3_key", pa.string()),
        ("status", pa.string()),
        ("error_code", pa.string()),
        ("error_message", pa.string()),
        ("processed_at_utc", pa.timestamp("us", tz="UTC")),
    ]
)

_WINDOWS_DRIVE_PATTERN = re.compile(r"^[A-Za-z]:")


class UnpackContext:
    """State shared across all archive levels of one batch."""

    def __init__(self, batch_id: str, source_zip_sha256: str, extract_root: Path):
        self.batch_id = batch_id
        self.source_zip_sha256 = source_zip_sha256
        self.extract_root = extract_root
        self.container_root = extract_root.parent / "containers"
        self.s3_client = boto3.client("s3", region_name=AWS_REGION)
        self.records: list[dict] = []
        self.seen_member_paths: set[str] = set()
        self.first_s3_key_by_hash: dict[str, str] = {}
        self.extracted_bytes = 0
        self.container_counter = 0


def build_parser() -> argparse.ArgumentParser:
    """Command-line arguments for this stage."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-id", required=True)
    parser.add_argument("--zip-path", default=None, help="Defaults to stage 01 output.")
    parser.add_argument(
        "--wrapper-depth", default="auto", help="'auto' or a non-negative integer."
    )
    return parser


def is_symlink(info: ZipInfo) -> bool:
    """True when the ZIP member is a symbolic link."""
    return stat.S_ISLNK((info.external_attr >> 16) & 0o170000)


def safe_member_parts(name: str) -> tuple[str, ...]:
    """Validate a member path (no traversal/absolute/NUL) and return its parts."""
    if "\x00" in name:
        raise ValueError("ZIP member path contains a NUL byte.")
    normalized = name.replace("\\", "/")
    if normalized.startswith("/") or _WINDOWS_DRIVE_PATTERN.match(normalized):
        raise ValueError("Absolute ZIP member path is not allowed.")
    parts = tuple(p for p in PurePosixPath(normalized).parts if p not in {"", "."})
    if not parts:
        raise ValueError("ZIP member path is empty.")
    if any(p == ".." for p in parts):
        raise ValueError("Parent traversal '..' is not allowed.")
    return parts


def detect_wrapper_depth(members: list[tuple[ZipInfo, tuple[str, ...]]]) -> int:
    """Depth of a common wrapper prefix; strip only when >=2 MasterIndex
    folders sit below it (a single-folder ZIP is ambiguous: use --wrapper-depth)."""
    candidate_extensions = ALLOWED_EXTENSIONS | {".zip"}
    candidates = [
        parts
        for info, parts in members
        if not info.is_dir()
        and Path(parts[-1]).suffix.lower() in candidate_extensions
        and len(parts) >= 2
    ]
    if not candidates:
        return 0
    max_possible_depth = min(len(parts) - 2 for parts in candidates)
    common_depth = 0
    while common_depth < max_possible_depth:
        expected = candidates[0][common_depth]
        if not all(parts[common_depth] == expected for parts in candidates):
            break
        common_depth += 1
    if common_depth == 0:
        return 0
    candidate_ids = {parts[common_depth] for parts in candidates}
    return common_depth if len(candidate_ids) >= 2 else 0


def classify_file_type(extension: str, is_directory: bool) -> str:
    """Map an extension to pdf/csv/zip/other/directory."""
    if is_directory:
        return "directory"
    return {".pdf": "pdf", ".csv": "csv", ".zip": "zip"}.get(extension, "other")


def extract_member(archive: ZipFile, info: ZipInfo, destination: Path) -> tuple[str, int]:
    """Stream one member to disk with hashing; return (sha256, size)."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial_path = destination.with_suffix(destination.suffix + ".part")
    partial_path.unlink(missing_ok=True)
    digest = hashlib.sha256()
    actual_size = 0
    try:
        with archive.open(info, "r") as source, partial_path.open("wb") as target:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                actual_size += len(chunk)
                digest.update(chunk)
                target.write(chunk)
        if actual_size != info.file_size:
            raise RuntimeError("Extracted size does not match the ZIP member metadata.")
        os.replace(partial_path, destination)
        return digest.hexdigest(), actual_size
    except Exception:
        partial_path.unlink(missing_ok=True)
        raise


def new_record(
    ctx: UnpackContext,
    member_path: str,
    info: ZipInfo,
    nesting_level: int,
    container_zip_path: str | None,
    wrapper_depth: int,
) -> dict:
    """Manifest row with defaults for one ZIP member."""
    filename = Path(member_path).name
    return {
        "batch_id": ctx.batch_id,
        "source_zip_sha256": ctx.source_zip_sha256,
        "zip_member_path": member_path,
        "nesting_level": nesting_level,
        "container_zip_path": container_zip_path,
        "wrapper_depth": wrapper_depth,
        "masterindex_id": None,
        "relative_path_under_masterindex": None,
        "filename": filename,
        "file_extension": Path(filename).suffix.lower(),
        "file_type": "directory" if info.is_dir() else "other",
        "file_size_bytes": int(info.file_size),
        "file_sha256": None,
        "local_staging_path": None,
        "target_s3_key": None,
        "is_duplicate_content": False,
        "duplicate_of_s3_key": None,
        "status": None,
        "error_code": None,
        "error_message": None,
        "processed_at_utc": utc_now(),
    }


def validate_member(ctx: UnpackContext, info: ZipInfo) -> tuple[str, ...]:
    """Safety checks for one member; returns its clean path parts."""
    parts = safe_member_parts(info.filename)
    if is_symlink(info):
        raise ValueError("Symbolic links are not allowed in the ZIP.")
    if info.file_size > MAX_SINGLE_FILE_BYTES:
        raise ValueError("ZIP member exceeds MAX_SINGLE_FILE_BYTES.")
    if not info.is_dir():
        ctx.extracted_bytes += info.file_size
        if ctx.extracted_bytes > MAX_UNCOMPRESSED_BYTES:
            raise ValueError("Total extracted size exceeds MAX_UNCOMPRESSED_BYTES.")
    return parts


def build_logical_path(
    ctx: UnpackContext,
    parts: tuple[str, ...],
    base_parts: tuple[str, ...],
    nesting_level: int,
    wrapper_depth: int,
) -> tuple[str, ...]:
    """Member path after wrapper stripping (top level only) and container
    prefixing; also rejects duplicate paths."""
    if nesting_level == 0 and wrapper_depth >= len(parts):
        raise ValueError("wrapper_depth removes the complete member path.")
    stripped = parts[wrapper_depth:] if nesting_level == 0 else parts
    logical_parts = base_parts + stripped
    normalized = "/".join(logical_parts)
    if normalized in ctx.seen_member_paths:
        raise ValueError("Duplicate normalized ZIP member path.")
    ctx.seen_member_paths.add(normalized)
    return logical_parts


def set_masterindex_fields(record: dict, logical_parts: tuple[str, ...]) -> str:
    """Fill MasterIndex columns from the logical path; return the ID."""
    if len(logical_parts) < 2:
        raise ValueError(
            "Cannot determine MasterIndex ID. Expected at least "
            "'<masterindex_id>/<filename>'."
        )
    masterindex_id = logical_parts[0].strip()
    if not masterindex_id:
        raise ValueError("MasterIndex ID is empty.")
    record["masterindex_id"] = masterindex_id
    record["relative_path_under_masterindex"] = "/".join(logical_parts[1:])
    return masterindex_id


def expand_nested_zip(
    ctx: UnpackContext,
    archive: ZipFile,
    info: ZipInfo,
    record: dict,
    logical_parts: tuple[str, ...],
    nesting_level: int,
) -> None:
    """Extract a nested ZIP and process its members recursively."""
    if nesting_level >= MAX_ZIP_NESTING:
        raise ValueError(
            f"Nested ZIP exceeds MAX_ZIP_NESTING={MAX_ZIP_NESTING}. "
            "Its contents were NOT extracted."
        )
    if len(logical_parts) >= 2:
        set_masterindex_fields(record, logical_parts)

    # Containers are staged flat: a mirrored tree would create a FILE named
    # like the ZIP that collides with the DIRECTORY needed for its contents.
    ctx.container_counter += 1
    local_path = (
        ctx.container_root
        / f"container_{ctx.container_counter:04d}_{Path(info.filename).name}"
    )
    file_sha256, actual_size = extract_member(archive, info, local_path)
    record["file_size_bytes"] = actual_size
    record["file_sha256"] = file_sha256
    record["local_staging_path"] = str(local_path)

    # A top-level ZIP represents one MasterIndex: its stem becomes the folder.
    nested_base = (
        (Path(logical_parts[0]).stem,) if len(logical_parts) == 1 else logical_parts
    )
    with ZipFile(local_path, "r") as nested:
        nested_files = process_archive(
            ctx,
            nested,
            base_parts=nested_base,
            nesting_level=nesting_level + 1,
            container_zip_path="/".join(logical_parts),
            wrapper_depth=0,
        )
    if nested_files == 0:
        raise ValueError("Nested ZIP contains no files (empty_nested_zip).")
    record["status"] = "expanded"


def extract_and_upload(
    ctx: UnpackContext,
    archive: ZipFile,
    info: ZipInfo,
    record: dict,
    logical_parts: tuple[str, ...],
    masterindex_id: str,
) -> None:
    """Extract a supported file to staging and upload it to the corpus."""
    local_path = ctx.extract_root.joinpath(*logical_parts)
    file_sha256, actual_size = extract_member(archive, info, local_path)
    record["file_size_bytes"] = actual_size
    record["file_sha256"] = file_sha256
    record["local_staging_path"] = str(local_path)

    normalized = "/".join(logical_parts)
    target_key = f"{CORPUS_ROOT_PREFIX}/unpacked/{ctx.batch_id}/{normalized}"
    record["target_s3_key"] = target_key

    duplicate_of = ctx.first_s3_key_by_hash.get(file_sha256)
    record["is_duplicate_content"] = duplicate_of is not None
    record["duplicate_of_s3_key"] = duplicate_of

    extension = record["file_extension"]
    ctx.s3_client.upload_file(
        str(local_path),
        PROJECT_BUCKET,
        target_key,
        ExtraArgs={
            "ContentType": "application/pdf" if extension == ".pdf" else "text/csv",
            "Metadata": {
                "sha256": file_sha256,
                "batch-id": ctx.batch_id,
                "masterindex-id": masterindex_id,
            },
        },
    )
    ctx.first_s3_key_by_hash.setdefault(file_sha256, target_key)
    record["status"] = "success"


def process_member(
    ctx: UnpackContext,
    archive: ZipFile,
    info: ZipInfo,
    base_parts: tuple[str, ...],
    nesting_level: int,
    container_zip_path: str | None,
    wrapper_depth: int,
) -> None:
    """Route one member: directory, nested ZIP, supported file, or skip."""
    display_path = (
        f"{container_zip_path}/{info.filename}" if container_zip_path else info.filename
    )
    record = new_record(
        ctx, display_path, info, nesting_level, container_zip_path, wrapper_depth
    )
    try:
        parts = validate_member(ctx, info)
        logical_parts = build_logical_path(
            ctx, parts, base_parts, nesting_level, wrapper_depth
        )
        extension = Path(logical_parts[-1]).suffix.lower()
        record["file_type"] = classify_file_type(extension, info.is_dir())

        if info.is_dir():
            record["status"] = "directory"
        elif extension == ".zip":
            expand_nested_zip(ctx, archive, info, record, logical_parts, nesting_level)
        else:
            masterindex_id = set_masterindex_fields(record, logical_parts)
            if extension not in ALLOWED_EXTENSIONS:
                record["status"] = "skipped"
                record["error_code"] = "unsupported_file_type"
            else:
                extract_and_upload(
                    ctx, archive, info, record, logical_parts, masterindex_id
                )

    except BadZipFile as error:
        record["status"] = "error"
        record["error_code"] = "invalid_nested_zip"
        record["error_message"] = str(error)
    except Exception as error:
        record["status"] = "error"
        message = str(error)
        record["error_code"] = (
            "empty_nested_zip" if "empty_nested_zip" in message
            else error.__class__.__name__
        )
        record["error_message"] = message

    ctx.records.append(record)


def process_archive(
    ctx: UnpackContext,
    archive: ZipFile,
    base_parts: tuple[str, ...],
    nesting_level: int,
    container_zip_path: str | None,
    wrapper_depth: int,
) -> int:
    """Process every member of one archive; return its file (non-dir) count."""
    file_count = 0
    for info in archive.infolist():
        if not info.is_dir():
            file_count += 1
        process_member(
            ctx, archive, info, base_parts, nesting_level, container_zip_path,
            wrapper_depth,
        )
    return file_count


def check_archive_limits(archive: ZipFile) -> int:
    """Enforce member-count and total-size limits; return total uncompressed size."""
    infos = archive.infolist()
    if len(infos) > MAX_ZIP_MEMBERS:
        raise RuntimeError(
            f"ZIP has {len(infos):,} members, exceeding MAX_ZIP_MEMBERS={MAX_ZIP_MEMBERS:,}."
        )
    total = sum(i.file_size for i in infos if not i.is_dir())
    if total > MAX_UNCOMPRESSED_BYTES:
        raise RuntimeError("ZIP uncompressed size exceeds MAX_UNCOMPRESSED_BYTES.")
    return total


def resolve_wrapper_depth(archive: ZipFile, argument: str) -> int:
    """Wrapper depth from --wrapper-depth, auto-detected when 'auto'."""
    if argument != "auto":
        depth = int(argument)
        if depth < 0:
            raise ValueError("wrapper_depth must be non-negative.")
        return depth
    validated = []
    for info in archive.infolist():
        try:
            validated.append((info, safe_member_parts(info.filename)))
        except Exception:
            continue  # unsafe members get their own error rows later
    return detect_wrapper_depth(validated)


def verify_zip_hash(zip_path: Path, ingestion_manifest_path: Path) -> str:
    """Hash the local ZIP and check it against stage 01's manifest if present."""
    source_zip_sha256 = sha256_file(zip_path)
    if ingestion_manifest_path.exists():
        expected = read_json(ingestion_manifest_path).get("source_sha256")
        if expected and expected != source_zip_sha256:
            raise RuntimeError("ZIP SHA-256 does not match ingestion_manifest.json.")
    return source_zip_sha256


def main() -> None:
    """Run stage 02 end to end."""
    args = build_parser().parse_args()
    batch_id = validate_batch_id(args.batch_id)

    batch_root = LOCAL_TEMP_ROOT / batch_id
    zip_path = Path(args.zip_path) if args.zip_path else batch_root / "input" / "source.zip"
    output_manifest_path = batch_root / "manifests" / "unpack_manifest.parquet"

    if not zip_path.is_file():
        raise FileNotFoundError(f"ZIP does not exist: {zip_path}")

    source_zip_sha256 = verify_zip_hash(
        zip_path, batch_root / "manifests" / "ingestion_manifest.json"
    )
    ctx = UnpackContext(batch_id, source_zip_sha256, batch_root / "extracted")

    try:
        with ZipFile(zip_path, "r") as archive:
            total_uncompressed = check_archive_limits(archive)
            wrapper_depth = resolve_wrapper_depth(archive, args.wrapper_depth)
            ensure_free_space(ctx.extract_root, total_uncompressed, reserve_bytes=1024**3)
            process_archive(
                ctx,
                archive,
                base_parts=(),
                nesting_level=0,
                container_zip_path=None,
                wrapper_depth=wrapper_depth,
            )
    except BadZipFile as error:
        raise RuntimeError(f"Invalid ZIP file: {zip_path}") from error

    write_parquet_records(output_manifest_path, ctx.records, UNPACK_SCHEMA)
    manifest_key = f"{CORPUS_ROOT_PREFIX}/manifests/{batch_id}/unpack_manifest.parquet"
    ctx.s3_client.upload_file(
        str(output_manifest_path),
        PROJECT_BUCKET,
        manifest_key,
        ExtraArgs={"ContentType": "application/octet-stream"},
    )

    success_count = sum(r["status"] == "success" for r in ctx.records)
    print(f"Unpack manifest: {output_manifest_path}")
    print(f"Successful files: {success_count}")
    print(f"Expanded nested ZIPs: {sum(r['status'] == 'expanded' for r in ctx.records)}")
    print(f"Errors: {sum(r['status'] == 'error' for r in ctx.records)}")
    print(f"S3 manifest: s3://{PROJECT_BUCKET}/{manifest_key}")

    if success_count == 0:
        raise RuntimeError("No supported files were unpacked successfully.")


if __name__ == "__main__":
    main()
