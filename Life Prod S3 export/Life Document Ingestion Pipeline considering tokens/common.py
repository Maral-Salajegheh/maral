"""Shared utilities for the ingestion pipeline: time, IDs, hashing,
JSON/Parquet writing, disk space, and S3 helpers."""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pyarrow as pa
import pyarrow.parquet as pq
from botocore.exceptions import ClientError

_BATCH_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


def utc_now() -> datetime:
    """Current UTC time as a timezone-aware datetime."""
    return datetime.now(timezone.utc)


def utc_now_iso() -> str:
    """Current UTC time as an ISO string."""
    return utc_now().isoformat()


def validate_batch_id(batch_id: str) -> str:
    """Reject batch IDs with unsafe characters; return the ID unchanged."""
    if not _BATCH_ID_PATTERN.fullmatch(batch_id):
        raise ValueError(
            "batch_id must start with a letter or number and contain only "
            "letters, numbers, '.', '_' or '-'. Maximum length: 128."
        )
    return batch_id


def build_batch_id(source_s3_uri: str) -> str:
    """Default batch ID from today's date and the ZIP filename stem."""
    stem = Path(source_s3_uri.rstrip("/").rsplit("/", 1)[-1]).stem
    safe_stem = re.sub(r"[^A-Za-z0-9._-]", "_", stem).strip("._-") or "batch"
    return validate_batch_id(f"life_{utc_now():%Y%m%d}_{safe_stem}"[:128])


def parse_s3_uri(uri: str) -> tuple[str, str]:
    """Split s3://bucket/key into (bucket, key)."""
    if not uri.startswith("s3://"):
        raise ValueError(f"Not an S3 URI: {uri}")
    bucket, separator, key = uri[5:].partition("/")
    if not bucket or not separator or not key:
        raise ValueError(f"S3 URI must include bucket and object key: {uri}")
    return bucket, key


def resolve_source_uri(name_or_uri: str, source_prefix: str) -> str:
    """Full s3:// URI from a bare ZIP filename, or pass through a full URI."""
    if name_or_uri.startswith("s3://"):
        return name_or_uri
    return f"{source_prefix}/{name_or_uri.lstrip('/')}"


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """SHA-256 of a file, streamed in chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_sha256(*parts: object) -> str:
    """Stable ID from parts; NUL separator prevents ambiguous concatenation."""
    digest = hashlib.sha256()
    for part in parts:
        digest.update(("" if part is None else str(part)).encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write JSON via a temporary file so readers never see partial output."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as output_file:
        json.dump(payload, output_file, ensure_ascii=False, indent=2, default=str)
        output_file.write("\n")
    os.replace(temporary_path, path)


def read_json(path: Path) -> dict[str, Any]:
    """Read a JSON file into a dict."""
    with path.open("r", encoding="utf-8") as input_file:
        return json.load(input_file)


def write_parquet_records(
    path: Path, records: Iterable[dict[str, Any]], schema: pa.Schema
) -> None:
    """Write records as Parquet via a temporary file (atomic replace)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    table = pa.Table.from_pylist(list(records), schema=schema)
    pq.write_table(table, temporary_path, compression="snappy")
    os.replace(temporary_path, path)


def ensure_free_space(
    directory: Path, required_bytes: int, reserve_bytes: int = 512 * 1024**2
) -> None:
    """Fail early when local disk space is insufficient."""
    directory.mkdir(parents=True, exist_ok=True)
    free_bytes = shutil.disk_usage(directory).free
    if free_bytes < required_bytes + reserve_bytes:
        raise RuntimeError(
            f"Insufficient local disk space. Required approximately "
            f"{required_bytes + reserve_bytes:,} bytes, available {free_bytes:,}."
        )


def get_s3_head(s3_client: Any, bucket: str, key: str) -> dict[str, Any] | None:
    """head_object that returns None instead of raising when the key is absent."""
    try:
        return s3_client.head_object(Bucket=bucket, Key=key)
    except ClientError as error:
        code = error.response.get("Error", {}).get("Code", "")
        if code in {"404", "NoSuchKey", "NotFound"}:
            return None
        raise