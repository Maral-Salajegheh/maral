"""Stage 01: register the source ZIP in the immutable raw S3 layer, download
it to local staging, validate it, and write ingestion_manifest.json."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from zipfile import BadZipFile, ZipFile

import boto3
from botocore.exceptions import BotoCoreError, ClientError

from common import (
    atomic_write_json,
    ensure_free_space,
    get_s3_head,
    parse_s3_uri,
    sha256_file,
    utc_now_iso,
    validate_batch_id,
)
from config import AWS_REGION, CORPUS_ROOT_PREFIX, LOCAL_TEMP_ROOT, PROJECT_BUCKET


def build_parser() -> argparse.ArgumentParser:
    """Command-line arguments for this stage."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-id", required=True)
    parser.add_argument("--source-s3-uri", required=True)
    parser.add_argument("--overwrite-raw", action="store_true")
    return parser


def check_source_object(s3_client, source_bucket: str, source_key: str) -> int:
    """Verify the source ZIP exists and is valid; return its size in bytes."""
    head = get_s3_head(s3_client, source_bucket, source_key)
    if head is None:
        raise FileNotFoundError(f"Source object does not exist: s3://{source_bucket}/{source_key}")
    size = int(head["ContentLength"])
    if size <= 0:
        raise RuntimeError("Source ZIP is empty.")
    if not source_key.lower().endswith(".zip"):
        raise RuntimeError("Source object key must end with '.zip'.")
    return size


def register_raw_zip(
    s3_client,
    source_bucket: str,
    source_key: str,
    source_size: int,
    raw_key: str,
    overwrite: bool,
) -> None:
    """Copy the source ZIP into the immutable raw layer, once per batch."""
    if source_bucket == PROJECT_BUCKET and source_key == raw_key:
        return
    existing = get_s3_head(s3_client, PROJECT_BUCKET, raw_key)
    if existing is not None and not overwrite:
        if int(existing["ContentLength"]) != source_size:
            raise RuntimeError(
                "Raw ZIP already exists for this batch_id with a different "
                "size. Use a new batch_id or --overwrite-raw."
            )
        return
    s3_client.copy(
        {"Bucket": source_bucket, "Key": source_key},
        PROJECT_BUCKET,
        raw_key,
        ExtraArgs={
            "MetadataDirective": "REPLACE",
            "Metadata": {"source-bucket": source_bucket, "source-key": source_key},
            "ContentType": "application/zip",
        },
    )


def download_raw_zip(s3_client, raw_key: str, destination: Path, expected_size: int) -> None:
    """Download the raw ZIP atomically and verify its size."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    ensure_free_space(destination.parent, expected_size)
    partial_path = destination.with_suffix(".zip.part")
    partial_path.unlink(missing_ok=True)
    s3_client.download_file(PROJECT_BUCKET, raw_key, str(partial_path))
    if partial_path.stat().st_size != expected_size:
        raise RuntimeError("Downloaded file size does not match the S3 object size.")
    os.replace(partial_path, destination)


def validate_zip(path: Path) -> int:
    """CRC-check the ZIP and return its member count."""
    try:
        with ZipFile(path, "r") as archive:
            bad_member = archive.testzip()
            if bad_member is not None:
                raise RuntimeError(f"ZIP CRC validation failed for member: {bad_member}")
            return len(archive.infolist())
    except BadZipFile as error:
        raise RuntimeError(f"Downloaded file is not a valid ZIP: {path}") from error


def write_manifest(
    s3_client,
    manifest_path: Path,
    batch_id: str,
    source_s3_uri: str,
    raw_key: str,
    source_key: str,
    size: int,
    sha256: str,
    member_count: int,
    local_zip_path: Path,
) -> None:
    """Write ingestion_manifest.json locally and upload it next to the raw ZIP."""
    manifest = {
        "batch_id": batch_id,
        "source_system": "aws_s3",
        "source_s3_uri": source_s3_uri,
        "raw_s3_uri": f"s3://{PROJECT_BUCKET}/{raw_key}",
        "source_filename": Path(source_key).name,
        "source_file_size_bytes": size,
        "source_sha256": sha256,
        "local_zip_path": str(local_zip_path),
        "zip_member_count": member_count,
        "downloaded_at_utc": utc_now_iso(),
        "status": "success",
    }
    atomic_write_json(manifest_path, manifest)
    s3_client.upload_file(
        str(manifest_path),
        PROJECT_BUCKET,
        f"{CORPUS_ROOT_PREFIX}/raw/{batch_id}/ingestion_manifest.json",
        ExtraArgs={"ContentType": "application/json"},
    )


def main() -> None:
    """Run stage 01 end to end."""
    args = build_parser().parse_args()
    batch_id = validate_batch_id(args.batch_id)
    source_bucket, source_key = parse_s3_uri(args.source_s3_uri)
    s3_client = boto3.client("s3", region_name=AWS_REGION)

    batch_root = LOCAL_TEMP_ROOT / batch_id
    local_zip_path = batch_root / "input" / "source.zip"
    manifest_path = batch_root / "manifests" / "ingestion_manifest.json"
    raw_key = f"{CORPUS_ROOT_PREFIX}/raw/{batch_id}/source.zip"

    try:
        source_size = check_source_object(s3_client, source_bucket, source_key)
        register_raw_zip(
            s3_client, source_bucket, source_key, source_size, raw_key, args.overwrite_raw
        )
        print(f"Downloading s3://{PROJECT_BUCKET}/{raw_key}")
        download_raw_zip(s3_client, raw_key, local_zip_path, source_size)
        source_sha256 = sha256_file(local_zip_path)
        member_count = validate_zip(local_zip_path)
        write_manifest(
            s3_client,
            manifest_path,
            batch_id,
            args.source_s3_uri,
            raw_key,
            source_key,
            source_size,
            source_sha256,
            member_count,
            local_zip_path,
        )
        print(f"Download completed: {local_zip_path}")
        print(f"SHA-256: {source_sha256}")
        print(f"ZIP members: {member_count}")
    except (ClientError, BotoCoreError) as error:
        raise RuntimeError(f"AWS operation failed: {error}") from error


if __name__ == "__main__":
    main()
