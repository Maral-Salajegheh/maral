"""Stage 01: download the source ZIP from the export bucket to local staging.

This is the only stage that reads the source bucket, so it is the only stage
that needs the temporary AWS tokens. It uploads nothing. When it finishes it
prints a reminder to remove the tokens before the upload stages.
"""
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
from config import AWS_REGION, LOCAL_TEMP_ROOT


def build_parser() -> argparse.ArgumentParser:
    """Command-line arguments for this stage."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-id", required=True)
    parser.add_argument("--source-s3-uri", required=True)
    return parser


def check_source_object(s3_client, bucket: str, key: str) -> int:
    """Verify the source ZIP exists and is valid; return its size in bytes."""
    head = get_s3_head(s3_client, bucket, key)
    if head is None:
        raise FileNotFoundError(f"Source object does not exist: s3://{bucket}/{key}")
    size = int(head["ContentLength"])
    if size <= 0:
        raise RuntimeError("Source ZIP is empty.")
    if not key.lower().endswith(".zip"):
        raise RuntimeError("Source object key must end with '.zip'.")
    return size


def download_zip(s3_client, bucket: str, key: str, destination: Path, expected_size: int) -> None:
    """Download the source ZIP atomically and verify its size."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    ensure_free_space(destination.parent, expected_size)
    partial_path = destination.with_suffix(".zip.part")
    partial_path.unlink(missing_ok=True)
    s3_client.download_file(bucket, key, str(partial_path))
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
    manifest_path: Path,
    batch_id: str,
    source_s3_uri: str,
    source_key: str,
    size: int,
    sha256: str,
    member_count: int,
    local_zip_path: Path,
) -> None:
    """Write ingestion_manifest.json locally (upload happens in stage 01b)."""
    manifest = {
        "batch_id": batch_id,
        "source_system": "aws_s3",
        "source_s3_uri": source_s3_uri,
        "source_filename": Path(source_key).name,
        "source_file_size_bytes": size,
        "source_sha256": sha256,
        "local_zip_path": str(local_zip_path),
        "zip_member_count": member_count,
        "downloaded_at_utc": utc_now_iso(),
        "status": "success",
    }
    atomic_write_json(manifest_path, manifest)


def print_credential_reminder(batch_id: str) -> None:
    """Warn the user to remove the tokens before the upload stages."""
    print("\n" + "=" * 72)
    print("  /!\\  DOWNLOAD DONE -- REMOVE YOUR AWS TOKENS BEFORE CONTINUING")
    print("=" * 72)
    print("  The next stages UPLOAD to the project bucket and must run WITHOUT")
    print("  tokens (they use the environment role). Remove the tokens, then")
    print("  continue from stage 01b:\n")
    print("      rm ~/.aws/credentials")
    print(f"      python run_pipeline.py --batch-id {batch_id} --from-stage 01b")
    print("=" * 72 + "\n")


def main() -> None:
    """Run stage 01 (download only)."""
    args = build_parser().parse_args()
    batch_id = validate_batch_id(args.batch_id)
    source_bucket, source_key = parse_s3_uri(args.source_s3_uri)
    s3_client = boto3.client("s3", region_name=AWS_REGION)

    batch_root = LOCAL_TEMP_ROOT / batch_id
    local_zip_path = batch_root / "input" / "source.zip"
    manifest_path = batch_root / "manifests" / "ingestion_manifest.json"

    try:
        source_size = check_source_object(s3_client, source_bucket, source_key)
        print(f"Downloading {args.source_s3_uri}")
        download_zip(s3_client, source_bucket, source_key, local_zip_path, source_size)
        source_sha256 = sha256_file(local_zip_path)
        member_count = validate_zip(local_zip_path)
        write_manifest(
            manifest_path,
            batch_id,
            args.source_s3_uri,
            source_key,
            source_size,
            source_sha256,
            member_count,
            local_zip_path,
        )
        print(f"Download completed: {local_zip_path}")
        print(f"SHA-256: {source_sha256}")
        print(f"ZIP members: {member_count}")
        print_credential_reminder(batch_id)
    except (ClientError, BotoCoreError) as error:
        raise RuntimeError(f"AWS operation failed: {error}") from error


if __name__ == "__main__":
    main()