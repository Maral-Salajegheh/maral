"""Stage 01b: upload the downloaded ZIP to the immutable raw layer.

Runs WITHOUT tokens (uses the environment role). Uploads the local source.zip
and its ingestion_manifest.json to the project bucket so the raw source is kept
permanently and traceably.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import boto3
from botocore.exceptions import BotoCoreError, ClientError

from common import get_s3_head, read_json, validate_batch_id
from config import AWS_REGION, LOCAL_TEMP_ROOT, PROJECT_BUCKET, RAW_DATA_PREFIX


def build_parser() -> argparse.ArgumentParser:
    """Command-line arguments for this stage."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-id", required=True)
    parser.add_argument("--overwrite-raw", action="store_true")
    return parser


def upload_raw_zip(s3_client, local_zip_path: Path, raw_key: str, size: int, overwrite: bool) -> None:
    """Upload the ZIP to the raw layer, once per batch."""
    existing = get_s3_head(s3_client, PROJECT_BUCKET, raw_key)
    if existing is not None and not overwrite:
        if int(existing["ContentLength"]) != size:
            raise RuntimeError(
                "Raw ZIP already exists for this batch_id with a different size. "
                "Use a new batch_id or --overwrite-raw."
            )
        return
    s3_client.upload_file(
        str(local_zip_path),
        PROJECT_BUCKET,
        raw_key,
        ExtraArgs={"ContentType": "application/zip"},
    )


def main() -> None:
    """Run stage 01b (raw registration)."""
    args = build_parser().parse_args()
    batch_id = validate_batch_id(args.batch_id)

    batch_root = LOCAL_TEMP_ROOT / batch_id
    local_zip_path = batch_root / "input" / "source.zip"
    manifest_path = batch_root / "manifests" / "ingestion_manifest.json"

    if not local_zip_path.is_file():
        raise FileNotFoundError(
            f"Downloaded ZIP not found: {local_zip_path}. Run stage 01 first."
        )
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Ingestion manifest not found: {manifest_path}.")

    size = read_json(manifest_path)["source_file_size_bytes"]
    raw_key = f"{RAW_DATA_PREFIX}/{batch_id}/source.zip"
    s3_client = boto3.client("s3", region_name=AWS_REGION)

    try:
        upload_raw_zip(s3_client, local_zip_path, raw_key, size, args.overwrite_raw)
        s3_client.upload_file(
            str(manifest_path),
            PROJECT_BUCKET,
            f"{RAW_DATA_PREFIX}/{batch_id}/ingestion_manifest.json",
            ExtraArgs={"ContentType": "application/json"},
        )
    except (ClientError, BotoCoreError) as error:
        raise RuntimeError(f"AWS operation failed: {error}") from error

    print(f"Raw ZIP registered: s3://{PROJECT_BUCKET}/{raw_key}")


if __name__ == "__main__":
    main()