"""Orchestrator: run the ingestion stages for one batch, in order, stopping at
the first failure. Writes run_summary.json locally.

After a successful run that includes stage 03, the local source.zip is deleted
when all of the following hold (use --keep-zip to keep it):
    - every row in page_inventory.parquet has status "success"
    - the raw copy exists in S3 with exactly the same size as the local ZIP
The manifests folder is kept. A deleted ZIP can be restored from the raw layer
(the command is printed) if stage 02 or 03 must run again.

Examples:
    # Step 1 (WITH tokens): download only, then stops and tells you to
    # remove the tokens.
    python run_pipeline.py --source-s3-uri export.zip

    # Step 2 (WITHOUT tokens, after `rm ~/.aws/credentials`): upload + process.
    python run_pipeline.py --batch-id life_20260804_export --from-stage 01b
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

import boto3
import pyarrow.parquet as pq
from botocore.exceptions import BotoCoreError, ClientError

from common import (
    atomic_write_json,
    build_batch_id,
    get_s3_head,
    resolve_source_uri,
    utc_now_iso,
    validate_batch_id,
)
from config import (
    AWS_REGION,
    LOCAL_OUTPUT_ROOT,
    LOCAL_TEMP_ROOT,
    PROJECT_BUCKET,
    RAW_DATA_PREFIX,
    SOURCE_ZIP_PREFIX,
)

SCRIPT_DIR = Path(__file__).resolve().parent
STAGES = [
    ("01", "01_download_batch_zip.py"),
    ("01b", "01b_register_raw_zip.py"),
    ("02", "02_build_document_inventory.py"),
    ("03", "03_render_pdf_pages.py"),
]
STAGE_NUMBERS = [number for number, _ in STAGES]


def build_parser() -> argparse.ArgumentParser:
    """Command-line arguments for the orchestrator."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--batch-id",
        default=None,
        help="Optional; derived from date and ZIP filename when omitted.",
    )
    parser.add_argument(
        "--source-s3-uri",
        default=None,
        help="ZIP filename (resolved against SOURCE_ZIP_PREFIX) or a full s3:// URI. Required when stage 01 runs.",
    )
    parser.add_argument("--from-stage", choices=STAGE_NUMBERS, default=STAGE_NUMBERS[0])
    parser.add_argument("--to-stage", choices=STAGE_NUMBERS, default=STAGE_NUMBERS[-1])
    parser.add_argument(
        "--extra-args-03", default="", help="Extra arguments for stage 03 (render)."
    )
    parser.add_argument(
        "--keep-zip", action="store_true", help="Do not delete the local source.zip after a clean render."
    )
    return parser


def resolve_batch_id(args: argparse.Namespace) -> str:
    """Batch ID from --batch-id, or derived from the source ZIP when omitted."""
    if args.batch_id:
        return validate_batch_id(args.batch_id)
    if not args.source_s3_uri:
        raise ValueError(
            "--batch-id is required when --source-s3-uri is not given "
            "(an existing batch must be named explicitly)."
        )
    batch_id = build_batch_id(args.source_s3_uri)
    print(f"Derived batch_id: {batch_id}")
    return batch_id


def select_stages(args: argparse.Namespace) -> list[tuple[str, str]]:
    """Stages to run, validated against the arguments."""
    from_index = STAGE_NUMBERS.index(args.from_stage)
    to_index = STAGE_NUMBERS.index(args.to_stage)
    if from_index > to_index:
        raise ValueError("--from-stage must not come after --to-stage.")
    selected = STAGES[from_index : to_index + 1]
    if any(n == "01" for n, _ in selected) and not args.source_s3_uri:
        raise ValueError("--source-s3-uri is required when stage 01 is included.")
    return selected


def stage_command(number: str, script: str, args: argparse.Namespace) -> list[str]:
    """Subprocess command for one stage."""
    command = [sys.executable, str(SCRIPT_DIR / script), "--batch-id", args.batch_id]
    if number == "01":
        command += ["--source-s3-uri", args.source_s3_uri]
    if number == "03" and args.extra_args_03:
        command += args.extra_args_03.split()
    return command


def run_stage(number: str, script: str, args: argparse.Namespace) -> dict:
    """Run one stage as a subprocess; return its summary entry."""
    command = stage_command(number, script, args)
    print(f"\n=== Stage {number}: {script} ===")
    print("Command:", " ".join(command))
    started = time.monotonic()
    result = subprocess.run(command)
    return {
        "stage": number,
        "script": script,
        "status": "success" if result.returncode == 0 else "failed",
        "return_code": result.returncode,
        "duration_seconds": round(time.monotonic() - started, 1),
    }


def count_unrendered_pages(batch_id: str) -> int | None:
    """Page rows whose status is not "success"; None when the inventory is missing."""
    path = LOCAL_OUTPUT_ROOT / batch_id / "page_inventory.parquet"
    if not path.is_file():
        return None
    statuses = pq.read_table(path, columns=["status"]).column("status").to_pylist()
    return sum(status != "success" for status in statuses)


def cleanup_local_zip(batch_id: str) -> dict:
    """Delete the local source.zip once the render is clean and S3 holds an identical-size copy."""
    zip_path = LOCAL_TEMP_ROOT / batch_id / "input" / "source.zip"
    raw_uri = f"s3://{PROJECT_BUCKET}/{RAW_DATA_PREFIX}/{batch_id}/source.zip"
    result = {"zip_path": str(zip_path), "raw_copy": raw_uri, "deleted": False}

    if not zip_path.is_file():
        result["reason"] = "local ZIP not found"
        return result

    unrendered = count_unrendered_pages(batch_id)
    if unrendered is None:
        result["reason"] = "page_inventory.parquet not found"
        return result
    if unrendered:
        result["reason"] = f"{unrendered} page rows are not 'success'"
        return result

    local_size = zip_path.stat().st_size
    try:
        s3_client = boto3.client("s3", region_name=AWS_REGION)
        head = get_s3_head(s3_client, PROJECT_BUCKET, f"{RAW_DATA_PREFIX}/{batch_id}/source.zip")
    except (ClientError, BotoCoreError) as error:
        result["reason"] = f"S3 check failed: {error}"
        return result
    if head is None:
        result["reason"] = "raw copy not found in S3"
        return result
    if int(head["ContentLength"]) != local_size:
        result["reason"] = f"S3 size {head['ContentLength']} differs from local size {local_size}"
        return result

    zip_path.unlink()
    try:
        zip_path.parent.rmdir()  # remove input/ if it is now empty
    except OSError:
        pass
    result.update({"deleted": True, "freed_bytes": local_size})
    return result


def print_cleanup(result: dict) -> None:
    """Report what happened to the local ZIP."""
    if result["deleted"]:
        print(f"\nLocal ZIP deleted ({result['freed_bytes'] / 1024**3:.1f} GiB freed): {result['zip_path']}")
        print("Restore it if stage 02 or 03 must run again:")
        print(f"  aws s3 cp {result['raw_copy']} {result['zip_path']}")
    else:
        print(f"\nLocal ZIP kept ({result['reason']}): {result['zip_path']}")


def main() -> None:
    """Run the selected stages and write the run summary."""
    args = build_parser().parse_args()
    if args.source_s3_uri:
        args.source_s3_uri = resolve_source_uri(args.source_s3_uri, SOURCE_ZIP_PREFIX)
    selected = select_stages(args)
    batch_id = resolve_batch_id(args)
    args.batch_id = batch_id

    summary = {
        "batch_id": batch_id,
        "started_at_utc": utc_now_iso(),
        "stages": [],
        "status": "running",
    }
    summary_path = LOCAL_TEMP_ROOT / batch_id / "manifests" / "run_summary.json"

    overall_success = True
    for number, script in selected:
        entry = run_stage(number, script, args)
        summary["stages"].append(entry)
        atomic_write_json(summary_path, summary)
        if entry["status"] == "failed":
            overall_success = False
            print(f"\nStage {number} failed (exit code {entry['return_code']}). Stopping.")
            break
        print(f"Stage {number} completed in {entry['duration_seconds']}s.")
        # Stop after the download stage so the user can remove the tokens
        # before the upload stages (01 prints the exact instructions).
        if number == "01":
            summary["status"] = "paused_after_download"
            summary["finished_at_utc"] = utc_now_iso()
            atomic_write_json(summary_path, summary)
            print(f"\nRun summary: {summary_path}")
            sys.exit(0)

    rendered = any(number == "03" for number, _ in selected)
    if overall_success and rendered and not args.keep_zip:
        summary["zip_cleanup"] = cleanup_local_zip(batch_id)
        print_cleanup(summary["zip_cleanup"])

    summary["status"] = "success" if overall_success else "failed"
    summary["finished_at_utc"] = utc_now_iso()
    atomic_write_json(summary_path, summary)

    print(f"\nRun summary: {summary_path}")
    print(f"Overall status: {summary['status']}")
    sys.exit(0 if overall_success else 1)


if __name__ == "__main__":
    main()
