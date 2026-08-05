"""Orchestrator: run the ingestion stages for one batch, in order, stopping at
the first failure. Writes run_summary.json locally and to S3.

Examples:
    # New batch: batch_id is derived from date and ZIP filename
    python run_pipeline.py --source-s3-uri s3://life-prod-02-out/export.zip

    # Re-run part of an existing batch: name it explicitly
    python run_pipeline.py --batch-id life_20260804_export --from-stage 04
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

import boto3

from common import atomic_write_json, build_batch_id, utc_now_iso, validate_batch_id
from config import AWS_REGION, CORPUS_ROOT_PREFIX, LOCAL_TEMP_ROOT, PROJECT_BUCKET

SCRIPT_DIR = Path(__file__).resolve().parent
STAGES = [
    ("01", "01_download_batch_zip.py"),
    ("02", "02_unpack_and_upload_documents.py"),
    ("03", "03_build_document_inventory.py"),
    ("04", "04_render_pdf_pages.py"),
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
        "--source-s3-uri", default=None, help="Required when stage 01 runs."
    )
    parser.add_argument("--from-stage", choices=STAGE_NUMBERS, default="01")
    parser.add_argument("--to-stage", choices=STAGE_NUMBERS, default="04")
    parser.add_argument(
        "--extra-args-04", default="", help="Extra arguments for stage 04."
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
    if number == "04" and args.extra_args_04:
        command += args.extra_args_04.split()
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


def upload_summary(summary_path: Path, batch_id: str) -> None:
    """Upload run_summary.json to S3; failure to upload is only a warning."""
    try:
        s3_client = boto3.client("s3", region_name=AWS_REGION)
        s3_client.upload_file(
            str(summary_path),
            PROJECT_BUCKET,
            f"{CORPUS_ROOT_PREFIX}/manifests/{batch_id}/run_summary.json",
            ExtraArgs={"ContentType": "application/json"},
        )
    except Exception as error:
        print(f"Warning: could not upload run summary to S3: {error}")


def main() -> None:
    """Run the selected stages and write the run summary."""
    args = build_parser().parse_args()
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

    summary["status"] = "success" if overall_success else "failed"
    summary["finished_at_utc"] = utc_now_iso()
    atomic_write_json(summary_path, summary)
    upload_summary(summary_path, batch_id)

    print(f"\nRun summary: {summary_path}")
    print(f"Overall status: {summary['status']}")
    sys.exit(0 if overall_success else 1)


if __name__ == "__main__":
    main()
