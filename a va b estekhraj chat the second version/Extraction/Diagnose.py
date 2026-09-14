"""Private MRZ diagnostics. Reads saved OCR only; makes no OCR or LLM calls.

Run from the project root:
    pixi run python Extraction/Diagnose.py
    pixi run python Extraction/Diagnose.py outputs/run_folder --row 2

Reports JSONL record numbers, never MID, paths, OCR text, field values or raw errors.
"""
import argparse
import json
import re
from collections import Counter
from pathlib import Path

import config
from mrz import inspect_mrz


def latest_run():
    runs = sorted(path for path in config.OUTPUT_DIR.glob("*") if (path / "pages.jsonl").is_file())
    if not runs:
        raise SystemExit("No saved pages.jsonl found. Supply a run folder or the JSONL file.")
    return runs[-1]


def safe_view(value):
    value = str(value or "")
    return value if re.fullmatch(r"rotation_(0|90|180|270)_(crop_\d+(_psm11)?|full)", value) else "unlabelled_view"


def report_page(page, index):
    audit = page.get("mrz") or {}
    stored = "accepted" if audit.get("parsed") else "not_accepted"
    if page.get("resolution_error") or page.get("metadata", {}).get("resolution_error"):
        print(f"Record {index}: image_resolution_failed")
        return "image_resolution_failed"
    print(f"Record {index}: saved MRZ result={stored}")
    attempts = audit.get("attempts") or []
    if not attempts:
        print("  No recorded OCR attempts; inspect image/toolchain locally.")
        return "no_recorded_attempts"
    outcomes = []
    for attempt_index, attempt in enumerate(attempts, 1):
        # Re-analyse old records with exactly the current parser; do not trust an old label.
        _, diagnostic = inspect_mrz(attempt.get("text") or "")
        reason = "ocr_error" if attempt.get("status") == "ocr_error" else diagnostic["reason"]
        outcomes.append(reason)
        lengths = ", ".join(map(str, diagnostic["line_lengths"])) or "none"
        print(f"  Attempt {attempt_index} {safe_view(attempt.get('view'))}: {reason}")
        print(f"    OCR line lengths (whitespace excluded): [{lengths}]")
        invalid = diagnostic["invalid_character_counts"]
        if any(invalid):
            print(f"    Invalid-character counts per line: {invalid}")
        for candidate in diagnostic["candidates"]:
            labels = ", ".join(candidate["failed_checks"]) or "passed"
            print(f"    {candidate['format']} from OCR line {candidate['start_line']}: {labels}")
    if "parsed" in outcomes and not audit.get("parsed"):
        print("  Current parser accepts at least one attempt; check other attempts for conflicts.")
    if audit.get("errors"):
        print(f"  Recorded technical/ambiguity errors: {len(audit['errors'])} (details withheld)")
    if audit.get("multiple_documents"):
        print("  Conflicting MRZ readings were recorded; review required.")
    return "saved_accepted" if audit.get("parsed") else "saved_not_accepted"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", nargs="?", type=Path, help="Run folder or pages.jsonl")
    parser.add_argument("--row", type=int, help="One-based nonempty JSONL record number, not PDF page number")
    args = parser.parse_args()
    if args.row is not None and args.row < 1:
        parser.error("--row must be positive")
    path = args.run or latest_run()
    if path.is_dir():
        path = path / "pages.jsonl"
    counts, index = Counter(), 0
    try:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                index += 1
                if args.row is not None and args.row != index:
                    continue
                try:
                    page = json.loads(line)
                    counts[report_page(page, index)] += 1
                except (ValueError, TypeError, AttributeError, KeyError):
                    print(f"Record {index}: invalid_audit_record (contents withheld)")
                    counts["invalid_audit_record"] += 1
    except OSError:
        raise SystemExit("Cannot open audit file. Check the supplied path locally.") from None
    if args.row is not None and args.row > index:
        raise SystemExit("Requested record is outside this file.")
    print("Summary:", dict(counts))
    print("Expected widths: TD1=30/30/30; TD2=36/36; TD3=44/44.")
    print("No MRZ-shaped OCR does not prove the image has no MRZ. Inspect the crop locally.")
    print("A checksum failure does not reveal the correct replacement character.")


if __name__ == "__main__":
    main()
