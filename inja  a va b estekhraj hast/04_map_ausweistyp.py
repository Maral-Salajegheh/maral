from __future__ import annotations

import argparse
from pathlib import Path

from Life.Extraction import config
from Life.Extraction.mrz_utils import map_ausweistyp
from Life.Extraction.utils import append_jsonl, page_key, read_latest_jsonl, utc_now_iso


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=config.MRZ_PARSED_JSONL)
    parser.add_argument("--output", type=Path, default=config.AUSWEISTYP_JSONL)
    args = parser.parse_args()
    records = read_latest_jsonl(args.input, page_key)
    done = read_latest_jsonl(args.output, page_key)
    for key, record in sorted(records.items()):
        previous = done.get(key)
        if (
            previous
            and previous.get("status") == "success"
            and previous.get("document_number") == record.get("document_number")
            and previous.get("mrz_ocr_text") == record.get("mrz_ocr_text")
            and previous.get("mrz_crop_path") == record.get("mrz_crop_path")
            and previous.get("detector_version") == record.get("detector_version")
        ):
            continue
        if record.get("status") != "success" or not record.get("checks_valid"):
            append_jsonl(args.output, {**record, "status": "failed", "needs_human_review": True, "error": record.get("error") or "MRZ checks did not pass", "processed_at_utc": utc_now_iso()})
            continue
        output = {**record, "ausweistyp": map_ausweistyp(str(record.get("document_code") or "")), "status": "success", "needs_human_review": False, "error": "", "processed_at_utc": utc_now_iso()}
        append_jsonl(args.output, output)
    print(f"Wrote Ausweistyp cache: {args.output}")


if __name__ == "__main__":
    main()
