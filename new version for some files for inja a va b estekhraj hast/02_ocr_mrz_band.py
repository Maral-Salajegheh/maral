from __future__ import annotations

import argparse
from pathlib import Path

from Life.Extraction import config
from Life.Extraction.utils import append_jsonl, int_value, page_key, read_latest_jsonl, run_mrz_ocr, utc_now_iso


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=config.MRZ_CROPS_JSONL)
    parser.add_argument("--output", type=Path, default=config.MRZ_OCR_JSONL)
    parser.add_argument("--force", action="store_true", help="Ignore cached results and reprocess every page")
    args = parser.parse_args()
    crops = read_latest_jsonl(args.input, page_key)
    done = read_latest_jsonl(args.output, page_key)
    for key, record in sorted(crops.items()):
        previous = None if args.force else done.get(key)
        if (
            previous
            and previous.get("status") == "success"
            and previous.get("mrz_crop_path") == record.get("mrz_crop_path")
            and previous.get("detector_version") == record.get("detector_version")
            and previous.get("mrz_ocr_stage_version") == config.MRZ_OCR_STAGE_VERSION
        ):
            continue
        if record.get("status") != "success":
            append_jsonl(args.output, {**record, "status": "failed", "needs_human_review": True, "error": record.get("error") or "MRZ crop failed", "processed_at_utc": utc_now_iso()})
            continue
        try:
            # Read the crop itself. Full-page detector text is only a last resort, because it
            # is recognized at page scale with no upscaling and loses MRZ characters.
            text, backend = run_mrz_ocr(Path(record["mrz_crop_path"]), int_value(record.get("line_count"), 2))
            if not text.strip() and record.get("mrz_detector_text"):
                text, backend = str(record["mrz_detector_text"]), f"{backend}+detector_text"
            output = {**record, "mrz_ocr_stage_version": config.MRZ_OCR_STAGE_VERSION, "mrz_ocr_backend": backend, "mrz_ocr_text": text,
                      "status": "success" if text.strip() else "failed", "needs_human_review": not text.strip(),
                      "error": "" if text.strip() else "MRZ crop produced no text", "processed_at_utc": utc_now_iso()}
        except Exception as error:
            output = {**record, "mrz_ocr_text": "", "status": "failed", "needs_human_review": True, "error": str(error), "processed_at_utc": utc_now_iso()}
        append_jsonl(args.output, output)
    print(f"Wrote MRZ OCR cache: {args.output}")


if __name__ == "__main__":
    main()
