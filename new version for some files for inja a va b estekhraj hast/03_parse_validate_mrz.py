from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from Life.Extraction import config
from Life.Extraction.mrz_utils import parse_with_repair
from Life.Extraction.utils import append_jsonl, page_key, read_latest_jsonl, utc_now_iso


# Re-read a band the OCR could not parse, but only when stage 01 cropped it successfully.
# A finger or glare over a character is worth a second look; a failed crop is not, because
# the model would just be describing whatever the wrong rectangle happened to contain.
_LLM_AVAILABLE = True


def llm_retry(record: dict[str, Any]) -> tuple[dict[str, Any] | None, str, str]:
    global _LLM_AVAILABLE
    crop_path = Path(str(record.get("mrz_crop_path") or ""))
    if not config.MRZ_LLM_FALLBACK or not _LLM_AVAILABLE or not crop_path.is_file():
        return None, "", ""
    try:
        from Life.Extraction.llm_mrz import transcribe_mrz

        text = transcribe_mrz(crop_path)
    except ImportError:
        # SecureGPT is not importable in this environment. Skip the retry for the rest of
        # the run and let the real parse error stand, instead of masking it with this one.
        _LLM_AVAILABLE = False
        return None, "", ""
    return parse_with_repair(text), text, "llm_vision"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=config.MRZ_OCR_JSONL)
    parser.add_argument("--output", type=Path, default=config.MRZ_PARSED_JSONL)
    parser.add_argument("--force", action="store_true", help="Ignore cached results and reprocess every page")
    args = parser.parse_args()
    records = read_latest_jsonl(args.input, page_key)
    done = read_latest_jsonl(args.output, page_key)
    for key, record in sorted(records.items()):
        # Validate the current input first: an upstream failure must not leave an old success in place.
        previous = None if args.force or record.get("status") != "success" else done.get(key)
        if (
            previous
            and previous.get("status") == "success"
            and previous.get("mrz_ocr_text") == record.get("mrz_ocr_text")
            and previous.get("mrz_crop_path") == record.get("mrz_crop_path")
            and previous.get("detector_version") == record.get("detector_version")
            and previous.get("mrz_parser_version") == config.MRZ_PARSER_VERSION
        ):
            continue
        if record.get("status") != "success":
            append_jsonl(args.output, {**record, "status": "failed", "needs_human_review": True, "error": record.get("error") or "MRZ OCR failed", "processed_at_utc": utc_now_iso()})
            continue
        try:
            parsed = parse_with_repair(record.get("mrz_ocr_text") or "")
            source, llm_text = "ocr", ""
            if not parsed:
                parsed, llm_text, source = llm_retry(record)
            if not parsed:
                raise ValueError("No checksum-valid TD1/TD2/TD3 MRZ parsed")
            extra = {"mrz_transcription_source": source or "ocr"}
            if llm_text:
                extra["mrz_llm_text"] = llm_text
            output = {**record, **parsed, **extra, "status": "success", "needs_human_review": False, "error": "", "processed_at_utc": utc_now_iso()}
        except Exception as error:
            output = {**record, "status": "failed", "needs_human_review": True, "error": str(error), "processed_at_utc": utc_now_iso()}
        append_jsonl(args.output, output)
    print(f"Wrote parsed MRZ cache: {args.output}")


if __name__ == "__main__":
    main()
