"""Explain every MRZ failure in a run. Reads pages.jsonl only; runs no OCR and no LLM.

    pixi run python Extraction/diagnose.py              # newest run
    pixi run python Extraction/diagnose.py <run_dir>    # a specific run
"""
import json
import re
import sys
from collections import Counter
from pathlib import Path

import config
from mrz import padded_variants, parse_td1, parse_td2, parse_td3, valid_date

FORMATS = (("TD1", 3, 30, parse_td1), ("TD2", 2, 36, parse_td2), ("TD3", 2, 44, parse_td3))
CHECK_NAMES = {"TD1": ["document_number", "date_of_birth", "expiry_date", "composite"],
               "TD2": ["document_number", "date_of_birth", "expiry_date", "composite"],
               "TD3": ["document_number", "date_of_birth", "expiry_date", "optional_data", "composite"]}


def latest_run():
    runs = sorted(path for path in config.OUTPUT_DIR.glob("*") if (path / "pages.jsonl").is_file())
    if not runs:
        raise SystemExit(f"No run with pages.jsonl under {config.OUTPUT_DIR}")
    return runs[-1]


def ocr_lines(page):
    for attempt in (page.get("mrz") or {}).get("attempts") or []:
        for line in attempt.get("text", "").splitlines():
            cleaned = re.sub(r"\s", "", line.upper())
            if cleaned:
                yield attempt.get("view", "?"), cleaned


# Windows of consecutive lines that could be one MRZ format, after filler restoration.
def candidate_windows(page):
    by_view = {}
    for view, line in ocr_lines(page):
        by_view.setdefault(view, []).append(line)
    for view, lines in by_view.items():
        for fmt, count, width, parser in FORMATS:
            for start in range(len(lines) - count + 1):
                window = lines[start:start + count]
                if not all(re.fullmatch(r"[A-Z0-9<]+", item) for item in window):
                    continue
                variants = [padded_variants(item, width) for item in window]
                if all(variants):
                    yield view, fmt, width, parser, window, [group[0] for group in variants]


# Which individual check digit fails, so a composite-only failure is visible as such.
def failing_checks(fmt, parser, lines):
    code, number, nationality, expiry, birth, sex, checks = parser(lines)
    failed = [name for name, ok in zip(CHECK_NAMES[fmt], checks) if not ok]
    if not valid_date(birth):
        failed.append("birth_not_a_date")
    if not valid_date(expiry):
        failed.append("expiry_not_a_date")
    if code[0] not in "IPAC":
        failed.append(f"document_code={code!r}")
    if sex not in "MF<":
        failed.append(f"sex={sex!r}")
    if not re.fullmatch(r"[A-Z<]{3}", nationality):
        failed.append(f"nationality={nationality!r}")
    return failed


def classify(page):
    if page.get("resolution_error"):
        return "image_not_resolved", page["resolution_error"]
    if (page.get("mrz") or {}).get("parsed"):
        return "parsed", ""
    if (page.get("mrz") or {}).get("multiple_documents"):
        return "multiple_mrz_on_page", ""
    attempts = (page.get("mrz") or {}).get("attempts") or []
    if not attempts:
        return "no_ocr_attempt", "; ".join((page.get("mrz") or {}).get("errors") or [])
    best = None
    for view, fmt, width, parser, window, restored in candidate_windows(page):
        failed = failing_checks(fmt, parser, restored)
        if not failed:
            return "would_parse_now", f"{fmt} in {view}"
        note = f"{fmt} in {view}: " + ", ".join(failed)
        if best is None or len(failed) < best[1]:
            best = (note, len(failed), failed)
    if best:
        # Only the whole-line composite failed: every business field validated on its own.
        reason = "composite_only" if best[2] == ["composite"] else "checks_failed"
        return reason, best[0]
    lengths = sorted({len(line) for _view, line in ocr_lines(page)}, reverse=True)[:6]
    return "no_mrz_shaped_text", f"longest OCR lines: {lengths}"


# A stored run cannot see a tool installed after it finished; say so rather than
# letting an old failure look like a current one.
def stale_run_warning(counts, notes):
    import shutil

    if not counts.get("no_ocr_attempt"):
        return
    missing_then = " ".join(notes)
    now = []
    if "tesseract" in missing_then.lower() and shutil.which(config.TESSERACT_CMD):
        now.append("Tesseract")
    if "cv2" in missing_then:
        try:
            import cv2  # noqa: F401

            now.append("OpenCV")
        except ImportError:
            pass
    if now:
        print(f"\nNOTE: {' and '.join(now)} is available now but was missing when this run "
              f"was written. Re-run extract.py and diagnose the new run.")


def main():
    run = Path(sys.argv[1]) if len(sys.argv) > 1 else latest_run()
    print(f"Run: {run}\n")
    counts, notes = Counter(), []
    for line in (run / "pages.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        page = json.loads(line)
        reason, detail = classify(page)
        counts[reason] += 1
        if reason == "no_ocr_attempt":
            notes.append(detail)
        meta = page.get("metadata") or page
        print(f"{meta.get('masterindex_id')} p{meta.get('page_number')}  {reason}"
              + (f"  |  {detail[:110]}" if detail else ""))
    stale_run_warning(counts, notes)
    print("\nSUMMARY")
    for reason, count in counts.most_common():
        print(f"  {count:4d}  {reason}")
    print("""
  image_not_resolved   metadata/path problem, no OCR ran. Fix METADATA_FILES / IMAGE_ROOTS.
  no_ocr_attempt       Tesseract or OpenCV missing, or the image failed to open.
  no_mrz_shaped_text   nothing near 30/36/44 characters was read. Inspect the saved crops.
  composite_only       every field check passed; only the whole-line composite failed.
  checks_failed        a data field's check digit failed: characters were misread.
  would_parse_now      parses with the current code; this run predates the change.""")


if __name__ == "__main__":
    main()
    
    
    
    
    
    cd ~/Projects/life-docai
unset TESSERACT_CMD
pixi run bash -c 'echo "[$TESSERACT_CMD]"; tesseract --version | head -1'


grep -rn TESSERACT_CMD ~/.bashrc ~/.bash_profile ~/.profile pixi.toml 2>/dev/null