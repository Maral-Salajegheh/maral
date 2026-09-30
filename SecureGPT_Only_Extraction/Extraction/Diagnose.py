"""Report saved extraction statuses without printing personal field values."""
import argparse
import json
from collections import Counter
from pathlib import Path
import config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", nargs="?", type=Path)
    args = parser.parse_args()
    path = args.run
    if path is None:
        runs = [p for p in config.OUTPUT_DIR.glob("*") if (p / "summary.json").is_file()]
        if not runs:
            parser.error("No completed or interrupted SecureGPT-only output found")
        path = max(runs, key=lambda p: (p / "summary.json").stat().st_mtime)
    if path.is_dir():
        path = path / "pages.jsonl"
    counts = Counter()
    with path.open(encoding="utf-8") as handle:
        for index, line in enumerate(handle, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            status = record.get("llm_status", "unknown")
            counts[status] += 1
            print(f"Record {index}: status={status}, observations={len(record.get('cards', []))}")
    print("Page statuses:", dict(counts))
    summary = path.parent / "summary.json"
    if summary.exists():
        value = json.loads(summary.read_text(encoding="utf-8"))
        print("Document counts:", {k: value.get(k) for k in
                                  ("documents", "ready", "review_documents", "unresolved_records", "excluded_cards")})


if __name__ == "__main__":
    main()
