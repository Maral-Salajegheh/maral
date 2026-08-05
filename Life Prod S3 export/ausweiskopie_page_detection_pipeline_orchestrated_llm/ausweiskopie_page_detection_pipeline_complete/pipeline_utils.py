from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


def utc_now_iso() -> str:
    """Return the current UTC timestamp in ISO format."""
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    """Return the SHA-256 checksum of one file."""
    digest = hashlib.sha256()

    with path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    """Append one JSON object immediately."""
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("a", encoding="utf-8") as output_file:
        output_file.write(
            json.dumps(record, ensure_ascii=False, default=str) + "\n"
        )


def read_latest_jsonl(
    path: Path,
    key_function: Callable[[dict[str, Any]], tuple[Any, ...]],
) -> dict[tuple[Any, ...], dict[str, Any]]:
    """Keep the latest valid record for every supplied key."""
    latest: dict[tuple[Any, ...], dict[str, Any]] = {}

    if not path.exists():
        return latest

    with path.open("r", encoding="utf-8") as input_file:
        for line in input_file:
            line = line.strip()
            if not line:
                continue

            try:
                record = json.loads(line)
                key = key_function(record)
            except (
                json.JSONDecodeError,
                KeyError,
                TypeError,
                ValueError,
            ):
                continue

            latest[key] = record

    return latest


def page_record_key(
    record: dict[str, Any],
) -> tuple[str, str, int, str]:
    """
    Return a stable page key.

    The image checksum prevents stale successful results from being reused
    when a page image changes.
    """
    return (
        str(record["masterindex_id"]),
        str(record["pdf_path_in_zip"]),
        int(record["source_page_number"]),
        str(record.get("image_sha256") or "NO_IMAGE"),
    )


def build_ausweis_segments(
    pages: list[dict[str, Any]],
) -> list[dict[str, int]]:
    """
    Group consecutive positive pages without crossing PDF boundaries.

    Each row must contain page, label, and pdf_path_in_zip.
    """
    segments: list[dict[str, int]] = []
    start_page: int | None = None
    previous_page: int | None = None
    current_pdf: str | None = None

    for page in sorted(pages, key=lambda item: item["page"]):
        page_number = int(page["page"])
        label = str(page["label"])
        pdf_path = str(page.get("pdf_path_in_zip") or "")

        continues_current_segment = (
            start_page is not None
            and previous_page is not None
            and page_number == previous_page + 1
            and pdf_path == current_pdf
        )

        if label == "ausweiskopie":
            if start_page is None:
                start_page = page_number
                current_pdf = pdf_path
            elif not continues_current_segment:
                segments.append({
                    "start_page": start_page,
                    "end_page": int(previous_page),
                })
                start_page = page_number
                current_pdf = pdf_path

            previous_page = page_number
            continue

        if start_page is not None:
            segments.append({
                "start_page": start_page,
                "end_page": int(previous_page),
            })
            start_page = None
            previous_page = None
            current_pdf = None

    if start_page is not None:
        segments.append({
            "start_page": start_page,
            "end_page": int(previous_page),
        })

    return segments
