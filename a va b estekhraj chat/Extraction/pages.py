"""Resolve prediction rows through existing metadata; never guess between images."""
import csv
import json
from collections import defaultdict
from decimal import Decimal, InvalidOperation
from pathlib import Path

import config


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def page_number(value):
    try:
        number = Decimal(str(value))
        if not number.is_finite() or number != int(number) or number < 1:
            raise ValueError("Invalid page number")
        return int(number)
    except (InvalidOperation, ValueError, OverflowError) as error:
        raise ValueError(f"Invalid page_number: {value!r}") from error


def key(row):
    mid = str(row.get("masterindex_id") or "").strip()
    if not mid:
        raise ValueError("Missing masterindex_id")
    return mid, page_number(row.get("page_number"))


def metadata_rows():
    files = [path for path in config.METADATA_FILES if path.is_file()]
    if not files:
        raise FileNotFoundError(
            "No extraction metadata file exists. Checked:\n- "
            + "\n- ".join(str(path) for path in config.METADATA_FILES)
        )
    for path in files:
        print(f"Metadata: {path}")
        if path.suffix == ".csv":
            rows = read_csv(path)
        else:
            rows = []
            with path.open(encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, 1):
                    if line.strip():
                        try:
                            rows.append(json.loads(line))
                        except json.JSONDecodeError as error:
                            raise ValueError(f"Invalid JSON in {path}, line {line_number}") from error
        for row in rows:
            yield {**row, "_metadata_source": str(path)}


def metadata_index():
    indexed = defaultdict(list)
    for row in metadata_rows():
        indexed[key(row)].append(row)
    return indexed


def image_path(text):
    path = Path(text)
    candidates = [path] if path.is_absolute() else [root / path for root in config.IMAGE_ROOTS]
    found = {p.resolve() for p in candidates if p.is_file()}
    if len(found) != 1:
        raise ValueError(f"Image path is missing or ambiguous: {text}")
    return found.pop()


def matching_metadata(row, index):
    matches = index.get(key(row), [])
    for name in ("pdf_path_in_zip", "image_sha256"):
        if row.get(name):
            matches = [m for m in matches if not m.get(name) or m[name] == row[name]]
    return matches


def resolve(row, index):
    matches = matching_metadata(row, index)
    text = row.get("resolved_image_path") or row.get("image_path")
    if not text:
        paths = {m["image_path"] for m in matches if m.get("image_path")}
        if not paths:
            raise ValueError(f"Metadata lookup found no image path for {key(row)}")
        if len(paths) > 1:
            raise ValueError(f"Metadata lookup found {len(paths)} image paths for {key(row)}")
        text = paths.pop()
    path = image_path(text)
    matching = [m for m in matches if m.get("image_path") in {text, str(path)}]
    enriched = dict(row)
    for name in ("pdf_path_in_zip", "source_page_number", "image_sha256", "_metadata_source"):
        values = {str(m[name]) for m in matching if m.get(name) is not None and str(m[name])}
        if not enriched.get(name) and len(values) == 1:
            enriched[name] = values.pop()
        elif not enriched.get(name) and len(values) > 1:
            raise ValueError(f"Conflicting metadata for {key(row)}: {name}")
    return {**enriched, "masterindex_id": key(row)[0], "page_number": key(row)[1],
            "resolved_image_path": str(path)}


def load_pages(input_csv):
    index, pages, seen = metadata_index(), [], set()
    for row in read_csv(input_csv):
        if "predicted_page_sst" not in row:
            raise ValueError("Prediction CSV must contain predicted_page_sst")
        if row["predicted_page_sst"].strip() != "G07":
            continue
        try:
            page = resolve(row, index)
        except Exception as error:
            pages.append({**row, "resolution_error": str(error)})
            continue
        identity = (page["masterindex_id"], page.get("pdf_path_in_zip"), page["resolved_image_path"])
        if identity not in seen:
            seen.add(identity)
            pages.append(page)
    return pages
