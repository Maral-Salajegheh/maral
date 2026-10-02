"""Prepare Life page rows from the mapping and rendered-page inventory."""

import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow.parquet as pq

from .config import CLASSES, PipelineConfig

REQUIRED_PAGE_COLUMNS = {
    "masterindex_id", "stack_id", "process_id", "doc_id", "subdoc_idx",
    "image_id", "seqno", "pdf_page_number", "page_number", "n_pages",
    "sst", "sfdoc_class", "page_class", "page_class_source", "label_tier",
    "training_label_quality", "is_first_page", "is_last_page", "mapping_status",
    "is_usable", "is_training_eligible",
}
PAGE_FIELDS = [
    "masterindex_id", "stack_id", "process_id", "doc_id", "subdoc_idx",
    "image_id", "seqno", "page_number", "pdf_page_number", "image_path",
    "image_sha256", "quality_status", "label_tier", "training_label_quality",
    "sst", "sfdoc_class", "page_class", "page_class_source", "is_first_page",
    "is_last_page",
]
VERIFY_FIELDS = [
    "masterindex_id", "doc_id", "image_id", "seqno", "pdf_page_number",
    "sfdoc_class", "page_class", "image_path", "image_sha256",
]


def read_csv(path: Path):
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in fields} for row in rows)


def group_by(rows, key):
    grouped = defaultdict(list)
    for row in rows:
        grouped[str(row[key])].append(row)
    return grouped


def load_mapped_pages(config: PipelineConfig):
    rows = read_csv(config.pages_csv)
    if not rows:
        raise ValueError(f"No rows found in {config.pages_csv}")
    missing = REQUIRED_PAGE_COLUMNS - set(rows[0])
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    for row in rows:
        if row["is_usable"].lower() != "true":
            raise ValueError("life_mid_pages.csv contains an unusable row")
        if row["page_class"] not in CLASSES:
            raise ValueError(f"Unexpected class: {row['page_class']}")
        if row["page_class"] == "TB0" and row["sst"] != "A00":
            raise ValueError("TB0 label found with non-A00 SST")
    return rows


def resolve_image_rows(config: PipelineConfig, pages):
    inventory = pq.read_table(
        config.page_inventory,
        columns=["masterindex_id", "source_page_number", "image_path", "image_sha256", "status", "quality_status"],
    ).to_pylist()
    by_mid = group_by([row for row in inventory if row.get("status") == "success"], "masterindex_id")
    matched, skipped = [], []
    for row in pages:
        out = dict(row)
        images = by_mid.get(row["masterindex_id"])
        if not images:
            out["match_status"] = "MID_NOT_RENDERED"
            skipped.append(out)
            continue
        candidates = [image for image in images if str(image["source_page_number"]) == str(row["pdf_page_number"])]
        if len(candidates) != 1:
            out["match_status"] = "RENDERED_PAGE_UNMATCHED" if not candidates else "RENDERED_PAGE_AMBIGUOUS"
            out["rendered_page_count"] = str(len(images))
            skipped.append(out)
            continue
        image = candidates[0]
        out.update({
            "image_path": image["image_path"],
            "image_sha256": image["image_sha256"],
            "quality_status": str(image.get("quality_status") or ""),
            "match_status": "MATCHED",
        })
        matched.append(out)
    return matched, skipped, inventory


def build_report(pages, matched, skipped, inventory):
    rendered_mids = {str(row["masterindex_id"]) for row in inventory if row.get("status") == "success"}
    by_status = Counter(row["match_status"] for row in skipped)
    image_sha_counts = Counter(row["image_sha256"] for row in matched)
    return {
        "mapped": {"mids": len({row["masterindex_id"] for row in pages}), "stacks": len({row["stack_id"] for row in pages}), "pages": len(pages)},
        "rendered_mids": len(rendered_mids),
        "matched_rows": len(matched),
        "skipped_rows": len(skipped),
        "skipped_by_status": dict(by_status),
        "matched_pages_per_class": dict(Counter(row["page_class"] for row in matched)),
        "skipped_pages_per_class": dict(Counter(row["page_class"] for row in skipped)),
        "skipped_pages_by_status_and_class": {
            status: dict(Counter(row["page_class"] for row in skipped if row["match_status"] == status))
            for status in sorted(by_status)
        },
        "quality_status": dict(Counter(str(row.get("quality_status") or "UNSET") for row in inventory)),
        "duplicate_image_paths": [path for path, count in Counter(row["image_path"] for row in matched).items() if count > 1],
        "duplicate_image_sha256_row_count": sum(count for count in image_sha_counts.values() if count > 1),
        "label_tier_matched": dict(Counter(row["training_label_quality"] for row in matched)),
        "sst_page_class": {f"{sst} x {label}": count for (sst, label), count in Counter((row["sst"], row["page_class"]) for row in pages).items()},
    }


def write_verification_sheet(config: PipelineConfig, rows):
    selected, selected_mids, tb0_mids = [], set(), set()
    for row in rows:
        if row["page_class"] == "TB0" and row["masterindex_id"] not in selected_mids:
            selected.append(row); selected_mids.add(row["masterindex_id"]); tb0_mids.add(row["masterindex_id"])
        if len(tb0_mids) >= 5:
            break
    for label in CLASSES:
        for row in rows:
            if row["page_class"] == label and row["masterindex_id"] not in selected_mids:
                selected.append(row); selected_mids.add(row["masterindex_id"])
            if len(selected_mids) >= 20:
                break
        if len(selected_mids) >= 20:
            break
    if len(selected_mids) < 20 or len(tb0_mids) < 5:
        raise ValueError(f"Verification requires 20 MIDs and 5 TB0 MIDs; got {len(selected_mids)} and {len(tb0_mids)}")
    write_csv(config.verification_sheet, selected, VERIFY_FIELDS)


def prepare_verified_data(config: PipelineConfig):
    config.classification_output.mkdir(parents=True, exist_ok=True)
    pages = load_mapped_pages(config)
    matched, skipped, inventory = resolve_image_rows(config, pages)
    report = build_report(pages, matched, skipped, inventory)
    config.validation_report.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    write_csv(config.matched_pages_csv, matched, PAGE_FIELDS + ["match_status"])
    write_csv(config.skipped_pages_csv, skipped, PAGE_FIELDS + ["match_status", "rendered_page_count"])
    write_verification_sheet(config, matched)
    return report


def load_existing_splits(path: Path):
    if not path.exists():
        return {}
    return {row["stack_id"]: row["split"] for row in read_csv(path)}


def assign_splits(rows, split_file: Path, seed=13, train_ratio=0.8, val_ratio=0.1):
    existing = load_existing_splits(split_file)
    stack_rows = group_by(rows, "stack_id")
    assignments = {stack: split for stack, split in existing.items() if stack in stack_rows}
    new_stacks = [stack for stack in stack_rows if stack not in assignments]
    rng = random.Random(seed); rng.shuffle(new_stacks)
    targets = {"train": train_ratio, "val": val_ratio, "test": 1 - train_ratio - val_ratio}
    counts = Counter(assignments.values())
    for stack in new_stacks:
        split = min(targets, key=lambda name: counts[name] / max(targets[name], 1e-9))
        assignments[stack] = split; counts[split] += 1
    tb0 = Counter(assignments[stack] for stack, stack_pages in stack_rows.items() if any(row["page_class"] == "TB0" for row in stack_pages))
    if tb0["val"] < 10 or tb0["test"] < 10:
        raise ValueError(f"Validation/test need at least 10 TB0 stacks: {dict(tb0)}")
    write_csv(split_file, [{"stack_id": stack, "split": assignments[stack]} for stack in sorted(stack_rows)], ["stack_id", "split"])
    return assignments
