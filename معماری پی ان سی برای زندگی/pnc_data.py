"""Validated Life population and complete-MID sequence construction."""

import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow.parquet as pq

from .config import CLASSES, CLASS_TO_ID, PipelineConfig
from .data import import_assign_splits, read_csv, write_csv

IGNORE_INDEX = -100
MODEL_FIELDS = [
    "page_identity",
    "masterindex_id",
    "stack_id",
    "pdf_page_number",
    "image_path",
    "image_sha256",
    "has_label",
    "page_class",
    "class_target",
    "is_first_page",
    "segmentation_target",
    "is_last_page",
    "last_page_target",
    "doc_id",
    "image_id",
    "seqno",
    "sfdoc_class",
    "label_tier",
    "training_label_quality",
    "placement",
]


def _bool(value):
    return str(value).lower() == "true"


def population_hash(rows):
    digest = hashlib.sha256()
    for row in sorted(rows, key=lambda item: (item["masterindex_id"], int(item["pdf_page_number"]))):
        digest.update(
            f"{row['page_identity']}{row['masterindex_id']}{row['stack_id']}{row['pdf_page_number']}{row['has_label']}\n".encode()
        )
    return digest.hexdigest()


def _write_new(path, rows, fields):
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing file: {path}")
    write_csv(path, rows, fields)


def validate_position_sets(label_positions, image_positions):
    label_positions = set(label_positions)
    image_positions = set(image_positions)
    missing = label_positions - image_positions
    if missing:
        raise ValueError(f"label_without_image:{sorted(missing)}")
    if label_positions and label_positions != set(range(1, max(label_positions) + 1)):
        raise ValueError("nonconsecutive_label_positions")
    extras = sorted(image_positions - label_positions)
    labelled_end = max(label_positions) if label_positions else 0
    expected = list(range(labelled_end + 1, labelled_end + 1 + len(extras)))
    if extras != expected:
        raise ValueError(f"ambiguous_unmatched_rendered:{extras}")
    return extras


def build_model_population(config: PipelineConfig):
    labels = read_csv(config.pages_csv)
    inventory = pq.read_table(
        config.page_inventory,
        columns=["masterindex_id", "source_page_number", "image_path", "image_sha256", "status"],
    ).to_pylist()
    labels_by_mid = defaultdict(list)
    images_by_mid = defaultdict(list)
    for row in labels:
        labels_by_mid[row["masterindex_id"]].append(row)
    for image in inventory:
        if image.get("status") == "success":
            images_by_mid[str(image["masterindex_id"])].append(image)

    model_rows = []
    invalid = []
    matched_label_rows = 0
    unmatched_rendered = 0
    safely_placeable = 0
    fully_matched_mids = 0
    duplicate_label_links = 0
    rendered_mid_missing_label_image = 0
    ambiguous_unmatched = 0

    for mid, images in sorted(images_by_mid.items()):
        mid_labels = labels_by_mid.get(mid, [])
        labels_by_page = defaultdict(list)
        images_by_page = defaultdict(list)
        for row in mid_labels:
            if not row.get("pdf_page_number"):
                invalid.append({"masterindex_id": mid, "reason": "null_pdf_page_number"})
                labels_by_page = None
                break
            labels_by_page[int(row["pdf_page_number"])].append(row)
        if labels_by_page is None:
            continue
        for image in images:
            images_by_page[int(image["source_page_number"])].append(image)
        if any(len(rows) != 1 for rows in labels_by_page.values()):
            duplicate_label_links += sum(len(rows) for rows in labels_by_page.values() if len(rows) > 1)
            invalid.append({"masterindex_id": mid, "reason": "duplicate_label_position"})
            continue
        if any(len(rows) != 1 for rows in images_by_page.values()):
            invalid.append({"masterindex_id": mid, "reason": "duplicate_rendered_position"})
            continue

        label_positions = set(labels_by_page)
        image_positions = set(images_by_page)
        try:
            extras = validate_position_sets(label_positions, image_positions)
        except ValueError as error:
            reason, _, positions = str(error).partition(":")
            if reason == "label_without_image":
                rendered_mid_missing_label_image += len(label_positions - image_positions)
            if reason == "ambiguous_unmatched_rendered":
                ambiguous_unmatched += len(image_positions - label_positions)
            invalid.append({"masterindex_id": mid, "reason": reason, "positions": positions})
            continue

        if not extras:
            fully_matched_mids += 1
        unmatched_rendered += len(extras)
        safely_placeable += len(extras)
        stack_ids = {row["stack_id"] for row in mid_labels}
        if len(stack_ids) != 1:
            invalid.append({"masterindex_id": mid, "reason": "missing_or_multiple_stack_ids", "stack_ids": sorted(stack_ids)})
            continue
        stack_id = next(iter(stack_ids))
        for position in sorted(image_positions):
            image = images_by_page[position][0]
            identity = str(image["image_path"])
            if position in labels_by_page:
                label = labels_by_page[position][0]
                matched_label_rows += 1
                out = {
                    "page_identity": identity,
                    "masterindex_id": mid,
                    "stack_id": stack_id,
                    "pdf_page_number": str(position),
                    "image_path": image["image_path"],
                    "image_sha256": image.get("image_sha256") or "",
                    "has_label": "True",
                    "page_class": label["page_class"],
                    "class_target": str(CLASS_TO_ID[label["page_class"]]),
                    "is_first_page": label["is_first_page"],
                    "segmentation_target": str(int(_bool(label["is_first_page"]))),
                    "is_last_page": label["is_last_page"],
                    "last_page_target": str(int(_bool(label["is_last_page"]))),
                    "doc_id": label["doc_id"],
                    "image_id": label["image_id"],
                    "seqno": label["seqno"],
                    "sfdoc_class": label["sfdoc_class"],
                    "label_tier": label["label_tier"],
                    "training_label_quality": label["training_label_quality"],
                    "placement": "labelled",
                }
            else:
                out = {
                    "page_identity": identity,
                    "masterindex_id": mid,
                    "stack_id": stack_id,
                    "pdf_page_number": str(position),
                    "image_path": image["image_path"],
                    "image_sha256": image.get("image_sha256") or "",
                    "has_label": "False",
                    "page_class": "",
                    "class_target": str(IGNORE_INDEX),
                    "is_first_page": "",
                    "segmentation_target": str(IGNORE_INDEX),
                    "is_last_page": "",
                    "last_page_target": str(IGNORE_INDEX),
                    "doc_id": "",
                    "image_id": "",
                    "seqno": "",
                    "sfdoc_class": "",
                    "label_tier": "",
                    "training_label_quality": "",
                    "placement": "trailing_unmatched",
                }
            model_rows.append(out)

    requested_mids = set(images_by_mid)
    mapping_mids = set(labels_by_mid)
    report = {
        "total_analyse_db_rows": len(labels),
        "total_rendered_images": sum(len(rows) for rows in images_by_mid.values()),
        "matched_labelled_images": matched_label_rows,
        "unmatched_rendered_images": unmatched_rendered,
        "unmatched_rendered_percentage": 100 * unmatched_rendered / max(1, sum(len(rows) for rows in images_by_mid.values())),
        "analyse_db_rows_mid_never_rendered": sum(len(rows) for mid, rows in labels_by_mid.items() if mid not in images_by_mid),
        "rendered_mid_label_rows_without_image": rendered_mid_missing_label_image,
        "rendered_images_matched_multiple_label_rows": duplicate_label_links,
        "requested_sample_mids": len(requested_mids),
        "rendered_mids_found": len(images_by_mid),
        "fully_matched_mids": fully_matched_mids,
        "missing_mids": len(requested_mids - mapping_mids),
        "invalid_or_incomplete_mids": len(invalid),
        "final_usable_complete_mid_sequences": len({row["masterindex_id"] for row in model_rows}),
        "unmatched_rendered_images_safely_placeable": safely_placeable,
        "unmatched_rendered_images_ambiguous": ambiguous_unmatched,
        "population_hash": population_hash(model_rows),
    }
    return model_rows, invalid, report


def prepare_model_population(config: PipelineConfig):
    rows, invalid, report = build_model_population(config)
    if report["rendered_mid_label_rows_without_image"] or report["rendered_images_matched_multiple_label_rows"]:
        raise RuntimeError(json.dumps(report, indent=2))
    if invalid:
        if not config.invalid_mids_report.exists():
            config.invalid_mids_report.write_text(json.dumps(invalid, indent=2), encoding="utf-8")
            raise RuntimeError(f"Unresolved invalid MIDs: {len(invalid)}; see {config.invalid_mids_report}")
    if config.population_report.exists():
        raise FileExistsError(f"Refusing to overwrite existing file: {config.population_report}")
    config.population_report.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return rows, report


def load_model_pages(config: PipelineConfig):
    rows = read_csv(config.model_pages_csv)
    validate_complete_mids(rows)
    return rows


def validate_complete_mids(rows):
    by_mid = defaultdict(list)
    for row in rows:
        by_mid[row["masterindex_id"]].append(row)
    errors = []
    for mid, mid_rows in by_mid.items():
        ordered = sorted(mid_rows, key=lambda row: int(row["pdf_page_number"]))
        positions = [int(row["pdf_page_number"]) for row in ordered]
        if positions != list(range(1, len(positions) + 1)):
            errors.append({"masterindex_id": mid, "reason": "nonconsecutive_model_sequence", "positions": positions})
        identities = [row["page_identity"] for row in ordered]
        if len(identities) != len(set(identities)):
            errors.append({"masterindex_id": mid, "reason": "duplicate_page_identity"})
    if errors:
        raise ValueError(f"Invalid MID sequences: {errors[:10]}")
    return by_mid


def select_complete_mids(rows, limit_mids=None):
    by_mid = validate_complete_mids(rows)
    selected = sorted(by_mid)
    if limit_mids is not None:
        selected = selected[:limit_mids]
    return [row for mid in selected for row in sorted(by_mid[mid], key=lambda item: int(item["pdf_page_number"]))]


def split_assignments(config: PipelineConfig, rows, seed=13):
    labelled = [row for row in rows if row["has_label"].lower() == "true"]
    return assign_splits(labelled, config.split_file, seed=seed)


def split_report(rows, assignments):
    by_split = defaultdict(list)
    for row in rows:
        by_split[assignments[row["stack_id"]]].append(row)
    report = {}
    for split, split_rows in by_split.items():
        mids = defaultdict(list)
        for row in split_rows:
            mids[row["masterindex_id"]].append(row)
        labelled = [row for row in split_rows if row["has_label"].lower() == "true"]
        doc_keys = {(row["masterindex_id"], row["doc_id"]) for row in labelled}
        internal_first = 0
        for mid_rows in mids.values():
            ordered = sorted(mid_rows, key=lambda row: int(row["pdf_page_number"]))
            internal_first += sum(int(row["segmentation_target"]) == 1 for row in ordered[1:])
        report[split] = {
            "pages": len(split_rows),
            "labelled_pages": len(labelled),
            "mids": len(mids),
            "single_page_mids": sum(len(value) == 1 for value in mids.values()),
            "multi_page_mids": sum(len(value) > 1 for value in mids.values()),
            "single_document_mids": sum(len({row["doc_id"] for row in value if row["has_label"].lower() == "true"}) == 1 for value in mids.values()),
            "multi_document_mids": sum(len({row["doc_id"] for row in value if row["has_label"].lower() == "true"}) > 1 for value in mids.values()),
            "internal_first_page_positives": internal_first,
            "page_class_counts": dict(Counter(row["page_class"] for row in labelled)),
        }
    stacks = {split: {row["stack_id"] for row in split_rows} for split, split_rows in by_split.items()}
    mids = {split: {row["masterindex_id"] for row in split_rows} for split, split_rows in by_split.items()}
    report["overlap"] = {
        "stack_overlap": sum(len(stacks[a] & stacks[b]) for a, b in [("train", "val"), ("train", "test"), ("val", "test")]),
        "mid_overlap": sum(len(mids[a] & mids[b]) for a, b in [("train", "val"), ("train", "test"), ("val", "test")]),
    }
    return report