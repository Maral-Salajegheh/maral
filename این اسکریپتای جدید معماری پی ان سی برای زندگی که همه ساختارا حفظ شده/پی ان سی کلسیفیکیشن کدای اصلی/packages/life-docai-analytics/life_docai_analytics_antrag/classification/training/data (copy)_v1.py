"""Stack-level Life dataset (PnC sequence unit): one sequence per stack_id, pages in image_id order.

image_id is the raw position of a page in the incoming stack before segmentation. masterindex_id and
pdf_page_number are used only to find the rendered image of a page, never to order pages.

Document boundaries come from the document key, compared between consecutive raw pages:
  primary:  stack_id + doc_id + sfdoc_class
  fallback: stack_id + masterindex_id + process_id + doc_id + subdoc_idx + sfdoc_class
The fallback is used only for a primary key that is ambiguous (it covers several fallback keys),
incomplete (an empty field) or anomalous (its pages are not one contiguous run in raw order).
"""

import csv
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow.parquet as pq

from .config import CLASS_TO_ID, PipelineConfig

IGNORE_INDEX = -100
PRIMARY_KEY = ("stack_id", "doc_id", "sfdoc_class")
FALLBACK_KEY = ("stack_id", "masterindex_id", "process_id", "doc_id", "subdoc_idx", "sfdoc_class")
MODEL_FIELDS = [
    "page_identity", "stack_id", "stack_position", "image_id", "masterindex_id", "pdf_page_number",
    "image_path", "image_sha256", "has_label", "page_class", "class_target",
    "is_first_page", "segmentation_target", "is_last_page", "last_page_target", "document_key_source",
    "process_id", "doc_id", "subdoc_idx", "sfdoc_class", "label_tier", "training_label_quality",
]


# ---------------------------------------------------------------------------
# CSV helpers and splits
# ---------------------------------------------------------------------------

def read_csv(path: Path):
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in fields} for row in rows)


def _write_new(path, rows, fields):
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing file: {path}")
    write_csv(path, rows, fields)


def assign_splits(rows, split_file: Path, seed=13, train_ratio=0.8, val_ratio=0.1):
    """Leakage-safe split by stack. Existing assignments are kept, also for stacks not in rows."""
    existing = {row["stack_id"]: row["split"] for row in read_csv(split_file)} if split_file.exists() else {}
    stack_rows = defaultdict(list)
    for row in rows:
        stack_rows[row["stack_id"]].append(row)
    assignments = {stack: split for stack, split in existing.items() if stack in stack_rows}
    new_stacks = [stack for stack in stack_rows if stack not in assignments]
    rng = random.Random(seed); rng.shuffle(new_stacks)
    targets = {"train": train_ratio, "val": val_ratio, "test": 1 - train_ratio - val_ratio}
    counts = Counter(assignments.values())
    for stack in new_stacks:
        split = min(targets, key=lambda name: counts[name] / max(targets[name], 1e-9))
        assignments[stack] = split; counts[split] += 1
    tb0 = Counter(assignments[stack] for stack, pages in stack_rows.items() if any(r["page_class"] == "TB0" for r in pages))
    if tb0["val"] < 10 or tb0["test"] < 10:
        raise ValueError(f"Validation/test need at least 10 TB0 stacks: {dict(tb0)}")
    if new_stacks:
        merged = {**existing, **assignments}
        tmp = split_file.with_suffix(".csv.tmp")
        write_csv(tmp, [{"stack_id": stack, "split": merged[stack]} for stack in sorted(merged)], ["stack_id", "split"])
        tmp.replace(split_file)
    return assignments


def split_assignments(config: PipelineConfig, rows, seed=13):
    return assign_splits(rows, config.split_file, seed=seed)


# ---------------------------------------------------------------------------
# Stack population
# ---------------------------------------------------------------------------

def _true(value):
    return str(value).strip().lower() == "true"


def population_hash(rows):
    digest = hashlib.sha256()
    for row in sorted(rows, key=lambda item: (item["stack_id"], int(item["stack_position"]))):
        digest.update(f'{row["page_identity"]}|{row["stack_id"]}|{row["stack_position"]}|{row["class_target"]}|{row["segmentation_target"]}\n'.encode())
    return digest.hexdigest()


def resolve_document_keys(pages):
    """Document key per page of one stack (pages already in raw order)."""
    primary = [tuple(str(p.get(f, "")).strip() for f in PRIMARY_KEY) for p in pages]
    fallback = [tuple(str(p.get(f, "")).strip() for f in FALLBACK_KEY) for p in pages]
    fallbacks_of, positions = defaultdict(set), defaultdict(list)
    for index, (pk, fk) in enumerate(zip(primary, fallback)):
        fallbacks_of[pk].add(fk); positions[pk].append(index)
    ambiguous = {pk for pk, fks in fallbacks_of.items() if len(fks) > 1}
    incomplete = {pk for pk in positions if any(part == "" for part in pk)}
    non_contiguous = {pk for pk, pos in positions.items() if pos != list(range(pos[0], pos[-1] + 1))}
    use_fallback = ambiguous | incomplete | non_contiguous
    keys = [fk if pk in use_fallback else pk for pk, fk in zip(primary, fallback)]
    sources = ["fallback" if pk in use_fallback else "primary" for pk in primary]
    stats = {"ambiguous": len(ambiguous), "incomplete": len(incomplete - ambiguous),
             "non_contiguous": len(non_contiguous - ambiguous - incomplete)}
    return keys, sources, stats


def _incomplete_stacks(documents, mid_to_stack):
    """Stacks with at least one mapped document that is not usable (its pages are missing)."""
    incomplete = set()
    for doc in documents:
        if _true(doc.get("is_usable")):
            continue
        if doc.get("stack_id"):
            incomplete.add(doc["stack_id"].lower())
        stack = mid_to_stack.get(doc.get("masterindex_id"))
        if stack:
            incomplete.add(stack.lower())
    return incomplete


def build_stack_population(config: PipelineConfig):
    labels = read_csv(config.pages_csv)
    documents = read_csv(config.documents_csv)
    inventory = pq.read_table(
        config.page_inventory,
        columns=["masterindex_id", "source_page_number", "image_path", "image_sha256", "status"],
    ).to_pylist()

    image_at = {}
    rendered_per_mid = Counter()
    for image in inventory:
        if image.get("status") == "success":
            image_at[(str(image["masterindex_id"]), int(image["source_page_number"]))] = image
            rendered_per_mid[str(image["masterindex_id"])] += 1

    by_stack = defaultdict(list)
    mid_to_stack = {}
    for row in labels:
        by_stack[row["stack_id"]].append(row)
        mid_to_stack[row["masterindex_id"]] = row["stack_id"]
    incomplete = _incomplete_stacks(documents, mid_to_stack)

    rows, excluded = [], []
    key_stats, boundary_changes = Counter(), 0
    for stack_id, pages in sorted(by_stack.items()):
        reason = None
        if stack_id.lower() in incomplete:
            reason = "stack_has_unusable_document"
        elif any(not str(r.get("image_id", "")).strip().isdigit() for r in pages):
            reason = "non_numeric_image_id"
        if reason is None:
            pages = sorted(pages, key=lambda r: int(r["image_id"]))  # raw order
            if len({r["image_id"] for r in pages}) != len(pages):
                reason = "duplicate_image_id"
        images = []
        if reason is None:
            for r in pages:
                position = str(r.get("pdf_page_number", "")).strip()
                images.append(image_at.get((r["masterindex_id"], int(position))) if position.isdigit() else None)
            if any(image is None for image in images):
                reason = "image_missing"
            elif len({image["image_path"] for image in images}) != len(images):
                reason = "image_used_twice"
        if reason is not None:
            excluded.append({"stack_id": stack_id, "reason": reason, "pages": len(pages)})
            continue

        keys, sources, stats = resolve_document_keys(pages)
        key_stats.update(stats)
        for pos, (label, image) in enumerate(zip(pages, images)):
            first = pos == 0 or keys[pos] != keys[pos - 1]                  # boundary in raw order
            last = pos == len(pages) - 1 or keys[pos] != keys[pos + 1]
            boundary_changes += first != _true(label["is_first_page"])
            rows.append({
                "page_identity": str(image["image_path"]),
                "stack_id": stack_id,
                "stack_position": str(pos + 1),
                "image_id": label["image_id"],
                "masterindex_id": label["masterindex_id"],
                "pdf_page_number": label["pdf_page_number"],
                "image_path": str(image["image_path"]),
                "image_sha256": image.get("image_sha256") or "",
                "has_label": "True",
                "page_class": label["page_class"],
                "class_target": str(CLASS_TO_ID[label["page_class"]]),
                "is_first_page": str(first),
                "segmentation_target": str(int(first)),
                "is_last_page": str(last),
                "last_page_target": str(int(last)),
                "document_key_source": sources[pos],
                "process_id": label["process_id"],
                "doc_id": label["doc_id"],
                "subdoc_idx": label["subdoc_idx"],
                "sfdoc_class": label["sfdoc_class"],
                "label_tier": label["label_tier"],
                "training_label_quality": label["training_label_quality"],
            })

    used_mids = Counter(row["masterindex_id"] for row in rows)
    report = {
        "analyse_db_rows": len(labels),
        "stacks_total": len(by_stack),
        "stacks_used": len({row["stack_id"] for row in rows}),
        "stacks_excluded": dict(Counter(e["reason"] for e in excluded)),
        "pages_used": len(rows),
        "pages_excluded": sum(e["pages"] for e in excluded),
        "page_class_counts": dict(Counter(row["page_class"] for row in rows)),
        "documents": sum(int(row["segmentation_target"]) for row in rows),
        "primary_keys_resolved_by_fallback": dict(key_stats),
        "pages_on_fallback_key": sum(row["document_key_source"] == "fallback" for row in rows),
        "first_page_labels_changed_vs_mapping": boundary_changes,
        "rendered_pages_without_row_in_used_mids": sum(rendered_per_mid[m] - used_mids[m] for m in used_mids),
        "population_hash": population_hash(rows),
    }
    return rows, excluded, report


def _file_record(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return {"path": str(Path(path).resolve()), "size_bytes": Path(path).stat().st_size, "sha256": digest.hexdigest()}


def prepare_model_population(config: PipelineConfig, seed=13):
    for path in (config.model_pages_csv, config.population_report, config.excluded_stacks_report):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite existing file: {path}")
    rows, excluded, report = build_stack_population(config)
    report["batch_id"] = config.batch_id
    report["split_seed"] = seed
    report["inputs"] = {name: _file_record(path) for name, path in
                        [("pages", config.pages_csv), ("documents", config.documents_csv), ("page_inventory", config.page_inventory)]}
    if not rows:
        raise RuntimeError(f"No usable stacks: {json.dumps(report, indent=2)}")
    config.classification_output.mkdir(parents=True, exist_ok=True)
    _write_new(config.model_pages_csv, rows, MODEL_FIELDS)
    config.excluded_stacks_report.write_text(json.dumps(excluded, indent=2), encoding="utf-8")
    config.population_report.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return rows, report


# ---------------------------------------------------------------------------
# Loading and selecting stacks
# ---------------------------------------------------------------------------

def validate_complete_stacks(rows):
    by_stack = defaultdict(list)
    for row in rows:
        by_stack[row["stack_id"]].append(row)
    errors = []
    for stack_id, stack_rows in by_stack.items():
        ordered = sorted(stack_rows, key=lambda row: int(row["stack_position"]))
        if [int(row["stack_position"]) for row in ordered] != list(range(1, len(ordered) + 1)):
            errors.append({"stack_id": stack_id, "reason": "nonconsecutive_stack_positions"})
        image_ids = [int(row["image_id"]) for row in ordered]
        if image_ids != sorted(set(image_ids)):
            errors.append({"stack_id": stack_id, "reason": "positions_not_in_image_id_order"})
        if len({row["page_identity"] for row in ordered}) != len(ordered):
            errors.append({"stack_id": stack_id, "reason": "duplicate_page_identity"})
    if errors:
        raise ValueError(f"Invalid stack sequences: {errors[:10]}")
    return by_stack


def load_model_pages(config: PipelineConfig):
    rows = read_csv(config.model_pages_csv)
    validate_complete_stacks(rows)
    return rows


def select_complete_stacks(rows, limit_stacks=None):
    by_stack = validate_complete_stacks(rows)
    selected = sorted(by_stack)
    if limit_stacks is not None:
        selected = selected[:limit_stacks]
    return [row for stack in selected for row in sorted(by_stack[stack], key=lambda item: int(item["stack_position"]))]


def rows_for_split(rows, assignments, split, limit_stacks=None):
    return select_complete_stacks([row for row in rows if assignments[row["stack_id"]] == split], limit_stacks)


def training_subset(config, rows, limit_stacks=None, training_limit_stacks=None, seed=13):
    """All stacks, the first N stacks, or N train stacks + N//4 val stacks (smoke runs)."""
    if limit_stacks and training_limit_stacks:
        raise ValueError("Choose limit_stacks or training_limit_stacks")
    if not training_limit_stacks:
        return select_complete_stacks(rows, limit_stacks)
    assignments = split_assignments(config, rows, seed)
    return (rows_for_split(rows, assignments, "train", training_limit_stacks)
            + rows_for_split(rows, assignments, "val", max(1, training_limit_stacks // 4)))


def build_stack_sequences(pages):
    by_stack = defaultdict(list)
    for page in pages:
        by_stack[page["stack_id"]].append(page)
    sequences = []
    for stack_id, stack_pages in sorted(by_stack.items()):
        ordered = sorted(stack_pages, key=lambda page: int(page["stack_position"]))
        if [int(page["stack_position"]) for page in ordered] != list(range(1, len(ordered) + 1)):
            raise ValueError(f"Stack {stack_id} has invalid positions")
        sequences.append(ordered)
    return sequences


def split_report(rows, assignments):
    by_split = defaultdict(list)
    for row in rows:
        by_split[assignments[row["stack_id"]]].append(row)
    report = {}
    for split, split_rows in by_split.items():
        stacks = defaultdict(list)
        for row in split_rows:
            stacks[row["stack_id"]].append(row)
        report[split] = {
            "stacks": len(stacks),
            "pages": len(split_rows),
            "single_page_stacks": sum(len(v) == 1 for v in stacks.values()),
            "documents": sum(int(row["segmentation_target"]) for row in split_rows),
            "internal_first_page_positives": sum(
                int(row["segmentation_target"]) for row in split_rows if row["stack_position"] != "1"),
            "page_class_counts": dict(Counter(row["page_class"] for row in split_rows)),
        }
    stacks = {split: {row["stack_id"] for row in split_rows} for split, split_rows in by_split.items()}
    pairs = [("train", "val"), ("train", "test"), ("val", "test")]
    report["stack_overlap"] = sum(len(stacks.get(a, set()) & stacks.get(b, set())) for a, b in pairs)
    return report
