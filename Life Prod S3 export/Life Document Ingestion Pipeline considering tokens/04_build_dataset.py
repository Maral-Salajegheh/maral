"""Stage 04: merge several ingestion batches into one dataset (local only).

Every batch stays untouched. The dataset is a new folder
<LOCAL_OUTPUT_ROOT>/<dataset_id>/ holding document_inventory.parquet and
page_inventory.parquet in exactly the batch schema, so downstream code reads a
dataset the same way it reads a batch (pass the dataset_id as --batch-id).

Rows keep their original image_path. Pages of the first batch therefore keep
the keys under which their OCR text and CLIP vectors are already cached, and
only pages of the new batches are computed downstream.

One document per masterindex_id. The first batch in --batch-ids that holds a
successful row for a MasterIndex wins. A MasterIndex delivered again with a
different PDF (other pdf_sha256) is reported as a conflict in
dataset_manifest.json; the earlier version is kept.

The order of --batch-ids decides which version wins, and with it the
image_path of every page. Rebuilding a dataset is therefore only allowed when
the previous batch list is a prefix of the new one (batches can be appended,
never reordered or removed).

Run from the ingestion directory (output paths are relative to it):

    python 04_build_dataset.py --dataset-id ds_life_A00_G07_MAD --batch-ids <old_batch> <new_batch_1> <new_batch_2>
"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from common import atomic_write_json, read_json, utc_now_iso, validate_batch_id
from config import LOCAL_OUTPUT_ROOT

DOCUMENT_FILE = "document_inventory.parquet"
PAGE_FILE = "page_inventory.parquet"
MANIFEST_FILE = "dataset_manifest.json"


def build_parser() -> argparse.ArgumentParser:
    """Command-line arguments for this stage."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--batch-ids", required=True, nargs="+", help="Oldest first; the first batch wins duplicates.")
    return parser


def check_arguments(dataset_id: str, batch_ids: list[str]) -> None:
    """Dataset and batches live side by side under LOCAL_OUTPUT_ROOT, so names must not collide."""
    if len(set(batch_ids)) != len(batch_ids):
        raise ValueError("--batch-ids contains a batch twice.")
    if dataset_id in batch_ids:
        raise ValueError("--dataset-id must differ from every batch_id (it would overwrite that batch's inventory).")
    dataset_dir = LOCAL_OUTPUT_ROOT / dataset_id
    manifest_path = dataset_dir / MANIFEST_FILE
    if manifest_path.exists():
        previous = read_json(manifest_path)["batch_ids"]
        if batch_ids[: len(previous)] != previous:
            raise RuntimeError(
                f"Dataset {dataset_id} was built from {previous}. Batches can only be appended, "
                "never reordered or removed (that would change image_path keys of cached pages). "
                "Use a new --dataset-id."
            )
    elif (dataset_dir / PAGE_FILE).exists():
        raise RuntimeError(f"{dataset_dir} holds a page inventory but no {MANIFEST_FILE}; it is probably a batch.")


def read_batch(batch_id: str) -> tuple[pa.Table, pa.Table]:
    """The document and page inventory of one batch."""
    batch_dir = LOCAL_OUTPUT_ROOT / batch_id
    for name in (DOCUMENT_FILE, PAGE_FILE):
        if not (batch_dir / name).is_file():
            raise FileNotFoundError(f"Missing {batch_dir / name}. Run stages 02 and 03 for {batch_id} first.")
    return pq.read_table(batch_dir / DOCUMENT_FILE), pq.read_table(batch_dir / PAGE_FILE)


def choose_documents(documents: dict[str, pa.Table], batch_ids: list[str]) -> tuple[dict[str, str], list[dict], Counter]:
    """masterindex_id -> winning batch_id, plus conflicts and duplicate counts.

    Pass 1 picks the first batch with a successful row; pass 2 falls back to the
    first batch at all, so failed MasterIndexes stay visible in the dataset.
    """
    rows = {b: documents[b].select(["masterindex_id", "pdf_sha256", "status"]).to_pylist() for b in batch_ids}
    winner: dict[str, str] = {}
    winner_sha: dict[str, str | None] = {}
    for successful_only in (True, False):
        for batch_id in batch_ids:
            for row in rows[batch_id]:
                mid = row["masterindex_id"]
                if mid in winner or (successful_only and row["status"] != "success"):
                    continue
                winner[mid], winner_sha[mid] = batch_id, row["pdf_sha256"]

    conflicts, duplicates = [], Counter()
    for batch_id in batch_ids:
        for row in rows[batch_id]:
            mid = row["masterindex_id"]
            if winner[mid] == batch_id:
                continue
            duplicates[batch_id] += 1
            if row["pdf_sha256"] and winner_sha[mid] and row["pdf_sha256"] != winner_sha[mid]:
                conflicts.append({"masterindex_id": mid, "kept_batch_id": winner[mid], "kept_pdf_sha256": winner_sha[mid],
                                  "dropped_batch_id": batch_id, "dropped_pdf_sha256": row["pdf_sha256"]})
    return winner, conflicts, duplicates


def keep_winners(table: pa.Table, batch_id: str, winner: dict[str, str]) -> pa.Table:
    """Rows of one batch whose MasterIndex this batch won."""
    mids = [mid for mid, b in winner.items() if b == batch_id]
    return table.filter(pc.is_in(table["masterindex_id"], value_set=pa.array(mids, pa.string())))


def concat_same_schema(tables: list[pa.Table], name: str) -> pa.Table:
    """Concatenate batch tables; all batches must share one schema."""
    first = tables[0].schema
    for table in tables[1:]:
        if not table.schema.equals(first, check_metadata=False):
            raise RuntimeError(f"{name} schemas differ between batches; rebuild the batches with the same pipeline version.")
    return pa.concat_tables(tables)


def check_pages(pages: pa.Table) -> None:
    """Fail on anything that would break the image_path-keyed caches downstream."""
    rendered = pages.filter(pc.equal(pages["status"], "success"))
    page_ids = [p for p in rendered["corpus_page_id"].to_pylist() if p]
    if len(page_ids) != len(set(page_ids)):
        raise RuntimeError("Duplicate corpus_page_id after merging.")
    paths = [p for p in rendered["image_path"].to_pylist() if p]
    if len(paths) != len(set(paths)):
        raise RuntimeError("Duplicate image_path after merging.")
    if len({Path(p).is_absolute() for p in paths}) > 1:
        raise RuntimeError("Mixed absolute and relative image_path values; render every batch with the same RENDER_ROOT.")
    for column in ("render_config_version", "render_format"):
        values = set(rendered[column].to_pylist())
        if len(values) > 1:
            raise RuntimeError(f"Batches were rendered with different {column}: {sorted(values)}.")


def main() -> None:
    """Run stage 04 (merge, local output)."""
    args = build_parser().parse_args()
    dataset_id = validate_batch_id(args.dataset_id)
    batch_ids = [validate_batch_id(b) for b in args.batch_ids]
    check_arguments(dataset_id, batch_ids)

    documents, pages = {}, {}
    for batch_id in batch_ids:
        documents[batch_id], pages[batch_id] = read_batch(batch_id)

    winner, conflicts, duplicates = choose_documents(documents, batch_ids)
    merged_documents = concat_same_schema([keep_winners(documents[b], b, winner) for b in batch_ids], DOCUMENT_FILE)
    merged_pages = concat_same_schema([keep_winners(pages[b], b, winner) for b in batch_ids], PAGE_FILE)
    check_pages(merged_pages)

    dataset_dir = LOCAL_OUTPUT_ROOT / dataset_id
    dataset_dir.mkdir(parents=True, exist_ok=True)
    for table, name in ((merged_documents, DOCUMENT_FILE), (merged_pages, PAGE_FILE)):
        temporary = dataset_dir / (name + ".tmp")
        pq.write_table(table, temporary, compression="snappy")
        temporary.replace(dataset_dir / name)

    kept = Counter(winner.values())
    manifest = {
        "dataset_id": dataset_id,
        "batch_ids": batch_ids,
        "built_at_utc": utc_now_iso(),
        "masterindex_count": len(winner),
        "page_rows": merged_pages.num_rows,
        "per_batch": {b: {"masterindex_kept": kept[b], "masterindex_duplicates_dropped": duplicates[b]} for b in batch_ids},
        "conflict_count": len(conflicts),
        "conflicts": conflicts,
    }
    atomic_write_json(dataset_dir / MANIFEST_FILE, manifest)

    print(f"Dataset: {dataset_dir}")
    print(f"MasterIndex: {len(winner):,}   page rows: {merged_pages.num_rows:,}")
    for b in batch_ids:
        print(f"  {b}: kept {kept[b]:,}, duplicates dropped {duplicates[b]:,}")
    if conflicts:
        print(f"WARNING: {len(conflicts)} MasterIndex delivered again with a different PDF; see {MANIFEST_FILE}.")


if __name__ == "__main__":
    main()
