"""Rank the A00 pages of a batch for GwG identification tables from their OCR text (P1-P4).
"""

import argparse
import csv
import hashlib
import json
import random
import re
import tempfile
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import pyarrow.parquet as pq

from ..training.config import PipelineConfig
from ..training.data import read_csv
from ..training.ocr import load_ocr_cache

REQUIRED_COLUMNS = {"masterindex_id", "stack_id", "pdf_page_number", "sst", "doc_id", "is_first_page"}

# Patterns run on normalised text: lower case, umlauts folded (ä->a, ß->ss), single spaces.
FIELD_PATTERNS = {
    "name": r"\bname\b",
    "vorname": r"\bvorname",
    "geburtsdatum": r"geburtsdatum",
    "geburtsort": r"geburtsort",
    "staatsangehoerigkeit": r"staatsangeh",
    "anschrift": r"anschrift|wohnort|wohnsitz",
    "ausweisart": r"ausweisart|legitimationsart|art des ausweis|personalausweis|reisepass",
    "ausweisnummer": r"ausweis\s*-?\s*(?:nummer|nr\b)|reisepass\s*-?\s*(?:nummer|nr\b)",
    "behoerde": r"ausstellende beh(?:o|oe)rde|ausgestellt von|ausstellungsbeh(?:o|oe)rde",
    "gueltig_bis": r"gultig bis|gueltig bis|gultigkeit",
    "wirtschaftlich_berechtigt": r"wirtschaftlich\w* berechtig",
    "pep": r"politisch\w* exponiert|\bpep\b",
    "gwg": r"geldwasche|geldwaesche|\bgwg\b",
}
# VL is a specific product hint; generic Lebensversicherung is not sufficient.
PRODUCT_PATTERNS = {
    "risiko": r"risikolebensversicherung|risiko-lebensversicherung|risikoversicherung",
    "buendel": r"bundelantrag|bundel-antrag|bundelvertrag|kombiantrag",
    "vl": r"\bvl\b|verm(?:o|oe)genswirksam",
}
FIELD_RX = {key: re.compile(pattern) for key, pattern in FIELD_PATTERNS.items()}
PRODUCT_RX = {key: re.compile(pattern) for key, pattern in PRODUCT_PATTERNS.items()}

IDENTIFICATION_PATTERNS = {
    "gwg_identification": r"\b(?:identifizierung|identifikation|legitimation)\b.{0,100}?(?:geldw(?:a|ae)sche(?:gesetz)?|\bgwg\b)",
    "identified_by": r"\bhat\s+sich\s+ausgewiesen\s+durch\b",
}
IDENTIFICATION_RX = {key: re.compile(pattern) for key, pattern in IDENTIFICATION_PATTERNS.items()}
SPECIFIC_FIELDS = ("ausweisart", "ausweisnummer", "behoerde", "gueltig_bis")
GENERAL_FIELDS = ("name", "vorname", "geburtsdatum", "geburtsort", "staatsangehoerigkeit", "anschrift")
LOCAL_WINDOW_CHARS = 800  # OCR-text proximity only; does not establish a visual table.

REVIEW_COLUMNS = ("page_identity", "masterindex_id", "stack_id", "pdf_page_number", "image_path", "sst", "page_class",
                  "priority", "priority_reason", "score", "context_score", "identification_signals",
                  "local_specific_fields", "matched_fields", "product_hint")
REVIEW_LABEL_COLUMNS = ("is_gwg_page", "table_completeness", "product_type", "gwg_layout", "reviewer", "comment")


# ---------------------------------------------------------------------------
# Ranking rules
# ---------------------------------------------------------------------------

def rank_evidence(text, fields):
    """Return an explainable text priority; personal details cannot create P1/P2."""
    signals = [key for key, rx in IDENTIFICATION_RX.items() if rx.search(text)]
    specific = [key for key in SPECIFIC_FIELDS if key in fields]
    general = [key for key in GENERAL_FIELDS if key in fields]
    local_fields = []
    # Prefer a cluster following an explicit identification phrase. Header/field
    # order can be damaged by OCR, so page-wide evidence remains eligible for P2.
    for rx in IDENTIFICATION_RX.values():
        for match in rx.finditer(text):
            window = text[max(0, match.start() - 100):match.end() + LOCAL_WINDOW_CHARS]
            found = [key for key in SPECIFIC_FIELDS if FIELD_RX[key].search(window)]
            if len(found) > len(local_fields):
                local_fields = found
    if signals and len(local_fields) >= 3:
        priority, reason, base = "P1", "identification_phrase_and_local_document_fields", 30
    elif len(specific) >= 2 or (signals and specific):
        priority, reason, base = "P2", "document_fields_or_partial_identification_section", 20
    elif signals or any(key in fields for key in ("gwg", "pep", "wirtschaftlich_berechtigt")):
        priority, reason, base = "P3", "gwg_related_text_without_sufficient_document_fields", 10
    else:
        priority, reason, base = "P4", "general_or_insufficient_evidence", 0
    # Bounded within-tier score: no number of general fields can change the tier.
    detail = min(9, 2 * len(specific) + min(2, len(general)) + bool(signals))
    return {
        "priority": priority,
        "priority_reason": reason,
        "identification_signals": "|".join(signals),
        "n_specific_fields": len(specific),
        "local_specific_fields": "|".join(local_fields),
        "score": base + detail,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def fail(message):
    print(f"ERROR: {message}", file=sys.stderr)
    raise SystemExit(1)


def normalise(text):
    text = (text or "").lower()
    for old, new in (("ä", "a"), ("ö", "o"), ("ü", "u"), ("ß", "ss")):
        text = text.replace(old, new)
    return re.sub(r"\s+", " ", text)


def alnum_upper(text):
    return re.sub(r"[^0-9A-Za-z]", "", text or "").upper()


def sha256_file(path, chunk=1024 * 1024):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def describe(path):
    path = Path(path)
    return {"path": str(path.resolve()), "size_bytes": path.stat().st_size, "sha256": sha256_file(path)}


def write_csv(path, rows, fields):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# Upstream inputs
# ---------------------------------------------------------------------------

def check_upstream(config, args):
    missing = []
    if not args.pages_file.is_file():
        missing.append(f"{args.pages_file}  (raw Life mapping)")
    if not config.page_inventory.is_file():
        missing.append(f"{config.page_inventory}  (rendered-page inventory for this batch)")
    if not args.ocr_text.is_file():
        missing.append(f"{args.ocr_text}  (OCR text; run training.main compute_ocr or pass --ocr-text)")
    if args.ausweis_csv and not Path(args.ausweis_csv).is_file():
        missing.append(f"{args.ausweis_csv}  (--ausweis-csv)")
    if missing:
        fail("missing upstream files:\n  " + "\n  ".join(missing))


def iter_mapping_rows(path):
    """Stream the full mapping (CSV or Parquet); never load the model/training manifest."""
    if path.suffix.lower() == ".parquet":
        source = pq.ParquetFile(path)
        absent = REQUIRED_COLUMNS - set(source.schema_arrow.names)
        if absent:
            fail(f"{path} lacks columns: {sorted(absent)}")
        for batch in source.iter_batches(batch_size=65536):
            yield from batch.to_pylist()
    else:
        with path.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            absent = REQUIRED_COLUMNS - set(reader.fieldnames or [])
            if absent:
                fail(f"{path} lacks columns: {sorted(absent)}")
            yield from reader


def load_images(inventory_path):
    """(masterindex_id, page number) -> image_path for the successfully rendered pages of the batch."""
    images = {}
    columns = ["masterindex_id", "source_page_number", "image_path", "status"]
    for image in pq.read_table(inventory_path, columns=columns).to_pylist():
        if image["status"] != "success":
            continue
        key = (str(image["masterindex_id"]).strip(), int(image["source_page_number"]))
        if key in images:
            fail(f"ambiguous rendered page in batch inventory: {key}")
        if not image["image_path"]:
            fail(f"empty image_path in batch inventory: {key}")
        images[key] = str(image["image_path"])
    return images


def load_pages(config, path):
    """Select original SST A00/G07 and join only the batch's rendered images.

    No page-class, prediction, training-eligibility or first-page filter is used.
    MID + page number locates the image; stack is retained in the page identity.
    """
    images = load_images(config.page_inventory)
    matched, seen = [], set()
    counts = Counter()
    for raw in iter_mapping_rows(path):
        counts["mapping_rows"] += 1
        sst = str(raw["sst"] or "").strip().upper()
        if sst not in {"A00", "G07"}:
            continue
        counts[f"raw_{sst}_pages"] += 1
        mid = str(raw["masterindex_id"] or "").strip()
        stack = str(raw["stack_id"] or "").strip()
        try:
            number = int(str(raw["pdf_page_number"]).strip())
        except (TypeError, ValueError):
            fail(f"invalid pdf_page_number in raw mapping for MID {mid}")
        if not mid or not stack or number < 1:
            fail(f"invalid page key in raw mapping: {(stack, mid, number)}")
        key = (mid, number)
        if key not in images:
            counts[f"{sst}_pages_without_batch_image"] += 1
            continue
        if key in seen:
            fail(f"ambiguous mapping for rendered page: {key}")
        seen.add(key)
        row = {k: "" if v is None else str(v) for k, v in raw.items()}
        row.update(sst=sst, masterindex_id=mid, stack_id=stack, pdf_page_number=str(number),
                   image_path=images[key], page_identity=f"{stack}|{mid}|{number}")
        matched.append(row)
        counts[f"rendered_{sst}_pages"] += 1
    if not counts["mapping_rows"]:
        fail(f"{path} is empty")
    print(f"Raw mapping / batch coverage: {dict(counts)}", flush=True)
    return matched, dict(counts)


def has_ocr_text(text):
    """Missing, empty and whitespace-only OCR provide no usable text."""
    return isinstance(text, str) and bool(text.strip())


def load_ocr(path, a00_rows, max_missing):
    """OCR text per image_path; stops if too many A00 pages have no text."""
    print(f"Loading OCR text: {path}", flush=True)
    ocr = load_ocr_cache(path)
    without_ocr = sum(not has_ocr_text(ocr.get(r["image_path"])) for r in a00_rows)
    share_missing = without_ocr / len(a00_rows)
    print(f"OCR text: {len(ocr):,} pages; A00 pages: {len(a00_rows):,}  without OCR: {without_ocr:,} ({share_missing:.2%})")
    if share_missing > max_missing:
        fail(f"OCR coverage too low ({share_missing:.2%} > {max_missing:.2%}); "
             "check --ocr-text and cached image-path keys; raise --max-missing-ocr only intentionally")
    return ocr, without_ocr


def load_ausweis(path):
    """stack_id -> set of normalised ID numbers / birth dates (min. 6 characters)."""
    rows = read_csv(Path(path))
    if not rows or "stack_id" not in rows[0]:
        fail(f"{path} needs a stack_id column")
    value_columns = [c for c in ("document_number", "birth_date") if c in rows[0]]
    if not value_columns:
        fail(f"{path} needs document_number and/or birth_date")
    tokens = defaultdict(set)
    for row in rows:
        for column in value_columns:
            token = alnum_upper(row.get(column))
            if len(token) >= 6:
                tokens[row["stack_id"]].add(token)
    return tokens


# ---------------------------------------------------------------------------
# Signals and score
# ---------------------------------------------------------------------------

def product_hints(a00_rows, ocr):
    """(masterindex_id, doc_id) -> product names found on the document's first page."""
    hints = {}
    for row in a00_rows:
        if str(row["is_first_page"]).strip().lower() not in {"true", "1"}:
            continue
        text = normalise(ocr.get(row["image_path"], ""))
        hints[(row["masterindex_id"], row["doc_id"])] = [k for k, rx in PRODUCT_RX.items() if rx.search(text)]
    return hints


def score_pages(a00_rows, ocr, g07_stacks, g07_mids, ausweis, hints, args):
    scored = []
    for row in a00_rows:
        raw = ocr.get(row["image_path"])
        text = normalise(raw)
        fields = [key for key, rx in FIELD_RX.items() if rx.search(text)]
        compact = alnum_upper(raw)
        id_match = any(token in compact for token in ausweis.get(row["stack_id"], ()))
        stack_g07 = row["stack_id"] in g07_stacks
        evidence = rank_evidence(text, fields)
        # Context breaks ties only; G07 or an ID/date match cannot promote a page.
        context_score = (args.g07_bonus if stack_g07 else 0) + (args.id_match_bonus if id_match else 0)
        scored.append({
            "page_identity": row["page_identity"],
            "masterindex_id": row["masterindex_id"],
            "stack_id": row["stack_id"],
            "doc_id": row["doc_id"],
            "pdf_page_number": int(row["pdf_page_number"]),
            "image_path": row["image_path"],
            "sst": row["sst"],
            "page_class": row.get("page_class", ""),
            "training_label_quality": row.get("training_label_quality", ""),
            "has_ocr": has_ocr_text(raw),
            "n_fields": len(fields),
            "matched_fields": "|".join(fields),
            "stack_has_g07": stack_g07,
            "mid_has_g07": row["masterindex_id"] in g07_mids,
            "id_match": id_match,
            "product_hint": "|".join(hints.get((row["masterindex_id"], row["doc_id"]), [])),
            **evidence,
            "context_score": context_score,
        })
    scored.sort(key=lambda r: (-r["score"], -r["context_score"], r["masterindex_id"], r["pdf_page_number"]))
    return scored


# ---------------------------------------------------------------------------
# Review sheet
# ---------------------------------------------------------------------------

def build_review_sheet(scored, threshold, per_group, rng):
    """Top pages, pages on both sides of the threshold, and a random sample."""
    chosen, taken = [], set()

    def add(rows, group):
        for row in rows:
            if row["page_identity"] not in taken and sum(g == group for _, g in chosen) < per_group:
                chosen.append((row, group)); taken.add(row["page_identity"])

    add(scored, "top")
    # Include both sides of the boundary even when the tier scores have gaps.
    below = [r for r in scored if r["score"] < threshold and r["page_identity"] not in taken]
    above = [r for r in scored if r["score"] >= threshold and r["page_identity"] not in taken]
    rng.shuffle(below); rng.shuffle(above)
    below.sort(key=lambda r: threshold - r["score"])
    above.sort(key=lambda r: r["score"] - threshold)
    add(below[:(per_group + 1) // 2], "near_threshold")
    add(above[:per_group // 2], "near_threshold")
    add(sorted(below + above, key=lambda r: abs(r["score"] - threshold)), "near_threshold")
    add(rng.sample(scored, len(scored)), "random")
    return chosen


def review_rows(review):
    return [{**{k: row[k] for k in REVIEW_COLUMNS}, "review_group": group, **{k: "" for k in REVIEW_LABEL_COLUMNS}}
            for row, group in review]


def check_review_images(config, review, require):
    missing = [row["image_path"] for row, _ in review if not (config.render_root / row["image_path"]).is_file()]
    if missing:
        print(f"WARNING: {len(missing):,} review-sheet page images not found (first: {missing[0]})")
        if require:
            fail("missing page images and --require-images is set")
    return missing


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def build_manifest(args, config, mapping_counts, a00_rows, without_ocr, scored, review, missing_images, hints):
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "batch_id": args.batch_id,
        "run_name": args.run_name,
        "script": describe(Path(__file__)),
        "parameters": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()
                       if k not in {"batch_id", "run_name"}},
        "inputs": {
            "raw_pages": describe(args.pages_file),
            "page_inventory": describe(config.page_inventory),
            "ocr_text": describe(args.ocr_text),
            **({"ausweis_csv": describe(args.ausweis_csv)} if args.ausweis_csv else {}),
        },
        "mapping_counts": mapping_counts,
        "counts": {
            "a00_pages": len(a00_rows),
            "a00_pages_without_ocr": without_ocr,
            "pages_above_threshold": sum(r["score"] >= args.threshold for r in scored),
            "review_sheet_pages": len(review),
            "review_group_counts": dict(Counter(group for _, group in review)),
            "review_images_missing": len(missing_images),
            "priority_counts": dict(sorted(Counter(r["priority"] for r in scored).items())),
            "score_distribution": dict(sorted(Counter(r["score"] for r in scored).items())),
            "product_hint_documents": dict(Counter(h for v in hints.values() for h in (v or ["none"]))),
        },
    }


def write_outputs(out_dir, scored, review, manifest):
    """Write to a temporary folder, then rename (no half-written run)."""
    out_dir = Path(out_dir)
    if out_dir.exists():
        fail(f"output exists, choose another --run-name: {out_dir}")
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{out_dir.name}-", dir=out_dir.parent) as temp:
        tmp_dir = Path(temp)
        write_csv(tmp_dir / "gwg_page_scores.csv", scored, list(scored[0]))
        rows = review_rows(review)
        write_csv(tmp_dir / "gwg_review_sheet.csv", rows, list(rows[0]))
        (tmp_dir / "gwg_manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
        if out_dir.exists():
            fail(f"output appeared during this run: {out_dir}")
        tmp_dir.rename(out_dir)



# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--batch-id", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--pages-file", type=Path, help="raw life_mid_pages.csv or .parquet; default: config.pages_csv")
    parser.add_argument("--ocr-text", type=Path, help="OCR text JSONL; default: output/<batch_id>/ocr_text.jsonl")
    parser.add_argument("--threshold", type=int, default=20,
                        help="review-sheet boundary: 20 = between P2 and P3; 10 = between P3 and P4")
    parser.add_argument("--review-per-group", type=int, default=30)
    parser.add_argument("--g07-bonus", type=int, default=3, help="tie-break only; does not change priority or score")
    parser.add_argument("--id-match-bonus", type=int, default=5, help="tie-break only; does not change priority or score")
    parser.add_argument("--ausweis-csv")
    parser.add_argument("--max-missing-ocr", type=float, default=0.01, help="allowed share of A00 pages without OCR")
    parser.add_argument("--require-images", action="store_true", help="fail if a review-sheet page image is missing")
    parser.add_argument("--seed", type=int, default=13)
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    if args.review_per_group < 1:
        parser.error("--review-per-group must be positive")

    if not 0 <= args.max_missing_ocr <= 1:
        parser.error("--max-missing-ocr must be between 0 and 1")

    config = PipelineConfig(batch_id=args.batch_id)
    args.pages_file = args.pages_file or config.pages_csv
    args.ocr_text = args.ocr_text or config.ocr_text_file
    out_dir = config.classification_output / "gwg" / args.run_name
    if out_dir.exists():
        fail(f"output exists, choose another --run-name: {out_dir}")
    check_upstream(config, args)

    # 1. Load and validate upstream
    pages, mapping_counts = load_pages(config, args.pages_file)
    a00_rows = [r for r in pages if r["sst"] == "A00"]
    if not a00_rows:
        fail("no raw sst=A00 pages matched the rendered images in this batch")
    g07_rows = [r for r in pages if r["sst"] == "G07"]
    if not g07_rows:
        print("WARNING: no G07 pages; the G07 signal is inactive")
    ocr, without_ocr = load_ocr(args.ocr_text, a00_rows, args.max_missing_ocr)
    ausweis = load_ausweis(args.ausweis_csv) if args.ausweis_csv else {}

    # 2. Signals and ranking
    hints = product_hints(a00_rows, ocr)
    scored = score_pages(a00_rows, ocr, {r["stack_id"] for r in g07_rows},
                         {r["masterindex_id"] for r in g07_rows}, ausweis, hints, args)

    # 3. Review sheet
    review = build_review_sheet(scored, args.threshold, args.review_per_group, random.Random(args.seed))
    missing_images = check_review_images(config, review, args.require_images)

    # 4. Output
    manifest = build_manifest(args, config, mapping_counts, a00_rows, without_ocr, scored, review, missing_images, hints)
    write_outputs(out_dir, scored, review, manifest)

    print(f"priorities: {manifest['counts']['priority_counts']}")
    print(f"review sheet: {len(review)} pages {manifest['counts']['review_group_counts']}")
    print(f"output: {out_dir}")


if __name__ == "__main__":
    main()