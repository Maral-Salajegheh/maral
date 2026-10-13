"""Rank A00 pages for GwG identification tables and build the LLM labelling input.

Upstream (checked before any work; the run stops with a clear message if missing):
  pnc_model_pages_v2.csv        from pnc_prepare        page labels, MIDs, stacks
  pnc_ocr_cache_full_v2.jsonl   from pnc_inputs ocr     page text
  rendered page images          from ingestion stage 03 (checked for selected pages)
Optional:
  --ausweis-csv                 Ausweis extraction results; columns: stack_id plus
                                document_number and/or birth_date

Downstream, written to <classification_output>/gwg/<run_name>/ (never overwritten):
  gwg_page_scores.csv    every A00 page with its signals and score
  gwg_llm_input.jsonl    one JSON record per page to send to the LLM
  gwg_review_sheet.csv   90 pages for hand labelling (GOLD), empty label columns
  gwg_manifest.json      inputs (size, sha256), parameters, counts, LLM output schema

Run from life-docai:
  export PYTHONPATH=.
  python -m Antrag.Classification.gwg_candidates --run-name gwg_v1
"""

import argparse
import csv
import hashlib
import json
import random
import re
import shutil
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from .config import PipelineConfig
from .data import read_csv
from .ocr import load_ocr_cache
from .pnc_data import population_hash, validate_complete_mids

REQUIRED_COLUMNS = {
    "page_identity", "masterindex_id", "stack_id", "pdf_page_number", "image_path",
    "has_label", "page_class", "doc_id", "is_first_page", "training_label_quality",
}

# Patterns run on normalised text: lower case, umlauts folded (ä->a, ß->ss), single spaces.
FIELD_PATTERNS = {
    "name": r"\bname\b",
    "vorname": r"\bvorname",
    "geburtsdatum": r"geburtsdatum",
    "geburtsort": r"geburtsort",
    "staatsangehoerigkeit": r"staatsangeh",
    "anschrift": r"anschrift|wohnort|wohnsitz",
    "ausweisart": r"ausweisart|legitimationsart|art des ausweis",
    "ausweisnummer": r"ausweisnummer|ausweis-nr|ausweis nr|reisepassnummer",
    "behoerde": r"ausstellende behorde|ausgestellt von|ausstellungsbehorde",
    "gueltig_bis": r"gultig bis|gueltig bis|gultigkeit",
    "wirtschaftlich_berechtigt": r"wirtschaftlich\w* berechtig",
    "pep": r"politisch\w* exponiert|\bpep\b",
    "gwg": r"geldwasche|geldwaesche|\bgwg\b",
}
# "lv" excludes "risikolebensversicherung", which would otherwise match both.
PRODUCT_PATTERNS = {
    "risiko": r"risikolebensversicherung|risiko-lebensversicherung|risikoversicherung",
    "buendel": r"bundelantrag|bundel-antrag|bundelvertrag|kombiantrag",
    "lv": r"(?<!risiko)(?<!risiko-)lebensversicherung",
}
FIELD_RX = {key: re.compile(pattern) for key, pattern in FIELD_PATTERNS.items()}
PRODUCT_RX = {key: re.compile(pattern) for key, pattern in PRODUCT_PATTERNS.items()}

LLM_OUTPUT_SCHEMA = {
    "is_gwg_page": "bool - page contains the GwG identification table of a person",
    "table_completeness": "full | partial | none",
    "product_type": "LV | Risiko | Buendel | other | unknown",
    "evidence": "short quote or field names that support the decision",
    "confidence": "float 0-1",
}
MAX_OCR_CHARS = 20000


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
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size_bytes": stat.st_size,
        "modified_utc": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
        "sha256": sha256_file(path),
    }


# ---------------------------------------------------------------------------
# Upstream checks
# ---------------------------------------------------------------------------

def check_upstream(config, args):
    missing = []
    if not config.model_pages_csv.is_file():
        missing.append(f"{config.model_pages_csv}  (run: python -m Antrag.Classification.pnc_prepare)")
    if not config.pnc_ocr_cache.is_file():
        missing.append(f"{config.pnc_ocr_cache}  (run: python -m Antrag.Classification.pnc_inputs ocr)")
    if args.ausweis_csv and not Path(args.ausweis_csv).is_file():
        missing.append(f"{args.ausweis_csv}  (--ausweis-csv)")
    if missing:
        fail("missing upstream files:\n  " + "\n  ".join(missing))


def load_pages(config):
    rows = read_csv(config.model_pages_csv)
    if not rows:
        fail(f"{config.model_pages_csv} is empty")
    absent = REQUIRED_COLUMNS - set(rows[0])
    if absent:
        fail(f"{config.model_pages_csv} lacks columns: {sorted(absent)}")
    validate_complete_mids(rows)
    return rows


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
        if str(row["is_first_page"]).lower() != "true":
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
        score = len(fields) + (args.g07_bonus if stack_g07 else 0) + (args.id_match_bonus if id_match else 0)
        scored.append({
            "page_identity": row["page_identity"],
            "masterindex_id": row["masterindex_id"],
            "stack_id": row["stack_id"],
            "doc_id": row["doc_id"],
            "pdf_page_number": int(row["pdf_page_number"]),
            "image_path": row["image_path"],
            "training_label_quality": row["training_label_quality"],
            "has_ocr": raw is not None,
            "n_fields": len(fields),
            "matched_fields": "|".join(fields),
            "stack_has_g07": stack_g07,
            "mid_has_g07": row["masterindex_id"] in g07_mids,
            "id_match": id_match,
            "product_hint": "|".join(hints.get((row["masterindex_id"], row["doc_id"]), [])),
            "score": score,
        })
    scored.sort(key=lambda r: (-r["score"], r["masterindex_id"], r["pdf_page_number"]))
    return scored


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------

def select_for_llm(scored, threshold, n_random, rng):
    above = [r for r in scored if r["score"] >= threshold]
    below = [r for r in scored if r["score"] < threshold]
    sample = rng.sample(below, min(n_random, len(below)))
    probability_below = len(sample) / len(below) if below else 0.0
    selected = [(r, "above_threshold", 1.0) for r in above]
    selected += [(r, "random_below_threshold", probability_below) for r in sample]
    return selected, len(above), len(below)


def build_review_sheet(scored, threshold, per_group, rng):
    chosen, taken = [], set()

    def add(rows, group):
        for row in rows:
            if row["page_identity"] not in taken and sum(g == group for _, g in chosen) < per_group:
                chosen.append((row, group)); taken.add(row["page_identity"])

    add(scored, "top")
    near = [r for r in scored if threshold - 2 <= r["score"] < threshold + 2]
    add(rng.sample(near, len(near)), "near_threshold")
    add(rng.sample(scored, len(scored)), "random")
    return chosen


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def write_csv(path, rows, fields):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--batch-id", default=PipelineConfig.batch_id)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--threshold", type=int, default=6, help="Score from which every page goes to the LLM")
    parser.add_argument("--random-below", type=int, default=300, help="Random pages below the threshold (recall check)")
    parser.add_argument("--review-per-group", type=int, default=30)
    parser.add_argument("--g07-bonus", type=int, default=3)
    parser.add_argument("--id-match-bonus", type=int, default=5)
    parser.add_argument("--ausweis-csv")
    parser.add_argument("--max-missing-ocr", type=float, default=0.01, help="Allowed share of A00 pages without OCR")
    parser.add_argument("--require-images", action="store_true", help="Fail if a selected page image is missing")
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()

    config = PipelineConfig(batch_id=args.batch_id)
    out_dir = config.classification_output / "gwg" / args.run_name
    if out_dir.exists():
        fail(f"output exists, choose another --run-name: {out_dir}")
    check_upstream(config, args)

    # 1. Load and validate upstream
    pages = load_pages(config)
    a00_rows = [r for r in pages if r["page_class"] == "A00"]
    if not a00_rows:
        fail("no A00 pages in the model population")
    g07_rows = [r for r in pages if r["page_class"] == "G07"]
    if not g07_rows:
        print("WARNING: no G07 pages; the G07 signal is inactive")
    ocr = load_ocr_cache(config.pnc_ocr_cache)
    without_ocr = sum(r["image_path"] not in ocr for r in a00_rows)
    share_missing = without_ocr / len(a00_rows)
    print(f"A00 pages: {len(a00_rows):,}  without OCR: {without_ocr:,} ({share_missing:.2%})")
    if share_missing > args.max_missing_ocr:
        fail(f"OCR coverage too low ({share_missing:.2%} > {args.max_missing_ocr:.2%}); "
             "complete the cache with pnc_inputs ocr or raise --max-missing-ocr")
    ausweis = load_ausweis(args.ausweis_csv) if args.ausweis_csv else {}

    # 2. Signals and ranking
    hints = product_hints(a00_rows, ocr)
    scored = score_pages(a00_rows, ocr, {r["stack_id"] for r in g07_rows},
                         {r["masterindex_id"] for r in g07_rows}, ausweis, hints, args)

    # 3. Selection for the LLM and for hand review
    rng = random.Random(args.seed)
    selected, n_above, n_below = select_for_llm(scored, args.threshold, args.random_below, rng)
    review = build_review_sheet(scored, args.threshold, args.review_per_group, rng)
    review_ids = {row["page_identity"] for row, _ in review}

    missing_images = []
    for row, _, _ in selected:
        if not (config.render_root / row["image_path"]).is_file():
            missing_images.append(row["image_path"])
    if missing_images:
        print(f"WARNING: {len(missing_images):,} selected page images not found (first: {missing_images[0]})")
        if args.require_images:
            fail("missing page images and --require-images is set")

    # 4. Write to a temporary folder, then rename (no half-written run)
    tmp_dir = out_dir.with_name(out_dir.name + ".tmp")
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True)

    write_csv(tmp_dir / "gwg_page_scores.csv", scored, list(scored[0]))

    missing_set = set(missing_images)
    with (tmp_dir / "gwg_llm_input.jsonl").open("w", encoding="utf-8") as handle:
        for row, reason, probability in selected:
            text = ocr.get(row["image_path"], "")
            record = {
                **row,
                "selection_reason": reason,
                "inclusion_probability": probability,
                "in_review_sheet": row["page_identity"] in review_ids,
                "image_abs_path": str((config.render_root / row["image_path"]).resolve()),
                "image_exists": row["image_path"] not in missing_set,
                "ocr_text": text[:MAX_OCR_CHARS],
                "ocr_truncated": len(text) > MAX_OCR_CHARS,
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    review_rows = [{**{k: row[k] for k in ("page_identity", "masterindex_id", "stack_id", "pdf_page_number",
                                            "image_path", "score", "matched_fields", "product_hint")},
                    "review_group": group, "is_gwg_page": "", "table_completeness": "", "product_type": "",
                    "reviewer": "", "comment": ""}
                   for row, group in review]
    write_csv(tmp_dir / "gwg_review_sheet.csv", review_rows, list(review_rows[0]))

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "batch_id": args.batch_id,
        "run_name": args.run_name,
        "parameters": {k: v for k, v in vars(args).items() if k not in {"batch_id", "run_name"}},
        "inputs": {
            "model_pages_csv": describe(config.model_pages_csv),
            "ocr_cache": describe(config.pnc_ocr_cache),
            **({"ausweis_csv": describe(args.ausweis_csv)} if args.ausweis_csv else {}),
        },
        "model_population_hash": population_hash(pages),
        "field_patterns": FIELD_PATTERNS,
        "product_patterns": PRODUCT_PATTERNS,
        "counts": {
            "a00_pages": len(a00_rows),
            "a00_pages_without_ocr": without_ocr,
            "pages_above_threshold": n_above,
            "pages_below_threshold": n_below,
            "llm_input_pages": len(selected),
            "review_sheet_pages": len(review),
            "selected_images_missing": len(missing_images),
            "score_distribution": dict(sorted(Counter(r["score"] for r in scored).items())),
            "product_hint_documents": dict(Counter(h for v in hints.values() for h in (v or ["none"]))),
        },
        "llm_output_schema": LLM_OUTPUT_SCHEMA,
        "notes": "inclusion_probability < 1 marks the random recall sample; weight its "
                 "positives by 1/inclusion_probability to estimate pages the ranking misses.",
    }
    (tmp_dir / "gwg_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp_dir.rename(out_dir)

    print(f"pages above threshold {args.threshold}: {n_above:,}  random below: {len(selected) - n_above:,}")
    print(f"review sheet: {len(review)} pages")
    print(f"output: {out_dir}")


if __name__ == "__main__":
    main()
