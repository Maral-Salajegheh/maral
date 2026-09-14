"""Run from the Pixi project root: pixi run python Extraction/extract.py"""
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import config
from documents import empty_fields, group_pages, normalise_number, comparable, identity_conflicts, card_records
from llm import Extractor, FIELDS, TYPE_CODES
from mrz import read_mrz
from pages import load_pages


def empty_page(metadata):
    return {"metadata": metadata, "fields": empty_fields(), "issues": [],
            "document_kind": "unknown", "type_evidence": None, "multiple_documents": False,
            "mrz": None, "mrz_status": "not_run", "llm": None,
            "llm_status": "not_run", "llm_requested_fields": []}


def set_field(page, name, value, source):
    page["fields"][name] = {"value": value, "source": source,
                            "page": page["metadata"].get("resolved_image_path"),
                            "card_index": page.get("card_index")}


def mrz_identity(parsed):
    return {"issuing_state": parsed.get("issuing_state"),
            "holder_name": (parsed.get("name") or "").replace("<", " ").strip() or None,
            "birth_date": parsed.get("birth_date")}


def attach_mrz(card, reading):
    parsed = reading["parsed"]
    card["mrz"] = reading
    card["mrz_status"] = "success"
    if parsed.get("needs_review"):
        card["issues"].append("mrz_trailing_filler_restored_review")
    for name, value in parsed["fields"].items():
        old = card["fields"][name]["value"]
        if old and value and comparable(name, old) != comparable(name, value):
            card["issues"].append("mrz_llm_field_conflict: " + name)
        if value:
            set_field(card, name, value, "mrz")
    identity = mrz_identity(parsed)
    for name, value in identity.items():
        if value:
            card["identity"][name] = value
    code = TYPE_CODES.get(card["document_kind"])
    if code and ((parsed["document_code"][0] == "P") != (code == "R")):
        card["issues"].append("mrz_visual_type_conflict")


def extract_page(metadata, llm, debug_dir):
    page = empty_page(metadata)
    page["cards"] = []
    if metadata.get("resolution_error"):
        page["resolution_error"] = metadata["resolution_error"]
        page["issues"].append("image_resolution_failed")
        page["mrz_status"] = page["llm_status"] = "skipped"
        return page
    path = Path(metadata["resolved_image_path"])
    page["mrz"] = read_mrz(path, debug_dir)
    readings = page["mrz"].get("results", [])
    if not readings and page["mrz"].get("parsed"):
        readings = [{"parsed": page["mrz"]["parsed"], "locations": []}]
    page["mrz_status"] = "success" if readings else "failed"
    page["mrz_reason"] = page["mrz"].get("reason", "parsed" if readings else "not_parsed")
    # All six fields are requested per card: even when MRZ succeeds we need the
    # independently read document number to assign that MRZ to the correct card.
    page["llm_requested_fields"] = list(FIELDS)
    try:
        answer = llm.read(path, list(FIELDS))
        page["llm"] = answer
        for index, item in enumerate(answer["documents"], 1):
            card = empty_page(dict(metadata))
            card.update({"card_index": index, "identity": dict(item["identity"]),
                         "document_kind": item["document_kind"], "type_evidence": item["type_evidence"],
                         "llm": item, "llm_status": "success", "llm_requested_fields": list(FIELDS)})
            for name, value in item["fields"].items():
                set_field(card, name, value, "llm")
            code = TYPE_CODES.get(card["document_kind"])
            if code and item["fields"].get("ausweistyp") != code:
                card["issues"].append("llm_type_code_conflict")
            set_field(card, "ausweistyp", code, "visual_type_mapping")
            page["cards"].append(card)
        page["llm_status"] = "success"
    except Exception as error:
        page["llm_status"] = "failed"
        page["issues"].append("llm_failed: " + str(error))

    # Match by document number only when unique on both sides, with no known
    # identity contradiction. Never use page position or crop coordinates to join.
    unmatched = []
    for reading in readings:
        parsed = reading["parsed"]
        number = normalise_number(parsed["fields"]["ausweisnummer"])
        matches = [c for c in page["cards"]
                   if normalise_number(c["fields"]["ausweisnummer"]["value"]) == number]
        alternatives = [r for r in readings
                        if normalise_number(r["parsed"]["fields"]["ausweisnummer"]) == number]
        if len(matches) == len(alternatives) == 1:
            card = matches[0]
            # Excluded cards cannot receive MRZ values or donate anything.
            if card["document_kind"] == "not_accepted":
                continue
            conflicts = identity_conflicts(card.get("identity", {}), mrz_identity(parsed))
            if not conflicts:
                attach_mrz(card, reading)
                continue
        for card in matches:
            if card["document_kind"] != "not_accepted":
                card["issues"].append("mrz_card_association_unresolved")
        unmatched.append(reading)

    page["unmatched_mrz"] = unmatched
    if unmatched:
        # An unmatched reading can signal a missed card or a wrong LLM number.
        # Do not emit other accepted observations on this image as ready yet.
        for card in page["cards"]:
            if card["document_kind"] != "not_accepted":
                card["issues"].append("unmatched_mrz_on_page")
    # Preserve unmatched MRZ data for review without attaching it to an arbitrary
    # LLM card or losing it after a provider failure.
    for reading in unmatched:
        card = empty_page(dict(metadata))
        card.update({"card_index": len(page["cards"]) + 1, "identity": {},
                     "association_unresolved": True,
                     "issues": ["mrz_card_association_unresolved"] + page["issues"]})
        attach_mrz(card, reading)
        page["cards"].append(card)
    page["multiple_documents"] = len(page["cards"]) > 1
    if not page["cards"] and not page["issues"]:
        page["issues"].append("no_documents_detected")
    # Compatibility summary only; grouping always consumes cards, never this
    # page-wide projection. Multi-card pages deliberately have no shared fields.
    if len(page["cards"]) == 1:
        card = page["cards"][0]
        for key in ("fields", "document_kind", "type_evidence"):
            page[key] = card[key]
        page["issues"] = list(dict.fromkeys(page["issues"] + card["issues"]))
    return page


def write_jsonl(path, records):
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_documents_csv(path, documents):
    columns = ["masterindex_id", "document_id", "pdf_paths", "page_numbers",
               *FIELDS, "status", "needs_human_review", "issues"]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for document in documents:
            row = {key: document[key] for key in columns if key not in FIELDS}
            row.update({name: document["fields"][name]["value"] for name in FIELDS})
            row["issues"] = "; ".join(document["issues"])
            row["page_numbers"] = ",".join(map(str, document["page_numbers"]))
            row["pdf_paths"] = "; ".join(document["pdf_paths"])
            writer.writerow(row)


def save_results(output, pages, model):
    documents, unresolved = group_pages(pages)
    write_jsonl(output / "documents.jsonl", documents)
    write_documents_csv(output / "documents.csv", documents)
    write_documents_csv(output / "partner_candidates.csv", [d for d in documents if d["status"] == "ready"])
    write_jsonl(output / "unresolved_pages.jsonl", unresolved)
    excluded = [card for card in card_records(pages) if card["document_kind"] == "not_accepted"]
    write_jsonl(output / "excluded_cards.jsonl", excluded)
    summary = {"cards": sum(len(p.get("cards", [p])) for p in pages), "excluded_cards": len(excluded),
               "unresolved_cards": len(unresolved), "pages": len(pages), "documents": len(documents),
               "unresolved_pages": len({(str(p["metadata"].get("masterindex_id")),
                                         str(p["metadata"].get("pdf_path_in_zip")),
                                         str(p["metadata"].get("page_number")),
                                         str(p["metadata"].get("resolved_image_path"))) for p in unresolved}),
               "ready": sum(d["status"] == "ready" for d in documents), "llm_model": model,
               "image_resolution_failed": sum(p.get("resolution_error") is not None for p in pages),
               "mrz_success": sum(p["mrz_status"] == "success" for p in pages),
               "mrz_failed": sum(p["mrz_status"] == "failed" for p in pages),
               "llm_success": sum(p["llm_status"] == "success" for p in pages),
               "llm_failed": sum(p["llm_status"] == "failed" for p in pages),
               "created_at_utc": datetime.now(timezone.utc).isoformat()}
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Pages: {len(pages)} | Documents: {len(documents)} | Unresolved: {len(unresolved)} | Ready: {summary['ready']}")


# Report the MRZ toolchain up front. Missing OpenCV or Tesseract silently turns every
# page into an LLM fallback, which looks like an MRZ accuracy problem and is not one.
def check_mrz_toolchain():
    import shutil
    import subprocess

    try:
        import cv2

        opencv = cv2.__version__
    except ImportError:
        opencv = None
    binary = shutil.which(config.TESSERACT_CMD) or (
        config.TESSERACT_CMD if Path(config.TESSERACT_CMD).is_file() else None)
    version = None
    if binary:
        try:
            version = subprocess.run([binary, "--version"], capture_output=True, text=True,
                                     timeout=10, check=True).stdout.splitlines()[0]
        except Exception as error:
            version = None
    print(f"OpenCV: {opencv or 'MISSING - full-page OCR remains available'}")
    print(f"Tesseract: {version or 'MISSING - no MRZ OCR, every page will use the LLM'}")
    return bool(version)


def page_limit():
    # Try the MRZ path on a few pages without paying for a whole LLM run.
    for index, argument in enumerate(sys.argv):
        if argument == "--limit":
            return int(sys.argv[index + 1])
    return None


def main():
    if not check_mrz_toolchain():
        print("WARNING: Tesseract is unavailable or unusable; MRZ OCR cannot run. "
              "Missing fields will come from the LLM "
              "with no checksum. Install with: pixi add py-opencv tesseract tesseract-data-eng")
    pages = load_pages(config.INPUT_CSV)
    limit = page_limit()
    if limit:
        pages = pages[:limit]
        print(f"--limit {limit}: processing the first {len(pages)} pages only")
    if not pages:
        raise ValueError("No G07 pages in the prediction CSV")
    unresolved_images = [page for page in pages if page.get("resolution_error")]
    if len(unresolved_images) == len(pages):
        raise RuntimeError(
            f"Image resolution failed for all {len(pages)} G07 pages. "
            f"First error: {unresolved_images[0]['resolution_error']}"
        )
    if unresolved_images:
        print(f"WARNING: image resolution failed for {len(unresolved_images)}/{len(pages)} pages")
    llm = Extractor()
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output = config.OUTPUT_DIR / run_id
    debug_root = config.CACHE_DIR / run_id
    output.mkdir(parents=True, exist_ok=False)
    debug_root.mkdir(parents=True, exist_ok=False)
    results = []
    with (output / "pages.jsonl").open("w", encoding="utf-8") as audit:
        for index, metadata in enumerate(pages, 1):
            result = extract_page(metadata, llm, debug_root / "mrz" / f"page_{index:06d}")
            audit.write(json.dumps(result, ensure_ascii=False) + "\n")
            audit.flush()
            results.append(result)
            first_issue = result["issues"][0] if result["issues"] else "none"
            print(f"Page {index}/{len(pages)} | MRZ={result['mrz_status']} "
                  f"({result.get('mrz_reason', 'skipped')}) | LLM={result['llm_status']} "
                  f"| first_issue={first_issue}")
    save_results(output, results, llm.metadata())
    print(f"Output: {output}")
    print(f"Debug cache: {debug_root}")


if __name__ == "__main__":
    main()