"""Run from the project root: pixi run python Extraction/extract.py."""
import argparse
import csv
import fcntl
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import config
from documents import empty_fields, group_pages, card_records
from llm import Extractor, decode
from response_model import FIELDS
from pages import load_pages


def empty_page(metadata):
    return {"metadata": metadata, "fields": empty_fields(), "issues": [],
            "document_kind": "unknown", "type_evidence": None,
            "multiple_documents": False, "llm": None, "llm_status": "not_run",
            "llm_requested_fields": list(FIELDS)}


def set_field(card, name, value, source="llm"):
    card["fields"][name] = {"value": value, "source": source,
                            "page": card["metadata"].get("resolved_image_path"),
                            "card_index": card.get("card_index")}


def extract_page(metadata, llm):
    page = empty_page(metadata)
    page["cards"] = []
    if metadata.get("resolution_error"):
        page.update(resolution_error=metadata["resolution_error"], llm_status="skipped")
        page["issues"].append("image_resolution_failed")
        return page
    try:
        answer = decode(llm.read(Path(metadata["resolved_image_path"]), FIELDS))
        page["llm"] = answer
        for index, item in enumerate(answer["documents"], 1):
            card = empty_page(dict(metadata))
            card.update(card_index=index, identity=item["identity"],
                        document_kind=item["document_kind"], type_evidence=item["type_evidence"],
                        llm=item, llm_status="success")
            for name, value in item["fields"].items():
                set_field(card, name, value)
            page["cards"].append(card)
        page["llm_status"] = "success"
        page["multiple_documents"] = len(page["cards"]) > 1
        if not page["cards"]:
            page["issues"].append("no_documents_detected")
        if len(page["cards"]) == 1:
            for name in ("fields", "document_kind", "type_evidence"):
                page[name] = page["cards"][0][name]
    except Exception as error:
        page["llm_status"] = "failed"
        page["issues"].append("llm_failed: " + type(error).__name__)
        page["error"] = str(error)
    return page


def atomic_text(path, text):
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def write_jsonl(path, records):
    atomic_text(path, "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records))


def write_documents_csv(path, documents):
    import io
    columns = ["masterindex_id", "document_id", "pdf_paths", "page_numbers", *FIELDS,
               "status", "needs_human_review", "issues"]
    buffer = io.StringIO(newline="")
    buffer.write("\ufeff")
    writer = csv.DictWriter(buffer, fieldnames=columns)
    writer.writeheader()
    for document in documents:
        row = {key: document[key] for key in columns if key not in FIELDS}
        row.update({name: document["fields"][name]["value"] for name in FIELDS})
        row.update(issues="; ".join(document["issues"]),
                   page_numbers=",".join(map(str, document["page_numbers"])),
                   pdf_paths="; ".join(document["pdf_paths"]))
        writer.writerow(row)
    atomic_text(path, buffer.getvalue())


def save_results(output, pages, model):
    documents, unresolved = group_pages(pages)
    excluded = [c for c in card_records(pages) if c["document_kind"] == "not_accepted"]
    write_jsonl(output / "pages.jsonl", pages)
    write_jsonl(output / "documents.jsonl", documents)
    write_documents_csv(output / "documents.csv", documents)
    write_documents_csv(output / "partner_candidates.csv", [d for d in documents if d["status"] == "ready"])
    write_jsonl(output / "unresolved_pages.jsonl", unresolved)
    write_jsonl(output / "excluded_cards.jsonl", excluded)
    summary = {"pages": len(pages), "cards": sum(len(p["cards"]) for p in pages),
               "documents": len(documents), "excluded_cards": len(excluded),
               "unresolved_records": len(unresolved),
               "review_documents": sum(d["status"] == "review" for d in documents),
               "ready": sum(d["status"] == "ready" for d in documents),
               "llm_success": sum(p["llm_status"] == "success" for p in pages),
               "llm_failed": sum(p["llm_status"] == "failed" for p in pages),
               "image_resolution_failed": sum(bool(p.get("resolution_error")) for p in pages),
               "llm_model": model, "pipeline_version": config.PIPELINE_VERSION,
               "created_at_utc": datetime.now(timezone.utc).isoformat()}
    atomic_text(output / "summary.json", json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Pages: {len(pages)} | Documents: {len(documents)} | Ready: {summary['ready']} | Review: {summary['review_documents']}")


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fingerprint():
    import securegpt_client as api
    api.validate_configuration()
    files = sorted(config.EXTRACTION_DIR.glob("*.py"))
    value = {"code": [(p.name, file_digest(p)) for p in files],
             "model": api.MODEL_NAME, "version": api.MODEL_VERSION}
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()[:20]


def checkpoint_key(metadata):
    value = dict(metadata)
    if not metadata.get("resolution_error"):
        # Hash actual bytes, not just a potentially stale metadata hash.
        value["actual_image_sha256"] = file_digest(metadata["resolved_image_path"])
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def process_pages(pages, output, cache, make_extractor=Extractor, refresh=False):
    results, llm = [], None
    try:
        for index, metadata in enumerate(pages, 1):
            try:
                key = checkpoint_key(metadata)
            except OSError as error:
                metadata = {**metadata, "resolution_error": str(error)}
                key = checkpoint_key(metadata)
            checkpoint = cache / (key + ".json")
            result = None
            if checkpoint.exists() and not refresh:
                try:
                    saved = json.loads(checkpoint.read_text(encoding="utf-8"))
                    if saved.get("llm_status") == "success":
                        decode(saved["llm"])
                        result = saved
                except (ValueError, KeyError, TypeError):
                    pass
            reused = result is not None
            if result is None:
                if not metadata.get("resolution_error") and llm is None:
                    llm = make_extractor()
                result = extract_page(metadata, llm)
                result["model_metadata"] = llm.metadata() if llm else {}
                atomic_text(checkpoint, json.dumps(result, ensure_ascii=False))
            results.append(result)
            print(f"Page {index}/{len(pages)} | LLM={result['llm_status']} | cached={reused}")
    finally:
        if results:
            model = llm.metadata() if llm else results[0].get("model_metadata", {})
            save_results(output, results, model)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--refresh", action="store_true", help="Reprocess selected pages, replacing their checkpoints")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    pages = load_pages(config.INPUT_CSV)
    if args.limit is not None:
        pages = pages[:args.limit]
    if not pages:
        raise ValueError("No G07 pages in the prediction CSV")
    signature = fingerprint()
    cache = config.CACHE_DIR / ("securegpt_only_" + signature)
    # Results contain only the selected input pages, with separate full/limited runs.
    selection = hashlib.sha256(json.dumps(pages, sort_keys=True).encode()).hexdigest()[:12]
    output = config.OUTPUT_DIR / ("securegpt_only_" + signature + "_" + selection)
    cache.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)
    with (cache / ".run.lock").open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("Another extraction run is using this cache") from None
        process_pages(pages, output, cache, refresh=args.refresh)
    print(f"Output: {output}")
    print(f"Checkpoints: {cache}")


if __name__ == "__main__":
    main()
