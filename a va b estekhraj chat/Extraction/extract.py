"""Run from the Pixi project root: pixi run python Extraction/extract.py"""
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import config
from documents import empty_fields, group_pages
from llm import Extractor, FIELDS, TYPE_CODES
from mrz import read_mrz
from pages import load_pages


def empty_page(metadata):
    return {"metadata": metadata, "fields": empty_fields(), "issues": [],
            "document_kind": "unknown", "type_evidence": None, "multiple_documents": False,
            "mrz": None, "llm": None, "llm_requested_fields": []}


def set_field(page, name, value, source):
    page["fields"][name] = {"value": value, "source": source,
                            "page": page["metadata"].get("resolved_image_path")}


def apply_llm(page, answer):
    page["llm"] = answer
    page["document_kind"] = answer["document_kind"]
    page["type_evidence"] = answer["type_evidence"]
    page["multiple_documents"] |= answer["multiple_documents"]
    if page["multiple_documents"]:
        page["issues"].append("multiple_documents_on_image")
        return
    for name in page["llm_requested_fields"]:
        set_field(page, name, answer["fields"].get(name), "llm")
    code = TYPE_CODES.get(page["document_kind"])
    if code and answer["fields"].get("ausweistyp") != code:
        page["issues"].append("llm_type_code_conflict")
    set_field(page, "ausweistyp", code, "visual_type_mapping")
    parsed = page["mrz"]["parsed"]
    if parsed and code and ((parsed["document_code"][0] == "P") != (code == "R")):
        page["issues"].append("mrz_visual_type_conflict")


def extract_page(metadata, llm, debug_dir):
    page = empty_page(metadata)
    if metadata.get("resolution_error"):
        page["resolution_error"] = metadata["resolution_error"]
        page["issues"].append("image_resolution_failed")
        return page
    path = Path(metadata["resolved_image_path"])
    page["mrz"] = read_mrz(path, debug_dir)
    page["multiple_documents"] = page["mrz"]["multiple_documents"]
    parsed = page["mrz"]["parsed"]
    if parsed:
        for name, value in parsed["fields"].items():
            set_field(page, name, value, "mrz")
    missing = [name for name in FIELDS if not page["fields"][name]["value"]]
    page["llm_requested_fields"] = missing
    try:
        apply_llm(page, llm.read(path, missing))
    except Exception as error:
        page["issues"].append("llm_failed: " + str(error))
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
    summary = {"pages": len(pages), "documents": len(documents), "unresolved_pages": len(unresolved),
               "ready": sum(d["status"] == "ready" for d in documents), "llm_model": model,
               "created_at_utc": datetime.now(timezone.utc).isoformat()}
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Pages: {len(pages)} | Documents: {len(documents)} | Unresolved: {len(unresolved)} | Ready: {summary['ready']}")


def main():
    pages = load_pages(config.INPUT_CSV)
    if not pages:
        raise ValueError("No G07 pages in the prediction CSV")
    llm = Extractor()
    output = config.OUTPUT_DIR / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output.mkdir(parents=True, exist_ok=False)
    results = []
    with (output / "pages.jsonl").open("w", encoding="utf-8") as audit:
        for index, metadata in enumerate(pages, 1):
            result = extract_page(metadata, llm, output / "mrz_crops" / f"page_{index:06d}")
            audit.write(json.dumps(result, ensure_ascii=False) + "\n")
            audit.flush()
            results.append(result)
            print(f"Extracted page {index}/{len(pages)}")
    save_results(output, results, llm.metadata())
    print(f"Output: {output}")


if __name__ == "__main__":
    main()
