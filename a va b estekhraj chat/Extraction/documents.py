"""Group already extracted pages. No extra OCR/LLM calls and no fuzzy number matching."""
import hashlib
import re
from collections import defaultdict

from llm import FIELDS, TYPE_CODES
from mrz import valid_date


def empty_fields():
    return {name: {"value": None, "source": None, "page": None} for name in FIELDS}


def normalise_number(value):
    if not value:
        return None
    # Ignore whitespace/case only; do not delete uncertain punctuation or repair glyphs.
    text = re.sub(r"\s", "", str(value).upper())
    return text if re.fullmatch(r"[A-Z0-9]+", text) else None


def page_scope(page):
    meta = page["metadata"]
    return str(meta.get("masterindex_id") or ""), str(meta.get("pdf_path_in_zip") or "")


def comparable(name, value):
    text = " ".join(str(value or "").upper().split())
    if name == "ausweisnummer":
        return normalise_number(value)
    if name == "nationalitaet" and text in {"D", "D<<", "DEU", "DEUTSCH", "GERMAN", "DEUTSCHE"}:
        return "D"
    return text


def group_pages(pages):
    grouped, unresolved = defaultdict(list), []
    for page in pages:
        number = normalise_number(page["fields"]["ausweisnummer"]["value"])
        scope = page_scope(page)
        if not number or not scope[0] or page.get("multiple_documents") or page.get("resolution_error"):
            page["issues"].append("document_link_unresolved")
            unresolved.append(page)
            continue
        # The same physical ID can have its sides in separate PDFs inside one MID.
        # Never cross MID boundaries; retain all contributing PDF paths for audit.
        grouped[(scope[0], number)].append(page)
    documents = [build_document(key, members) for key, members in sorted(grouped.items())]
    return documents, unresolved


def merge_fields(pages, issues):
    merged = empty_fields()
    for page in pages:
        for name, entry in page["fields"].items():
            old, value = merged[name], entry["value"]
            if not value:
                continue
            same = comparable(name, old["value"]) == comparable(name, value)
            if old["value"] and not same:
                issues.append("field_conflict: " + name)
            if not old["value"] or entry["source"] == "mrz":
                merged[name] = dict(entry)
    return merged


def kind_of(pages, issues):
    kinds = {page["document_kind"] for page in pages} - {"unknown"}
    if len(kinds) > 1:
        issues.append("document_kind_conflict")
        return "unknown"
    return next(iter(kinds)) if kinds else "unknown"


def review_issues(values, kind, issues):
    missing = [name for name in FIELDS if not values[name]["value"]]
    if missing:
        issues.append("missing_fields: " + ", ".join(missing))
    expiry = values["gueltigkeitsdatum"]["value"]
    if expiry and expiry != "UNBEFRISTET" and not valid_date(expiry):
        issues.append("invalid_expiry_format")
    if kind == "unknown":
        issues.append("document_kind_unconfirmed")


def build_document(key, pages):
    issues = [issue for page in pages for issue in page["issues"]]
    fields = merge_fields(pages, issues)
    kind = kind_of(pages, issues)
    review_issues(fields, kind, issues)
    if normalise_number(fields["ausweisnummer"]["value"]) != key[1]:
        issues.append("document_anchor_conflict")
    status = "rejected" if kind == "not_accepted" else "ready" if kind in TYPE_CODES and not issues else "review"
    identifier = hashlib.sha256(repr(key).encode()).hexdigest()[:20]
    pdfs = sorted({p["metadata"]["pdf_path_in_zip"] for p in pages if p["metadata"].get("pdf_path_in_zip")})
    return {"document_id": identifier, "masterindex_id": key[0], "pdf_paths": pdfs,
            "document_anchor": key[1], "document_kind": kind, "fields": fields,
            "page_numbers": [p["metadata"].get("page_number") for p in pages],
            "grouping_method": "exact_extracted_number_within_mid",
            "status": status, "needs_human_review": status == "review", "issues": sorted(set(issues))}
