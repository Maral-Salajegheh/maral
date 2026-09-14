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


def identity_conflicts(first, second):
    conflicts = []
    for name in ("issuing_state", "holder_name", "birth_date"):
        a, b = first.get(name), second.get(name)
        if not a or not b:
            continue
        a, b = [" ".join(v.upper().replace("<", " ").split()) for v in (a, b)]
        if name == "issuing_state":
            a, b = ["D" if v == "DEU" else v for v in (a, b)]
        if a != b:
            conflicts.append(name)
    return conflicts


def card_records(pages):
    for page in pages:
        if "cards" in page:
            yield from page["cards"]
        else:
            yield page


def group_pages(pages):
    grouped, unresolved = defaultdict(list), []
    # Failed/empty pages have no card entries but still require an audit result.
    unresolved.extend(p for p in pages if "cards" in p and not p["cards"])
    for card in card_records(pages):
        if card["document_kind"] == "not_accepted":
            continue
        number = normalise_number(card["fields"]["ausweisnummer"]["value"])
        scope = page_scope(card)
        if (not number or not scope[0] or card.get("multiple_documents")
                or card.get("resolution_error") or card.get("association_unresolved")):
            if "document_link_unresolved" not in card["issues"]:
                card["issues"].append("document_link_unresolved")
            unresolved.append(card)
            continue
        grouped[(scope[0], number)].append(card)
    documents = []
    for key, members in sorted(grouped.items()):
        # Known identity/type contradictions with the same number must not mix
        # fields. Keep each observation separate for review instead of guessing.
        collision = any(identity_conflicts(a.get("identity", {}), b.get("identity", {}))
                        or (a["document_kind"] != b["document_kind"]
                            and "unknown" not in (a["document_kind"], b["document_kind"]))
                        for i, a in enumerate(members) for b in members[i+1:])
        if collision:
            for index, card in enumerate(members):
                card = dict(card, issues=card["issues"] + ["document_identity_conflict"])
                documents.append(build_document((*key, index), [card]))
        else:
            documents.append(build_document(key, members))
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
            "page_numbers": list(dict.fromkeys(p["metadata"].get("page_number") for p in pages)),
            "card_references": [{"pdf_path_in_zip": p["metadata"].get("pdf_path_in_zip"),
                                 "page_number": p["metadata"].get("page_number"),
                                 "image_path": p["metadata"].get("resolved_image_path"),
                                 "card_index": p.get("card_index")} for p in pages],
            "grouping_method": "exact_extracted_number_within_mid",
            "status": status, "needs_human_review": status == "review", "issues": sorted(set(issues))}
