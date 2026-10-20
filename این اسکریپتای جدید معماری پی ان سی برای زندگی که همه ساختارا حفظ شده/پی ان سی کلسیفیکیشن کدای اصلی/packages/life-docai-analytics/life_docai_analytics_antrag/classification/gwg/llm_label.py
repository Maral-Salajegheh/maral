"""Label GwG candidate pages with an LLM (vision + OCR text) and validate the answers.

Upstream (checked before any call):
  <classification_output>/gwg/<run_name>/gwg_llm_input.jsonl   from gwg_candidates
  <classification_output>/gwg/<run_name>/gwg_manifest.json     from gwg_candidates
  the page images listed in gwg_llm_input.jsonl                 from ingestion stage 03

Downstream (same run folder):
  gwg_llm_labels.jsonl   one validated answer per page; appended, so a re-run resumes
  gwg_llm_errors.jsonl   pages whose answer could not be parsed or validated
  gwg_llm_summary.json   counts, positives per selection reason, estimated misses

The client speaks the OpenAI-compatible chat API (Azure OpenAI / SecureGPT gateways
accept this format). Configure it with environment variables:
  LLM_BASE_URL, LLM_API_KEY, LLM_MODEL   (and LLM_API_VERSION for Azure-style gateways)

Run from the repository root (see the classification README for PYTHONPATH):
  export PYTHONPATH=packages/life-docai-analytics
  python -m life_docai_analytics_antrag.classification.gwg.llm_label --run-name gwg_v1 --dry-run   # writes requests only
  python -m life_docai_analytics_antrag.classification.gwg.llm_label --run-name gwg_v1
"""

import argparse
import base64
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

from ..training.config import PipelineConfig

PROMPT_VERSION = "gwg_v1"

SYSTEM_PROMPT = """Du prüfst einzelne Seiten aus Lebensversicherungsanträgen der AXA.

Aufgabe: Entscheide, ob die Seite die GwG-Identifizierungstabelle enthält, also den
Abschnitt, in dem die Identität einer Person nach dem Geldwäschegesetz festgehalten
wird und den die Sachbearbeitung mit der Ausweiskopie abgleicht.

Eine GwG-Seite (is_gwg_page = true) enthält eine Tabelle oder ein Formularfeld-Raster
mit persönlichen Daten einer Person, typischerweise mehrere von:
Name, Vorname, Geburtsdatum, Geburtsort, Staatsangehörigkeit, Anschrift,
Ausweisart, Ausweisnummer, ausstellende Behörde, gültig bis,
oft ergänzt um wirtschaftlich Berechtigte oder die Frage nach politisch exponierten
Personen (PEP). Die Felder können ausgefüllt oder leer sein.

KEINE GwG-Seite (is_gwg_page = false):
- Bedingungen, Merkblätter oder Datenschutzhinweise, die das GwG nur erwähnen
- Antragsseiten, die nur Name und Anschrift des Versicherungsnehmers enthalten
- Gesundheitsfragen, Bezugsrecht, Beitragsberechnung, Unterschriftenseiten ohne Ausweisdaten
- die Ausweiskopie selbst (Foto oder Scan eines Ausweises)

table_completeness:
- "full": Personendaten UND Ausweisdaten (Art/Nummer/Behörde/Gültigkeit) vorhanden
- "partial": nur ein Teil der Tabelle auf dieser Seite (z. B. fortgesetzt auf der nächsten)
- "none": keine GwG-Tabelle

product_type: Produkt, falls auf der Seite erkennbar ("LV", "Risiko", "Buendel"),
sonst "other" bei einem anderen erkennbaren Produkt oder "unknown".

Der OCR-Text kann Fehler enthalten (fehlende Umlaute, vertauschte Zeichen). Bei
Widerspruch zwischen Bild und OCR-Text gilt das Bild.

Gib in "evidence" nur Feldbezeichnungen oder Überschriften an, niemals persönliche
Daten wie Namen, Nummern oder Geburtsdaten.

Antworte ausschließlich mit einem JSON-Objekt:
{"is_gwg_page": true|false, "table_completeness": "full"|"partial"|"none",
 "product_type": "LV"|"Risiko"|"Buendel"|"other"|"unknown",
 "evidence": "<max. 200 Zeichen>", "confidence": <0.0-1.0>}"""

USER_TEMPLATE = """Seite {pdf_page_number} eines Dokuments.
Produkthinweis aus der ersten Dokumentseite (Regex, kann fehlen): {product_hint}

OCR-Text der Seite:
<<<
{ocr_text}
>>>"""

ALLOWED = {
    "table_completeness": {"full", "partial", "none"},
    "product_type": {"LV", "Risiko", "Buendel", "other", "unknown"},
}


def fail(message):
    print(f"ERROR: {message}", file=sys.stderr)
    raise SystemExit(1)


def read_jsonl(path):
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return rows


def append_jsonl(path, record):
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def build_messages(record, use_image=True):
    text = USER_TEMPLATE.format(
        pdf_page_number=record["pdf_page_number"],
        product_hint=record.get("product_hint") or "keiner",
        ocr_text=record.get("ocr_text") or "(kein OCR-Text)",
    )
    content = [{"type": "text", "text": text}]
    if use_image:
        suffix = Path(record["image_abs_path"]).suffix.lower().lstrip(".")
        mime = "image/jpeg" if suffix in {"jpg", "jpeg"} else "image/png"
        data = base64.b64encode(Path(record["image_abs_path"]).read_bytes()).decode("ascii")
        content.append({"type": "image_url", "image_url": {"url": f"data:{mime};base64,{data}"}})
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": content}]


def parse_answer(raw):
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.strip("`").split("\n", 1)[-1].rsplit("```", 1)[0]
    start, end = raw.find("{"), raw.rfind("}")
    if start < 0 or end < start:
        raise ValueError("no JSON object in answer")
    answer = json.loads(raw[start:end + 1])
    if not isinstance(answer.get("is_gwg_page"), bool):
        raise ValueError("is_gwg_page must be true/false")
    for key, allowed in ALLOWED.items():
        if answer.get(key) not in allowed:
            raise ValueError(f"{key}={answer.get(key)!r} not in {sorted(allowed)}")
    confidence = float(answer.get("confidence"))
    if not 0.0 <= confidence <= 1.0:
        raise ValueError("confidence outside 0-1")
    if answer["is_gwg_page"] != (answer["table_completeness"] != "none"):
        raise ValueError("is_gwg_page contradicts table_completeness")
    return {
        "is_gwg_page": answer["is_gwg_page"],
        "table_completeness": answer["table_completeness"],
        "product_type": answer["product_type"],
        "evidence": str(answer.get("evidence", ""))[:200],
        "confidence": confidence,
    }


def make_client():
    missing = [k for k in ("LLM_BASE_URL", "LLM_API_KEY", "LLM_MODEL") if not os.getenv(k)]
    if missing:
        fail(f"set environment variables: {', '.join(missing)} (or use --dry-run)")
    try:
        from openai import AzureOpenAI, OpenAI
    except ImportError:
        fail("package 'openai' is not installed (pip install openai)")
    if os.getenv("LLM_API_VERSION"):
        return AzureOpenAI(azure_endpoint=os.environ["LLM_BASE_URL"], api_key=os.environ["LLM_API_KEY"],
                           api_version=os.environ["LLM_API_VERSION"])
    return OpenAI(base_url=os.environ["LLM_BASE_URL"], api_key=os.environ["LLM_API_KEY"])


def call_llm(client, messages, retries=3):
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model=os.environ["LLM_MODEL"], messages=messages, temperature=0, max_tokens=300,
            )
            return response.choices[0].message.content or ""
        except Exception as error:  # network or rate limit: back off, then retry
            if attempt == retries - 1:
                raise
            wait = 2 ** (attempt + 2)
            print(f"  retry in {wait}s after {error.__class__.__name__}")
            time.sleep(wait)


def summarise(labels, inputs):
    by_id = {r["page_identity"]: r for r in inputs}
    counts = Counter()
    weighted_below = 0.0
    for label in labels:
        record = by_id.get(label["page_identity"])
        if record is None:
            continue
        reason = record["selection_reason"]
        counts[f"{reason}:total"] += 1
        if label["is_gwg_page"]:
            counts[f"{reason}:gwg"] += 1
            if reason == "random_below_threshold" and record["inclusion_probability"] > 0:
                weighted_below += 1 / record["inclusion_probability"]
    above = counts["above_threshold:gwg"]
    return {
        "labelled_pages": len(labels),
        "by_selection": dict(sorted(counts.items())),
        "gwg_pages_found_above_threshold": above,
        "estimated_gwg_pages_below_threshold": round(weighted_below, 1),
        "estimated_ranking_recall": round(above / (above + weighted_below), 3) if above + weighted_below else None,
        "product_types_of_gwg_pages": dict(Counter(l["product_type"] for l in labels if l["is_gwg_page"])),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--batch-id", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--limit", type=int, help="Label at most N pending pages (try 20 first)")
    parser.add_argument("--text-only", action="store_true", help="Send OCR text without the page image")
    parser.add_argument("--dry-run", action="store_true", help="Write the requests without calling the LLM")
    args = parser.parse_args()

    config = PipelineConfig(batch_id=args.batch_id)
    run_dir = config.classification_output / "gwg" / args.run_name
    input_path, manifest_path = run_dir / "gwg_llm_input.jsonl", run_dir / "gwg_manifest.json"
    for path in (input_path, manifest_path):
        if not path.is_file():
            fail(f"missing {path}  (run: python -m life_docai_analytics_antrag.classification.gwg.candidates --run-name {args.run_name})")

    inputs = read_jsonl(input_path)
    if not inputs:
        fail(f"{input_path} is empty")
    use_image = not args.text_only
    if use_image:
        missing = [r["image_abs_path"] for r in inputs if not Path(r["image_abs_path"]).is_file()]
        if missing:
            fail(f"{len(missing)} page images missing (first: {missing[0]}); fix the paths or use --text-only")

    labels_path, errors_path = run_dir / "gwg_llm_labels.jsonl", run_dir / "gwg_llm_errors.jsonl"
    done = {r["page_identity"] for r in read_jsonl(labels_path)}
    pending = [r for r in inputs if r["page_identity"] not in done]
    if args.limit:
        pending = pending[:args.limit]
    print(f"input pages: {len(inputs):,}  already labelled: {len(done):,}  to do now: {len(pending):,}")

    if args.dry_run:
        out = run_dir / "gwg_llm_requests_preview.jsonl"
        with out.open("w", encoding="utf-8") as handle:
            for record in pending[:20]:
                messages = build_messages(record, use_image=False)  # preview without base64 images
                handle.write(json.dumps({"page_identity": record["page_identity"], "messages": messages},
                                        ensure_ascii=False) + "\n")
        print(f"preview of up to 20 requests (images omitted): {out}")
        return

    client = make_client()
    for index, record in enumerate(pending, start=1):
        meta = {"page_identity": record["page_identity"], "masterindex_id": record["masterindex_id"],
                "pdf_page_number": record["pdf_page_number"], "prompt_version": PROMPT_VERSION,
                "model": os.environ["LLM_MODEL"], "with_image": use_image}
        raw = ""
        try:
            raw = call_llm(client, build_messages(record, use_image))
            append_jsonl(labels_path, {**meta, **parse_answer(raw)})
        except Exception as error:
            append_jsonl(errors_path, {**meta, "error": f"{error.__class__.__name__}: {error}",
                                       "raw": raw[:1000]})
        if index % 50 == 0:
            print(f"  {index}/{len(pending)}")

    summary = summarise(read_jsonl(labels_path), inputs)
    (run_dir / "gwg_llm_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
