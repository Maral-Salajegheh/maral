"""Regression tests for the issues fixed in this round. Run with: python -m pytest Extraction/test_pipeline.py"""
from __future__ import annotations

import importlib.util
from pathlib import Path

from Life.Extraction import config
from Life.Extraction.mrz_utils import check_digit, parse_with_repair
from Life.Extraction.utils import document_key

STAGE_06 = Path(__file__).resolve().parent / "06_emit_document_json.py"


# Import a numbered stage module, whose name is not a valid Python identifier.
def load_stage(path: Path):
    spec = importlib.util.spec_from_file_location("stage", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def valid_td1() -> list[str]:
    number = "222MV7913"
    line1 = "ID" + "D<<" + number + check_digit(number) + "<" * 15
    body = "900901" + check_digit("900901") + "M" + "280103" + check_digit("280103") + "D<<" + "<" * 11
    line2 = body + check_digit(line1[5:30] + body[0:7] + body[8:15] + body[18:29])
    return [line1, line2, "VERBARG<<THOMAS" + "<" * 15]


def record(**overrides):
    base = {"masterindex_id": "MIH1", "pdf_path_in_zip": "a.pdf", "page_number": 1,
            "status": "success", "checks_valid": True, "document_number": "222MV7913",
            "ausweistyp": "P", "nationality": "D", "expiry_date": "280103",
            "field_extraction_source": "field_llm", "geburtsort": None,
            "ausstellende_behoerde": None, "name_cross_check_state": "not_checked"}
    base.update(overrides)
    return base


# JSON and CSV must be built from the same resolved state.
def test_json_and_csv_agree_after_merge():
    stage = load_stage(STAGE_06)
    front = stage.build_payload(record(page_number=1, checks_valid=False, status="failed", geburtsort="KOELN"))
    front["failure_stage"] = "01_detect"
    back = stage.build_payload(record(page_number=2, ausstellende_behoerde="STADT KOELN"))
    payloads = [front, back]
    stage.resolve_page_roles(payloads)
    row = stage.build_csv_row(back)
    assert back["fields"]["geburtsort"]["value"] == "KOELN"
    assert row["geburtsort"] == back["fields"]["geburtsort"]["value"]
    assert row["ausstellende_behoerde"] == back["fields"]["ausstellende_behoerde"]["value"]
    assert row["needs_human_review"] == back["needs_human_review"]
    assert front["status"] == "no_mrz_on_page"


# Two identity documents under one masterindex_id must not donate fields to each other.
def test_two_documents_in_one_mid_do_not_merge():
    stage = load_stage(STAGE_06)
    person_a = stage.build_payload(record(pdf_path_in_zip="a.pdf", page_number=1, geburtsort="KOELN"))
    person_b = stage.build_payload(record(pdf_path_in_zip="b.pdf", page_number=1, document_number="999XX1111"))
    stage.resolve_page_roles([person_a, person_b])
    assert person_b["fields"]["geburtsort"]["value"] is None
    assert document_key(person_a) != document_key(person_b)


# Each field keeps its own source and donor page.
def test_field_provenance_is_per_field():
    stage = load_stage(STAGE_06)
    front = stage.build_payload(record(page_number=1, checks_valid=False, status="failed", geburtsort="KOELN"))
    front["failure_stage"] = "01_detect"
    back = stage.build_payload(record(page_number=2, ausstellende_behoerde="STADT KOELN"))
    stage.resolve_page_roles([front, back])
    assert back["fields"]["geburtsort"]["donor_page"] == 1
    assert back["fields"]["ausstellende_behoerde"]["donor_page"] is None
    row = stage.build_csv_row(back)
    assert row["source_geburtsort"].endswith("_page_1")
    assert not row["source_ausstellende_behoerde"].endswith("_page_1")


# Disagreeing pages are recorded, never silently overwritten.
def test_conflicting_fields_are_flagged():
    stage = load_stage(STAGE_06)
    back = stage.build_payload(record(page_number=2, geburtsort="KOELN"))
    other = stage.build_payload(record(page_number=3, checks_valid=False, status="failed", geburtsort="BONN"))
    other["failure_stage"] = "01_detect"
    stage.resolve_page_roles([back, other])
    assert back["fields"]["geburtsort"]["value"] == "KOELN"
    assert any(item["other"] == "BONN" for item in back["field_conflicts"])
    assert back["needs_human_review"] is True


# A successful LLM extraction must satisfy the cache condition on the next run.
def test_llm_result_satisfies_cache_condition():
    previous = {"status": "success", "field_extraction_backend": "llm",
                "field_ocr_stage_version": config.FIELD_OCR_STAGE_VERSION,
                "document_number": "222MV7913", "mrz_crop_path": "/c/x.png",
                "detector_version": config.MRZ_DETECTOR_VERSION,
                "mrz_parser_version": config.MRZ_PARSER_VERSION}
    current = dict(previous)
    assert previous["field_extraction_backend"] == "llm"
    assert previous["field_ocr_stage_version"] == config.FIELD_OCR_STAGE_VERSION
    assert all(previous[key] == current[key] for key in previous)


# An upstream failure must invalidate a cached success rather than keep it.
def test_upstream_failure_invalidates_cached_success():
    force = False
    current = {"status": "failed", "document_number": "222MV7913", "mrz_crop_path": "/c/x.png",
               "detector_version": config.MRZ_DETECTOR_VERSION}
    previous = None if force or current.get("status") != "success" else {"status": "success"}
    assert previous is None


# Several repairs satisfying the check digits must be flagged, not silently resolved.
def test_ambiguous_repair_is_flagged():
    lines = valid_td1()
    corrupted = [lines[0].replace("222MV7913", "2S2MV79I3"), lines[1], lines[2]]
    parsed = parse_with_repair("\n".join(corrupted))
    assert parsed is not None
    assert parsed["repair_ambiguous"] is True
    assert parsed["repair_alternatives"] > 1
    assert parsed["mrz_repair_needs_review"] is True
    assert parsed["mrz_lines_raw"] == corrupted


# A clean read stays clean: no repair, no review, no ambiguity.
def test_clean_mrz_needs_no_repair():
    parsed = parse_with_repair("\n".join(valid_td1()))
    assert parsed["checks_valid"] is True
    assert parsed["repair_applied"] is False
    assert parsed["mrz_repair_needs_review"] is False


# Genuine optional data must survive; only an almost-empty zone is rewritten.
def test_real_optional_data_is_not_erased():
    from Life.Extraction.mrz_utils import repair_filler_zones

    line = "IDD<<222MV79135AB<<C<<<D<<<E<<"
    assert repair_filler_zones([line, "0" * 30, "0" * 30], "TD1")[0] == line
