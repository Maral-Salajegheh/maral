"""Synthetic multi-card regressions; no provider calls or real personal data."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import documents
import extract
import mrz
from llm import FIELDS, decode
from test_pipeline import metadata, response
from test_mrz_regressions import td1
import test_mrz_regressions


def card(number, kind="personalausweis", name=None, birth=None, issuer=None, **fields):
    item = response(FIELDS)
    item.update(document_kind=kind, type_evidence="PERSONALAUSWEIS",
                identity={"holder_name": name, "birth_date": birth, "issuing_state": issuer})
    item["fields"].update(ausweisnummer=number, ausweistyp="P", gueltigkeitsdatum="300415", nationalitaet="USA")
    item["fields"].update(fields)
    return item


def page(items, number=1, readings=()):
    llm = Mock()
    llm.read.return_value = decode({"documents": copy.deepcopy(items)}, FIELDS)
    audit = {"parsed": None, "results": [{"parsed": p, "locations": []} for p in readings],
             "multiple_documents": len(readings) > 1, "errors": [], "attempts": []}
    with patch("extract.read_mrz", return_value=audit):
        result = extract.extract_page(metadata(number), llm, None)
    assert llm.read.call_count == 1
    return result


class MultiCardTests(unittest.TestCase):
    def test_two_fronts_two_backs_reversed_order(self):
        a = page([card("AAA", name="PERSON ONE", ausstellende_behoerde=None),
                  card("BBB", name="PERSON TWO", ausstellende_behoerde=None)])
        b = page([card("BBB", name="PERSON TWO", geburtsort=None, ausstellende_behoerde="B AUTHORITY"),
                  card("AAA", name="PERSON ONE", geburtsort=None, ausstellende_behoerde="A AUTHORITY")], 2)
        docs, pending = documents.group_pages([a, b])
        self.assertEqual(len(docs), 2)
        self.assertFalse(pending)
        for d in docs:
            self.assertEqual(d["status"], "ready")
            self.assertEqual(d["page_numbers"], [1, 2])
            self.assertEqual(d["fields"]["ausstellende_behoerde"]["value"], d["document_anchor"][0] + " AUTHORITY")
            self.assertEqual(len(d["card_references"]), 2)

    def test_excluded_cards_never_donate_even_same_number(self):
        for excluded_kind in ("VERSICHERTENKARTE", "FUEHRERSCHEIN"):
            excluded = card("AAA", "not_accepted", geburtsort="WRONG PLACE")
            excluded["type_evidence"] = excluded_kind
            p = page([excluded, card("AAA", geburtsort=None)])
            docs, _ = documents.group_pages([p])
            self.assertEqual(len(docs), 1)
            self.assertIsNone(docs[0]["fields"]["geburtsort"]["value"])
            self.assertEqual(docs[0]["status"], "review")

    def test_two_documents_same_person_stay_separate(self):
        p = page([card("AAA", name="PERSON ONE"), card("BBB", name="PERSON ONE")])
        docs, _ = documents.group_pages([p])
        self.assertEqual(len(docs), 2)
        self.assertTrue(all(d["status"] == "ready" for d in docs))

    def test_missing_number_not_linked_by_position_or_name(self):
        a = page([card(None, name="PERSON ONE", geburtsort="FRONT ONLY")])
        b = page([card("AAA", name="PERSON ONE", geburtsort=None)], 2)
        docs, pending = documents.group_pages([a, b])
        self.assertEqual(len(pending), 1)
        self.assertIsNone(docs[0]["fields"]["geburtsort"]["value"])

    def test_same_number_identity_collision_keeps_fields_separate(self):
        for identity_key in ("name", "birth", "issuer"):
            a = page([card("AAA", **{identity_key: "FIRST"}, geburtsort="ONE", ausstellende_behoerde=None)])
            b = page([card("AAA", **{identity_key: "SECOND"}, geburtsort=None, ausstellende_behoerde="TWO")], 2)
            docs, _ = documents.group_pages([a, b])
            self.assertEqual(len(docs), 2)
            self.assertEqual(len({d["document_id"] for d in docs}), 2)
            self.assertTrue(all(d["status"] == "review" for d in docs))
            self.assertIsNone(docs[0]["fields"]["ausstellende_behoerde"]["value"])
            self.assertIsNone(docs[1]["fields"]["geburtsort"]["value"])

    def test_mrz_assigned_by_number_not_order(self):
        readings = [mrz.parse_mrz("\n".join(td1(number=n))) for n in ("D23145890", "X12345678")]
        p = page([card("X12345678"), card("D23145890")], readings=readings)
        self.assertEqual(len(p["cards"]), 2)
        for c in p["cards"]:
            self.assertEqual(c["fields"]["ausweisnummer"]["source"], "mrz")
            self.assertEqual(c["mrz"]["parsed"]["fields"]["ausweisnummer"], c["fields"]["ausweisnummer"]["value"])
        docs, pending = documents.group_pages([p])
        self.assertEqual(len(docs), 2)
        self.assertFalse(pending)
        self.assertTrue(all(d["status"] == "ready" for d in docs))

    def test_conflicting_mrz_not_assigned_by_first_match(self):
        a, b = td1(), td1(); b[2] = "OTHER<<PERSON".ljust(30, "<")
        p = page([card("D23145890")], readings=[mrz.parse_mrz("\n".join(x)) for x in (a, b)])
        self.assertEqual(p["cards"][0]["fields"]["ausweisnummer"]["source"], "llm")
        docs, pending = documents.group_pages([p])
        self.assertEqual(len(pending), 2)
        self.assertEqual(docs[0]["status"], "review")

    def test_unmatched_mrz_blocks_ready_on_image(self):
        p = page([card("WRONGNUMBER")], readings=[mrz.parse_mrz("\n".join(td1()))])
        docs, pending = documents.group_pages([p])
        self.assertEqual(docs[0]["status"], "review")
        self.assertEqual(len(pending), 1)

    def test_mrz_identity_conflict_is_unresolved(self):
        p = page([card("D23145890", name="OTHER PERSON")], readings=[mrz.parse_mrz("\n".join(td1()))])
        self.assertEqual(p["cards"][0]["fields"]["ausweisnummer"]["source"], "llm")
        self.assertEqual(len(p["unmatched_mrz"]), 1)

    def test_all_excluded_saved_separately(self):
        p = page([card("AAA", "not_accepted"), card("BBB", "not_accepted")])
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            extract.save_results(out, [p], {})
            self.assertEqual((out / "documents.jsonl").read_text(), "")
            self.assertEqual((out / "unresolved_pages.jsonl").read_text(), "")
            self.assertEqual(len((out / "excluded_cards.jsonl").read_text().splitlines()), 2)
            summary = json.loads((out / "summary.json").read_text())
            self.assertEqual(summary["pages"], 1)
            self.assertEqual(summary["excluded_cards"], 2)

    def test_empty_page_remains_in_audit(self):
        docs, pending = documents.group_pages([page([])])
        self.assertFalse(docs)
        self.assertEqual(len(pending), 1)

    def test_response_schema_rejects_legacy_and_malformed_cards(self):
        for value in (response(FIELDS), {"documents": {}}, {"documents": [None]},
                      {"documents": [dict(card("AAA"), identity={})]}):
            with self.assertRaises(ValueError):
                decode(value, FIELDS)

    def test_scan_preserves_distinct_readings_and_locations(self):
        texts = ["\n".join(td1(number=n)) for n in ("D23145890", "X12345678")]
        audit, calls = test_mrz_regressions.MRZRegressionTests().scan(texts)
        self.assertEqual(calls, 2)
        self.assertEqual(len(audit["results"]), 2)
        self.assertTrue(all(r["locations"][0]["region"] for r in audit["results"]))

    def test_multiple_windows_in_one_crop_are_retained(self):
        text = "\n".join(td1() + td1(number="X12345678"))
        audit, calls = test_mrz_regressions.MRZRegressionTests().scan([text], count=1)
        self.assertEqual(calls, 1)
        self.assertEqual(len(audit["results"]), 2)

    def test_duplicate_readings_are_deduplicated(self):
        text = "\n".join(td1())
        audit, _ = test_mrz_regressions.MRZRegressionTests().scan([text, text])
        self.assertEqual(len(audit["results"]), 1)
        self.assertEqual(len(audit["results"][0]["locations"]), 2)


if __name__ == "__main__":
    unittest.main()
