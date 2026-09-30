"""Offline synthetic tests; no AXA credentials, real documents, or API calls."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch
from pydantic import ValidationError
from PIL import Image

import config
import documents
import extract
import llm
import pages
from response_model import FIELDS, TYPE_CODES, ExtractionResponse


def card(number="AAA", kind="personalausweis", **fields):
    values = dict(zip(FIELDS, ["BONN", number, TYPE_CODES.get(kind), "300415", "CITY BONN", "D"]))
    values.update(fields)
    return {"document_kind": kind, "type_evidence": "VISIBLE DOCUMENT TITLE" if kind in TYPE_CODES else None,
            "fields": values, "identity": {"holder_name": None, "issuing_state": None, "birth_date": None}}


def meta(n=1):
    return {"masterindex_id": "MID", "pdf_path_in_zip": "a.pdf", "page_number": n,
            "resolved_image_path": f"/page{n}.png"}


def page(items, n=1):
    client = Mock()
    client.read.return_value = {"documents": copy.deepcopy(items)}
    return extract.extract_page(meta(n), client)


def fake_extractor(client):
    obj = llm.Extractor.__new__(llm.Extractor)
    obj.client, obj.call_mode = client, None
    obj.api = SimpleNamespace(normalize_page_image=lambda p: "image", MAX_ATTEMPTS=5,
                              RETRY_DELAYS_SECONDS=[], is_retryable_securegpt_error=lambda e: False)
    return obj


class PipelineTests(unittest.TestCase):
    def test_all_kinds_and_codes(self):
        for kind in [*TYPE_CODES, "unknown", "not_accepted"]:
            self.assertEqual(llm.decode({"documents": [card(kind=kind)]})["documents"][0]["document_kind"], kind)

    def test_all_fields_required_but_nullable(self):
        c = card(kind="unknown", **dict.fromkeys(FIELDS))
        self.assertIsNotNone(ExtractionResponse.model_validate({"documents": [c]}))
        del c["fields"]["geburtsort"]
        with self.assertRaises(ValidationError):
            llm.decode({"documents": [c]})

    def test_forbid_extra_fields(self):
        c = card(); c["fields"]["invented"] = "value"
        with self.assertRaises(ValidationError): llm.decode({"documents": [c]})

    def test_reject_wrong_code(self):
        with self.assertRaises(ValidationError):
            llm.decode({"documents": [card(ausweistyp="R")]})

    def test_reject_numeric_number(self):
        with self.assertRaises(ValidationError): llm.decode({"documents": [card(123)]})

    def test_dates_and_unlimited(self):
        for value in [None, "UNBEFRISTET", "300415"]:
            llm.decode({"documents": [card(gueltigkeitsdatum=value)]})
        with self.assertRaises(ValidationError):
            llm.decode({"documents": [card(gueltigkeitsdatum="309932")]})

    def test_evidence_required(self):
        c = card(); c["type_evidence"] = " "
        with self.assertRaises(ValidationError): llm.decode({"documents": [c]})

    def test_response_forms(self):
        response = {"documents": [card()]}
        for value in [response, json.dumps(response), ExtractionResponse.model_validate(response),
                      '```json\n'+json.dumps(response)+'\n```']:
            self.assertEqual(llm.decode(value), response)

    def test_same_image_front_back(self):
        p = page([card(ausstellende_behoerde=None), card(geburtsort=None)])
        docs, unresolved = documents.group_pages([p])
        self.assertEqual(len(docs), 1)
        self.assertEqual(docs[0]["status"], "ready")
        self.assertEqual(len(docs[0]["card_references"]), 2)
        self.assertFalse(unresolved)

    def test_reversed_multicard_sides(self):
        a = page([card("AAA", ausstellende_behoerde=None), card("BBB", ausstellende_behoerde=None)])
        b = page([card("BBB", geburtsort=None), card("AAA", geburtsort=None)], 2)
        docs, pending = documents.group_pages([a,b])
        self.assertEqual(len(docs), 2)
        self.assertTrue(all(d["status"] == "ready" for d in docs))
        self.assertFalse(pending)

    def test_excluded_card_cannot_donate(self):
        p = page([card(geburtsort=None), card(kind="not_accepted", geburtsort="WRONG")])
        docs, _ = documents.group_pages([p])
        self.assertEqual(len(docs), 1)
        self.assertIsNone(docs[0]["fields"]["geburtsort"]["value"])

    def test_unknown_side_can_merge(self):
        docs, _ = documents.group_pages([page([card(kind="unknown", geburtsort=None)]), page([card()], 2)])
        self.assertEqual(docs[0]["status"], "ready")

    def test_missing_number_unresolved(self):
        docs, pending = documents.group_pages([page([card(None)])])
        self.assertFalse(docs); self.assertEqual(len(pending), 1)

    def test_identity_conflict_never_mixes(self):
        a,b = card(geburtsort=None),card(ausstellende_behoerde=None)
        a["identity"]["holder_name"]="PERSON ONE"; b["identity"]["holder_name"]="PERSON TWO"
        docs, _ = documents.group_pages([page([a]),page([b],2)])
        self.assertEqual(len(docs),2)
        self.assertTrue(all(d["status"] == "review" for d in docs))
        self.assertIsNone(docs[0]["fields"]["geburtsort"]["value"])

    def test_field_conflict_requires_review(self):
        docs, _ = documents.group_pages([page([card()]),page([card(geburtsort="BERLIN")],2)])
        self.assertIn("field_conflict: geburtsort",docs[0]["issues"])
        self.assertEqual(docs[0]["status"],"review")

    def test_no_merge_across_mid(self):
        a,b=page([card()]),page([card()])
        b["cards"][0]["metadata"]["masterindex_id"]="OTHER"
        self.assertEqual(len(documents.group_pages([a,b])[0]),2)

    def test_empty_and_failed_pages_audited(self):
        self.assertEqual(len(documents.group_pages([page([])])[1]),1)
        client=Mock();client.read.side_effect=RuntimeError("offline")
        p=extract.extract_page(meta(),client)
        self.assertEqual(p["llm_status"],"failed")
        self.assertEqual(len(documents.group_pages([p])[1]),1)

    def test_compatible_modes(self):
        for unsupported, expected in [(set(),"full"),({"image_detail"},"no_detail"),
                                       ({"image_detail","response_model"},"json_fallback")]:
            def call(**kwargs):
                for key in unsupported:
                    if key in kwargs: raise TypeError(f"new_chat() got an unexpected keyword argument '{key}'")
                return {"documents":[card()]}
            client=Mock(); client.new_chat.side_effect=call
            obj=fake_extractor(client)
            obj.read(Path("unused"));self.assertEqual(obj.call_mode,expected)
            count=client.new_chat.call_count
            obj.read(Path("unused"));self.assertEqual(client.new_chat.call_count,count+1)

    def test_internal_typeerror_does_not_fallback(self):
        client=Mock();client.new_chat.side_effect=TypeError("internal bug")
        with self.assertRaises(TypeError): fake_extractor(client).read(Path("unused"))
        self.assertEqual(client.new_chat.call_count,1)

    def test_schema_retry_bounded(self):
        client=Mock();client.new_chat.return_value={"documents":[{"bad":"value"}]}
        with self.assertRaises(ValidationError): fake_extractor(client).read(Path("unused"))
        self.assertEqual(client.new_chat.call_count,2)

    def test_schema_retry_recovers(self):
        client=Mock();client.new_chat.side_effect=[{"bad":1},{"documents":[card()]}]
        self.assertEqual(len(fake_extractor(client).read(Path("unused"))["documents"]),1)

    def test_metadata_lookup_and_g07_filter(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);Image.new("RGB",(10,10)).save(root/"a.png")
            (root/"input.csv").write_text("masterindex_id,page_number,predicted_page_sst\nMID,1.0,G07\nMID,2,AB1\n")
            (root/"labels.jsonl").write_text(json.dumps({"masterindex_id":"MID","page_number":1,"image_path":"a.png"}))
            with patch.object(config,"METADATA_FILES",[root/"labels.jsonl"]),patch.object(config,"IMAGE_ROOTS",[root]):
                result=pages.load_pages(root/"input.csv")
            self.assertEqual(len(result),1)
            self.assertEqual(result[0]["resolved_image_path"],str(root/"a.png"))

    def test_resume_and_image_change(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);out=root/"out";cache=root/"cache";out.mkdir();cache.mkdir()
            image=root/"a.png";Image.new("RGB",(10,10)).save(image)
            m={**meta(),"resolved_image_path":str(image)}
            client=Mock();client.read.return_value={"documents":[card()]};client.metadata.return_value={"model":"fake"}
            factory=Mock(return_value=client)
            extract.process_pages([m],out,cache,factory)
            extract.process_pages([m],out,cache,factory)
            self.assertEqual(client.read.call_count,1)
            Image.new("RGB",(11,10)).save(image)
            extract.process_pages([m],out,cache,factory)
            self.assertEqual(client.read.call_count,2)
            extract.process_pages([m],out,cache,factory,refresh=True)
            self.assertEqual(client.read.call_count,3)

    def test_interrupt_resume_and_failed_retry(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);out=root/"out";cache=root/"cache";out.mkdir();cache.mkdir()
            image=root/"a.png";Image.new("RGB",(10,10)).save(image)
            selected=[{**meta(n),"resolved_image_path":str(image)} for n in (1,2)]
            client=Mock();client.metadata.return_value={"model":"fake"}
            client.read.side_effect=[{"documents":[card()]},KeyboardInterrupt()]
            with self.assertRaises(KeyboardInterrupt):extract.process_pages(selected,out,cache,lambda:client)
            self.assertTrue((out/"pages.jsonl").exists())
            client.read.side_effect=[RuntimeError("offline")]
            extract.process_pages(selected,out,cache,lambda:client)
            self.assertEqual(client.read.call_count,3)
            client.read.side_effect=[{"documents":[card()]}]
            extract.process_pages(selected,out,cache,lambda:client)
            self.assertEqual(client.read.call_count,4)
            self.assertEqual(len((out/"documents.jsonl").read_text().splitlines()),1)


if __name__ == "__main__":
    unittest.main()
