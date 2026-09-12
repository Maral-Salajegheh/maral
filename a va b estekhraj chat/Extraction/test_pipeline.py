import csv
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from PIL import Image
import config
import documents
import extract
import mrz
import pages
from llm import FIELDS, decode

TD3 = "P<UTOERIKSSON<<ANNA<MARIA<<<<<<<<<<<<<<<<<<<\nL898902C36UTO7408122F1204159ZE184226B<<<<<10"


def metadata(number=1, pdf="a.pdf"):
    return {"masterindex_id": "MID", "pdf_path_in_zip": pdf, "page_number": number,
            "resolved_image_path": f"/page{number}.png"}


def response(keys):
    values = dict(zip(FIELDS, ["BONN", "L898902C3", "R", "120415", "CITY BONN", "UTO"]))
    return {"document_kind": "reisepass", "type_evidence": "PASSPORT",
            "multiple_documents": False, "fields": {key: values[key] for key in keys}}


def result(number=1, anchor="L898902C3", pdf="a.pdf"):
    page = extract.empty_page(metadata(number, pdf))
    page["document_kind"] = "reisepass"
    for name, value in response(FIELDS)["fields"].items():
        extract.set_field(page, name, value, "llm")
    extract.set_field(page, "ausweisnummer", anchor, "llm")
    return page


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.llm = Mock()
        self.llm.read.side_effect = lambda path, keys: response(keys)

    def extraction(self, parsed=None):
        audit = {"parsed": parsed, "multiple_documents": False, "errors": [], "attempts": [],
                 "rotation": None, "ocr_text": ""}
        with patch("extract.read_mrz", return_value=audit):
            return extract.extract_page(metadata(), self.llm, None)

    def test_mrz_success_requests_only_remaining_fields(self):
        page = self.extraction(mrz.parse_mrz(TD3))
        self.assertEqual(set(page["llm_requested_fields"]), {"geburtsort", "ausweistyp", "ausstellende_behoerde"})
        self.assertEqual(page["fields"]["ausweisnummer"]["source"], "mrz")

    def test_mrz_failure_requests_all_six(self):
        page = self.extraction()
        self.assertEqual(set(page["llm_requested_fields"]), set(FIELDS))
        docs, unresolved = documents.group_pages([page])
        self.assertEqual(docs[0]["status"], "ready")
        self.assertFalse(unresolved)

    def test_llm_failure_preserves_mrz(self):
        self.llm.read.side_effect = RuntimeError("offline")
        page = self.extraction(mrz.parse_mrz(TD3))
        docs, _ = documents.group_pages([page])
        self.assertEqual(docs[0]["fields"]["ausweisnummer"]["value"], "L898902C3")
        self.assertEqual(docs[0]["status"], "review")

    def test_two_sides_one_row_no_duplicate_llm_calls(self):
        front, back = result(1), result(2)
        front["fields"]["ausstellende_behoerde"]["value"] = None
        back["fields"]["geburtsort"]["value"] = None
        docs, pending = documents.group_pages([front, back])
        self.assertEqual(len(docs), 1)
        self.assertEqual(docs[0]["status"], "ready")
        self.assertEqual(docs[0]["page_numbers"], [1, 2])
        self.assertFalse(pending)

    def test_two_people_in_one_pdf_not_merged(self):
        docs, _ = documents.group_pages([result(1, "AAA"), result(2, "AAA"), result(3, "BBB")])
        self.assertEqual(len(docs), 2)

    def test_same_id_across_pdf_can_merge_but_not_across_mid(self):
        docs, pending = documents.group_pages([result(1), result(2, pdf="b.pdf"), result(3, pdf="")])
        self.assertEqual(len(docs), 1)
        self.assertFalse(pending)
        a, b = result(1), result(2)
        b["metadata"]["masterindex_id"] = "OTHER"
        docs, _ = documents.group_pages([a, b])
        self.assertEqual(len(docs), 2)

    def test_unreadable_anchor_not_a_fake_document(self):
        docs, pending = documents.group_pages([result(1, None), result(2, None)])
        self.assertFalse(docs)
        self.assertEqual(len(pending), 2)

    def test_multiple_documents_never_merged(self):
        page = result()
        page["multiple_documents"] = True
        docs, pending = documents.group_pages([page])
        self.assertFalse(docs)
        self.assertEqual(len(pending), 1)

    def test_field_conflict_review_and_mrz_preserved(self):
        a, b = result(1), result(2)
        extract.set_field(b, "gueltigkeitsdatum", "300101", "mrz")
        docs, _ = documents.group_pages([a, b])
        self.assertEqual(docs[0]["fields"]["gueltigkeitsdatum"]["value"], "300101")
        self.assertIn("field_conflict: gueltigkeitsdatum", docs[0]["issues"])

    def test_german_nationality_code_matches_printed_word(self):
        a, b = result(1), result(2)
        extract.set_field(a, "nationalitaet", "DEUTSCH", "llm")
        extract.set_field(b, "nationalitaet", "D", "mrz")
        docs, _ = documents.group_pages([a, b])
        self.assertEqual(docs[0]["status"], "ready")
        self.assertEqual(docs[0]["fields"]["nationalitaet"]["value"], "D")

    def test_prediction_without_paths_uses_original_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            Image.new("RGB", (20, 20)).save(root / "a.png")
            prediction = root / "predictions.csv"
            prediction.write_text("masterindex_id,page_number,predicted_page_sst\nMID,1.0,G07\nMID,2,AB1\n")
            label = root / "corpus_page_labels.jsonl"
            row = {"masterindex_id": "MID", "page_number": 1, "image_path": "a.png", "pdf_path_in_zip": "a.pdf"}
            label.write_text(json.dumps(row) + "\n")
            with patch.object(config, "DATA_DIRS", [root]), patch.object(config, "INVENTORY_FILES", []), \
                 patch.object(config, "IMAGE_ROOTS", [root]):
                selected = pages.load_pages(prediction)
            self.assertEqual(len(selected), 1)
            self.assertEqual(selected[0]["resolved_image_path"], str(root / "a.png"))
            self.assertEqual(selected[0]["pdf_path_in_zip"], "a.pdf")

    def test_ambiguous_metadata_does_not_guess(self):
        index = {("MID", 1): [{"image_path": "a.png"}, {"image_path": "b.png"}]}
        with self.assertRaises(ValueError):
            pages.resolve({"masterindex_id": "MID", "page_number": 1}, index)

    def test_page_number_validation(self):
        self.assertEqual(pages.page_number("2.0"), 2)
        for value in (None, "NaN", "1.5", "0"):
            with self.assertRaises(ValueError):
                pages.page_number(value)

    def test_whole_run_json_csv_same_document_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            extract.save_results(output, [result(1), result(2), result(3, None)], {"model": "fake"})
            doc = json.loads((output / "documents.jsonl").read_text())
            with (output / "documents.csv").open(encoding="utf-8-sig") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 1)
            for name in FIELDS:
                self.assertEqual(rows[0][name], doc["fields"][name]["value"])
            self.assertEqual(len((output / "unresolved_pages.jsonl").read_text().splitlines()), 1)

    def test_strict_mrz_and_llm_parsing(self):
        self.assertIsNotNone(mrz.parse_mrz(TD3))
        self.assertIsNone(mrz.parse_mrz(TD3[:-2]))
        answer = response(FIELDS)
        self.assertEqual(decode(json.dumps(answer), FIELDS)["document_kind"], "reisepass")
        del answer["fields"]["geburtsort"]
        with self.assertRaises(ValueError):
            decode(answer, FIELDS)

    def test_mrz_crop_used_before_full_frame_and_audited(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "page.png"
            image = Image.new("RGB", (800, 600), "white")
            image.save(path)
            with patch("mrz.proposed_crops", return_value=[(image, {"surface": "test", "box": [0, 0, 1, 1]})]), \
                 patch("mrz.tesseract", return_value=TD3) as ocr:
                audit = mrz.read_mrz(path)
            self.assertEqual(ocr.call_count, 1)
            self.assertIn("crop", audit["attempts"][0]["view"])
            self.assertIsNotNone(audit["parsed"])

    def test_multiple_mrz_audit_text_not_lost(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "page.png"
            Image.new("RGB", (200, 100)).save(path)
            text = TD3 + "\n" + TD3.replace("UTO740812", "USA740812")
            with patch("mrz.proposed_crops", return_value=[]), patch("mrz.tesseract", return_value=text):
                audit = mrz.read_mrz(path)
            self.assertTrue(audit["multiple_documents"])
            self.assertEqual(audit["ocr_text"], text)
            self.assertIsNone(audit["parsed"])

    def test_full_page_ocr_input_is_saved_in_debug_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path, debug = root / "page.png", root / "debug"
            Image.new("RGB", (200, 100), "white").save(path)
            with patch("mrz.proposed_crops", return_value=[]), \
                 patch("mrz.tesseract", return_value="not an mrz"):
                mrz.read_mrz(path, debug)
            self.assertTrue((debug / "rotation_0_full.png").is_file())

    def test_all_image_resolution_failures_stop_before_llm(self):
        failed = [{"masterindex_id": "MID", "page_number": 1,
                   "resolution_error": "metadata not found"}]
        with patch("extract.load_pages", return_value=failed), \
             patch("extract.Extractor") as constructor:
            with self.assertRaisesRegex(RuntimeError, "Image resolution failed for all"):
                extract.main()
        constructor.assert_not_called()

    def test_one_command_from_prediction_to_document(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("front.png", "back.png"):
                Image.new("RGB", (40, 40)).save(root / name)
            prediction = root / "predictions.csv"
            prediction.write_text("masterindex_id,page_number,predicted_page_sst\nMID,1,G07\nMID,2,G07\n")
            meta = [{"masterindex_id": "MID", "page_number": i, "image_path": name, "pdf_path_in_zip": "a.pdf"}
                    for i, name in enumerate(("front.png", "back.png"), 1)]
            (root / "corpus_page_labels.jsonl").write_text("\n".join(json.dumps(row) for row in meta))
            audit = {"parsed": None, "multiple_documents": False, "errors": []}
            self.llm.metadata.return_value = {"model": "fake"}
            with patch.object(config, "INPUT_CSV", prediction), patch.object(config, "DATA_DIRS", [root]), \
                 patch.object(config, "INVENTORY_FILES", []), patch.object(config, "IMAGE_ROOTS", [root]), \
                 patch.object(config, "OUTPUT_DIR", root / "outputs"), \
                 patch.object(config, "CACHE_DIR", root / "cache"), patch("extract.Extractor", return_value=self.llm), \
                 patch("extract.read_mrz", return_value=audit):
                extract.main()
            self.assertEqual(self.llm.read.call_count, 2)
            output = next((root / "outputs").iterdir())
            self.assertEqual(len((output / "documents.jsonl").read_text().splitlines()), 1)
            doc = json.loads((output / "documents.jsonl").read_text())
            self.assertEqual(doc["page_numbers"], [1, 2])


if __name__ == "__main__":
    unittest.main()
