"""Offline regressions for the first pipeline; all data are synthetic."""
import contextlib
import io
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from PIL import Image
import config
import Diagnose
import extract
import mrz
from test_pipeline import TD3, metadata, response


def td1(nationality="USA", number="D23145890"):
    a = "I<UTO" + number + mrz.digit(number) + "<" * 15
    b = "740812" + mrz.digit("740812") + "F300415" + mrz.digit("300415") + nationality + "<" * 11
    b += mrz.digit(a[5:30] + b[:7] + b[8:15] + b[18:29])
    return [a, b, "ERIKSSON<<ANNA<MARIA".ljust(30, "<")]


def td2():
    a = "I<UTOERIKSSON<<ANNA<MARIA".ljust(36, "<")
    b = "D23145890" + mrz.digit("D23145890") + "UTO740812" + mrz.digit("740812") + "F300415" + mrz.digit("300415") + "<" * 7
    b += mrz.digit(b[:10] + b[13:20] + b[21:35])
    return [a, b]


class MRZRegressionTests(unittest.TestCase):
    def test_clean_formats(self):
        for text, fmt in [("\n".join(td1()), "TD1"), ("\n".join(td2()), "TD2"), (TD3, "TD3")]:
            self.assertEqual(mrz.parse_mrz(text)["format"], fmt)

    def test_blank_lines_do_not_break_parse(self):
        self.assertIsNotNone(mrz.parse_mrz("\n\n".join(td1())))

    def test_trailing_filler_restoration_requires_review(self):
        lines = td1(); lines[0] = lines[0][:-4]
        result = mrz.parse_mrz("\n".join(lines))
        self.assertTrue(result["needs_review"])
        self.assertEqual(result["fields"]["nationalitaet"], "USA")

    def test_deleted_nationality_character_is_not_repaired(self):
        lines = td1(); lines[1] = lines[1][:17] + lines[1][18:]
        self.assertIsNone(mrz.parse_mrz("\n".join(lines)))

    def test_lower_line_padding_is_rejected_and_explained(self):
        lines = td1(); lines[1] = lines[1][:-5] + lines[1][-1]
        result, diag = mrz.inspect_mrz("\n".join(lines))
        self.assertIsNone(result)
        self.assertEqual(diag["reason"], "line_length_mismatch")
        self.assertEqual(diag["line_lengths"], [30, 26, 30])

    def test_checksum_failure_has_a_reason(self):
        lines = td1(); lines[0] = lines[0][:14] + str((int(lines[0][14]) + 1) % 10) + lines[0][15:]
        result, diag = mrz.inspect_mrz("\n".join(lines))
        self.assertIsNone(result)
        self.assertIn("document_number_checksum", diag["best_candidate"]["failed_checks"])

    def test_invalid_character_is_not_replaced(self):
        lines = td1(); lines[2] = lines[2][:-1] + "?"
        result, diag = mrz.inspect_mrz("\n".join(lines))
        self.assertIsNone(result)
        self.assertEqual(diag["invalid_character_counts"], [0, 0, 1])

    def test_invalid_name_rejected(self):
        lines = td1(); lines[2] = "1" * 30
        self.assertIsNone(mrz.parse_mrz("\n".join(lines)))

    def test_impossible_date_rejected_with_correct_checksums(self):
        lines = td1()
        b = lines[1][:8] + "309932" + mrz.digit("309932") + lines[1][15:29]
        lines[1] = b + mrz.digit(lines[0][5:30] + b[:7] + b[8:15] + b[18:29])
        self.assertIsNone(mrz.parse_mrz("\n".join(lines)))

    def test_conflicting_windows_are_not_silently_chosen(self):
        a, b = td1(), td1(); b[2] = "OTHER<<PERSON".ljust(30, "<")
        with self.assertRaises(ValueError):
            mrz.parse_mrz("\n".join(a + b))

    def test_diagnostics_do_not_contain_personal_values(self):
        text = "\n".join(td1())
        _, diag = mrz.inspect_mrz(text)
        self.assertNotIn("D23145890", json.dumps(diag))
        out = io.StringIO()
        page = {"metadata": {"masterindex_id": "SECRET_MID"}, "mrz": {
            "attempts": [{"view": "rotation_0_crop_0", "text": text}], "errors": ["SECRET_ERROR"]}}
        with contextlib.redirect_stdout(out):
            Diagnose.report_page(page, 2)
        for secret in ["D23145890", "ERIKSSON", "SECRET_MID", "SECRET_ERROR", "740812"]:
            self.assertNotIn(secret, out.getvalue())
        self.assertIn("30, 30, 30", out.getvalue())

    def scan(self, responses, count=2):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "page.png"
            image = Image.new("RGB", (100, 100), "white"); image.save(path)
            crops = [(image, {"surface": "test", "box": [i, 0, i+1, 1]}) for i in range(count)]
            with patch("mrz.proposed_crops", return_value=crops), patch("mrz.tesseract", side_effect=responses) as ocr:
                audit = mrz.read_mrz(path)
            return audit, ocr.call_count

    def test_second_distinct_crop_checked_after_success(self):
        audit, calls = self.scan(["\n".join(td1()), "\n".join(td1(number="X12345678"))])
        self.assertEqual(calls, 2)
        self.assertTrue(audit["multiple_documents"])
        self.assertIsNone(audit["parsed"])

    def test_identical_valid_crops_are_not_multiple_documents(self):
        audit, calls = self.scan([TD3, TD3])
        self.assertEqual(calls, 2)
        self.assertFalse(audit["multiple_documents"])
        self.assertIsNotNone(audit["parsed"])

    def test_timeout_recorded_and_later_crop_tried(self):
        audit, calls = self.scan([subprocess.TimeoutExpired("tesseract", 30), "", TD3])
        self.assertEqual(calls, 3)
        self.assertEqual(audit["attempts"][0]["status"], "ocr_error")
        self.assertIsNotNone(audit["parsed"])

    def test_sparse_text_retry_is_audited(self):
        audit, calls = self.scan(["", TD3], count=1)
        self.assertEqual(calls, 2)
        self.assertEqual(audit["attempts"][1]["psm"], "11")
        self.assertIsNotNone(audit["parsed"])

    def test_full_page_llm_fallback_remains(self):
        llm = Mock(); llm.read.side_effect = lambda path, requested: {"documents": [response(requested)]}
        with patch("extract.read_mrz", return_value={"parsed": None, "multiple_documents": False,
                                                      "reason": "validation_failed"}):
            page = extract.extract_page(metadata(), llm, None)
        self.assertEqual(len(page["llm_requested_fields"]), 6)
        self.assertEqual(page["llm_status"], "success")
        self.assertEqual(page["mrz_reason"], "validation_failed")

    def test_restored_parse_adds_document_review_issue(self):
        lines = td1(); lines[0] = lines[0][:-4]
        llm = Mock(); llm.read.side_effect = lambda path, requested: {"documents": [response(requested)]}
        with patch("extract.read_mrz", return_value={"parsed": mrz.parse_mrz("\n".join(lines)),
                                                      "multiple_documents": False}):
            page = extract.extract_page(metadata(), llm, None)
        self.assertTrue(any("mrz_trailing_filler_restored_review" in c["issues"] for c in page["cards"]))

    def test_toolchain_does_not_accept_failed_version_command(self):
        with patch("shutil.which", return_value="/fake/tesseract"), patch("subprocess.run", side_effect=subprocess.CalledProcessError(1, "tesseract")):
            self.assertFalse(extract.check_mrz_toolchain())


if __name__ == "__main__":
    unittest.main()
