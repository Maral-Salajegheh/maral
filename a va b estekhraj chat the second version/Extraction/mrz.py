"""MRZ OCR, conservative parsing, and diagnostics without personal values."""
from datetime import datetime
from pathlib import Path
import re
import subprocess
import tempfile

from itertools import product

from PIL import Image, ImageDraw, ImageOps

import config


def digit(text):
    values = [0 if c == "<" else int(c) if c.isdigit() else ord(c) - 55 for c in text]
    return str(sum(v * (7, 3, 1)[i % 3] for i, v in enumerate(values)) % 10)


def valid_date(value):
    if not re.fullmatch(r"[0-9]{6}", value):
        return False
    try:
        datetime(2000 + int(value[:2]), int(value[2:4]), int(value[4:]))
        return True
    except ValueError:
        return False


def check(value, actual, optional=False):
    if optional and set(value) == {"<"} and actual == "<":
        return True
    return digit(value) == actual


def parse_td1(lines):
    a, b, _ = lines
    checks = [check(a[5:14], a[14]), check(b[:6], b[6]),
              check(b[8:14], b[14]),
              check(a[5:30] + b[:7] + b[8:15] + b[18:29], b[29])]
    return a[:2], a[5:14], b[15:18], b[8:14], b[:6], b[7], checks


def parse_td2(lines):
    a, b = lines
    checks = [check(b[:9], b[9]), check(b[13:19], b[19]),
              check(b[21:27], b[27]),
              check(b[:10] + b[13:20] + b[21:35], b[35])]
    return a[:2], b[:9], b[10:13], b[21:27], b[13:19], b[20], checks


def parse_td3(lines):
    a, b = lines
    checks = [check(b[:9], b[9]), check(b[13:19], b[19]),
              check(b[21:27], b[27]), check(b[28:42], b[42], optional=True),
              check(b[:10] + b[13:20] + b[21:43], b[43])]
    return a[:2], b[:9], b[10:13], b[21:27], b[13:19], b[20], checks


FORMATS = (("TD1", 3, 30, parse_td1), ("TD2", 2, 36, parse_td2),
           ("TD3", 2, 44, parse_td3))
CHECK_NAMES = {"TD1": ("document_number", "birth_date", "expiry_date", "composite"),
               "TD2": ("document_number", "birth_date", "expiry_date", "composite"),
               "TD3": ("document_number", "birth_date", "expiry_date", "optional_data", "composite")}


def cleaned_lines(text):
    # Ignore empty OCR lines, but never replace a glyph or join separate text lines.
    return [value for raw in text.splitlines()
            if (value := re.sub(r"\s", "", raw.upper()))]


def validation_failures(lines, fmt, parser):
    """Return only fixed reason labels, never field contents."""
    code, number, nationality, expiry, birth, sex, checks = parser(lines)
    failed = [name + "_checksum" for name, ok in zip(CHECK_NAMES[fmt], checks) if not ok]
    if not valid_date(birth):
        failed.append("birth_date_format")
    if not valid_date(expiry):
        failed.append("expiry_date_format")
    allowed = "P" if fmt == "TD3" else "ACI"
    if code[0] not in allowed or not re.fullmatch(r"[A-Z<]{2}", code):
        failed.append("document_code_format")
    if not re.fullmatch(r"[A-Z]{1,3}<{0,2}", lines[0][2:5]):
        failed.append("issuing_state_format")
    if sex not in "MF<":
        failed.append("sex_format")
    if not re.fullmatch(r"[A-Z0-9]+<*", number):
        failed.append("document_number_format")
    if not re.fullmatch(r"(?:[A-Z]{1,3}<{0,2}|<<<)", nationality):
        failed.append("nationality_format")
    name = lines[2] if fmt == "TD1" else lines[0][5:]
    if not re.fullmatch(r"[A-Z<]+", name) or not name.strip("<"):
        failed.append("name_format")
    # This release deliberately leaves extended document numbers to the full-page LLM.
    check_char = lines[0][14] if fmt == "TD1" else lines[1][9]
    if check_char == "<" and number.strip("<"):
        failed.append("extended_document_number_not_supported")
    return failed


def parse_window(lines, fmt, parser):
    if validation_failures(lines, fmt, parser):
        return None
    code, number, nationality, expiry, birth, sex, _ = parser(lines)
    name = lines[2] if fmt == "TD1" else lines[0][5:]
    return {"format": fmt, "lines": lines, "document_code": code,
            "issuing_state": lines[0][2:5], "birth_date": birth, "sex": sex,
            "name": name.rstrip("<"),
            "fields": {"ausweisnummer": number.rstrip("<"),
                       "nationalitaet": nationality.rstrip("<") or None,
                       "gueltigkeitsdatum": expiry}}


def line_variants(line, width, fmt, index):
    if len(line) == width:
        return [line]
    # Only right-pad an already visible trailing filler run on the TD1 upper/name
    # line or TD2/TD3 name line. Never pad the lower data line, trim edges, insert
    # before a check digit, or substitute O/0. Even these restorations need review.
    can_pad = (fmt == "TD1" and index in (0, 2)) or (fmt != "TD1" and index == 0)
    missing = width - len(line)
    if (can_pad and 0 < missing <= config.MRZ_MAX_MISSING_FILLER
            and line.endswith("<<<")):
        return [line.ljust(width, "<")]
    return []


def result_identity(parsed):
    return (tuple(sorted(parsed["fields"].items())), parsed["document_code"],
            parsed.get("issuing_state"), parsed.get("birth_date"),
            parsed.get("sex"), parsed.get("name"))


def inspect_mrz(text, readings=None):
    """Parse and explain the same candidates; diagnostics contain no OCR values."""
    lines = cleaned_lines(text)
    diagnostic = {"line_lengths": [len(line) for line in lines],
                  "invalid_character_counts": [sum(c not in "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789<"
                                                    for c in line) for line in lines],
                  "reason": "empty_ocr" if not lines else "no_mrz_shaped_text",
                  "candidates": []}
    found = []
    for fmt, count, width, parser in FORMATS:
        for start in range(len(lines) - count + 1):
            window = lines[start:start + count]
            # A loose window is useful for diagnostics, not permission to repair it.
            if any(abs(len(line) - width) > 10 for line in window):
                continue
            report = {"format": fmt, "start_line": start + 1,
                      "line_lengths": [len(line) for line in window], "failed_checks": []}
            if not all(re.fullmatch(r"[A-Z0-9<]+", line) for line in window):
                report["failed_checks"] = ["invalid_characters"]
            else:
                variants = [line_variants(line, width, fmt, i) for i, line in enumerate(window)]
                if not all(variants):
                    report["failed_checks"] = ["line_lengths"]
                else:
                    for candidate in product(*variants):
                        candidate = list(candidate)
                        report["failed_checks"] = validation_failures(candidate, fmt, parser)
                        result = parse_window(candidate, fmt, parser)
                        if result:
                            result.update({"filler_restored": candidate != window,
                                           "needs_review": candidate != window, "ocr_lines": window})
                            found.append(result)
            diagnostic["candidates"].append(report)
    if readings is not None:
        readings.extend(found)
    if len({result_identity(item) for item in found}) > 1:
        diagnostic["reason"] = "multiple_valid_mrz"
        return None, diagnostic
    if found:
        diagnostic["reason"] = "parsed"
        return found[0], diagnostic
    reports = diagnostic["candidates"]
    if reports:
        # Exact/allowed widths provide stronger evidence than an ill-shaped window.
        best = min(reports, key=lambda r: ("line_lengths" in r["failed_checks"],
                                           "invalid_characters" in r["failed_checks"],
                                           len(r["failed_checks"])))
        diagnostic["best_candidate"] = best
        failed = best["failed_checks"]
        diagnostic["reason"] = ("line_length_mismatch" if failed == ["line_lengths"] else
                                "invalid_characters" if failed == ["invalid_characters"] else
                                "validation_failed")
    return None, diagnostic


def parse_mrz(text):
    parsed, diagnostic = inspect_mrz(text)
    if diagnostic["reason"] == "multiple_valid_mrz":
        raise ValueError("Several different valid MRZ readings; review the page")
    return parsed


def tesseract(image, directory, psm=None):
    import config
    path = Path(directory) / "ocr.png"
    ImageOps.expand(image, border=20, fill="white").save(path)
    command = [config.TESSERACT_CMD, str(path), "stdout", "--psm",
               psm or config.MRZ_TESSERACT_PSM, "-c",
               "tessedit_char_whitelist=" + config.MRZ_TESSERACT_WHITELIST]
    return subprocess.run(command, capture_output=True, text=True, check=True,
                          timeout=config.MRZ_OCR_TIMEOUT_SECONDS).stdout


def proposed_crops(page, debug_dir=None, prefix="view"):
    from mrz_morph import morphological_candidates
    import config
    selected = []
    for item in morphological_candidates(page):
        duplicate = False
        for old in selected:
            if item["surface"] != old["surface"]:
                continue
            intersection = max(0, min(item["x1"], old["x1"]) - max(item["x0"], old["x0"])) * max(0, min(item["y1"], old["y1"]) - max(item["y0"], old["y0"]))
            area = (item["x1"] - item["x0"]) * (item["y1"] - item["y0"])
            old_area = (old["x1"] - old["x0"]) * (old["y1"] - old["y0"])
            if intersection / max(area + old_area - intersection, 1) >= 0.85:
                duplicate = True
                break
        if not duplicate:
            selected.append(item)
        if len(selected) >= config.MRZ_MAX_CROPS:
            break
    for index, item in enumerate(selected):
        surface = item["surface_image"]
        box = (item["x0"], item["y0"], item["x1"], item["y1"])
        if debug_dir:
            debug_dir.mkdir(parents=True, exist_ok=True)
            surface.save(debug_dir / f"{prefix}_{item['surface']}_surface.png")
            overlay = surface.copy()
            ImageDraw.Draw(overlay).rectangle(box, outline="red", width=5)
            overlay.save(debug_dir / f"{prefix}_candidate_{index}_overlay.png")
        crop = surface.crop(box)
        scale = min(4.0, max(1.0, 2000 / max(crop.width, 1)))
        crop = crop.resize((max(1, round(crop.width * scale)), max(1, round(crop.height * scale))), Image.Resampling.LANCZOS)
        yield crop, {"box": box, "surface": item["surface"]}


def scan_view(image, directory, audit, label, debug_dir, region=None, psm=None):
    attempt = {"view": label, "region": region, "text": "", "status": "started",
               "psm": psm or config.MRZ_TESSERACT_PSM}
    audit["attempts"].append(attempt)
    try:
        if debug_dir:
            debug_dir.mkdir(parents=True, exist_ok=True)
            image.save(debug_dir / (label + ".png"))
        text = tesseract(image, directory) if psm is None else tesseract(image, directory, psm=psm)
        attempt["text"] = text
    except Exception as error:
        # Store technical detail locally; Diagnose.py never prints the error contents.
        attempt.update({"status": "ocr_error", "error_type": type(error).__name__,
                        "error": str(error), "stderr": str(getattr(error, "stderr", "") or "")})
        audit["errors"].append(f"{label}: {type(error).__name__}")
        return None
    readings = []
    parsed, diagnostic = inspect_mrz(text, readings)
    for reading in readings:
        audit.setdefault("readings", []).append({"parsed": reading, "view": label, "region": region})
    attempt.update({"diagnostics": diagnostic, "status": diagnostic["reason"]})
    if diagnostic["reason"] == "multiple_valid_mrz":
        audit["multiple_documents"] = True
        # Separate readings are retained below; association belongs to extraction.
    return parsed


def select_result(found, audit, angle):
    distinct = {}
    for reading in audit.get("readings", []):
        parsed = reading["parsed"]
        key = result_identity(parsed)
        if key not in distinct:
            distinct[key] = {"parsed": parsed, "locations": []}
        elif not parsed.get("filler_restored"):
            distinct[key]["parsed"] = parsed
        distinct[key]["locations"].append({"view": reading["view"], "region": reading["region"]})
    audit["results"] = list(distinct.values())
    if distinct:
        found = [item["parsed"] for item in audit["results"]]
    if len({result_identity(p) for p in found}) > 1:
        audit["multiple_documents"] = True
        # Different results can be different cards, or inconsistent OCR of one card.
    if found and not audit["multiple_documents"]:
        # Prefer a complete read of the same identity over a padded one.
        audit["parsed"] = min(found, key=lambda p: bool(p.get("filler_restored")))
        audit["rotation"] = angle
    audit["ocr_text"] = "\n".join(item["text"] for item in audit["attempts"])
    reasons = {item["status"] for item in audit["attempts"]}
    audit["reason"] = ("multiple_valid_mrz" if audit["multiple_documents"] else
                       "parsed" if audit["parsed"] else
                       "validation_failed" if "validation_failed" in reasons else
                       "line_length_mismatch" if "line_length_mismatch" in reasons else
                       "invalid_characters" if "invalid_characters" in reasons else
                       "no_mrz_shaped_text" if "no_mrz_shaped_text" in reasons else
                       "empty_ocr" if "empty_ocr" in reasons else "ocr_or_image_error")
    return audit


def read_mrz(image_path, debug_dir=None):
    audit = {"parsed": None, "rotation": None, "ocr_text": "", "attempts": [],
             "errors": [], "multiple_documents": False}
    try:
        with Image.open(image_path) as image:
            source = ImageOps.exif_transpose(image).convert("RGB")
        with tempfile.TemporaryDirectory() as directory:
            return scan_rotations(source, directory, audit, debug_dir)
    except Exception as error:
        audit["errors"].append(type(error).__name__ + ": " + str(error))
        return select_result([], audit, None)


def scan_rotations(source, directory, audit, debug_dir):
    for angle in (0, 90, 180, 270):
        page, found = source.rotate(angle, expand=True), []
        try:
            crops = list(proposed_crops(page, debug_dir, f"rotation_{angle}"))
        except Exception as error:
            audit["errors"].append("MRZ localisation: " + type(error).__name__)
            crops = []
        # Check every shortlisted crop at this orientation, even after one succeeds.
        for index, (crop, region) in enumerate(crops):
            label = f"rotation_{angle}_crop_{index}"
            parsed = scan_view(crop, directory, audit, label, debug_dir, region)
            if not parsed and audit["attempts"][-1]["status"] != "multiple_valid_mrz":
                # Sparse-text segmentation can separate MRZ from printed text in a crop.
                # This is another OCR read, not a character repair or a checksum bypass.
                parsed = scan_view(crop, directory, audit, label + "_psm11", debug_dir, region, psm="11")
            if parsed:
                found.append(parsed)
        if not found and not audit["multiple_documents"]:
            parsed = scan_view(page, directory, audit, f"rotation_{angle}_full", debug_dir)
            if parsed:
                found.append(parsed)
        if found or audit["multiple_documents"]:
            return select_result(found, audit, angle)
    return select_result([], audit, None)
