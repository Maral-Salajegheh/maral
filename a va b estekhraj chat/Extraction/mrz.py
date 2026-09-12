"""Conservative TD1/TD2/TD3 parsing. Never guess or repair characters."""
from datetime import datetime
from pathlib import Path
import os
import re
import subprocess
import tempfile

from PIL import Image, ImageOps


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


def parse_window(lines, fmt, parser):
    code, number, nationality, expiry, birth, sex, checks = parser(lines)
    if not all(checks) or not valid_date(birth) or not valid_date(expiry):
        return None
    if code[0] not in "IPAC" or sex not in "MF<" or not number.strip("<"):
        return None
    if fmt == "TD3" and code[0] != "P":
        return None
    if not re.fullmatch(r"[A-Z<]{3}", nationality):
        return None
    return {"format": fmt, "lines": lines, "document_code": code,
            "fields": {"ausweisnummer": number.replace("<", ""),
                       "nationalitaet": nationality.replace("<", "") or None,
                       "gueltigkeitsdatum": expiry}}


def parse_mrz(text):
    # Remove OCR whitespace only. Do not pad, truncate, or substitute unknown glyphs.
    lines = [re.sub(r"\s", "", line.upper()) for line in text.splitlines()]
    found = []
    for fmt, count, width, parser in (("TD1", 3, 30, parse_td1),
                                     ("TD2", 2, 36, parse_td2), ("TD3", 2, 44, parse_td3)):
        for start in range(len(lines) - count + 1):
            window = lines[start:start + count]
            if not all(len(s) == width and re.fullmatch(r"[A-Z0-9<]+", s) for s in window):
                continue
            result = parse_window(window, fmt, parser)
            if result:
                found.append(result)
    identities = {tuple(item["fields"].items()) + (("code", item["document_code"]),) for item in found}
    if len(identities) > 1:
        raise ValueError("Several different valid MRZs on one page; split the image first")
    return found[0] if found else None


def tesseract(image, directory):
    path = Path(directory) / "ocr.png"
    ImageOps.expand(image, border=20, fill="white").save(path)
    command = [os.getenv("TESSERACT_CMD", "tesseract"), str(path), "stdout", "--psm", "6",
               "-c", "tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789<"]
    return subprocess.run(command, capture_output=True, text=True, check=True, timeout=30).stdout


def proposed_crops(page):
    from mrz_morph import morphological_candidates
    import config
    for item in morphological_candidates(page)[:config.MRZ_MAX_CROPS]:
        surface = item["surface_image"]
        box = (item["x0"], item["y0"], item["x1"], item["y1"])
        crop = surface.crop(box)
        scale = min(4.0, max(1.0, 2000 / max(crop.width, 1)))
        crop = crop.resize((max(1, round(crop.width * scale)), max(1, round(crop.height * scale))), Image.Resampling.LANCZOS)
        yield crop, {"box": box, "surface": item["surface"]}


def scan_view(image, directory, audit, label, debug_dir, region=None):
    if debug_dir:
        debug_dir.mkdir(parents=True, exist_ok=True)
        image.save(debug_dir / (label + ".png"))
    text = tesseract(image, directory)
    attempt = {"view": label, "region": region, "text": text}
    audit["attempts"].append(attempt)
    try:
        return parse_mrz(text)
    except ValueError as error:
        audit["multiple_documents"] = True
        audit["errors"].append(str(error))
        return None


def select_result(found, audit, angle):
    identities = {tuple(sorted(p["fields"].items())) for p in found}
    if len(identities) > 1:
        audit["multiple_documents"] = True
        audit["errors"].append("Different valid MRZs across crops")
    if found and not audit["multiple_documents"]:
        audit["parsed"], audit["rotation"] = found[0], angle
    audit["ocr_text"] = "\n".join(item["text"] for item in audit["attempts"])
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
        audit["errors"].append(str(error))
        return select_result([], audit, None)


def scan_rotations(source, directory, audit, debug_dir):
    morphology_available = True
    for angle in (0, 90, 180, 270):
        page, found = source.rotate(angle, expand=True), []
        try:
            crops = list(proposed_crops(page)) if morphology_available else []
        except Exception as error:
            audit["errors"].append("MRZ localisation: " + str(error))
            morphology_available = False
            crops = []
        for index, (crop, region) in enumerate(crops):
            parsed = scan_view(crop, directory, audit, f"rotation_{angle}_crop_{index}", debug_dir, region)
            if parsed:
                found.append(parsed)
        if not found and not audit["multiple_documents"]:
            parsed = scan_view(page, directory, audit, f"rotation_{angle}_full", None)
            if parsed:
                found.append(parsed)
        if found or audit["multiple_documents"]:
            return select_result(found, audit, angle)
    return select_result([], audit, None)
