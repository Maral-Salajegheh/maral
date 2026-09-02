from __future__ import annotations

import os
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PACKAGE_ROOT.parent
DATA_DIR = PACKAGE_ROOT / "data"
RESULTS_DIR = PACKAGE_ROOT / "results"
EXTRACTION_DIR = PACKAGE_ROOT / "Extraction"
CACHE_DIR = EXTRACTION_DIR / "cache"
OUTPUT_DIR = EXTRACTION_DIR / "outputs"

INPUT_PAGES_CSV = RESULTS_DIR / "variant_a_vision_clip_sample_all_test_pages.csv"
CANDIDATES_CSV = OUTPUT_DIR / "01_g07_candidates.csv"
MRZ_CROPS_JSONL = OUTPUT_DIR / "02_mrz_crops.jsonl"
MRZ_OCR_JSONL = OUTPUT_DIR / "03_mrz_ocr.jsonl"
MRZ_PARSED_JSONL = OUTPUT_DIR / "04_mrz_parsed.jsonl"
AUSWEISTYP_JSONL = OUTPUT_DIR / "05_ausweistyp.jsonl"
FIELD_EXTRACTION_JSONL = OUTPUT_DIR / "06_field_extraction.jsonl"
FIELD_OCR_JSONL = FIELD_EXTRACTION_JSONL
FINAL_JSON_DIR = OUTPUT_DIR / "documents"
FINAL_CSV_PATH = OUTPUT_DIR / "06_documents_flat.csv"

MRZ_CROP_DIR = CACHE_DIR / "mrz_crops"
SURFACE_DIR = CACHE_DIR / "surfaces"
MRZ_DEBUG_DIR = CACHE_DIR / "mrz_overlays"
FIELD_CROP_DIR = CACHE_DIR / "field_crops"

RENDERED_PAGES_DIR = Path.home() / "Projects/life-docai/ausweiskopie_page_detection/Ausweiskopie/RenderedPages"
SECUREGPT_WRAPPER_DIR = Path.home() / "Projects/life-docai/ausweiskopie_page_detection"

IMAGE_ROOTS = [
    RENDERED_PAGES_DIR,
    REPO_ROOT,
    PACKAGE_ROOT,
    Path.home() / "Projects/life-docai",
    Path.home() / "Projects/life-docai/ausweiskopie_page_detection",
    Path.home() / "Projects/life-docai/Life Prod S3 export/Life Document Ingestion Pipeline",
]

TESSERACT_CMD = os.environ.get("TESSERACT_CMD", "tesseract")
# psm 6 only. The character whitelist is ignored by the LSTM engine, and forcing --oem 0
# to honour it needs legacy tessdata that is usually absent. Characters are normalized after OCR.
MRZ_TESSERACT_CONFIG = "--psm 6"
MRZ_OCR_BACKEND = os.environ.get("MRZ_OCR_BACKEND", "auto").lower()
FIELD_TESSERACT_CONFIG = "--psm 6"
FIELD_TESSERACT_LANG = os.environ.get("FIELD_TESSERACT_LANG", "deu+eng")
FIELD_EXTRACTION_BACKEND = os.environ.get("FIELD_EXTRACTION_BACKEND", "llm").lower()
FIELD_OCR_STAGE_VERSION = "rapidocr_fallback_v2"

MRZ_DETECTOR_VERSION = "rapidocr_lines_v3"
MRZ_OCR_STAGE_VERSION = "rapidocr_crop_v3"

# Detector: a band is accepted at MRZ_MIN_GROUP_SCORE, and rotation search stops at MRZ_GOOD_SCORE.
MRZ_MIN_GROUP_SCORE = 1.20
MRZ_GOOD_SCORE = 1.60
MRZ_CANONICAL_WIDTHS = (30, 36, 44)
MRZ_WIDTH_TOLERANCE = 10

# Stage 02 upscales the crop so each MRZ line is roughly this tall before recognition.
MRZ_TARGET_LINE_HEIGHT = 48
MRZ_MAX_UPSCALE = 4.0
MRZ_CROP_BORDER = 40

# Morphological detector: all kernel sizes assume the page is resized to this width.
MRZ_WORK_WIDTH = 900
MRZ_MORPH_MIN_SCORE = 1.60
MRZ_MORPH_MAX_CANDIDATES = 6
MRZ_BOX_PAD_RATIO = 0.06

# Card detection. Photographs of an ID on a table or carpet only work if the card is
# located and flattened first; searching the raw frame finds carpet texture instead.
CARD_WORK_WIDTH = 700
CARD_MIN_AREA_RATIO = 0.08
MRZ_CARD_SURFACE_BONUS = 0.75
MRZ_MAX_DESKEW_DEGREES = 30.0

# Re-read a correctly cropped but unreadable band with the vision model. Gated on a
# successful crop, so a detection failure is never handed to the model to guess at.
MRZ_LLM_FALLBACK = os.environ.get("MRZ_LLM_FALLBACK", "1") not in {"0", "false", "False"}

# Glyphs RapidOCR returns in place of the MRZ filler character. Substituted, never deleted.
MRZ_CHAR_FIXES = {
    ">": "<", "«": "<", "»": "<", "‹": "<", "›": "<", "^": "<", "~": "<",
    "(": "<", ")": "<", "[": "<", "]": "<", "{": "<", "}": "<", "|": "<",
    "/": "<", "\\": "<", "-": "<", "—": "<", "_": "<", "=": "<", "+": "<",
    "*": "<", ".": "<", ",": "<", ":": "<", ";": "<", "'": "<", '"': "<",
    "!": "<", "?": "<", " ": "<",
}

OCR_CONFUSIONS = {"O": "0", "0": "O", "I": "1", "1": "I", "S": "5", "5": "S", "B": "8", "8": "B", "Z": "2", "2": "Z"}
