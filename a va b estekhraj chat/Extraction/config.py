"""Run from the project root. Edit corpus paths here, not in terminal commands."""
import os
from pathlib import Path

EXTRACTION_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EXTRACTION_DIR.parent
INPUT_CSV = EXTRACTION_DIR / "Input/variant_a_vision_clip_sample_all_test_pages.csv"
OUTPUT_DIR = EXTRACTION_DIR / "outputs"
CACHE_DIR = EXTRACTION_DIR / "cache"

# Existing project locations shown in the repository. These are inputs, not new folders.
INGESTION_DIR = (PROJECT_ROOT / "Life Prod S3 export"
                 / "Life Document Ingestion Pipeline")
AUSWEIS_OUTPUT_DIR = PROJECT_ROOT / "ausweiskopie_page_detection" / "outputs"

METADATA_FILES = [
    INGESTION_DIR / "g07_page_labels.jsonl",
    INGESTION_DIR / "other_page_labels.jsonl",
    AUSWEIS_OUTPUT_DIR / "AB1_page_labels.jsonl",
    AUSWEIS_OUTPUT_DIR / "ab1_pseudo_documents.csv",
]
AUSWEIS_RENDERED_PAGES_DIR = (PROJECT_ROOT / "ausweiskopie_page_detection"
                              / "Ausweiskopie" / "RenderedPages")
LIFE_RENDERED_PAGES_DIR = INGESTION_DIR / "RenderedPages"
IMAGE_ROOTS = [AUSWEIS_RENDERED_PAGES_DIR, LIFE_RENDERED_PAGES_DIR,
               INGESTION_DIR, AUSWEIS_OUTPUT_DIR,
               PROJECT_ROOT, PROJECT_ROOT.parent,
               Path.home() / "Projects/life-docai",
               Path.home() / "Projects/life-docai/ausweiskopie_page_detection",
               Path.home() / "Projects/life-docai/Life Prod S3 export/Life Document Ingestion Pipeline"]

# Image localisation settings retained from the supplied morphology module.
CARD_WORK_WIDTH = 700
CARD_MIN_AREA_RATIO = 0.08
MRZ_WORK_WIDTH = 900
MRZ_BOX_PAD_RATIO = 0.06
MRZ_CARD_SURFACE_BONUS = 0.75
MRZ_MAX_DESKEW_DEGREES = 30.0
MRZ_MAX_CROPS = 4

# OCR settings belong here so the runtime configuration is visible in one place.
TESSERACT_CMD = os.getenv("TESSERACT_CMD", "tesseract")
MRZ_TESSERACT_PSM = "6"
MRZ_TESSERACT_WHITELIST = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789<"
MRZ_OCR_TIMEOUT_SECONDS = 30
