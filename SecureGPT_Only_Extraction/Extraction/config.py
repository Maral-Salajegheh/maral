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
                 / "Life Document Ingestion Pipeline" / "output")
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

# Same output/cache roots; separate checkpoints from the previous MRZ runs.
PIPELINE_VERSION = "securegpt-only-v1"
