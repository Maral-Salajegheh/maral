"""Central configuration for the ingestion pipeline. All values can be
overridden with environment variables."""
from __future__ import annotations

import os
from pathlib import Path

AWS_REGION = os.getenv("AWS_REGION", "eu-central-1")
PROJECT_BUCKET = os.getenv("PROJECT_BUCKET", "ap-mlops-life-ai")
CORPUS_ROOT_PREFIX = os.getenv("CORPUS_ROOT_PREFIX", "data/document_corpus").strip("/")
LOCAL_TEMP_ROOT = Path(os.getenv("LOCAL_TEMP_ROOT", "/tmp/life-document-ai"))

ALLOWED_EXTENSIONS = {".pdf", ".csv"}
MAX_ZIP_MEMBERS = int(os.getenv("MAX_ZIP_MEMBERS", "500000"))
MAX_UNCOMPRESSED_BYTES = int(os.getenv("MAX_UNCOMPRESSED_BYTES", str(500 * 1024**3)))
MAX_SINGLE_FILE_BYTES = int(os.getenv("MAX_SINGLE_FILE_BYTES", str(20 * 1024**3)))
# Maximum ZIP-in-ZIP depth; deeper containers get explicit error rows.
MAX_ZIP_NESTING = int(os.getenv("MAX_ZIP_NESTING", "3"))

DEFAULT_RENDER_DPI = int(os.getenv("RENDER_DPI", "200"))
DEFAULT_RENDER_FORMAT = os.getenv("RENDER_FORMAT", "png").lower()
DEFAULT_RENDER_VERSION = os.getenv("RENDER_VERSION", "v1")
# Adaptive zoom: short side is upscaled to at least MIN, long side capped at MAX.
TARGET_MIN_SHORT_SIDE_PX = int(os.getenv("TARGET_MIN_SHORT_SIDE_PX", "1024"))
TARGET_MAX_LONG_SIDE_PX = int(os.getenv("TARGET_MAX_LONG_SIDE_PX", "2400"))
