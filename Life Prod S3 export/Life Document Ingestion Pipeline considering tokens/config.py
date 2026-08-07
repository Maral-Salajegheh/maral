"""Central configuration for the ingestion pipeline. All values can be
overridden with environment variables."""
from __future__ import annotations

import os
from pathlib import Path

AWS_REGION = os.getenv("AWS_REGION", "eu-central-1")
PROJECT_BUCKET = os.getenv("PROJECT_BUCKET", "ap-mlops-life-ai")
# Only the raw ZIPs are kept in S3, under this prefix.
RAW_DATA_PREFIX = os.getenv("RAW_DATA_PREFIX", "life_prod_raw_data").strip("/")
LOCAL_TEMP_ROOT = Path(os.getenv("LOCAL_TEMP_ROOT", "/tmp/life-document-ai"))
# Manifests and rendered pages stay local, under the current working directory.
LOCAL_OUTPUT_ROOT = Path(os.getenv("LOCAL_OUTPUT_ROOT", "output"))
RENDER_ROOT = Path(os.getenv("RENDER_ROOT", "RenderedPages"))
# Default location of incoming Life Prod export ZIPs. A bare filename given on
# the command line is resolved against this prefix.
SOURCE_ZIP_PREFIX = os.getenv(
    "SOURCE_ZIP_PREFIX", "s3://itecmcm-prod-prod-flexporter-life-prod/02_OUT"
).rstrip("/")

ALLOWED_EXTENSIONS = {".pdf", ".csv"}
MAX_ZIP_MEMBERS = int(os.getenv("MAX_ZIP_MEMBERS", "500000"))

# Baseline DPI for normal A4 pages. Confirmed identity pages may later be
# rendered at 300 DPI for field extraction.
DEFAULT_RENDER_DPI = int(os.getenv("RENDER_DPI", "200"))
DEFAULT_RENDER_FORMAT = os.getenv("RENDER_FORMAT", "png").lower()
DEFAULT_RENDER_VERSION = os.getenv("RENDER_VERSION", "v1")

# Adaptive zoom targets. A directly scanned ID card (~54x86 mm) renders far
# too small at a fixed DPI; the zoom upscales the short side to at least
# TARGET_MIN_SHORT_SIDE_PX and caps the long side at TARGET_MAX_LONG_SIDE_PX.
# Normal A4 at 200 DPI (~1654x2339 px) is unaffected.
TARGET_MIN_SHORT_SIDE_PX = int(os.getenv("TARGET_MIN_SHORT_SIDE_PX", "1024"))
TARGET_MAX_LONG_SIDE_PX = int(os.getenv("TARGET_MAX_LONG_SIDE_PX", "2400"))

# Quality thresholds for flagging unusable renders before downstream use.
MIN_IMAGE_WIDTH = int(os.getenv("MIN_IMAGE_WIDTH", "500"))
MIN_IMAGE_HEIGHT = int(os.getenv("MIN_IMAGE_HEIGHT", "500"))
BLANK_WHITE_PIXEL_RATIO = float(os.getenv("BLANK_WHITE_PIXEL_RATIO", "0.995"))
LOW_CONTRAST_STDDEV = float(os.getenv("LOW_CONTRAST_STDDEV", "8.0"))