"""Configuration for the page-classification pipeline (stack-level, PnC structure)."""

import os
from dataclasses import dataclass
from pathlib import Path

CLASSES = ("A00", "G07", "MAD", "TB0")
CLASS_TO_ID = {label: idx for idx, label in enumerate(CLASSES)}
ID_TO_CLASS = {idx: label for label, idx in CLASS_TO_ID.items()}

ROOT_MODELS_DIR = Path(os.getenv("PNC_ROOT_MODELS_DIR", "/home/shared_folders/pnc_claims_docai/usecases/docai/pretrained-models"))
TEXT_MODEL_NAME = os.getenv("BERT_MODEL_PATH", str(ROOT_MODELS_DIR / "gbert-large"))
CLIP_MODEL_NAME = "ViT-H-14"
CLIP_MODEL_PATH = os.getenv("CLIP_MODEL_PATH", str(ROOT_MODELS_DIR / "CLIP-ViT-H-14-laion2B-s32B-b79K/open_clip_pytorch_model.safetensors"))

# Folder names inside the package life_docai_analytics_antrag/.
DATASETS_DIR = "datasets"
MAPPING_DIR = "mapping"
INGESTION_DIR = "aws_download_ingestion"
CLASSIFICATION_DIR = "classification"
RESULTS_DIR = "results"


@dataclass(frozen=True)
class PipelineConfig:
    batch_id: str = "life_20260918_A00_G07_MAD_6000_sample_MID"
    # package root: packages/life-docai-analytics/life_docai_analytics_antrag/ (this file is classification/training/config.py)
    root: Path = Path(__file__).resolve().parents[2]

    # ---- upstream inputs -------------------------------------------------
    @property
    def mapping_output(self) -> Path:
        return self.root / DATASETS_DIR / MAPPING_DIR / "output"

    @property
    def ingestion_root(self) -> Path:
        return self.root / DATASETS_DIR / INGESTION_DIR

    @property
    def batch_output(self) -> Path:
        return self.ingestion_root / "output" / self.batch_id

    @property
    def render_root(self) -> Path:
        # image_path values in page_inventory.parquet are relative to this folder
        return self.ingestion_root

    @property
    def pages_csv(self) -> Path:
        return self.mapping_output / "life_mid_pages.csv"

    @property
    def documents_csv(self) -> Path:
        return self.mapping_output / "life_mid_documents.csv"

    @property
    def page_inventory(self) -> Path:
        return self.batch_output / "page_inventory.parquet"

    # ---- dataset (prepare) ------------------------------------------------
    @property
    def classification_output(self) -> Path:
        return self.root / CLASSIFICATION_DIR / "output" / self.batch_id

    @property
    def model_pages_csv(self) -> Path:
        """One row per page, stacks in original page order, with all targets."""
        return self.classification_output / "stack_pages.csv"

    @property
    def population_report(self) -> Path:
        return self.classification_output / "stack_population_report.json"

    @property
    def excluded_stacks_report(self) -> Path:
        return self.classification_output / "excluded_stacks.json"

    @property
    def split_file(self) -> Path:
        return self.classification_output / "stack_splits.csv"

    @property
    def split_report(self) -> Path:
        return self.classification_output / "stack_split_report.json"

    # ---- page caches, keyed by image_path ------------------------------------
    @property
    def ocr_text_file(self) -> Path:
        """OCR text per page (JSONL, appended)."""
        return self.classification_output / "ocr_text.jsonl"

    @property
    def clip_features_dir(self) -> Path:
        """CLIP image vectors per page (write-once shards + index)."""
        return self.classification_output / "clip_features"

    # ---- Stage 1 -> Stage 2 ---------------------------------------------------
    def stage1_features_dir(self, run_name: str, modality: str) -> Path:
        """Frozen Stage 1 page representations, one row per page in stack order."""
        return self.classification_output / f"stage1_features_{modality}_{run_name}"

    # ---- training outputs -------------------------------------------------------
    @property
    def checkpoints_dir(self) -> Path:
        return self.classification_output / "checkpoints"

    @property
    def results_dir(self) -> Path:
        return self.root / CLASSIFICATION_DIR / RESULTS_DIR / self.batch_id
