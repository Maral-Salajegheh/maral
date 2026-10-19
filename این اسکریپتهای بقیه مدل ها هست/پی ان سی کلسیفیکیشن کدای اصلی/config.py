"""Configuration for the Life page-classification pipeline (stack-level, PnC structure)."""

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

# Repository folder names (first letter lower case).
ANTRAG_DIR = "antrag"
DATASETS_DIR = "datasets"
MAPPING_DIR = "mapping"
INGESTION_DIR = "aWS_download_ingestion"
CLASSIFICATION_DIR = "classification"
RESULTS_DIR = "results"


@dataclass(frozen=True)
class PipelineConfig:
    batch_id: str = "life_20260918_A00_G07_MAD_6000_sample_MID"
    root: Path = Path(__file__).resolve().parents[3]  # life-docai/antrag/classification/training/config.py

    # ---- upstream inputs -------------------------------------------------
    @property
    def mapping_output(self) -> Path:
        return self.root / ANTRAG_DIR / DATASETS_DIR / MAPPING_DIR / "output"

    @property
    def ingestion_root(self) -> Path:
        return self.root / ANTRAG_DIR / DATASETS_DIR / INGESTION_DIR

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

    @property
    def document_inventory(self) -> Path:
        return self.batch_output / "document_inventory.parquet"

    # ---- classification outputs ------------------------------------------
    @property
    def classification_output(self) -> Path:
        return self.root / ANTRAG_DIR / CLASSIFICATION_DIR / "output" / self.batch_id

    @property
    def split_file(self) -> Path:
        return self.classification_output / "stack_splits.csv"

    @property
    def checkpoints_dir(self) -> Path:
        return self.classification_output / "checkpoints"

    # Page caches are keyed by image_path, so the v2 caches are reused unchanged.
    @property
    def pnc_ocr_cache(self) -> Path:
        return self.classification_output / "pnc_ocr_cache_full_v2.jsonl"

    @property
    def pnc_clip_feature_file(self) -> Path:
        return self.classification_output / "pnc_clip_features_full_v2.pt"

    # Stack-level population (v3): one sequence per stack_id, ordered by image_id.
    @property
    def model_pages_csv(self) -> Path:
        return self.classification_output / "pnc_stack_pages_v3.csv"

    @property
    def population_report(self) -> Path:
        return self.classification_output / "pnc_stack_population_report_v3.json"

    @property
    def excluded_stacks_report(self) -> Path:
        return self.classification_output / "pnc_excluded_stacks_v3.json"

    @property
    def split_report(self) -> Path:
        return self.classification_output / "pnc_stack_split_report_v3.json"

    @property
    def results_dir(self) -> Path:
        return self.root / ANTRAG_DIR / RESULTS_DIR / self.batch_id
