"""Configuration for the Life page-classification pipeline."""

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


@dataclass(frozen=True)
class PipelineConfig:
    batch_id: str = "life_20260918_A00_G07_MAD_6000_sample_MID"
    root: Path = Path(__file__).resolve().parents[2]

    @property
    def mapping_output(self) -> Path:
        return self.root / "Antrag" / "Datasets" / "Mapping" / "output"

    @property
    def ingestion_root(self) -> Path:
        return self.root / "Antrag" / "Datasets" / "AWS_download_ingestion"

    @property
    def batch_output(self) -> Path:
        return self.ingestion_root / "output" / self.batch_id

    @property
    def render_root(self) -> Path:
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

    @property
    def classification_output(self) -> Path:
        return self.root / "Antrag" / "Classification" / "output" / self.batch_id

    @property
    def matched_pages_csv(self) -> Path:
        return self.classification_output / "matched_pages.csv"

    @property
    def skipped_pages_csv(self) -> Path:
        return self.classification_output / "skipped_pages.csv"

    @property
    def validation_report(self) -> Path:
        return self.classification_output / "life_page_validation_report.json"

    @property
    def verification_sheet(self) -> Path:
        return self.classification_output / "life_page_verification_20.csv"

    @property
    def split_file(self) -> Path:
        return self.classification_output / "stack_splits.csv"

    @property
    def ocr_cache(self) -> Path:
        return self.classification_output / "ocr_cache.jsonl"

    @property
    def clip_feature_file(self) -> Path:
        return self.classification_output / "clip_features.pt"

    @property
    def checkpoints_dir(self) -> Path:
        return self.classification_output / "checkpoints"

    @property
    def pnc_ocr_cache(self) -> Path:
        return self.classification_output / "pnc_ocr_cache_full_v2.jsonl"

    @property
    def pnc_clip_feature_file(self) -> Path:
        return self.classification_output / "pnc_clip_features_full_v2.pt"

    @property
    def model_pages_csv(self) -> Path:
        return self.classification_output / "pnc_model_pages_v2.csv"

    @property
    def population_report(self) -> Path:
        return self.classification_output / "pnc_population_report_v2.json"

    @property
    def invalid_mids_report(self) -> Path:
        return self.classification_output / "pnc_invalid_mids_v2.json"

    def stage1_feature_file(self, modality: str) -> Path:
        return self.classification_output / f"pnc_stage1_features_{modality}.pt"

    @property
    def results_dir(self) -> Path:
        return self.root / "Antrag" / "Results" / self.batch_id

    @property
    def results_summary_csv(self) -> Path:
        return self.results_dir / "smoke_test_summary.csv"