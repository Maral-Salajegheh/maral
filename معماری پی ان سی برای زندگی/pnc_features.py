"""Extract validated Stage 2 inputs from a trained Stage 1 checkpoint."""

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .config import PipelineConfig
from .ocr import load_ocr_cache
from .pnc_data import load_model_pages, population_hash, select_complete_mids, split_assignments
from .pnc_stage1 import Stage1Dataset, collate_stage1, file_sha256, load_clip, load_stage1_checkpoint


def extract_features(config, checkpoint, output, batch_size=8, limit_mids=None, training_limit_mids=None, seed=13, device=None):
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite feature cache: {output}")
    rows = load_model_pages(config)
    if limit_mids and training_limit_mids:
        raise ValueError("Choose limit_mids or training_limit_mids")
    if training_limit_mids:
        assignments = split_assignments(config, rows, seed)
        selected = []
        for split in ["train", "val"]:
            split_rows = [row for row in rows if assignments[row["stack_id"]] == split]
            count = training_limit_mids if split == "train" else max(1, training_limit_mids // 4)
            selected.extend(select_complete_mids(split_rows, count))
    else:
        selected = select_complete_mids(rows, limit_mids)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, checkpoint_payload = load_stage1_checkpoint(checkpoint, device)
    metadata = checkpoint_payload["metadata"]
    if metadata["population_hash"] != population_hash(rows):
        raise ValueError("Stage 1 checkpoint population is incompatible with current model population")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval()
    modality = metadata["modality"]
    ##ocr = load_ocr_cache(config.ocr_cache) if modality in {"text", "fusion"} else {}
    ocr = load_ocr_cache(config.pnc_ocr_cache) if modality in {"text", "fusion"} else {}
    clip = load_clip(config) if modality in {"vision", "fusion"} else {}
    dataset = Stage1Dataset(selected, model, modality, ocr, clip)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda batch: collate_stage1(batch, model, modality),
    )
    feature_blocks = []
    page_metadata = []
    with torch.no_grad():
        for batch in loader:
            visual = batch.get("visual")
            text = batch.get("text")
            if visual is not None:
                visual = visual.to(device)
            if text is not None:
                text = {key: value.to(device) for key, value in text.items()}
            trained_representation = model(visual, text, return_embeddings=True)
            if modality == "fusion":
                features = torch.cat((trained_representation[:, :1024], visual), dim=1)
            else:
                features = trained_representation
            feature_blocks.append(features.cpu())
            for row in batch["rows"]:
                page_metadata.append({
                    "page_identity": row["page_identity"],
                    "masterindex_id": row["masterindex_id"],
                    "stack_id": row["stack_id"],
                    "pdf_page_number": int(row["pdf_page_number"]),
                    "has_label": row["has_label"].lower() == "true",
                    "class_target": int(row["class_target"]),
                    "segmentation_target": int(row["segmentation_target"]),
                    "last_page_target": int(row["last_page_target"]),
                    "page_class": row["page_class"],
                    "training_label_quality": row["training_label_quality"],
                    "placement": row["placement"],
                })
    feature_tensor = torch.cat(feature_blocks) if feature_blocks else torch.zeros((0, 2048 if modality == "fusion" else 1024))
    expected_dim = 2048 if modality == "fusion" else 1024
    if feature_tensor.shape != (len(page_metadata), expected_dim):
        raise ValueError(f"Feature shape mismatch: {feature_tensor.shape}, expected {(len(page_metadata), expected_dim)}")
    cache = {
        "metadata": {
            "format": "PnC_trained_stage1_features_v2",
            "stage1_checkpoint": str(Path(checkpoint).resolve()),
            "stage1_checkpoint_sha256": file_sha256(checkpoint),
            "modality": modality,
            "feature_dim": expected_dim,
            "page_count": len(page_metadata),
            "mid_count": len({page["masterindex_id"] for page in page_metadata}),
            "full_population_hash": population_hash(rows),
            "selected_population_hash": population_hash(selected),
            "ordering": "masterindex_id,pdf_page_number",
        },
        "pages": page_metadata,
        "features": feature_tensor,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cache, output)
    print(output)
    return cache


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-id", default=PipelineConfig.batch_id)
    parser.add_argument("--stage1-checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--limit-mids", type=int)
    parser.add_argument("--training-limit-mids", type=int)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--device")
    args = parser.parse_args()
    extract_features(
        PipelineConfig(batch_id=args.batch_id), args.stage1_checkpoint, args.output,
        args.batch_size, args.limit_mids, args.training_limit_mids, args.seed, args.device,
    )


if __name__ == "__main__":
    main()
