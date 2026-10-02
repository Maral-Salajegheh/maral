"""Precompute CLIP visual features for verified pages."""

import argparse
from pathlib import Path

import open_clip
import torch
from PIL import Image
from safetensors.torch import load_file
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from .config import CLIP_MODEL_NAME, CLIP_MODEL_PATH, PipelineConfig
from .data import read_csv

class ImagePathDataset(Dataset):
    def __init__(self, rows, root, transform):
        self.rows = rows
        self.root = root
        self.transform = transform
    def __len__(self):
        return len(self.rows)
    def __getitem__(self, idx):
        row = self.rows[idx]
        path = self.root / row["image_path"]
        image = Image.open(path).convert("RGB")
        return idx, self.transform(image)

def _save_features_atomic(config, features):
    tmp_path = config.clip_feature_file.with_suffix(".pt.tmp")
    torch.save(features, tmp_path)
    tmp_path.replace(config.clip_feature_file)


def precompute_clip(config, limit=None, batch_size=32, device=None, checkpoint_every=50, rows=None):
    rows = read_csv(config.matched_pages_csv) if rows is None else rows
    if limit:
        rows = rows[:limit]
    config.clip_feature_file.parent.mkdir(parents=True, exist_ok=True)
    features = {}
    if config.clip_feature_file.exists():
        features = torch.load(config.clip_feature_file, map_location="cpu")
        print(f"resuming_clip_features={len(features)}")
    pending_rows = [row for row in rows if row["image_path"] not in features]
    if not pending_rows:
        print(f"clip_features={config.clip_feature_file} rows={len(features)}")
        return
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, _, preprocess = open_clip.create_model_and_transforms(CLIP_MODEL_NAME)
    model.load_state_dict(load_file(CLIP_MODEL_PATH))
    visual = model.visual.eval().requires_grad_(False).to(device)
    dataset = ImagePathDataset(pending_rows, config.render_root, preprocess)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    with torch.no_grad():
        for batch_index, (indices, images) in enumerate(tqdm(loader, desc="clip"), start=1):
            out = visual(images.to(device)).cpu()
            for index, vector in zip(indices.tolist(), out):
                features[pending_rows[index]["image_path"]] = vector
            if batch_index % checkpoint_every == 0:
                _save_features_atomic(config, features)
    _save_features_atomic(config, features)
    print(f"clip_features={config.clip_feature_file} rows={len(features)}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-id", default=PipelineConfig.batch_id)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device")
    args = parser.parse_args()
    precompute_clip(PipelineConfig(batch_id=args.batch_id), args.limit, args.batch_size, args.device)

if __name__ == "__main__":
    main()
