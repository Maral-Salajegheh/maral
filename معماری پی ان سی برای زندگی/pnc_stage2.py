"""Corrected PnC Stage 2 sequence-level model training, evaluation, and inference."""

import argparse
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from .config import CLASSES, ID_TO_CLASS, IGNORE_INDEX, PipelineConfig
from .models import SequenceModelRNN
from .pnc_data import load_model_pages, population_hash, select_complete_mids, split_assignments
from .pnc_metrics import binary_metrics, classification_metrics, write_csv_new, write_json_new
from .pnc_stage1 import file_sha256


class MIDSequenceDataset(Dataset):
    def __init__(self, sequences, features):
        self.sequences = sequences
        self.features = features

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, index):
        sequence = self.sequences[index]
        features = [self.features[page["page_identity"]] for page in sequence]
        class_targets = [page["class_target"] for page in sequence]
        segmentation_targets = [page["segmentation_target"] for page in sequence]
        return {
            "rows": sequence,
            "features": torch.stack(features),
            "class_targets": torch.tensor(class_targets, dtype=torch.long),
            "segmentation_targets": torch.tensor(segmentation_targets, dtype=torch.long),
        }


def load_feature_cache(config, feature_cache, stage1_checkpoint):
    payload = torch.load(feature_cache, map_location="cpu")
    metadata = payload.get("metadata", {})
    if metadata.get("stage1_checkpoint") != str(Path(stage1_checkpoint).resolve()):
        raise ValueError(f"Feature cache stage1_checkpoint mismatch: {feature_cache}")
    if metadata.get("stage1_checkpoint_sha256") != file_sha256(stage1_checkpoint):
        raise ValueError(f"Feature cache stage1_checkpoint SHA256 mismatch: {feature_cache}")
    
    cache = payload["features"]
    pages = load_model_pages(config)
    enriched = []
    current_by_identity = {page["page_identity"]: page for page in pages}
    seen = set()
    
    for index, page in enumerate(pages):
        identity = page["page_identity"]
        if identity in seen or identity not in current_by_identity:
            raise ValueError(f"Duplicate or stale page identity in feature cache: {identity}")
        seen.add(identity)
        current = current_by_identity[identity]
        for field in ["masterindex_id", "stack_id", "pdf_page_number", "has_label", "class_target", "segmentation_target"]:
            expected = str(current[field]).lower() if field == "has_label" else str(current[field])
            actual = str(page[field]).lower() if field == "has_label" else str(page[field])
            if expected != actual:
                raise ValueError(f"Feature cache metadata mismatch for {identity}: {field}")
        row = dict(page)
        row["feature_index"] = index
        enriched.append(row)
        
    selected_current = [current_by_identity[page["page_identity"]] for page in pages]
    if metadata.get("selected_population_hash") != population_hash(selected_current):
        raise ValueError("Feature cache selected population is stale or reordered")
        
    return cache, enriched, features


def build_mid_sequences(pages):
    by_mid = defaultdict(list)
    for page in pages:
        by_mid[page["masterindex_id"]].append(page)
    sequences = []
    for mid, mid_pages in sorted(by_mid.items()):
        ordered = sorted(mid_pages, key=lambda page: int(page["pdf_page_number"]))
        positions = [int(page["pdf_page_number"]) for page in ordered]
        if positions != list(range(1, len(positions) + 1)):
            raise ValueError(f"MID {mid} has invalid positions: {positions}")
        sequences.append(ordered)
    return sequences


def collate_sequences(batch):
    batch_size = len(batch)
    max_length = max(len(item["rows"]) for item in batch)
    dim = batch[0]["features"].shape[1]
    features = torch.zeros((batch_size, max_length, dim))
    class_targets = torch.full((batch_size, max_length), IGNORE_INDEX, dtype=torch.long)
    segmentation_targets = torch.full((batch_size, max_length), IGNORE_INDEX, dtype=torch.long)
    padding_mask = torch.ones((batch_size, max_length), dtype=torch.bool)
    rows = []
    for index, item in enumerate(batch):
        length = len(item["rows"])
        features[index, :length] = item["features"]
        class_targets[index, :length] = item["class_targets"]
        segmentation_targets[index, :length] = item["segmentation_targets"]
        segmentation_targets[index, 0] = IGNORE_INDEX
        padding_mask[index, :length] = False
        rows.append(item["rows"])
    return {
        "features": features,
        "class_targets": class_targets,
        "segmentation_targets": segmentation_targets,
        "padding_mask": padding_mask,
        "rows": rows,
    }


def _sequence_sampling(sequences):
    class_counts = Counter(
        page["class_target"] for sequence in sequences for page in sequence if page["class_target"] != IGNORE_INDEX
    )
    weights = []
    for sequence in sequences:
        labelled = [page["class_target"] for page in sequence if page["class_target"] != IGNORE_INDEX]
        weights.append(max([1 / class_counts[target] for target in labelled], default=0.0))
    weight_sum = sum(weights)
    probabilities = [weight / weight_sum for weight in weights]
    effective_pages = Counter()
    draws = len(sequences)
    for sequence, probability in zip(sequences, probabilities):
        for page in sequence:
            if page["class_target"] != IGNORE_INDEX:
                effective_pages[ID_TO_CLASS[page["class_target"]]] += draws * probability
    return weights, {
        "reference_method": "max inverse page-class frequency per sequence",
        "original_supervised_pages": {ID_TO_CLASS[index]: count for index, count in class_counts.items()},
        "sequence_sampling_count": draws,
        "sequence_sample_weights": {sequence[0]["masterindex_id"]: weight for sequence, weight in zip(sequences, weights)},
        "expected_supervised_pages_per_epoch": dict(effective_pages),
        "loss_weights": {label: 1.0 for label in CLASSES},
    }


def make_loader(sequences, features, batch_size, train, balance):
    dataset = MIDSequenceDataset(sequences, features)
    sampler = None
    report = None
    if train and balance:
        weights, report = _sequence_sampling(sequences)
        sampler = WeightedRandomSampler(weights, num_samples=len(sequences), replacement=True)
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler, shuffle=train and sampler is None, collate_fn=collate_sequences), report


def run_epoch(model, loader, optimizer, device):
    train = optimizer is not None
    model.train(train)
    criterion = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
    total_loss = 0.0; class_true = []; class_pred = []; seg_true = []; seg_pred = []
    for batch in loader:
        features = batch["features"].to(device); class_targets = batch["class_targets"].to(device); seg_targets = batch["segmentation_targets"].to(device)
        pred_seg, pred_class = model(features)
        class_loss = criterion(pred_class.flatten(0, 1), class_targets.flatten())
        if (seg_targets != IGNORE_INDEX).any():
            seg_loss = criterion(pred_seg.flatten(0, 1), seg_targets.flatten())
        else:
            seg_loss = pred_seg.sum() * 0
        loss = class_loss + seg_loss
        if train:
            optimizer.zero_grad(); loss.backward(); optimizer.step()
        total_loss += loss.item()
        class_mask = class_targets != IGNORE_INDEX; seg_mask = seg_targets != IGNORE_INDEX
        class_true.extend(class_targets[class_mask].cpu().tolist()); class_pred.extend(pred_class.argmax(-1)[class_mask].cpu().tolist())
        seg_true.extend(seg_targets[seg_mask].cpu().tolist()); seg_pred.extend(pred_seg.argmax(-1)[seg_mask].cpu().tolist())
    result = classification_metrics(class_true, class_pred)
    result["segmentation"] = binary_metrics(seg_true, seg_pred)
    result["loss"] = total_loss / max(len(loader), 1)
    return result


def _split_sequences(pages, assignments, split):
    return build_mid_sequences([page for page in pages if assignments[page["stack_id"]] == split])


def checkpoint_metadata(cache, feature_cache, stage1_checkpoint, input_dim):
    return {
        "architecture": "PnC_SequenceModelRNN_LifeMIDPort_v2",
        "classes": list(CLASSES),
        "class_to_id": {label: index for index, label in ID_TO_CLASS.items()},
        "modality": cache["metadata"]["modality"],
        "input_dim": input_dim,
        "hidden_dim": 1024,
        "stage1_checkpoint": str(Path(stage1_checkpoint).resolve()),
        "stage1_checkpoint_sha256": file_sha256(stage1_checkpoint),
        "feature_cache": str(Path(feature_cache).resolve()),
        "feature_cache_sha256": file_sha256(feature_cache),
        "feature_cache_metadata": cache["metadata"],
    }


def save_checkpoint_new(path, model, optimizer, epoch, best, metadata):
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite checkpoint: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch, "best_macro_f1": best, "metadata": metadata}, path)


def load_stage2_checkpoint(path, feature_cache, stage1_checkpoint, device):
    payload = torch.load(path, map_location=device)
    metadata = payload.get("metadata", {})
    if metadata.get("architecture") != "PnC_SequenceModelRNN_LifeMIDPort_v2":
        raise ValueError("Incompatible Stage 2 checkpoint")
    if metadata.get("stage1_checkpoint_sha256") != file_sha256(stage1_checkpoint):
        raise ValueError("Stage 1 checkpoint mismatch")
    if metadata.get("feature_cache_sha256") != file_sha256(feature_cache):
        raise ValueError("Stage 1 feature cache mismatch")
    model = SequenceModelRNN(False, len(CLASSES), input_dim=metadata["input_dim"]).to(device)
    model.load_state_dict(payload["model"])
    return model, payload


def train(args):
    config = PipelineConfig(batch_id=args.batch_id)
    cache, pages, features = load_feature_cache(config, args.feature_cache, args.stage1_checkpoint)
    assignments = split_assignments(config, load_model_pages(config), args.seed)
    train_sequences = _split_sequences(pages, assignments, "train")
    val_sequences = _split_sequences(pages, assignments, "val")
    if args.limit_mids:
        train_sequences = train_sequences[:args.limit_mids]
        val_sequences = val_sequences[:max(1, args.limit_mids // 4)]
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = SequenceModelRNN(False, len(CLASSES), lr=args.lr, input_dim=features.shape[1]).to(device)
    optimizer = model.get_optimizer()
    metadata = checkpoint_metadata(cache, args.feature_cache, args.stage1_checkpoint, features.shape[1])
    start = 0; best = -1.0
    if args.resume:
        model, payload = load_stage2_checkpoint(args.resume, args.feature_cache, args.stage1_checkpoint, device)
        optimizer = model.get_optimizer(); optimizer.load_state_dict(payload["optimizer"])
        start = payload["epoch"] + 1; best = payload["best_macro_f1"]
    train_loader, balance_report = make_loader(train_sequences, features, args.batch_size, True, args.balance_classes)
    val_loader, _ = make_loader(val_sequences, features, args.batch_size, False, False)
    if balance_report is None:
        _, balance_report = _sequence_sampling(train_sequences)
    mids_by_class = {label: set() for label in CLASSES}
    for sequence in train_sequences:
        for page in sequence:
            if page["class_target"] != IGNORE_INDEX:
                mids_by_class[ID_TO_CLASS[page["class_target"]]].add(page["masterindex_id"])
    balance_report["mids_containing_class"] = {label: len(mids) for label, mids in mids_by_class.items()}
    low = {label: {"pages": balance_report["original_supervised_pages"].get(label, 0), "mids": balance_report["mids_containing_class"].get(label, 0)}
           for label in CLASSES if balance_report["original_supervised_pages"].get(label, 0) < 500 or balance_report["mids_containing_class"].get(label, 0) < 50}
    print(json.dumps({"balance_report": balance_report, "low_support_classes": low}, indent=2))
    run = args.run_name or datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    for epoch in range(start, args.epochs):
        train_metrics = run_epoch(model, train_loader, optimizer, device)
        val_metrics = run_epoch(model, val_loader, None, device)
        result_path = config.results_dir / f"pnc_stage2_{run}_{cache['metadata']['modality']}_epoch_{epoch:03d}.json"
        write_json_new(result_path, {"epoch": epoch, "train": train_metrics, "val": val_metrics, "balance_report": balance_report, "low_support_classes": low})
        epoch_path = config.checkpoints_dir / f"pnc_stage2_{run}_{cache['metadata']['modality']}_epoch_{epoch:03d}.pt"
        save_checkpoint_new(epoch_path, model, optimizer, epoch, max(best, val_metrics["macro_f1"]), metadata)
        if val_metrics["macro_f1"] > best:
            best = val_metrics["macro_f1"]
            save_checkpoint_new(config.checkpoints_dir / f"pnc_stage2_{run}_{cache['metadata']['modality']}_best_epoch_{epoch:03d}.pt", model, optimizer, epoch, best, metadata)
        print(json.dumps({"result": str(result_path), "checkpoint": str(epoch_path), "val_macro_f1": val_metrics["macro_f1"]}))


def evaluate(args):
    config = PipelineConfig(batch_id=args.batch_id)
    cache, pages, features = load_feature_cache(config, args.feature_cache, args.stage1_checkpoint)
    assignments = split_assignments(config, load_model_pages(config), args.seed)
    sequences = _split_sequences(pages, assignments, args.split)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = load_stage2_checkpoint(args.checkpoint, args.feature_cache, args.stage1_checkpoint, device)
    loader, _ = make_loader(sequences, features, args.batch_size, False, False)
    write_json_new(Path(args.output), run_epoch(model, loader, None, device))
    print(args.output)


def predict(args):
    config = PipelineConfig(batch_id=args.batch_id)
    cache, pages, features = load_feature_cache(config, args.feature_cache, args.stage1_checkpoint)
    sequences = build_mid_sequences(pages)
    if args.limit_mids:
        sequences = sequences[:args.limit_mids]
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = load_stage2_checkpoint(args.checkpoint, args.feature_cache, args.stage1_checkpoint, device)
    model.eval()
    loader, _ = make_loader(sequences, features, args.batch_size, False, False)
    output = []
    with torch.no_grad():
        for batch in loader:
            pred_seg, pred_class = model(batch["features"].to(device))
            seg_probs = torch.softmax(pred_seg, -1).cpu(); class_probs = torch.softmax(pred_class, -1).cpu()
            for sequence_index, sequence in enumerate(batch["rows"]):
                for page_index, page in enumerate(sequence):
                    cp = class_probs[sequence_index, page_index]
                    sp = seg_probs[sequence_index, page_index]
                    pred = int(cp.argmax())
                    row = {key: page.get(key, "") for key in ["page_identity", "masterindex_id", "stack_id", "pdf_page_number", "has_label", "page_class", "segmentation_target", "placement"]}
                    row.update({"predicted_class": ID_TO_CLASS[pred], "document_start_probability": float(sp[1])})
                    for index, label in ID_TO_CLASS.items():
                        row[f"prob_{label}"] = float(cp[index])
                    output.append(row)
    write_csv_new(Path(args.output), output, list(output[0]))
    print(args.output)


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    def common(command):
        command.add_argument("--batch-id", default=PipelineConfig.batch_id)
        command.add_argument("--feature-cache", required=True)
        command.add_argument("--stage1-checkpoint", required=True)
        command.add_argument("--batch-size", type=int, default=4)
        command.add_argument("--device")
        command.add_argument("--seed", type=int, default=13)
    train_parser = sub.add_parser("train"); common(train_parser)
    train_parser.add_argument("--epochs", type=int, default=5); train_parser.add_argument("--lr", type=float, default=1e-4)
    train_parser.add_argument("--balance-classes", action="store_true"); train_parser.add_argument("--limit-mids", type=int); train_parser.add_argument("--resume"); train_parser.add_argument("--run-name")
    eval_parser = sub.add_parser("evaluate"); common(eval_parser)
    eval_parser.add_argument("--checkpoint", required=True); eval_parser.add_argument("--split", choices=["train", "val", "test"], default="test"); eval_parser.add_argument("--output", required=True)
    pred_parser = sub.add_parser("predict"); common(pred_parser)
    pred_parser.add_argument("--checkpoint", required=True); pred_parser.add_argument("--limit-mids", type=int); pred_parser.add_argument("--output", required=True)
    args = parser.parse_args()
    {"train": train, "evaluate": evaluate, "predict": predict}[args.command](args)


if __name__ == "__main__":
    main()