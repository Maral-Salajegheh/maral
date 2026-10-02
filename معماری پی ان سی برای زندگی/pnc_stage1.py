"""Corrected PnC Stage 1 training, evaluation, inference, and embedding extraction."""

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from .config import CLASSES, ID_TO_CLASS, TEXT_MODEL_NAME, PipelineConfig
from .models import PageClassifier
from .ocr import load_ocr_cache
from .pnc_data import load_model_pages, population_hash, select_complete_mids, split_assignments
from .pnc_metrics import binary_metrics, classification_metrics, write_csv_new, write_json_new


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_clip(config):
    if not config.pnc_clip_feature_file.exists():
        raise FileNotFoundError(f"Missing CLIP cache: {config.pnc_clip_feature_file}")
    return torch.load(config.pnc_clip_feature_file, map_location="cpu")


class Stage1Dataset(Dataset):
    def __init__(self, rows, model, modality, ocr_cache, clip_features):
        self.rows = rows
        self.model = model
        self.modality = modality
        self.ocr_cache = ocr_cache
        self.clip_features = clip_features

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        item = {"row": row}
        if self.modality in {"text", "fusion"}:
            if row["image_path"] not in self.ocr_cache:
                raise KeyError(f"Missing OCR text: {row['image_path']}")
            item["text"] = self.ocr_cache[row["image_path"]]
        if self.modality in {"vision", "fusion"}:
            if row["image_path"] not in self.clip_features:
                raise KeyError(f"Missing CLIP feature: {row['image_path']}")
            item["visual"] = self.clip_features[row["image_path"]].float()
        return item


def collate_stage1(batch, model, modality):
    output = {"rows": [item["row"] for item in batch]}
    if modality in {"text", "fusion"}:
        output["text"] = model.tokenizer(
            [item["text"] for item in batch],
            return_tensors="pt",
            padding="max_length",
            pad_to_multiple_of=8,
            truncation="longest_first",
        )
    if modality in {"vision", "fusion"}:
        output["visual"] = torch.stack([item["visual"] for item in batch])
    return output


def _rows_for_split(all_rows, assignments, split, limit_mids):
    split_rows = [row for row in all_rows if assignments[row["stack_id"]] == split]
    return select_complete_mids(split_rows, limit_mids)


def _labelled(rows):
    return [row for row in rows if row["has_label"].lower() == "true"]


def _sampling_report(rows, sampler_weights=None, draws=None):
    class_counts = Counter(row["page_class"] for row in rows)
    mids_by_class = {label: set() for label in CLASSES}
    for row in rows:
        mids_by_class[row["page_class"]].add(row["masterindex_id"])
    report = {
        "supervised_pages_per_class": dict(class_counts),
        "mids_containing_class": {label: len(mids) for label, mids in mids_by_class.items()},
        "loss_weights": {label: 1.0 for label in CLASSES},
        "effective_epoch_draws": draws or len(rows),
    }
    if sampler_weights is not None:
        totals = Counter()
        weight_sum = sum(sampler_weights)
        for row, weight in zip(rows, sampler_weights):
            totals[row["page_class"]] += (draws or len(rows)) * weight / weight_sum
        report["expected_sampled_pages_per_class"] = dict(totals)
        report["sample_weight_per_class"] = {
            label: (1 / class_counts[label] if class_counts[label] else 0.0) for label in CLASSES
        }
    return report


def make_loader(rows, model, modality, config, batch_size, train, balance):
    labelled = _labelled(rows)
    ocr = load_ocr_cache(config.pnc_ocr_cache) if modality in {"text", "fusion"} else {}
    clip = load_clip(config) if modality in {"vision", "fusion"} else {}
    dataset = Stage1Dataset(labelled, model, modality, ocr, clip)
    sampler = None
    weights = None
    if train and balance:
        counts = Counter(row["page_class"] for row in labelled)
        weights = [1 / counts[row["page_class"]] for row in labelled]
        sampler = WeightedRandomSampler(weights, num_samples=len(labelled), replacement=True)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=train and sampler is None,
        collate_fn=lambda batch: collate_stage1(batch, model, modality),
    )
    return loader, _sampling_report(labelled, weights, len(labelled))


def _autocast(device, use_fp16, use_bf16):
    enabled = use_fp16 or use_bf16
    device_type = torch.device(device).type
    if device_type == "cpu" and use_fp16:
        raise ValueError("fp16 requires CUDA")
    return torch.autocast(
        device_type=device_type,
        enabled=enabled,
        dtype=torch.bfloat16 if use_bf16 else torch.float16,
    )


def run_epoch(model, loader, optimizer, device, use_fp16=False, use_bf16=False):
    train = optimizer is not None
    model.train(train)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=train and use_fp16)
    total_loss = 0.0
    class_true = []; class_pred = []
    first_true = []; first_pred = []
    last_true = []; last_pred = []
    qualities = []
    for batch in loader:
        rows = batch["rows"]
        class_target = torch.tensor([int(row["class_target"]) for row in rows], device=device)
        first_target = torch.tensor([int(row["segmentation_target"]) for row in rows], device=device)
        last_target = torch.tensor([int(row["last_page_target"]) for row in rows], device=device)
        visual = batch.get("visual")
        if visual is not None:
            visual = visual.to(device)
        text = batch.get("text")
        if text is not None:
            text = {key: value.to(device) for key, value in text.items()}
        if train:
            optimizer.zero_grad()
        with torch.set_grad_enabled(train), _autocast(device, use_fp16, use_bf16):
            pred_first, pred_class, pred_last = model(visual, text)
            loss = criterion(pred_first, first_target) + criterion(pred_class, class_target) + criterion(pred_last, last_target)
        if train:
            if use_fp16:
                scaler.scale(loss).backward(); scaler.unscale_(optimizer)
            else:
                loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 20.0)
            if use_fp16:
                scaler.step(optimizer); scaler.update()
            else:
                optimizer.step()
        total_loss += loss.item() * len(rows)
        class_true.extend(class_target.cpu().tolist()); class_pred.extend(pred_class.argmax(-1).cpu().tolist())
        first_true.extend(first_target.cpu().tolist()); first_pred.extend(pred_first.argmax(-1).cpu().tolist())
        last_true.extend(last_target.cpu().tolist()); last_pred.extend(pred_last.argmax(-1).cpu().tolist())
        qualities.extend([row["training_label_quality"] for row in rows])
    result = classification_metrics(class_true, class_pred)
    result["loss"] = total_loss / max(len(class_true), 1)
    result["first_page"] = binary_metrics(first_true, first_pred)
    result["last_page"] = binary_metrics(last_true, last_pred)
    result["by_training_label_quality"] = {}
    for quality in sorted(set(qualities)):
        indices = [index for index, value in enumerate(qualities) if value == quality]
        result["by_training_label_quality"][quality] = classification_metrics(
            [class_true[index] for index in indices], [class_pred[index] for index in indices]
        )
    return result


def checkpoint_metadata(config, model, modality, rows):
    return {
        "architecture": "PnC_BertPageModel_LifePort_v2",
        "modality": modality,
        "classes": list(CLASSES),
        "class_to_id": {label: index for index, label in ID_TO_CLASS.items()},
        "embedding_dim": model.embedding_dim,
        "clip_input_dim": 1024,
        "population_hash": population_hash(rows),
        "population_pages": len(rows),
        "text_model_name": TEXT_MODEL_NAME,
    }


def save_checkpoint_new(path, model, optimizer, epoch, best_macro_f1, metadata):
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite checkpoint: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch,
        "best_macro_f1": best_macro_f1, "metadata": metadata,
    }, path)


def load_stage1_checkpoint(path, device):
    payload = torch.load(path, map_location=device)
    metadata = payload.get("metadata")
    if not metadata or metadata.get("architecture") != "PnC_BertPageModel_LifePort_v2":
        raise ValueError(f"Incompatible Stage 1 checkpoint: {path}")
    model = PageClassifier(
        TEXT_MODEL_NAME,
        len(CLASSES),
        modality=metadata["modality"],
        clip_input_dim=metadata["clip_input_dim"],
    ).to(device)
    model.load_state_dict(payload["model"])
    return model, payload


def train(args):
    if args.use_fp16 and args.use_bf16:
        raise ValueError("Choose at most one of --use-fp16 and --use-bf16")
    config = PipelineConfig(batch_id=args.batch_id)
    rows = load_model_pages(config)
    assignments = split_assignments(config, rows, args.seed)
    train_rows = _rows_for_split(rows, assignments, "train", args.limit_mids)
    val_limit = max(1, args.limit_mids // 4) if args.limit_mids else None
    val_rows = _rows_for_split(rows, assignments, "val", val_limit)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = PageClassifier(TEXT_MODEL_NAME, len(CLASSES), args.modality, lr=args.lr).to(device)
    optimizer = model.get_optimizer()
    start_epoch = 0; best = -1.0
    metadata = checkpoint_metadata(config, model, args.modality, rows)
    if args.resume:
        resumed, payload = load_stage1_checkpoint(args.resume, device)
        if payload["metadata"] != metadata:
            raise ValueError("Resume checkpoint metadata is incompatible with current run")
        model = resumed; optimizer = model.get_optimizer(); optimizer.load_state_dict(payload["optimizer"])
        start_epoch = payload["epoch"] + 1; best = payload["best_macro_f1"]
    train_loader, balance_report = make_loader(train_rows, model, args.modality, config, args.batch_size, True, args.balance_classes)
    val_loader, _ = make_loader(val_rows, model, args.modality, config, args.batch_size, False, False)
    low_support = {
        label: {"pages": balance_report["supervised_pages_per_class"].get(label, 0), "mids": balance_report["mids_containing_class"].get(label, 0)}
        for label in CLASSES
        if balance_report["supervised_pages_per_class"].get(label, 0) < 500 or balance_report["mids_containing_class"].get(label, 0) < 50
    }
    print(json.dumps({"balance_report": balance_report, "low_support_classes": low_support}, indent=2))
    run_name = args.run_name or datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    for epoch in range(start_epoch, args.epochs):
        train_metrics = run_epoch(model, train_loader, optimizer, device, args.use_fp16, args.use_bf16)
        val_metrics = run_epoch(model, val_loader, None, device, args.use_fp16, args.use_bf16)
        result = {"epoch": epoch, "train": train_metrics, "val": val_metrics, "balance_report": balance_report, "low_support_classes": low_support}
        result_path = config.results_dir / f"pnc_stage1_{run_name}_{args.modality}_epoch_{epoch:03d}.json"
        write_json_new(result_path, result)
        epoch_path = config.checkpoints_dir / f"pnc_stage1_{run_name}_{args.modality}_epoch_{epoch:03d}.pt"
        save_checkpoint_new(epoch_path, model, optimizer, epoch, max(best, val_metrics["macro_f1"]), metadata)
        if val_metrics["macro_f1"] > best:
            best = val_metrics["macro_f1"]
            best_path = config.checkpoints_dir / f"pnc_stage1_{run_name}_{args.modality}_best_epoch_{epoch:03d}.pt"
            save_checkpoint_new(best_path, model, optimizer, epoch, best, metadata)
        print(json.dumps({"result": str(result_path), "checkpoint": str(epoch_path), "val_macro_f1": val_metrics["macro_f1"]}))


def evaluate(args):
    config = PipelineConfig(batch_id=args.batch_id)
    rows = load_model_pages(config); assignments = split_assignments(config, rows, args.seed)
    selected = _rows_for_split(rows, assignments, args.split, args.limit_mids)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, payload = load_stage1_checkpoint(args.checkpoint, device)
    loader, _ = make_loader(selected, model, payload["metadata"]["modality"], config, args.batch_size, False, False)
    result = run_epoch(model, loader, None, device, args.use_fp16, args.use_bf16)
    output = Path(args.output)
    write_json_new(output, result)
    print(output)


def predict(args):
    config = PipelineConfig(batch_id=args.batch_id)
    rows = load_model_pages(config)
    if args.limit_mids:
        rows = select_complete_mids(rows, args.limit_mids)
    labelled = _labelled(rows)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, payload = load_stage1_checkpoint(args.checkpoint, device)
    loader, _ = make_loader(labelled, model, payload["metadata"]["modality"], config, args.batch_size, False, False)
    output_rows = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            visual = batch.get("visual"); text = batch.get("text")
            if visual is not None: visual = visual.to(device)
            if text is not None: text = {key: value.to(device) for key, value in text.items()}
            first, classes, last = model(visual, text)
            first_probs = torch.softmax(first, -1).cpu(); class_probs = torch.softmax(classes, -1).cpu(); last_probs = torch.softmax(last, -1).cpu()
            for row, fp, cp, lp in zip(batch["rows"], first_probs, class_probs, last_probs):
                pred = int(cp.argmax())
                record = {key: row.get(key, "") for key in ["page_identity", "masterindex_id", "stack_id", "pdf_page_number", "image_path", "page_class", "is_first_page", "is_last_page"]}
                record.update({"predicted_class": ID_TO_CLASS[pred], "first_page_probability": float(fp[1]), "last_page_probability": float(lp[1])})
                for index, label in ID_TO_CLASS.items(): record[f"prob_{label}"] = float(cp[index])
                output_rows.append(record)
    write_csv_new(Path(args.output), output_rows, list(output_rows[0]))


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    def common(command):
        command.add_argument("--batch-id", default=PipelineConfig.batch_id)
        command.add_argument("--batch-size", type=int, default=8)
        command.add_argument("--limit-mids", type=int)
        command.add_argument("--device")
        command.add_argument("--seed", type=int, default=13)
        command.add_argument("--use-fp16", action="store_true")
        command.add_argument("--use-bf16", action="store_true")
    train_parser = sub.add_parser("train"); common(train_parser)
    train_parser.add_argument("--modality", choices=["vision", "text", "fusion"], required=True)
    train_parser.add_argument("--epochs", type=int, default=5); train_parser.add_argument("--lr", type=float, default=1e-5)
    train_parser.add_argument("--balance-classes", action="store_true"); train_parser.add_argument("--resume"); train_parser.add_argument("--run-name")
    eval_parser = sub.add_parser("evaluate"); common(eval_parser)
    eval_parser.add_argument("--checkpoint", required=True); eval_parser.add_argument("--split", choices=["train", "val", "test"], default="test"); eval_parser.add_argument("--output", required=True)
    pred_parser = sub.add_parser("predict"); common(pred_parser)
    pred_parser.add_argument("--checkpoint", required=True); pred_parser.add_argument("--output", required=True)
    args = parser.parse_args()
    {"train": train, "evaluate": evaluate, "predict": predict}[args.command](args)


if __name__ == "__main__":
    main()