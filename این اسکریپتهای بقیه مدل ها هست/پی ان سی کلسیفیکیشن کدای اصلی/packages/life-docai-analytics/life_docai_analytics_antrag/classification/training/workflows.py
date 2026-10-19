"""Workflows (as in PnC workflows.py): page caches, Stage 1, Stage 1 -> Stage 2 features, Stage 2.

Every function takes the parsed CLI arguments from main.py.
"""

import hashlib
import json
import random
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Sampler, WeightedRandomSampler

from .config import CLASSES, CLIP_MODEL_NAME, CLIP_MODEL_PATH, ID_TO_CLASS, TEXT_MODEL_NAME, PipelineConfig
from .data import (IGNORE_INDEX, build_stack_sequences, load_model_pages, population_hash, rows_for_split,
                   select_complete_stacks, split_assignments, training_subset)
from .evaluation import binary_metrics, classification_metrics, write_csv_new, write_json_new
from .models import PageClassifier, SequenceModelRNN
from .ocr import build_ocr_cache, load_ocr_cache
from .storage import ClipFeatureStore, Stage1Features

STAGE1_ARCHITECTURE = "BertPageModel"
STAGE2_ARCHITECTURE = "SequenceModelRNN"
POPULATION = "stack"


def _config(args):
    return PipelineConfig(batch_id=args.batch_id)


def _device(args):
    return args.device or ("cuda" if torch.cuda.is_available() else "cpu")


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _check_new(*paths):
    for path in paths:
        if Path(path).exists():
            raise FileExistsError(f"Output already exists: {path}; use another --run-name or --output")


# ===========================================================================
# Page caches (OCR text, CLIP vectors), keyed by image_path
# ===========================================================================

def compute_ocr(args):
    config = _config(args)
    rows = training_subset(config, load_model_pages(config), args.limit_stacks, args.training_limit_stacks, args.seed)
    build_ocr_cache(config.ocr_text_file, rows, config.render_root, workers=args.workers)


def compute_visual_embeddings(args):
    """As PnC compute_visual_embeddings: frozen OpenCLIP ViT-H-14 visual tower, cached once.

    Vectors go into write-once shards, so adding pages never rewrites what is already stored.
    """
    import open_clip
    from PIL import Image
    from safetensors.torch import load_file
    from tqdm import tqdm

    config = _config(args)
    rows = training_subset(config, load_model_pages(config), args.limit_stacks, args.training_limit_stacks, args.seed)
    store = ClipFeatureStore(config.clip_features_dir)
    pending = [row for row in rows if row["image_path"] not in store]
    print(f"clip_cached={len(store)} pending={len(pending)}")
    if not pending:
        return
    model, _, preprocess = open_clip.create_model_and_transforms(CLIP_MODEL_NAME)
    model.load_state_dict(load_file(CLIP_MODEL_PATH))
    visual = model.visual.eval().requires_grad_(False).to(_device(args))

    class Pages(Dataset):
        def __len__(self):
            return len(pending)

        def __getitem__(self, index):
            with Image.open(config.render_root / pending[index]["image_path"]) as image:
                return index, preprocess(image.convert("RGB"))

    loader = DataLoader(Pages(), batch_size=args.batch_size, shuffle=False, num_workers=2)
    with torch.no_grad():
        for indices, images in tqdm(loader, desc="clip"):
            for index, vector in zip(indices.tolist(), visual(images.to(_device(args))).cpu()):
                store.add(pending[index]["image_path"], vector.numpy())
    store.flush()
    print(f"clip_features={config.clip_features_dir} rows={len(store)}")


def migrate_clip_cache(args):
    """One-time conversion of an old single-file CLIP cache ({image_path: tensor}) into shards."""
    config = _config(args)
    store = ClipFeatureStore(config.clip_features_dir)
    old = torch.load(args.source, map_location="cpu")
    added = 0
    for image_path, vector in old.items():
        if image_path not in store:
            store.add(image_path, vector.float().numpy()); added += 1
    store.flush()
    print(f"migrated={added} already_present={len(old) - added} total={len(store)} -> {config.clip_features_dir}")


def load_inputs(config, modality):
    ocr = load_ocr_cache(config.ocr_text_file) if modality in {"text", "fusion"} else {}
    clip = {}
    if modality in {"vision", "fusion"}:
        clip = ClipFeatureStore(config.clip_features_dir)   # memory-mapped, read page by page
        if not len(clip):
            raise FileNotFoundError(f"Missing CLIP features: {config.clip_features_dir}")
    return ocr, clip


# ===========================================================================
# Stage 1: page model
# ===========================================================================

class Stage1Dataset(Dataset):
    def __init__(self, rows, modality, ocr_cache, clip_features):
        self.rows, self.modality, self.ocr_cache, self.clip_features = rows, modality, ocr_cache, clip_features

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
        output["text"] = model.tokenizer([item["text"] for item in batch], return_tensors="pt",
                                         padding="max_length", pad_to_multiple_of=8, truncation="longest_first")
    if modality in {"vision", "fusion"}:
        output["visual"] = torch.stack([item["visual"] for item in batch])
    return output


def _stage1_loader(rows, model, modality, batch_size, train, balance, ocr, clip):
    sampler, weights = None, None
    counts = Counter(row["page_class"] for row in rows)
    if train and balance:  # as PnC: draw pages with p ~ 1 / count(class), with replacement
        weights = [1 / counts[row["page_class"]] for row in rows]
        sampler = WeightedRandomSampler(weights, num_samples=len(rows), replacement=True)
    loader = DataLoader(Stage1Dataset(rows, modality, ocr, clip), batch_size=batch_size, sampler=sampler,
                        shuffle=train and sampler is None, collate_fn=lambda batch: collate_stage1(batch, model, modality))
    stacks = {label: {r["stack_id"] for r in rows if r["page_class"] == label} for label in CLASSES}
    report = {"supervised_pages_per_class": dict(counts),
              "stacks_containing_class": {label: len(s) for label, s in stacks.items()}}
    if weights is not None:
        total = sum(weights)
        report["expected_sampled_pages_per_class"] = {
            label: len(rows) * sum(w for r, w in zip(rows, weights) if r["page_class"] == label) / total for label in CLASSES}
    return loader, report


def _autocast(device, use_fp16, use_bf16):
    device_type = torch.device(device).type
    if device_type == "cpu" and use_fp16:
        raise ValueError("fp16 requires CUDA")
    return torch.autocast(device_type=device_type, enabled=use_fp16 or use_bf16,
                          dtype=torch.bfloat16 if use_bf16 else torch.float16)


def stage1_epoch(model, loader, optimizer, device, use_fp16=False, use_bf16=False, scaler=None):
    train = optimizer is not None
    model.train(train)
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    class_true, class_pred, first_true, first_pred, last_true, last_pred, qualities = [], [], [], [], [], [], []
    for batch in loader:
        rows = batch["rows"]
        class_target = torch.tensor([int(r["class_target"]) for r in rows], device=device)
        first_target = torch.tensor([int(r["segmentation_target"]) for r in rows], device=device)
        last_target = torch.tensor([int(r["last_page_target"]) for r in rows], device=device)
        visual = batch.get("visual")
        visual = visual.to(device) if visual is not None else None
        text = batch.get("text")
        text = {key: value.to(device) for key, value in text.items()} if text is not None else None
        if train:
            optimizer.zero_grad()
        with torch.set_grad_enabled(train), _autocast(device, use_fp16, use_bf16):
            pred_first, pred_class, pred_last = model(visual, text)
            loss = criterion(pred_first, first_target) + criterion(pred_class, class_target) + criterion(pred_last, last_target)
        if train:
            (scaler.scale(loss) if scaler else loss).backward()
            if scaler:
                scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 20.0)
            if scaler:
                scaler.step(optimizer); scaler.update()
            else:
                optimizer.step()
        total_loss += loss.item() * len(rows)
        # As PnC evaluation: the first page of a stack is always a document start.
        predicted_first = [1 if r["stack_position"] == "1" else p for r, p in zip(rows, pred_first.argmax(-1).cpu().tolist())]
        class_true += class_target.cpu().tolist(); class_pred += pred_class.argmax(-1).cpu().tolist()
        first_true += first_target.cpu().tolist(); first_pred += predicted_first
        last_true += last_target.cpu().tolist(); last_pred += pred_last.argmax(-1).cpu().tolist()
        qualities += [r["training_label_quality"] for r in rows]
    result = classification_metrics(class_true, class_pred)
    result["loss"] = total_loss / max(len(class_true), 1)
    result["first_page"] = binary_metrics(first_true, first_pred)
    result["last_page"] = binary_metrics(last_true, last_pred)
    result["by_training_label_quality"] = {
        q: classification_metrics([t for t, v in zip(class_true, qualities) if v == q],
                                  [p for p, v in zip(class_pred, qualities) if v == q]) for q in sorted(set(qualities))}
    return result


def _stage1_metadata(model, modality, rows):
    return {"architecture": STAGE1_ARCHITECTURE, "population": POPULATION, "modality": modality,
            "classes": list(CLASSES), "class_to_id": {label: index for index, label in ID_TO_CLASS.items()},
            "embedding_dim": model.embedding_dim, "clip_input_dim": 1024,
            "population_hash": population_hash(rows), "population_pages": len(rows), "text_model_name": TEXT_MODEL_NAME}


def _save_checkpoint(path, model, optimizer, epoch, best, metadata):
    _check_new(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch,
                "best_macro_f1": best, "metadata": metadata}, path)


def load_stage1_checkpoint(path, device):
    payload = torch.load(path, map_location=device)
    metadata = payload.get("metadata") or {}
    if metadata.get("architecture") != STAGE1_ARCHITECTURE or metadata.get("population") != POPULATION:
        raise ValueError(f"Incompatible Stage 1 checkpoint (needs a stack-level checkpoint): {path}")
    model = PageClassifier(TEXT_MODEL_NAME, len(CLASSES), modality=metadata["modality"],
                           clip_input_dim=metadata["clip_input_dim"]).to(device)
    model.load_state_dict(payload["model"])
    return model, payload


def train_stage1(args):
    if args.use_fp16 and args.use_bf16:
        raise ValueError("Choose at most one of --use-fp16 and --use-bf16")
    config, device = _config(args), _device(args)
    rows = load_model_pages(config)
    assignments = split_assignments(config, rows, args.seed)
    train_rows = rows_for_split(rows, assignments, "train", args.limit_stacks)
    val_rows = rows_for_split(rows, assignments, "val", max(1, args.limit_stacks // 4) if args.limit_stacks else None)
    model = PageClassifier(TEXT_MODEL_NAME, len(CLASSES), args.modality, lr=args.lr).to(device)
    optimizer = model.get_optimizer()
    metadata = _stage1_metadata(model, args.modality, rows)
    start, best = 0, -1.0
    if args.resume:
        model, payload = load_stage1_checkpoint(args.resume, device)
        if payload["metadata"] != metadata:
            raise ValueError("Resume checkpoint metadata is incompatible with current run")
        optimizer = model.get_optimizer(); optimizer.load_state_dict(payload["optimizer"])
        start, best = payload["epoch"] + 1, payload["best_macro_f1"]
    scaler = torch.amp.GradScaler("cuda") if args.use_fp16 else None  # one scaler per run, as PnC
    ocr, clip = load_inputs(config, args.modality)
    train_loader, report = _stage1_loader(train_rows, model, args.modality, args.batch_size, True, args.balance_classes, ocr, clip)
    val_loader, _ = _stage1_loader(val_rows, model, args.modality, args.batch_size, False, False, ocr, clip)
    print(json.dumps(report, indent=2))
    run = args.run_name or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    for epoch in range(start, args.epochs):
        if args.lr_decay_epoch is not None and epoch == args.lr_decay_epoch:
            for group in optimizer.param_groups:  # PnC: lr -> 2e-6 after 40,000 iterations
                group["lr"] = args.lr_decay_to
            print(f"lr set to {args.lr_decay_to} from epoch {epoch}")
        result_path = config.results_dir / f"metrics_stage1_{args.modality}_{run}_epoch_{epoch:03d}.json"
        epoch_path = config.checkpoints_dir / f"model_stage1_{args.modality}_{run}_epoch_{epoch:03d}.pt"
        _check_new(result_path, epoch_path)
        train_metrics = stage1_epoch(model, train_loader, optimizer, device, args.use_fp16, args.use_bf16, scaler)
        val_metrics = stage1_epoch(model, val_loader, None, device, args.use_fp16, args.use_bf16)
        write_json_new(result_path, {"epoch": epoch, "train": train_metrics, "val": val_metrics,
                                     "balance_classes": args.balance_classes, "sampling": report})
        _save_checkpoint(epoch_path, model, optimizer, epoch, max(best, val_metrics["macro_f1"]), metadata)
        if val_metrics["macro_f1"] > best:
            best = val_metrics["macro_f1"]
            _save_checkpoint(config.checkpoints_dir / f"model_stage1_{args.modality}_{run}_best_epoch_{epoch:03d}.pt",
                             model, optimizer, epoch, best, metadata)
        print(json.dumps({"epoch": epoch, "val_macro_f1": val_metrics["macro_f1"], "checkpoint": str(epoch_path)}))


def evaluate_stage1(args):
    config, device = _config(args), _device(args)
    rows = load_model_pages(config)
    selected = rows_for_split(rows, split_assignments(config, rows, args.seed), args.split, args.limit_stacks)
    if not selected:
        raise ValueError(f"No pages in split {args.split}")
    model, payload = load_stage1_checkpoint(args.checkpoint, device)
    modality = payload["metadata"]["modality"]
    loader, _ = _stage1_loader(selected, model, modality, args.batch_size, False, False, *load_inputs(config, modality))
    write_json_new(Path(args.output), stage1_epoch(model, loader, None, device, args.use_fp16, args.use_bf16))
    print(args.output)


def predict_stage1(args):
    config, device = _config(args), _device(args)
    rows = select_complete_stacks(load_model_pages(config), args.limit_stacks)
    model, payload = load_stage1_checkpoint(args.checkpoint, device)
    modality = payload["metadata"]["modality"]
    loader, _ = _stage1_loader(rows, model, modality, args.batch_size, False, False, *load_inputs(config, modality))
    output = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            visual = batch.get("visual"); text = batch.get("text")
            visual = visual.to(device) if visual is not None else None
            text = {key: value.to(device) for key, value in text.items()} if text is not None else None
            first, classes, last = (torch.softmax(t, -1).cpu() for t in model(visual, text))
            for row, fp, cp, lp in zip(batch["rows"], first, classes, last):
                record = {k: row[k] for k in ["page_identity", "stack_id", "stack_position", "image_id", "masterindex_id",
                                              "pdf_page_number", "page_class", "is_first_page", "is_last_page"]}
                record.update({"predicted_class": ID_TO_CLASS[int(cp.argmax())],
                               "first_page_probability": 1.0 if row["stack_position"] == "1" else float(fp[1]),
                               "last_page_probability": float(lp[1])})
                record.update({f"prob_{label}": float(cp[index]) for index, label in ID_TO_CLASS.items()})
                output.append(record)
    if not output:
        raise ValueError("No pages to predict")
    write_csv_new(Path(args.output), output, list(output[0]))
    print(args.output)


# ===========================================================================
# Stage 1 -> Stage 2: frozen trained page representations (PnC precompute_bert_features)
# ===========================================================================

def default_features_dir(config, checkpoint, modality):
    name = Path(checkpoint).stem.removeprefix("model_stage1_").removeprefix(f"{modality}_")
    return config.stage1_features_dir(name, modality)


def precompute_features(args):
    config, device = _config(args), _device(args)
    rows = load_model_pages(config)
    selected = training_subset(config, rows, args.limit_stacks, args.training_limit_stacks, args.seed)
    model, payload = load_stage1_checkpoint(args.stage1_checkpoint, device)
    if payload["metadata"]["population_hash"] != population_hash(rows):
        raise ValueError("Stage 1 checkpoint population is incompatible with current model population")
    model.requires_grad_(False).eval()
    modality = payload["metadata"]["modality"]
    if modality == "vision":
        print("WARNING: vision Stage 1 returns the raw CLIP vector; Stage 2 gets no Stage 1 training (PnC: debug only)")
    output = Path(args.output) if args.output else default_features_dir(config, args.stage1_checkpoint, modality)
    dim = 2048 if modality == "fusion" else 1024
    writer = Stage1Features.create(output, len(selected), dim)
    ocr, clip = load_inputs(config, modality)
    loader = DataLoader(Stage1Dataset(selected, modality, ocr, clip), batch_size=args.batch_size, shuffle=False,
                        collate_fn=lambda batch: collate_stage1(batch, model, modality))
    pages, row = [], 0
    with torch.no_grad():
        for batch in loader:
            visual = batch.get("visual"); text = batch.get("text")
            visual = visual.to(device) if visual is not None else None
            text = {key: value.to(device) for key, value in text.items()} if text is not None else None
            h = model(visual, text, return_embeddings=True)
            block = (torch.cat((h[:, :1024], visual), dim=1) if modality == "fusion" else h).float().cpu().numpy()  # as PnC
            if block.shape[1] != dim:
                raise ValueError(f"Feature dimension {block.shape[1]}, expected {dim}")
            writer[row:row + len(block)] = block
            for r in batch["rows"]:
                pages.append({"row": row, "page_identity": r["page_identity"], "stack_id": r["stack_id"],
                              "stack_position": r["stack_position"], "image_id": r["image_id"],
                              "masterindex_id": r["masterindex_id"], "pdf_page_number": r["pdf_page_number"],
                              "class_target": r["class_target"], "segmentation_target": r["segmentation_target"],
                              "last_page_target": r["last_page_target"], "page_class": r["page_class"],
                              "training_label_quality": r["training_label_quality"]})
                row += 1
    if row != len(selected):
        raise ValueError(f"Wrote {row} feature rows, expected {len(selected)}")
    Stage1Features.finish(output, writer, pages, {
        "stage1_checkpoint": str(Path(args.stage1_checkpoint).resolve()),
        "stage1_checkpoint_sha256": file_sha256(args.stage1_checkpoint), "modality": modality,
        "feature_dim": dim, "page_count": len(pages), "stack_count": len({p["stack_id"] for p in pages}),
        "full_population_hash": population_hash(rows), "selected_population_hash": population_hash(selected),
        "ordering": "stack_id,stack_position (original page order); each stack is one contiguous block of rows"})
    print(output)


def load_stage1_features(config, directory, stage1_checkpoint):
    """Validate a feature directory; returns (store, sequences). Feature rows stay on disk."""
    store = Stage1Features(directory)
    metadata = store.metadata
    if metadata.get("stage1_checkpoint_sha256") != file_sha256(stage1_checkpoint):
        raise ValueError("Stage 1 checkpoint does not match the Stage 1 features")
    rows = load_model_pages(config)
    if metadata.get("full_population_hash") != population_hash(rows):
        raise ValueError("Stage 1 features were built for another population; rebuild them")
    if metadata.get("page_count") != len(store.pages) or tuple(store.features.shape) != (len(store.pages), metadata.get("feature_dim")):
        raise ValueError("Stage 1 feature files are inconsistent")
    current = {row["page_identity"]: row for row in rows}
    by_stack, seen = defaultdict(list), set()
    for page in store.pages:
        identity = page["page_identity"]
        if identity in seen or identity not in current:
            raise ValueError(f"Duplicate or stale page in Stage 1 features: {identity}")
        seen.add(identity)
        for field in ["stack_id", "stack_position", "image_id", "class_target", "segmentation_target"]:
            if str(current[identity][field]) != str(page[field]):
                raise ValueError(f"Stage 1 features disagree with the page table for {identity}: {field}")
        by_stack[page["stack_id"]].append({**page, "row": int(page["row"]), "stack_position": int(page["stack_position"]),
                                           "class_target": int(page["class_target"]),
                                           "segmentation_target": int(page["segmentation_target"])})
    if metadata.get("selected_population_hash") != population_hash([current[p["page_identity"]] for p in store.pages]):
        raise ValueError("Stage 1 features are stale or reordered")
    sequences = []
    for stack_id, pages in sorted(by_stack.items()):
        pages.sort(key=lambda p: p["stack_position"])
        if [p["stack_position"] for p in pages] != list(range(1, len(pages) + 1)) or \
                [p["row"] for p in pages] != list(range(pages[0]["row"], pages[0]["row"] + len(pages))):
            raise ValueError(f"Stack {stack_id} is not one ordered block in the Stage 1 features")
        sequences.append({"stack_id": stack_id, "start": pages[0]["row"], "end": pages[-1]["row"] + 1, "pages": pages})
    return store, sequences


# ===========================================================================
# Stage 2: sequence model over the full stack
# ===========================================================================

class StackSequenceDataset(Dataset):
    """Reads one stack's feature rows from the memory-mapped Stage 1 features."""

    def __init__(self, sequences, store):
        self.sequences, self.store = sequences, store

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, index):
        sequence = self.sequences[index]
        return {"rows": sequence["pages"],
                "features": self.store.rows(sequence["start"], sequence["end"]),
                "class_targets": torch.tensor([p["class_target"] for p in sequence["pages"]]),
                "segmentation_targets": torch.tensor([p["segmentation_target"] for p in sequence["pages"]])}


def collate_sequences(batch):
    size, length, dim = len(batch), max(len(item["rows"]) for item in batch), batch[0]["features"].shape[1]
    features = torch.zeros((size, length, dim))
    class_targets = torch.full((size, length), IGNORE_INDEX, dtype=torch.long)
    seg_targets = torch.full((size, length), IGNORE_INDEX, dtype=torch.long)
    for index, item in enumerate(batch):
        n = len(item["rows"])
        features[index, :n] = item["features"]
        class_targets[index, :n] = item["class_targets"]
        seg_targets[index, :n] = item["segmentation_targets"]
        seg_targets[index, 0] = IGNORE_INDEX  # as PnC: Y[:, 0] = -100, first page of the stack
    return {"features": features, "class_targets": class_targets, "segmentation_targets": seg_targets,
            "rows": [item["rows"] for item in batch]}


class LengthBatchSampler(Sampler):
    """Batches of stacks with similar length, at most max_pages padded pages per batch.

    With weights, each epoch first draws len(stacks) stacks with replacement (balancing), then groups them.
    Padding sits at the end and the GRU is unidirectional, so grouping does not change any prediction.
    """

    def __init__(self, lengths, max_pages, shuffle, weights=None, seed=13):
        self.lengths, self.max_pages, self.shuffle, self.weights = lengths, max_pages, shuffle, weights
        self.rng = random.Random(seed)

    def _batches(self):
        if self.weights is not None:
            drawn = torch.multinomial(torch.tensor(self.weights, dtype=torch.double), len(self.lengths), replacement=True).tolist()
        else:
            drawn = list(range(len(self.lengths)))
        drawn.sort(key=lambda i: self.lengths[i])
        batches, current, longest = [], [], 0
        for i in drawn:
            if current and max(longest, self.lengths[i]) * (len(current) + 1) > self.max_pages:
                batches.append(current); current, longest = [], 0
            current.append(i); longest = max(longest, self.lengths[i])
        if current:
            batches.append(current)
        if self.shuffle:
            self.rng.shuffle(batches)
        return batches

    def __iter__(self):
        return iter(self._batches())

    def __len__(self):
        return len(self._batches()) if self.weights is None else max(1, sum(self.lengths) // self.max_pages)


def _stage2_loader(sequences, store, max_pages, train, balance):
    weights, report = None, None
    if train and balance:  # as PnC: max inverse page-class frequency per sequence
        counts = Counter(p["class_target"] for s in sequences for p in s["pages"])
        weights = [max(1 / counts[p["class_target"]] for p in s["pages"]) for s in sequences]
        total = sum(weights)
        report = {"expected_supervised_pages_per_epoch": {
            ID_TO_CLASS[c]: len(sequences) * sum(w * sum(p["class_target"] == c for p in s["pages"])
                                                 for s, w in zip(sequences, weights)) / total for c in counts}}
    sampler = LengthBatchSampler([len(s["pages"]) for s in sequences], max_pages, shuffle=train, weights=weights)
    loader = DataLoader(StackSequenceDataset(sequences, store), batch_sampler=sampler, collate_fn=collate_sequences)
    return loader, report


def stage2_epoch(model, loader, optimizer, device):
    train = optimizer is not None
    model.train(train)
    criterion = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
    total_loss = 0.0
    class_true, class_pred, seg_true, seg_pred = [], [], [], []
    batches = 0
    with torch.set_grad_enabled(train):
        for batch in loader:
            batches += 1
            features = batch["features"].to(device)
            class_targets, seg_targets = batch["class_targets"].to(device), batch["segmentation_targets"].to(device)
            pred_seg, pred_class = model(features)
            loss = criterion(pred_class.flatten(0, 1), class_targets.flatten())
            if (seg_targets != IGNORE_INDEX).any():
                loss = loss + criterion(pred_seg.flatten(0, 1), seg_targets.flatten())
            if train:
                optimizer.zero_grad(); loss.backward(); optimizer.step()
            total_loss += loss.item()
            class_mask, seg_mask = class_targets != IGNORE_INDEX, seg_targets != IGNORE_INDEX
            class_true += class_targets[class_mask].cpu().tolist(); class_pred += pred_class.argmax(-1)[class_mask].cpu().tolist()
            seg_true += seg_targets[seg_mask].cpu().tolist(); seg_pred += pred_seg.argmax(-1)[seg_mask].cpu().tolist()
    result = classification_metrics(class_true, class_pred)
    result["segmentation"] = binary_metrics(seg_true, seg_pred)  # stack position 1 excluded
    result["loss"] = total_loss / max(batches, 1)
    return result


def _stage2_metadata(store, args, input_dim):
    return {"architecture": STAGE2_ARCHITECTURE, "classes": list(CLASSES),
            "class_to_id": {label: index for index, label in ID_TO_CLASS.items()},
            "modality": store.metadata["modality"], "input_dim": input_dim, "hidden_dim": 1024,
            "stage1_checkpoint": str(Path(args.stage1_checkpoint).resolve()),
            "stage1_checkpoint_sha256": file_sha256(args.stage1_checkpoint),
            "stage1_features": str(Path(args.stage1_features).resolve()),
            "stage1_features_metadata_sha256": store.metadata_sha256(),
            "stage1_features_metadata": store.metadata}


def load_stage2_checkpoint(args, store, device, path=None):
    payload = torch.load(path or args.checkpoint, map_location=device)
    metadata = payload.get("metadata", {})
    if metadata.get("architecture") != STAGE2_ARCHITECTURE:
        raise ValueError("Incompatible Stage 2 checkpoint")
    if metadata.get("stage1_checkpoint_sha256") != file_sha256(args.stage1_checkpoint):
        raise ValueError("Stage 1 checkpoint mismatch")
    if metadata.get("stage1_features_metadata_sha256") != store.metadata_sha256():
        raise ValueError("Stage 1 features mismatch")
    model = SequenceModelRNN(False, len(CLASSES), input_dim=metadata["input_dim"]).to(device)
    model.load_state_dict(payload["model"])
    return model, payload


def train_stage2(args):
    config, device = _config(args), _device(args)
    store, sequences = load_stage1_features(config, args.stage1_features, args.stage1_checkpoint)
    assignments = split_assignments(config, load_model_pages(config), args.seed)
    train_seq = [s for s in sequences if assignments[s["stack_id"]] == "train"]
    val_seq = [s for s in sequences if assignments[s["stack_id"]] == "val"]
    if args.limit_stacks:
        train_seq, val_seq = train_seq[:args.limit_stacks], val_seq[:max(1, args.limit_stacks // 4)]
    if not train_seq or not val_seq:
        raise ValueError("The Stage 1 features hold no train or no val stacks")
    dim = store.metadata["feature_dim"]
    model = SequenceModelRNN(False, len(CLASSES), lr=args.lr, input_dim=dim).to(device)
    optimizer = model.get_optimizer()
    metadata = _stage2_metadata(store, args, dim)
    start, best = 0, -1.0
    if args.resume:
        model, payload = load_stage2_checkpoint(args, store, device, path=args.resume)
        optimizer = model.get_optimizer(); optimizer.load_state_dict(payload["optimizer"])
        start, best = payload["epoch"] + 1, payload["best_macro_f1"]
    train_loader, report = _stage2_loader(train_seq, store, args.max_pages_per_batch, True, args.balance_classes)
    val_loader, _ = _stage2_loader(val_seq, store, args.max_pages_per_batch, False, False)
    summary = {"balance_classes": args.balance_classes, "sampling": report, "max_pages_per_batch": args.max_pages_per_batch,
               "train_stacks": len(train_seq), "val_stacks": len(val_seq)}
    print(json.dumps(summary, indent=2))
    run, modality = args.run_name or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"), store.metadata["modality"]
    for epoch in range(start, args.epochs):
        result_path = config.results_dir / f"metrics_final_{modality}_{run}_epoch_{epoch:03d}.json"
        epoch_path = config.checkpoints_dir / f"model_final_{modality}_{run}_epoch_{epoch:03d}.pt"
        _check_new(result_path, epoch_path)
        train_metrics = stage2_epoch(model, train_loader, optimizer, device)
        val_metrics = stage2_epoch(model, val_loader, None, device)
        write_json_new(result_path, {"epoch": epoch, "train": train_metrics, "val": val_metrics, **summary})
        _save_checkpoint(epoch_path, model, optimizer, epoch, max(best, val_metrics["macro_f1"]), metadata)
        if val_metrics["macro_f1"] > best:
            best = val_metrics["macro_f1"]
            _save_checkpoint(config.checkpoints_dir / f"model_final_{modality}_{run}_best_epoch_{epoch:03d}.pt",
                             model, optimizer, epoch, best, metadata)
        print(json.dumps({"epoch": epoch, "val_macro_f1": val_metrics["macro_f1"], "checkpoint": str(epoch_path)}))


def evaluate_stage2(args):
    config, device = _config(args), _device(args)
    store, sequences = load_stage1_features(config, args.stage1_features, args.stage1_checkpoint)
    assignments = split_assignments(config, load_model_pages(config), args.seed)
    sequences = [s for s in sequences if assignments[s["stack_id"]] == args.split]
    if not sequences:
        raise ValueError(f"The Stage 1 features hold no stacks of split {args.split}")
    model, _ = load_stage2_checkpoint(args, store, device)
    loader, _ = _stage2_loader(sequences, store, args.max_pages_per_batch, False, False)
    write_json_new(Path(args.output), stage2_epoch(model, loader, None, device))
    print(args.output)


def predict_stage2(args):
    config, device = _config(args), _device(args)
    store, sequences = load_stage1_features(config, args.stage1_features, args.stage1_checkpoint)
    if args.limit_stacks:
        sequences = sequences[:args.limit_stacks]
    model, _ = load_stage2_checkpoint(args, store, device)
    model.eval()
    loader, _ = _stage2_loader(sequences, store, args.max_pages_per_batch, False, False)
    output = []
    with torch.no_grad():
        for batch in loader:
            pred_seg, pred_class = model(batch["features"].to(device))
            seg_probs, class_probs = torch.softmax(pred_seg, -1).cpu(), torch.softmax(pred_class, -1).cpu()
            for s, pages in enumerate(batch["rows"]):
                for i, page in enumerate(pages):
                    cp = class_probs[s, i]
                    row = {k: page[k] for k in ["page_identity", "stack_id", "stack_position", "image_id", "masterindex_id",
                                                "pdf_page_number", "page_class", "segmentation_target"]}
                    row.update({"predicted_class": ID_TO_CLASS[int(cp.argmax())],
                                "document_start_probability": 1.0 if i == 0 else float(seg_probs[s, i][1])})  # as PnC
                    row.update({f"prob_{label}": float(cp[index]) for index, label in ID_TO_CLASS.items()})
                    output.append(row)
    if not output:
        raise ValueError("No pages to predict")
    output.sort(key=lambda r: (r["stack_id"], int(r["stack_position"])))   # batches are grouped by length
    write_csv_new(Path(args.output), output, list(output[0]))
    print(args.output)
