import datetime
import os

import mlflow
import numpy as np
import open_clip
import torch
import torch.nn as nn
from PIL import Image
from pnc_claims_classification.training.config import (
    CLIP_MODEL_PATH,
    SCHADEN_DATASET_SUFFIX,
)
from pnc_claims_classification.training.models import ShuffledSampler
from safetensors.torch import load_file
from sklearn.metrics import f1_score, precision_score, recall_score
from tqdm import tqdm


def compute_visual_embeddings(dataset, args):
    assert args.stage == "compute_visual_embeddings"
    assert (
        args.multimodal
    ), "Visual embeddings can only be computed for the multimodal model"
    assert (
        "val" not in dataset
    ), "Visual embeddings must be extracted before the train/val split"

    class PageDataset(torch.utils.data.Dataset):
        def __init__(self, processor, paths):
            self.processor = processor
            self.paths = np.array(paths)

        def __len__(self):
            return len(self.paths)

        def __getitem__(self, idx):
            path = self.paths[idx]
            with Image.open(path) as f:
                x = self.processor(f)

            return idx, x

    print("Loading CLIP...")
    clip_model, _, clip_processor = open_clip.create_model_and_transforms("ViT-H-14")
    clip_model.load_state_dict(load_file(CLIP_MODEL_PATH))
    clip_model = clip_model.visual
    clip_model.eval().requires_grad_(False).to(args.device)
    print("CLIP loaded successfully")

    for save_split in dataset.keys():
        print("Processing split", save_split)
        ds_paths = dataset[save_split]["pages_stacks"]
        ds = PageDataset(clip_processor, ds_paths)
        loader = torch.utils.data.DataLoader(
            ds, batch_size=4, num_workers=os.cpu_count(), pin_memory=True
        )

        all_features_clip = []
        all_paths = []
        with torch.no_grad():
            for idx, X_clip in tqdm(loader):
                X_clip = X_clip.to(args.device, non_blocking=True)
                if len(idx) == 1:
                    all_paths.append(ds.paths[idx.item()])
                else:
                    paths = list(ds.paths[idx])
                    all_paths += paths

                if clip_model is not None:
                    features_clip = clip_model(X_clip)
                    all_features_clip.append(features_clip.cpu())

        if len(all_features_clip) > 0:
            all_features_clip = torch.cat(all_features_clip, dim=0)
            print("clip features shape:", all_features_clip.shape)  # type: ignore[attr-defined]
            os.makedirs(args.output_dir, exist_ok=True)
            torch.save(
                (all_paths, all_features_clip),
                f"{args.output_dir}/clip_features_{save_split}_{args.dataset}_{SCHADEN_DATASET_SUFFIX}.bin",
            )


def load_visual_embeddings(dataset, args):
    assert (
        args.multimodal
    ), "Visual embeddings can only be loaded for the multimodal model"
    assert (
        "val" not in dataset
    ), "Visual embeddings must be loaded before the train/val split"

    print("Loading visual embeddings for multimodal model...")

    for split, d in dataset.items():
        paths, features = torch.load(
            f"{args.output_dir}/clip_features_{split}_{args.dataset}_{SCHADEN_DATASET_SUFFIX}.bin"
        )

        # Safety check
        assert paths == list(d["pages_stacks"])
        assert len(features) == len(d["pages_stacks"])

        d["pages_stacks_features"] = features
        docs_features = []
        stacks_features = []
        flattened_stacks_features = []
        doc_ptr = 0
        for stack in d["stacks"]:
            current_stack_features = []
            doc_ptr_start = doc_ptr
            for doc in stack:
                doc_len = len(doc)
                docs_features.append(features[doc_ptr : doc_ptr + doc_len])
                current_stack_features.append(docs_features[-1])
                doc_ptr += doc_len
            stacks_features.append(current_stack_features)
            flattened_stacks_features.append(features[doc_ptr_start:doc_ptr])
        d["docs_stacks_features"] = np.array(docs_features, dtype=object)
        d["stacks_features"] = np.array(stacks_features, dtype=object)
        d["flattened_stacks_features"] = np.array(
            flattened_stacks_features, dtype=object
        )
    return dataset


def train_stage1(
    model,
    dataset,
    args,
    n_iterations,
    print_train_metrics_every,
    save_progress_and_validate_every,
):
    use_fp16 = args.use_fp16
    use_bf16 = args.use_bf16
    assert not (use_fp16 and use_bf16)

    criterion = nn.CrossEntropyLoss()
    optimizer = model.get_optimizer()
    if use_fp16:
        scaler = torch.cuda.amp.GradScaler()
    else:
        scaler = None
    loss_curve = []
    acc_curve = []
    acc_curve_class = []
    acc_curve_lp = []
    grad_norms = []
    i = 0

    # Prepare data (single-page model)
    X_train = dataset["train"]["pages_stacks_text"]
    if model.multimodal:
        X_train_img = dataset["train"]["pages_stacks_features"]
    Y_train = torch.LongTensor(dataset["train"]["pages_stacks_isfirstpage"])
    Yl_train = torch.LongTensor(dataset["train"]["pages_stacks_islastpage"])
    C_train = torch.LongTensor(dataset["train"]["pages_stacks_classes"])

    # If we use the full dataset, we balance all classes
    balanced = args.balance_classes
    if balanced:
        freqs = 1 / np.bincount(C_train.numpy())
        sample_freqs = freqs[C_train][:-1]
        sample_freqs /= sample_freqs.sum()
        print("Balancing...")

    bs = args.batch_size
    print("Expected iterations per epoch", len(X_train) / bs)

    sampler = ShuffledSampler(len(X_train) - 1)

    os.makedirs(args.output_dir, exist_ok=True)
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    if model.multimodal:
        output_filename = f"model_{args.dataset}_stage1_multimodal_{current_time}.pth"
    else:
        output_filename = f"model_{args.dataset}_stage1_textonly_{current_time}.pth"
    output_filename = os.path.join(args.output_dir, output_filename)
    print("Model will be saved to", output_filename)

    model.train()
    for i in tqdm(range(n_iterations)):  # 70000
        if balanced:
            pages = np.random.choice(
                len(X_train) - 1, size=bs, replace=True, p=sample_freqs
            )
        else:
            pages = np.array(sampler(bs)) + 1
        X_text = list(X_train[pages])

        Y = Y_train[pages].to(args.device)
        Yl = Yl_train[pages].to(args.device)
        C = C_train[pages].to(args.device)

        if model.multimodal:
            X_img = X_train_img[pages].to(args.device)
        else:
            X_img = None

        X = model.tokenizer(
            X_text,
            return_tensors="pt",
            padding="max_length",
            pad_to_multiple_of=8,
            truncation="longest_first",
        )
        X = X.to(args.device)

        optimizer.zero_grad()
        with torch.cuda.amp.autocast(
            enabled=use_fp16 or use_bf16,
            dtype=torch.bfloat16 if use_bf16 else torch.float16,
        ):
            pred, pred_class, pred_lp = model(X_img, X)
            loss = criterion(pred, Y)
            if pred_class is not None:
                loss += criterion(pred_class, C)
            if pred_lp is not None:
                loss += criterion(pred_lp, Yl)
        if use_fp16:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
        else:
            loss.backward()
        grad_norms.append(nn.utils.clip_grad_norm_(model.parameters(), 20.0).item())
        if use_fp16:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        loss_curve.append(loss.item())
        acc = (pred.argmax(dim=-1) == Y).float().mean().item()
        if pred_lp is not None:
            acc_lp = (pred_lp.argmax(dim=-1) == Yl).float().mean().item()
        else:
            acc_lp = 0
        acc_curve.append(acc)
        acc_curve_lp.append(acc_lp)
        if pred_class is not None:
            acc_class = (pred_class.argmax(dim=-1) == C).float().mean().item()
            acc_curve_class.append(acc_class)
        if i % print_train_metrics_every == 0:
            mlflow.log_metrics(
                {
                    "loss": loss.item(),
                    "acc_first_page": acc,
                    "acc_last_page": acc_lp,
                    "acc_class": acc_class if pred_class is not None else 0,
                },
                step=i,
            )
            print(
                "loss",
                loss.item(),
                "acc_firstpage",
                acc,
                "acc_lastpage",
                acc_lp,
                "acc_class",
                acc_class if pred_class is not None else "",
            )
        if i == 40000:
            print("Decaying learning rate...")
            optimizer.param_groups[0]["lr"] = 2e-6
        if i > 0 and i % save_progress_and_validate_every == 0:
            output_filename_with_step = output_filename.replace(".pth", f"_{i}.pth")
            torch.save(
                {
                    "bert_model": model.state_dict(),
                },
                output_filename_with_step,
            )
            print("Current progress saved to", output_filename_with_step)
            mlflow.log_artifact(output_filename_with_step)
            print("Validating...")
            k = "val"
            split = dataset[k]
            with torch.no_grad():
                model.eval()
                C_target = split["pages_stacks_classes"]
                target = split["pages_stacks_text"]
                if model.multimodal:
                    target_img = split["pages_stacks_features"]
                Y_target = split["pages_stacks_isfirstpage"]
                Y_pred = []
                C_pred = []
                end = len(target) + 1
                for page2 in tqdm(range(1, end)):
                    page1 = page2 - 1

                    X_text = target[page1]
                    if model.multimodal:
                        X_img = target_img[page1 : page1 + 1].to(args.device)
                    else:
                        X_img = None

                    X = model.tokenizer(
                        X_text,
                        return_tensors="pt",
                        padding="longest",
                        pad_to_multiple_of=8,
                        truncation="longest_first",
                    )
                    X = X.to(args.device)

                    pred, pred_class, pred_lp = model(X_img, X)
                    pred = pred.argmax(dim=-1)
                    Y_pred += pred.tolist()
                    if pred_class is not None:
                        pred_class = pred_class.argmax(dim=-1)
                        C_pred += pred_class.tolist()

                pred_val_pw = np.array(Y_pred)
                pred_val_pw_C = np.array(C_pred)

            # Force first page in a stack to be segmented correctly (always 1)
            ptr = 0
            for stack in split["flattened_stacks"]:
                pred_val_pw[ptr] = 1
                ptr += len(stack)

            print(
                "Validation results (note: these metrics are only meant for monitoring purposes)"
            )

            print("---- SEGMENTATION ----")
            if pred_val_pw is not None:
                seg_accuracy = (Y_target == pred_val_pw).mean()
                seg_precision = precision_score(pred_val_pw, Y_target)
                seg_recall = recall_score(pred_val_pw, Y_target)
                seg_f1 = f1_score(pred_val_pw, Y_target)
                print(k, "accuracy", seg_accuracy)
                print(
                    k,
                    "precision",
                    seg_precision,
                    "recall",
                    seg_recall,
                    "f1",
                    seg_f1,
                )
                mlflow.log_metrics(
                    {
                        "seg_accuracy": seg_accuracy,
                        "seg_precision": seg_precision,
                        "seg_recall": seg_recall,
                        "seg_f1": seg_f1,
                    },
                    step=i,
                )
            else:
                print("Skipped.")

            print("---- CLASSIFICATION ----")
            class_accuracy = (C_target == pred_val_pw_C).mean()
            class_precision_macro = precision_score(
                pred_val_pw_C, C_target, average="macro"
            )
            class_recall_macro = recall_score(pred_val_pw_C, C_target, average="macro")
            class_f1_macro = f1_score(pred_val_pw_C, C_target, average="macro")
            class_precision = precision_score(
                pred_val_pw_C, C_target, average="weighted"
            )
            class_recall = recall_score(pred_val_pw_C, C_target, average="weighted")
            class_f1 = f1_score(pred_val_pw_C, C_target, average="weighted")
            print(k, "accuracy", class_accuracy)
            print(
                k,
                "precision",
                (class_precision_macro, class_precision),
                "recall",
                (class_recall_macro, class_recall),
                "f1",
                (class_f1_macro, class_f1),
            )
            mlflow.log_metrics(
                {
                    "class_accuracy": class_accuracy,
                    "class_precision_macro": class_precision_macro,
                    "class_recall_macro": class_recall_macro,
                    "class_f1_macro": class_f1_macro,
                    "class_precision": class_precision,
                    "class_recall": class_recall,
                    "class_f1": class_f1,
                },
                step=i,
            )
            model.train()

    print("Stage 1 finished. Saving...")
    torch.save(
        {
            "bert_model": model.state_dict(),
        },
        output_filename.replace(".pth", f"_{i}.pth"),
    )
    print("Model saved to", output_filename.replace(".pth", f"_{i}.pth"))
    mlflow.pytorch.log_model(model, "stage1_model")


def precompute_bert_features(bert_model, dataset, args, limit=-1, no_train=False):
    # Pre-compute single-page features for all splits

    if args.dummy_bert_features:
        print("Warning: BERT features are ignored due to --dummy_bert_features")

    prefetched_features = {}
    for k, split in dataset.items():
        if no_train and k == "train":
            continue
        print(f"Pre-computing BERT features for split {k}...")
        count = 0
        X_flattened = []
        with torch.no_grad():
            bert_model.eval()
            targets = split["flattened_stacks_text"]
            target_imgs = (
                split["flattened_stacks_features"] if bert_model.multimodal else None
            )
            if not bert_model.multimodal:
                target_imgs = targets
            for target, target_img in zip(tqdm(targets), target_imgs):
                if count == limit:
                    break
                count += 1
                feats_pred = []
                end = len(target) + 1
                for page2 in range(1, end):
                    page1 = page2 - 1

                    X1 = target[page1]
                    if bert_model.multimodal:
                        X1_img = target_img[page1].to(args.device)
                    else:
                        X1_img = None

                    if not args.dummy_bert_features:
                        X = bert_model.tokenizer(
                            X1,
                            return_tensors="pt",
                            padding="longest",
                            pad_to_multiple_of=8,
                            truncation="longest_first",
                        )
                        X = X.to(args.device)
                        pred = bert_model(X1_img, X, return_embeddings=True)
                    else:
                        pred = torch.zeros((1, 1024))
                    feats_pred.append(pred.cpu())
                if len(feats_pred) > 0:
                    feats_pred = torch.cat(feats_pred)
                else:
                    feats_pred = torch.zeros((0, 1024))
                X_flattened.append(feats_pred)

        prefetched_features[k] = X_flattened

    # Concatenate visual features to textual features
    if args.multimodal:
        for k, split in dataset.items():
            if k not in prefetched_features:
                continue
            for i in range(len(prefetched_features[k])):
                s = split["flattened_stacks_features"][i]
                if isinstance(s, list):
                    s = torch.stack(s, dim=0)
                prefetched_features[k][i] = torch.cat(
                    (prefetched_features[k][i][:, :1024], s),  # type: ignore[call-overload]
                    dim=1,
                )

    return prefetched_features


def train_stage2(
    bert_model,
    model,
    prefetched_features,
    dataset,
    args,
    n_iterations,
    print_train_metrics_every,
    validate_every,
):
    assert model.multimodal == args.multimodal

    criterion = nn.CrossEntropyLoss()
    optimizer = model.get_optimizer()
    loss_curve = []
    acc_curve = []

    bs = args.batch_size
    model.train()
    X_flattened_train = prefetched_features["train"]
    Y_flattened = dataset["train"]["flattened_stacks_isfirstpage"]
    C_flattened = dataset["train"]["flattened_stacks_classes"]
    if len(X_flattened_train) < len(Y_flattened):
        # The dataset has been limited
        Y_flattened = Y_flattened[: len(X_flattened_train)]
        C_flattened = C_flattened[: len(X_flattened_train)]
    elif len(X_flattened_train) > len(Y_flattened):
        raise RuntimeError(
            "The dataset labels are not aligned to the prefetched features"
        )

    balanced = args.balance_classes
    if balanced:
        flattened_classes = [item for sublist in C_flattened for item in sublist]

        class_counts = np.bincount(flattened_classes)

        # weights array for each sample in `flattened_stacks_classes`
        sample_freqs_list = []
        for classes in C_flattened:
            weights = [1 / class_counts[class_idx] for class_idx in classes]
            max_weight = np.max(weights)
            sample_freqs_list.append(max_weight)

        sample_freqs = np.array(sample_freqs_list)

        # normalize to sum to 1
        sample_freqs /= sample_freqs.sum()
        print("Balancing...")

    sampler = ShuffledSampler(len(X_flattened_train))
    print("Expected iterations per epoch", len(X_flattened_train) / bs)

    os.makedirs(args.output_dir, exist_ok=True)
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    if model.multimodal:
        output_filename = f"model_{args.dataset}_final_multimodal_{current_time}.pth"
    else:
        output_filename = f"model_{args.dataset}_final_textonly_{current_time}.pth"
    output_filename = os.path.join(args.output_dir, output_filename)
    print("Model will be saved to", output_filename)

    print("Training Stage 2 (sequence model)")
    for i in range(n_iterations):  # 15000
        # Collate
        if balanced:
            idx = np.random.choice(
                len(X_flattened_train), size=bs, replace=True, p=sample_freqs
            )
        else:
            idx = np.array(sampler(bs))

        lengths = []
        for idx_ in idx:
            lengths.append(len(X_flattened_train[idx_]))
        max_len = max(lengths)
        X = torch.zeros((bs, max_len, model.input_dim))
        Y = torch.ones((bs, max_len), dtype=torch.long) * -100
        C = torch.ones((bs, max_len), dtype=torch.long) * -100
        for j, idx_ in enumerate(idx):
            sample = X_flattened_train[idx_].float()
            X[j, : len(sample)] = sample
            Y[j, : len(sample)] = torch.LongTensor(Y_flattened[idx_])
            C[j, : len(sample)] = torch.LongTensor(C_flattened[idx_])
        X = X.to(args.device)
        Y = Y.to(args.device)
        C = C.to(args.device)

        Y[:, 0] = -100  # Do not segment first page

        pred, pred_class = model(X)
        loss = criterion(pred.flatten(0, 1), Y.flatten(0, 1))
        if pred_class is not None:
            loss += criterion(pred_class.flatten(0, 1), C.flatten(0, 1))
        optimizer.zero_grad()
        loss.backward()
        loss_curve.append(loss.item())
        optimizer.step()
        with torch.no_grad():
            pred_match = (pred.argmax(dim=-1) == Y).flatten().float()
            pred_mask = (Y != -100).flatten()
            acc = pred_match[pred_mask].mean().item()
            acc_curve.append(acc)
            if pred_class is not None:
                pred_class_match = (pred_class.argmax(dim=-1) == C).flatten().float()
                acc_class = pred_class_match[pred_mask].mean().item()
                # acc_curve_class.append(acc_class)
        if i % print_train_metrics_every == 0:
            mlflow.log_metrics(
                {
                    "loss": loss.item(),
                    "acc_first_page": acc,
                    "acc_class": acc_class if pred_class is not None else 0,
                },
                step=i,
            )
            print(
                f"[{i}] loss {loss.item()} segmentation_accuracy {acc}, class accuracy {acc_class}"
            )
        if i > 0 and i % validate_every == 0:
            # Validate
            print(
                "Validation results (note: these metrics are only meant for monitoring purposes)"
            )
            use_gt_segmentation = False
            k = "val"
            split = dataset[k]
            X_flattened_val = prefetched_features[k]
            with torch.no_grad():
                model.eval()
                target = X_flattened_val

                Y_target = split["pages_stacks_isfirstpage"]
                C_target = split["pages_stacks_classes"]
                Y_pred = []
                C_pred = []
                for sx in target:
                    if isinstance(sx, list):
                        sx = torch.stack(sx, dim=0)
                    X = sx.to(args.device).unsqueeze(0)
                    pred, pred_class = model(X)
                    pred = pred.argmax(dim=-1)[0]
                    if use_gt_segmentation:
                        pred[:] = 0
                    pred = pred.tolist()
                    pred[0] = (
                        1  # Force firstpage prediction on first page of each stack
                    )
                    Y_pred += pred
                    C_pred += pred_class.argmax(dim=-1)[0].tolist()
                pred_seq = np.array(Y_pred)
                pred_seq_C = np.array(C_pred)

            print("---- SEGMENTATION ----")
            seg_accuracy = (Y_target == pred_seq).mean()
            seg_precision = precision_score(pred_seq, Y_target)
            seg_recall = recall_score(pred_seq, Y_target)
            seg_f1 = f1_score(pred_seq, Y_target)
            print(k, "accuracy", seg_accuracy)
            print(
                k,
                "precision",
                seg_precision,
                "recall",
                seg_recall,
                "f1",
                seg_f1,
            )
            mlflow.log_metrics(
                {
                    "seg_accuracy": seg_accuracy,
                    "seg_precision": seg_precision,
                    "seg_recall": seg_recall,
                    "seg_f1": seg_f1,
                },
                step=i,
            )

            print("---- CLASSIFICATION ----")
            class_accuracy = (C_target == pred_seq_C).mean()
            class_precision_macro = precision_score(
                pred_seq_C, C_target, average="macro"
            )
            class_recall_macro = recall_score(pred_seq_C, C_target, average="macro")
            class_f1_macro = f1_score(pred_seq_C, C_target, average="macro")
            class_precision = precision_score(pred_seq_C, C_target, average="weighted")
            class_recall = recall_score(pred_seq_C, C_target, average="weighted")
            class_f1 = f1_score(pred_seq_C, C_target, average="weighted")
            print(k, "accuracy", class_accuracy)
            print(
                k,
                "precision",
                (class_precision_macro, class_precision),
                "recall",
                (class_recall_macro, class_recall),
                "f1",
                (class_f1_macro, class_f1),
            )
            mlflow.log_metrics(
                {
                    "class_accuracy": class_accuracy,
                    "class_precision_macro": class_precision_macro,
                    "class_recall_macro": class_recall_macro,
                    "class_f1_macro": class_f1_macro,
                    "class_precision": class_precision,
                    "class_recall": class_recall,
                    "class_f1": class_f1,
                },
                step=i,
            )
            model.train()

    print("Stage 2 finished. Saving...")
    torch.save(
        {
            "bert_model": bert_model.state_dict(),
            "model": model.state_dict(),
        },
        output_filename,
    )
    print("Model saved to", output_filename)
    mlflow.pytorch.log_model(model, "final_model")
