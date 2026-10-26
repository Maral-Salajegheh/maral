import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import classification_report
from tqdm import tqdm


def evaluate_stage1(model, dataset, args):
    # Validate on all splits and store results
    saved_results_seg = {}  # type: ignore[var-annotated]  # Cached segmentation results
    saved_results_c = {}  # type: ignore[var-annotated] # Cached classification results
    saved_results_c_probs = {}  # type: ignore[var-annotated]  # Cached classification probabilities
    for split in dataset.keys():
        saved_results_seg[split] = {}
        saved_results_c[split] = {}
        saved_results_c_probs[split] = {}

    for k, split in dataset.items():
        if k == "train":
            continue
        print(f"Computing results for split {k}...")
        with torch.no_grad():
            model.eval()
            target = split["pages_stacks_text"]
            if model.multimodal:
                target_img = split["pages_stacks_features"]
            Y_pred = []
            C_pred = []
            C_probs_pred = []
            end = len(target) + 1
            for page2 in tqdm(range(1, end)):
                page1 = page2 - 1

                X1 = target[page1]
                if model.multimodal:
                    X1_img = target_img[page1 : page1 + 1].to(args.device)
                else:
                    X1_img = None

                X = model.tokenizer(
                    X1,
                    return_tensors="pt",
                    padding="longest",
                    pad_to_multiple_of=8,
                    truncation="longest_first",
                )
                X = X.to(args.device)

                pred, pred_class, _ = model(X1_img, X)
                pred = pred.argmax(dim=-1)
                Y_pred += pred.tolist()
                if pred_class is not None:
                    C_probs_pred.append(
                        nn.functional.softmax(pred_class, dim=-1).cpu().numpy()
                    )
                    pred_class = pred_class.argmax(dim=-1)
                    C_pred += pred_class.tolist()

            pred_val_pw = np.array(Y_pred) if pred is not None else None
            pred_val_pw_C = np.array(C_pred)
            pred_val_pw_C_probs = np.concatenate(C_probs_pred, axis=0)

        # Force first page in a stack to be segmented correctly (always 1)
        ptr = 0
        for stack in split["flattened_stacks"]:
            pred_val_pw[ptr] = 1  # type: ignore[index]
            ptr += len(stack)

        saved_results_seg[k]["stage1_bert_singlepage"] = pred_val_pw
        saved_results_c[k]["stage1_bert_singlepage"] = pred_val_pw_C
        saved_results_c_probs[k]["stage1_bert_singlepage"] = pred_val_pw_C_probs

    return saved_results_seg, saved_results_c, saved_results_c_probs


def evaluate_stage2(model, prefetched_features, dataset, args, evaluate_on_train=False):
    # Validate on all splits and store results
    saved_results_seg = {}  # type: ignore[var-annotated]  # Cached segmentation results
    saved_results_c = {}  # type: ignore[var-annotated]  # Cached classification results
    saved_results_c_probs = {}  # type: ignore[var-annotated]  # Cached classification probabilities
    for split in dataset.keys():
        saved_results_seg[split] = {}
        saved_results_c[split] = {}
        saved_results_c_probs[split] = {}

    for k, split in dataset.items():
        if k == "train" and not evaluate_on_train:
            continue
        print(f"Computing results for split {k}...")
        assert k in prefetched_features

        with torch.no_grad():
            model.eval()
            target = prefetched_features[k]

            Y_pred = []
            C_pred = []
            C_probs_pred = []
            for sx in target:
                if isinstance(sx, list):
                    sx = torch.stack(sx, dim=0)
                X = sx.to(args.device).unsqueeze(0)
                pred, pred_class = model(X)
                pred = pred.argmax(dim=-1)[0]
                pred = pred.tolist()
                pred[0] = 1  # Force firstpage on first item
                Y_pred += pred
                C_pred += pred_class.argmax(dim=-1)[0].tolist()
                C_probs_pred.append(
                    nn.functional.softmax(pred_class, dim=-1)[0].cpu().numpy()
                )
            pred_seq = np.array(Y_pred)
            pred_seq_C = np.array(C_pred)
            if len(C_probs_pred) > 0:
                pred_seq_C_probs = np.concatenate(C_probs_pred, axis=0)
            else:
                pred_seq_C_probs = None

        saved_results_seg[k]["stage2_rnn_sequence"] = pred_seq
        saved_results_c[k]["stage2_rnn_sequence"] = pred_seq_C
        saved_results_c_probs[k]["stage2_rnn_sequence"] = pred_seq_C_probs

    return saved_results_seg, saved_results_c, saved_results_c_probs


def generate_report(
    dataset,
    id_remapper,
    id_remapper_names,
    saved_results_seg,
    saved_results_c,
    saved_results_c_probs,
    confidence_threshold: float = 0,
    confidence_percentile: float = 0,
):
    report = {}
    most_common_errors = {}
    pred_tuples = {}

    for k in dataset.keys():
        for method in saved_results_seg[k].keys():
            C_target = dataset[k]["pages_stacks_classes"].copy()
            gt_seq_seg = dataset[k]["pages_stacks_isfirstpage"]
            pred_seq_seg = saved_results_seg[k][method]
            pred_seq_C = saved_results_c[k][method]
            pred_seq_C_probs = saved_results_c_probs[k][method]

            # Set to true for debugging / compute an upper bound
            use_gt_segmentation = False

            # Aggregate by voting (probability pooling)
            chunked_seqs = []  # type: ignore[var-annotated]
            current_chunk = None
            current_chunk_probs = None
            target_seg = gt_seq_seg if use_gt_segmentation else pred_seq_seg
            for is_firstpage, cl, probs in zip(
                target_seg, pred_seq_C, pred_seq_C_probs
            ):
                if is_firstpage:
                    if current_chunk is not None:
                        chunked_seqs.append((current_chunk, current_chunk_probs))
                    current_chunk = []
                    current_chunk_probs = []
                current_chunk.append(cl)
                current_chunk_probs.append(probs)
            chunked_seqs.append((current_chunk, current_chunk_probs))

            n_valid = 0
            pred_seq_C_voted = []
            pred_seq_C_probs_voted = []
            agreement_feature = []
            for chunk, probs in chunked_seqs:
                if (np.array(chunk) == chunk[0]).all():
                    n_valid += 1
                    agreement_feature += [True] * len(chunk)
                else:
                    agreement_feature += [False] * len(chunk)
                probs = np.array(probs).mean(axis=0)
                elem = np.argmax(probs)
                pred_seq_C_voted += [elem] * len(chunk)
                pred_seq_C_probs_voted += [list(probs)] * len(chunk)
            print(
                "Coherent segmented sequences before page aggregation:",
                n_valid / len(chunked_seqs),
            )

            if confidence_percentile > 0 and confidence_percentile <= 100:
                # take the [confidence_percentile] most confident predictions where confidence means low entropy of the class probabilities
                entropy_values = -(
                    pred_seq_C_probs * np.log(pred_seq_C_probs + 1e-12)
                ).sum(axis=1)
                num_most_confident_samples = int(
                    len(entropy_values) * confidence_percentile / 100
                )
                idxs_of_confident_classifications = np.argsort(entropy_values)[
                    :num_most_confident_samples
                ]
            elif confidence_threshold >= 0 and confidence_threshold <= 1:
                highest_probability_values = pred_seq_C_probs.max(axis=1)
                idxs_of_confident_classifications = (
                    highest_probability_values > confidence_threshold
                )
            else:
                raise ValueError(
                    "Either confidence_percentile or p_threshold must be set to a valid value."
                )

            pred_seq_seg = pred_seq_seg[idxs_of_confident_classifications]
            pred_seq_C = pred_seq_C[idxs_of_confident_classifications]
            pred_seq_C_probs = pred_seq_C_probs[idxs_of_confident_classifications]
            C_target = C_target[idxs_of_confident_classifications]

            pred_tuple = (pred_seq_seg, pred_seq_C, pred_seq_C_probs, C_target)
            pred_tuples[k] = pred_tuple

            # Classification report
            classif_report = classification_report(
                list(map(id_remapper_names.get, C_target)),
                list(map(id_remapper_names.get, pred_seq_C)),
                zero_division=np.nan,
                output_dict=True,
            )
            cls_df = (
                pd.DataFrame(classif_report)
                .T.round(2)
                .sort_values(by="support", ascending=False)
                .astype({"support": int})
            )
            cls_df["share_support"] = cls_df["support"] / cls_df["support"].sum()
            cls_df["cumsum_share_support"] = cls_df["share_support"].cumsum()

            report[k] = cls_df.round(4)

            # Most common errors
            df = pd.DataFrame.from_dict(
                {
                    "True class": list(map(id_remapper_names.get, C_target)),
                    "Predicted class": list(map(id_remapper_names.get, pred_seq_C)),
                }
            )

            df = (
                df.groupby(["True class", "Predicted class"])
                .size()
                .to_frame("size")
                .reset_index()
            )
            df = df[df["True class"] != df["Predicted class"]]
            df = df.sort_values(by="size", ascending=False)

            most_common_errors[k] = df

    return (
        report,
        most_common_errors,
        pred_tuples,
    )
