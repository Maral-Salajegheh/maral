"""Metrics, confidence reports and immutable result writing (counterpart of PnC evaluation.py)."""
import csv
import json
import math
from pathlib import Path

from .config import CLASS_TO_ID, CLASSES, ID_TO_CLASS
from .data import read_csv


def classification_metrics(y_true, y_pred):
    per_class = {}
    weighted_sum = 0.0
    total = len(y_true)
    matrix = [[0 for _ in CLASSES] for _ in CLASSES]
    for truth, pred in zip(y_true, y_pred):
        matrix[truth][pred] += 1
    present_f1 = []
    for index, label in ID_TO_CLASS.items():
        tp = matrix[index][index]
        fp = sum(matrix[row][index] for row in range(len(CLASSES)) if row != index)
        fn = sum(matrix[index][column] for column in range(len(CLASSES)) if column != index)
        support = sum(matrix[index])
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[label] = {"precision": precision, "recall": recall, "f1": f1, "support": support}
        weighted_sum += f1 * support
        if support or tp + fp:  # as PnC/sklearn: only classes that occur in truth or prediction
            present_f1.append(f1)
    return {
        "per_class": per_class,
        "macro_f1": sum(present_f1) / len(present_f1) if present_f1 else 0.0,
        "macro_f1_classes": len(present_f1),
        "weighted_f1": weighted_sum / max(total, 1),
        "confusion_matrix": matrix,
        "mad_recall": per_class["MAD"]["recall"],
        "mad_f1": per_class["MAD"]["f1"],
    }


def binary_metrics(y_true, y_pred):
    if not y_true:
        return {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0, "support": 0}
    tp = sum(a == 1 and b == 1 for a, b in zip(y_true, y_pred))
    fp = sum(a == 0 and b == 1 for a, b in zip(y_true, y_pred))
    fn = sum(a == 1 and b == 0 for a, b in zip(y_true, y_pred))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "accuracy": sum(a == b for a, b in zip(y_true, y_pred)) / len(y_true),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "support": len(y_true),
    }


# ===========================================================================
# Confidence report (PnC generate_report: confidence threshold or percentile, probability pooling)
# ===========================================================================

REVIEW_FIELDS = ["stack_id", "stack_position", "masterindex_id", "pdf_page_number", "image_path", "predicted_class",
                 "confidence", "second_class", "second_probability", "reviewed_class", "reviewer", "comment"]

def read_predictions(path):
    """Rows of a predict_stage1/predict_stage2 CSV, ordered by stack and position, with class probabilities."""
    rows = read_csv(path)
    for row in rows:
        row["probs"] = [float(row[f"prob_{label}"]) for label in CLASSES]
        row["stack_position"] = int(row["stack_position"])
    return sorted(rows, key=lambda row: (row["stack_id"], row["stack_position"]))


def start_probability(row):
    """Document-start probability; predict_stage2 and predict_stage1 name the column differently."""
    return float(row.get("document_start_probability") or row["first_page_probability"])


def predicted_documents(rows):
    """Split ordered rows into predicted documents (start probability >= 0.5 is the argmax of the start head)."""
    documents, current = [], []
    for row in rows:
        if current and (row["stack_position"] == 1 or start_probability(row) >= 0.5):
            documents.append(current)
            current = []
        current.append(row)
    if current:
        documents.append(current)
    return documents


def pool_documents(rows):
    """PnC probability pooling: each page takes the mean class probabilities of its predicted document."""
    pooled = []
    for document in predicted_documents(rows):
        mean = [sum(row["probs"][i] for row in document) / len(document) for i in range(len(CLASSES))]
        pooled += [{**row, "probs": mean} for row in document]
    return pooled


def level_rows(rows, level):
    """Page probabilities, or each page with the mean probabilities of its predicted document."""
    return pool_documents(rows) if level == "document" else rows


def predicted_id(row):
    return max(range(len(CLASSES)), key=lambda index: row["probs"][index])


def confidence(row):
    """Highest class probability (PnC confidence_threshold)."""
    return max(row["probs"])


def entropy(row):
    """Entropy of the class probabilities (PnC confidence_percentile); low means confident."""
    return -sum(p * math.log(p + 1e-12) for p in row["probs"])


def threshold_auto(rows, threshold):
    """PnC: a page is processed automatically when its highest probability is above the threshold."""
    if not 0 <= threshold <= 1:
        raise ValueError(f"threshold must be between 0 and 1: {threshold}")
    return [confidence(row) > threshold for row in rows]


def percentile_auto(rows, percentile):
    """PnC: the given percent of pages with the lowest entropy is processed automatically."""
    if not 0 < percentile <= 100:
        raise ValueError(f"confidence percentile must be in (0, 100]: {percentile}")
    keep = int(len(rows) * percentile / 100)
    chosen = set(sorted(range(len(rows)), key=lambda index: entropy(rows[index]))[:keep])
    return [index in chosen for index in range(len(rows))]


def routing_metrics(rows, auto_flags):
    """Coverage, errors on the automatic and the reviewed part, and metrics on the automatic part."""
    auto = [row for row, flag in zip(rows, auto_flags) if flag]
    truth = [CLASS_TO_ID[row["page_class"]] for row in auto]
    pred = [predicted_id(row) for row in auto]
    errors_auto = sum(t != p for t, p in zip(truth, pred))
    errors_total = sum(CLASS_TO_ID[row["page_class"]] != predicted_id(row) for row in rows)
    return {"coverage": len(auto) / len(rows) if rows else 0.0,
            "auto_pages": len(auto), "review_pages": len(rows) - len(auto),
            "errors_auto": errors_auto, "errors_review": errors_total - errors_auto,
            "classification": classification_metrics(truth, pred)}


def confidence_report(rows, thresholds, percentile=0):
    """As PnC: the percentile rule when percentile > 0, otherwise one result per threshold."""
    if percentile > 0:
        return [{"percentile": percentile, **routing_metrics(rows, percentile_auto(rows, percentile))}]
    return [{"threshold": threshold, **routing_metrics(rows, threshold_auto(rows, threshold))} for threshold in thresholds]


def summary_rows(results):
    """One flat row per threshold (or percentile): the readable form of the report, for the CSV and the table."""
    rows = []
    for result in results:
        rule = "percentile" if "percentile" in result else "threshold"
        per_class = result["classification"]["per_class"]
        row = {rule: result[rule], "coverage": round(result["coverage"], 4), "auto": result["auto_pages"],
               "review": result["review_pages"], "err_auto": result["errors_auto"],
               "err_review": result["errors_review"], "macro_f1": round(result["classification"]["macro_f1"], 4)}
        row.update({f"f1_{label}": round(per_class[label]["f1"], 4) for label in CLASSES})
        rows.append(row)
    return rows


def format_confidence_table(rows):
    """Plain-text table of summary_rows."""
    header = list(rows[0])
    lines = ["  ".join(f"{name:>10}" for name in header)]
    for row in rows:
        lines.append("  ".join(f"{row[name]:>10}" for name in header))
    return "\n".join(lines)


def review_row(row):
    """One page for the reviewer: prediction, two most likely classes, empty columns to fill in."""
    ranked = sorted(range(len(CLASSES)), key=lambda index: -row["probs"][index])
    return {"stack_id": row["stack_id"], "stack_position": row["stack_position"],
            "masterindex_id": row["masterindex_id"], "pdf_page_number": row["pdf_page_number"],
            "image_path": row["page_identity"], "predicted_class": CLASSES[ranked[0]],
            "confidence": round(row["probs"][ranked[0]], 4), "second_class": CLASSES[ranked[1]],
            "second_probability": round(row["probs"][ranked[1]], 4),
            "reviewed_class": "", "reviewer": "", "comment": ""}


def review_rows(rows, threshold):
    """Pages that go to human review at a threshold, least confident first. The true label is left out."""
    flags = threshold_auto(rows, threshold)
    queue = [row for row, auto in zip(rows, flags) if not auto]
    queue.sort(key=lambda row: (confidence(row), row["stack_id"], row["stack_position"]))
    return [review_row(row) for row in queue]


# ===========================================================================
# Immutable result writing
# ===========================================================================

def write_json_new(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing result: {path}")
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")


def write_csv_new(path: Path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing result: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)