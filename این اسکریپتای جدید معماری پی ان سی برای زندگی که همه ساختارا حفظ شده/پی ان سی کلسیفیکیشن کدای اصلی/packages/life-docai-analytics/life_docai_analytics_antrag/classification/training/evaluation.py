"""Metrics, confidence reports and immutable result writing (counterpart of PnC evaluation.py)."""

import csv
import json
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
# Confidence report (PnC generate_report: confidence threshold, probability pooling)
# ===========================================================================

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


def predicted_id(row):
    return max(range(len(CLASSES)), key=lambda index: row["probs"][index])


def threshold_metrics(rows, threshold):
    """Metrics for pages processed automatically at a threshold (PnC: highest probability > threshold)."""
    auto = [row for row in rows if max(row["probs"]) > threshold]
    truth = [CLASS_TO_ID[row["page_class"]] for row in auto]
    pred = [predicted_id(row) for row in auto]
    errors_auto = sum(t != p for t, p in zip(truth, pred))
    errors_total = sum(CLASS_TO_ID[row["page_class"]] != predicted_id(row) for row in rows)
    return {"threshold": threshold, "coverage": len(auto) / len(rows) if rows else 0.0,
            "auto_pages": len(auto), "review_pages": len(rows) - len(auto),
            "errors_auto": errors_auto, "errors_review": errors_total - errors_auto,
            "classification": classification_metrics(truth, pred)}


def confidence_report(rows, thresholds, level):
    """One threshold_metrics result per threshold, on page or pooled document probabilities."""
    rows = pool_documents(rows) if level == "document" else rows
    return [threshold_metrics(rows, threshold) for threshold in thresholds]


def format_confidence_table(results):
    """Plain-text table: coverage, routed errors and F1 per class for each threshold."""
    header = ["threshold", "coverage", "auto", "review", "err_auto", "err_review", "macro_f1"] + list(CLASSES)
    lines = ["  ".join(f"{name:>10}" for name in header)]
    for result in results:
        per_class = result["classification"]["per_class"]
        values = [result["threshold"], result["coverage"], result["auto_pages"], result["review_pages"],
                  result["errors_auto"], result["errors_review"], result["classification"]["macro_f1"]]
        values += [per_class[label]["f1"] for label in CLASSES]
        lines.append("  ".join(f"{value:>10.4f}" if isinstance(value, float) else f"{value:>10}" for value in values))
    return "\n".join(lines)


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