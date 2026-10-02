"""Metrics and immutable result writing for corrected PnC runs."""

import csv
import json
from pathlib import Path

from .config import CLASSES, ID_TO_CLASS


def classification_metrics(y_true, y_pred):
    per_class = {}
    f1_values = []
    weighted_sum = 0.0
    total = len(y_true)
    matrix = [[0 for _ in CLASSES] for _ in CLASSES]
    for truth, pred in zip(y_true, y_pred):
        matrix[truth][pred] += 1
    for index, label in ID_TO_CLASS.items():
        tp = matrix[index][index]
        fp = sum(matrix[row][index] for row in range(len(CLASSES)) if row != index)
        fn = sum(matrix[index][column] for column in range(len(CLASSES)) if column != index)
        support = sum(matrix[index])
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[label] = {"precision": precision, "recall": recall, "f1": f1, "support": support}
        f1_values.append(f1)
        weighted_sum += f1 * support
    return {
        "per_class": per_class,
        "macro_f1": sum(f1_values) / len(f1_values),
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
