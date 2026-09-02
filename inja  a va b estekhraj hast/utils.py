from __future__ import annotations

import csv
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
from PIL import Image, ImageOps

from Life.Extraction import config

_RAPIDOCR_ENGINE: Any | None = None


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def int_value(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"CSV not found: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


def read_latest_jsonl(path: Path, key_fn: Callable[[dict[str, Any]], tuple[Any, ...]]) -> dict[tuple[Any, ...], dict[str, Any]]:
    latest: dict[tuple[Any, ...], dict[str, Any]] = {}
    if not path.is_file():
        return latest
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                latest[key_fn(record)] = record
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                continue
    return latest


def page_key(record: dict[str, Any]) -> tuple[str, str, int, str]:
    return (
        str(record["masterindex_id"]),
        str(record.get("pdf_path_in_zip") or ""),
        int_value(record.get("page_number")),
        str(record.get("image_sha256") or "NO_IMAGE"),
    )


def resolve_image_path(path_text: str) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        if path.is_file():
            return path
        raise FileNotFoundError(f"Rendered PNG not found: {path_text}")
    for root in config.IMAGE_ROOTS:
        candidate = root / path_text
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Rendered PNG not found: {path_text}")


def load_page_metadata() -> dict[tuple[str, int], list[dict[str, str]]]:
    rows: list[dict[str, str]] = []
    for path in sorted(config.DATA_DIR.glob("*_page_labels.jsonl")):
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    rows.append(json.loads(line))
    pseudo_path = config.DATA_DIR / "ab1_pseudo_documents.csv"
    if pseudo_path.is_file():
        rows.extend(read_csv(pseudo_path))
    by_key: dict[tuple[str, int], list[dict[str, str]]] = {}
    for row in rows:
        key = (str(row["masterindex_id"]), int_value(row["page_number"]))
        by_key.setdefault(key, []).append(row)
    return by_key


def rapidocr_engine() -> Any:
    global _RAPIDOCR_ENGINE
    if _RAPIDOCR_ENGINE is None:
        from rapidocr_onnxruntime import RapidOCR

        _RAPIDOCR_ENGINE = RapidOCR()
    return _RAPIDOCR_ENGINE


# Run RapidOCR and return one box per detected text region. The angle classifier is off
# because it reverses MRZ lines, which are almost all filler and have no reliable direction cue.
def ocr_boxes(image: Image.Image) -> list[dict[str, Any]]:
    result, _elapsed = rapidocr_engine()(np.asarray(image.convert("RGB")), use_cls=False)
    boxes = []
    for item in result or []:
        if len(item) < 2 or not item[1]:
            continue
        xs = [float(point[0]) for point in item[0]]
        ys = [float(point[1]) for point in item[0]]
        boxes.append({"text": str(item[1]), "x0": min(xs), "y0": min(ys), "x1": max(xs), "y1": max(ys)})
    return boxes


# Join boxes that share a baseline into one text line, read left to right.
def merge_boxes_into_lines(boxes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    lines: list[dict[str, Any]] = []
    for box in sorted(boxes, key=lambda item: item["y0"] + item["y1"]):
        height = box["y1"] - box["y0"]
        target = None
        for line in lines:
            overlap = min(line["y1"], box["y1"]) - max(line["y0"], box["y0"])
            if overlap > 0.5 * min(height, line["y1"] - line["y0"]):
                target = line
                break
        if target is None:
            lines.append({**box, "parts": [box]})
        else:
            target["parts"].append(box)
            target["x0"] = min(target["x0"], box["x0"])
            target["x1"] = max(target["x1"], box["x1"])
            target["y0"] = min(target["y0"], box["y0"])
            target["y1"] = max(target["y1"], box["y1"])
    for line in lines:
        line["text"] = "".join(part["text"] for part in sorted(line["parts"], key=lambda item: item["x0"]))
        line.pop("parts")
    return sorted(lines, key=lambda item: item["y0"])


# Upscale a crop to a readable character height and surround it with white margin,
# because both OCR engines clip text that touches the image edge.
def prepare_mrz_crop(image_path: Path, line_count: int) -> Image.Image:
    image = Image.open(image_path).convert("RGB")
    wanted = config.MRZ_TARGET_LINE_HEIGHT * max(line_count, 1)
    scale = min(max(wanted / max(image.height, 1), 1.0), config.MRZ_MAX_UPSCALE)
    if scale > 1.0:
        image = image.resize((int(image.width * scale), int(image.height * scale)), Image.LANCZOS)
    return ImageOps.expand(image, border=config.MRZ_CROP_BORDER, fill="white")


def tesseract_available() -> bool:
    command = config.TESSERACT_CMD
    return Path(command).is_file() if "/" in command else shutil.which(command) is not None


def run_tesseract(image_path: Path, config_text: str, lang: str | None = None) -> str:
    command = [config.TESSERACT_CMD, str(image_path), "stdout"]
    if lang:
        command.extend(["-l", lang])
    command.extend(config_text.split())
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    return result.stdout


# Cut a wide line into pieces the recognizer can handle, splitting at low-ink columns.
def split_columns(image: Image.Image, max_aspect: float = 7.0) -> list[Image.Image]:
    pieces = max(1, int(np.ceil((image.width / max(image.height, 1)) / max_aspect)))
    if pieces == 1:
        return [image]
    ink = (np.asarray(image.convert("L")) < 160).sum(axis=0)
    step = image.width / pieces
    cuts = [0]
    for index in range(1, pieces):
        centre = int(index * step)
        window = range(max(centre - 12, 1), min(centre + 12, image.width - 1))
        cuts.append(min(window, key=lambda column: ink[column]))
    cuts.append(image.width)
    return [image.crop((cuts[i], 0, cuts[i + 1], image.height)) for i in range(len(cuts) - 1)]


# RapidOCR fallback for the MRZ band: one line at a time, each split into narrow chunks.
def rapidocr_mrz_lines(image: Image.Image) -> str:
    texts = []
    for line in merge_boxes_into_lines(ocr_boxes(image)):
        strip = image.crop((0, max(int(line["y0"]) - 4, 0), image.width, min(int(line["y1"]) + 4, image.height)))
        texts.append("".join("".join(str(box["text"]) for box in ocr_boxes(chunk)) for chunk in split_columns(strip)))
    return "\n".join(text for text in texts if text)


# Read the MRZ band. Tesseract handles OCR-B far better than RapidOCR's general recognizer,
# which drops and duplicates characters inside long runs of filler.
def run_mrz_ocr(image_path: Path, line_count: int = 2) -> tuple[str, str]:
    prepared = prepare_mrz_crop(image_path, line_count)
    prepared_path = image_path.with_name(image_path.stem + "_prepared.png")
    prepared.save(prepared_path)
    backend = config.MRZ_OCR_BACKEND
    if backend in {"auto", "tesseract"} and tesseract_available():
        text = run_tesseract(prepared_path, config.MRZ_TESSERACT_CONFIG)
        if text.strip() or backend == "tesseract":
            return text, "tesseract"
    if backend == "tesseract":
        raise FileNotFoundError(f"Tesseract not found: {config.TESSERACT_CMD}")
    return rapidocr_mrz_lines(prepared), "rapidocr_onnxruntime_chunked"


def run_rapidocr(image_path: Path) -> str:
    lines = merge_boxes_into_lines(ocr_boxes(Image.open(image_path)))
    return "\n".join(line["text"].upper() for line in lines)


def run_field_ocr(image_path: Path) -> tuple[str, str]:
    if tesseract_available():
        return run_tesseract(image_path, config.FIELD_TESSERACT_CONFIG, config.FIELD_TESSERACT_LANG), "tesseract"
    return run_rapidocr(image_path), "rapidocr_onnxruntime"
