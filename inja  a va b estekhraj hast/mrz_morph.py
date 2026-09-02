from __future__ import annotations

from typing import Any

import cv2
import numpy as np
from PIL import Image

from Life.Extraction import config


# Work at a fixed width so kernel sizes mean the same thing on every scan resolution.
def to_working_gray(image: Image.Image) -> tuple[np.ndarray, float]:
    scale = config.MRZ_WORK_WIDTH / max(image.width, 1)
    size = (config.MRZ_WORK_WIDTH, max(int(image.height * scale), 1))
    resized = image.convert("L").resize(size, Image.LANCZOS)
    return np.asarray(resized), scale


# Blackhat brings out dark glyphs on a light card; the wide kernel then fuses a
# line of characters into one solid blob, because MRZ has no word gaps to break it.
def band_mask(gray: np.ndarray) -> np.ndarray:
    smooth = cv2.GaussianBlur(gray, (3, 3), 0)
    blackhat = cv2.morphologyEx(smooth, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (21, 7)))
    gradient = np.absolute(cv2.Sobel(blackhat, cv2.CV_32F, 1, 0, ksize=-1))
    span = gradient.max() - gradient.min()
    gradient = np.uint8(255 * (gradient - gradient.min()) / (span if span else 1))
    closed = cv2.morphologyEx(gradient, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (21, 5)))
    mask = cv2.threshold(closed, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    mask = remove_frame_lines(mask)
    # Close horizontally only, so stacked lines stay separate blobs and can be grouped.
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (31, 3)))


# Drop vertical rules such as the card outline. They bridge the end of one text line
# into the next during horizontal closing and merge the whole card into one component.
# Only vertical lines are removed: a horizontal rule looks exactly like a line of text.
def remove_frame_lines(mask: np.ndarray) -> np.ndarray:
    vertical = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 25)))
    return cv2.subtract(mask, cv2.dilate(vertical, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1))))


# One text-line blob, kept only if it is wide, short and solidly filled.
def line_blob(contour: np.ndarray, gray: np.ndarray) -> dict[str, Any] | None:
    x, y, width, height = cv2.boundingRect(contour)
    if width < gray.shape[1] * 0.12 or height < 5:
        return None
    aspect = width / height
    fill = cv2.contourArea(contour) / max(width * height, 1)
    if not 6.0 <= aspect <= 60.0 or fill < 0.40:
        return None
    ink = float((gray[y : y + height, x : x + width] < np.percentile(gray, 40)).mean())
    return {"x": x, "y": y, "w": width, "h": height, "aspect": aspect, "fill": fill, "ink": ink}


# Two or three blobs stacked closely, of near-identical width and height. Prose lines
# vary in width because they end wherever the sentence ends; MRZ lines never do.
def blob_groups(blobs: list[dict[str, Any]]) -> list[tuple[float, list[dict[str, Any]]]]:
    blobs = sorted(blobs, key=lambda item: item["y"])
    groups = []
    for size in (3, 2):
        for start in range(len(blobs) - size + 1):
            group = blobs[start : start + size]
            widths = [item["w"] for item in group]
            heights = [item["h"] for item in group]
            gaps = [group[i + 1]["y"] - (group[i]["y"] + group[i]["h"]) for i in range(size - 1)]
            if min(gaps) < -2 or max(gaps) > max(heights) * 2.0:
                continue
            if min(widths) / max(widths) < 0.85 or min(heights) / max(heights) < 0.55:
                continue
            overlap = min(item["x"] + item["w"] for item in group) - max(item["x"] for item in group)
            if overlap < min(widths) * 0.7:
                continue
            uniformity = min(widths) / max(widths)
            score = size * 0.5 + uniformity * 2.0 + min(float(np.mean([item["ink"] for item in group])) / 0.25, 1.0)
            groups.append((score, group))
    return sorted(groups, key=lambda item: item[0], reverse=True)


# Candidate MRZ bands in original-image coordinates, best first.
def morphological_candidates(image: Image.Image) -> list[dict[str, Any]]:
    gray, scale = to_working_gray(image)
    contours, _ = cv2.findContours(band_mask(gray), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    blobs = [blob for blob in (line_blob(contour, gray) for contour in contours) if blob]
    results = []
    for score, group in blob_groups(blobs):
        height = max(item["h"] for item in group)
        pad_x, pad_y = int(max(item["w"] for item in group) * 0.03), int(height * 0.9)
        results.append({
            "x0": max(int((min(item["x"] for item in group) - pad_x) / scale), 0),
            "y0": max(int((min(item["y"] for item in group) - pad_y) / scale), 0),
            "x1": min(int((max(item["x"] + item["w"] for item in group) + pad_x) / scale), image.width),
            "y1": min(int((max(item["y"] + item["h"] for item in group) + pad_y) / scale), image.height),
            "morph_score": round(score, 3),
            "line_count": len(group),
            "detector": "morphological",
        })
    return results
