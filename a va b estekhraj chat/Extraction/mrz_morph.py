from __future__ import annotations

from typing import Any

import cv2
import numpy as np
from PIL import Image

import config

CARD_ASPECT = 85.6 / 53.98


# Order a quadrilateral as top-left, top-right, bottom-right, bottom-left.
def order_quad(points: np.ndarray) -> np.ndarray:
    points = points.reshape(4, 2).astype("float32")
    total = points.sum(axis=1)
    diff = np.diff(points, axis=1).ravel()
    return np.array([points[np.argmin(total)], points[np.argmin(diff)], points[np.argmax(total)], points[np.argmax(diff)]], dtype="float32")


# The card is the largest bright four-sided shape in the frame. Finding it first is what
# keeps the MRZ search off carpet, wood grain and table edges.
def find_card_quad(image: Image.Image) -> np.ndarray | None:
    scale = config.CARD_WORK_WIDTH / max(image.width, 1)
    small = np.asarray(image.convert("L").resize((config.CARD_WORK_WIDTH, max(int(image.height * scale), 1)), Image.LANCZOS))
    edges = cv2.Canny(cv2.bilateralFilter(small, 9, 75, 75), 30, 120)
    edges = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    frame_area = small.shape[0] * small.shape[1]
    for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:8]:
        area = cv2.contourArea(contour)
        if area < frame_area * config.CARD_MIN_AREA_RATIO or area > frame_area * 0.98:
            continue
        approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            return order_quad(approx) / scale
    return None


# Flatten the card to a front-on rectangle, removing rotation and perspective at once.
def warp_card(image: Image.Image, quad: np.ndarray) -> Image.Image:
    width = int(max(np.linalg.norm(quad[1] - quad[0]), np.linalg.norm(quad[2] - quad[3])))
    height = int(max(np.linalg.norm(quad[3] - quad[0]), np.linalg.norm(quad[2] - quad[1])))
    if width < height:
        quad = np.roll(quad, -1, axis=0)
        width, height = height, width
    height = max(int(width / CARD_ASPECT), 1)
    target = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype="float32")
    warped = cv2.warpPerspective(np.asarray(image.convert("RGB")), cv2.getPerspectiveTransform(quad, target), (width, height))
    return Image.fromarray(warped)


# Text-response mask used for skew estimation. Raw ink is dominated by the card outline
# and photo block; blackhat keeps only small dark glyph strokes.
def skew_mask(image: Image.Image) -> np.ndarray:
    width, height = image.size
    inner = image.crop((int(width * 0.08), int(height * 0.08), int(width * 0.92), int(height * 0.92)))
    scale = 400 / max(inner.width, 1)
    small = np.asarray(inner.convert("L").resize((400, max(int(inner.height * scale), 1)), Image.LANCZOS))
    blackhat = cv2.morphologyEx(cv2.GaussianBlur(small, (3, 3), 0), cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (13, 5)))
    return cv2.threshold(blackhat, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]


# Straight text puts its ink in a few sharp rows; slanted text smears it across many.
def profile_sharpness(mask: np.ndarray, angle: float) -> float:
    matrix = cv2.getRotationMatrix2D((mask.shape[1] / 2, mask.shape[0] / 2), angle, 1.0)
    rotated = cv2.warpAffine(mask, matrix, (mask.shape[1], mask.shape[0]), flags=cv2.INTER_NEAREST)
    return float(np.diff(rotated.sum(axis=1).astype("float64")).var())


# Coarse sweep then a fine pass, because a residual degree or two still smears a long line.
def skew_angle(image: Image.Image) -> float:
    mask = skew_mask(image)
    if not np.any(mask):
        return 0.0
    limit = config.MRZ_MAX_DESKEW_DEGREES
    coarse = max(np.arange(-limit, limit + 0.5, 1.0), key=lambda angle: profile_sharpness(mask, angle))
    fine = max(np.arange(coarse - 1.0, coarse + 1.01, 0.2), key=lambda angle: profile_sharpness(mask, angle))
    return float(fine)


# Rotate an image onto its text baseline, leaving it untouched when already straight.
def deskew(image: Image.Image) -> tuple[Image.Image, float]:
    angle = skew_angle(image)
    if abs(angle) < 1.0:
        return image, 0.0
    # OpenCV's trial angle is already the correction angle; PIL uses the same sign.
    return image.rotate(angle, expand=True, resample=Image.BICUBIC, fillcolor=(255, 255, 255)), angle


# Work at a fixed width so kernel sizes mean the same thing on every input resolution.
def to_working_gray(image: Image.Image) -> tuple[np.ndarray, float]:
    scale = config.MRZ_WORK_WIDTH / max(image.width, 1)
    size = (config.MRZ_WORK_WIDTH, max(int(image.height * scale), 1))
    return np.asarray(image.convert("L").resize(size, Image.LANCZOS)), scale


# Drop vertical rules such as the card outline; during horizontal closing they bridge the
# end of one text line into the next and merge the whole card into one component.
def remove_frame_lines(mask: np.ndarray) -> np.ndarray:
    vertical = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 25)))
    return cv2.subtract(mask, cv2.dilate(vertical, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1))))


# Blackhat brings out dark glyphs on a light card; closing sideways fuses a line of
# characters into one blob, because MRZ has no word gaps to break it apart.
def band_mask(gray: np.ndarray) -> np.ndarray:
    smooth = cv2.GaussianBlur(gray, (3, 3), 0)
    blackhat = cv2.morphologyEx(smooth, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (21, 7)))
    gradient = np.absolute(cv2.Sobel(blackhat, cv2.CV_32F, 1, 0, ksize=-1))
    span = gradient.max() - gradient.min()
    gradient = np.uint8(255 * (gradient - gradient.min()) / (span if span else 1))
    closed = cv2.morphologyEx(gradient, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (21, 5)))
    mask = cv2.threshold(closed, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    return cv2.morphologyEx(remove_frame_lines(mask), cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (31, 3)))


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


# Two or three blobs stacked closely, of near-identical width and height. Prose lines vary
# in width because they end where the sentence ends; MRZ lines never do.
def blob_groups(blobs: list[dict[str, Any]], page_height: int) -> list[tuple[float, list[dict[str, Any]]]]:
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
            depth = (group[0]["y"] + group[-1]["y"]) / (2 * max(page_height, 1))
            score = size * 0.5 + (min(widths) / max(widths)) * 2.0 + min(float(np.mean([item["ink"] for item in group])) / 0.25, 1.0) + (0.4 if depth > 0.45 else 0.0)
            groups.append((score, group))
    return sorted(groups, key=lambda item: item[0], reverse=True)


# Candidate MRZ bands inside one already-flattened image, in that image's coordinates.
def bands_in_image(image: Image.Image) -> list[dict[str, Any]]:
    gray, scale = to_working_gray(image)
    contours, _ = cv2.findContours(band_mask(gray), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    blobs = [blob for blob in (line_blob(contour, gray) for contour in contours) if blob]
    results = []
    for score, group in blob_groups(blobs, gray.shape[0]):
        pad_x = int(max(item["w"] for item in group) * config.MRZ_BOX_PAD_RATIO)
        pad_y = int(max(item["h"] for item in group) * 0.9)
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


# Search inside the flattened card when one is found, and only fall back to the raw frame
# when it is not. On a photo of a card on carpet, the fallback is what finds carpet.
def candidate_surfaces(image: Image.Image) -> list[tuple[str, Image.Image]]:
    candidates = []
    quad = find_card_quad(image)
    if quad is not None:
        candidates.append(("card", warp_card(image, quad)))
    candidates.append(("frame", image))
    surfaces = []
    for name, surface in candidates:
        straight, angle = deskew(surface)
        label = f"{name}_deskew" if angle else name
        surfaces.extend([(label, straight), (f"{label}_180", straight.rotate(180, expand=True))])
    return surfaces


# Candidate bands, each carrying the surface it came from so the caller crops the right image.
def morphological_candidates(image: Image.Image) -> list[dict[str, Any]]:
    results = []
    for name, surface in candidate_surfaces(image):
        bonus = config.MRZ_CARD_SURFACE_BONUS if name.startswith("card") else 0.0
        for band in bands_in_image(surface):
            results.append({**band, "surface": name, "surface_image": surface, "morph_score": round(band["morph_score"] + bonus, 3)})
    return sorted(results, key=lambda item: item["morph_score"], reverse=True)
