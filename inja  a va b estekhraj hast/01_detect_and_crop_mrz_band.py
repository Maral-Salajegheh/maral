from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from Life.Extraction import config
from Life.Extraction.mrz_morph import morphological_candidates
from Life.Extraction.mrz_utils import normalize_line
from Life.Extraction.utils import append_jsonl, int_value, load_page_metadata, merge_boxes_into_lines, ocr_boxes, page_key, read_csv, read_latest_jsonl, resolve_image_path, utc_now_iso, write_csv

METADATA_FIELDS = ["pdf_path_in_zip", "source_page_number", "image_path", "image_sha256", "render_status", "quality_status"]


# Fill in corpus metadata for a page, whether or not the input CSV already carries an image path.
def resolve_page(row: dict[str, str], metadata: dict[tuple[str, int], list[dict[str, str]]]) -> dict[str, Any]:
    enriched = dict(row)
    key = (row["masterindex_id"], int_value(row["page_number"]))
    usable = [item for item in metadata.get(key, []) if item.get("image_path")]
    unique_paths = sorted({item["image_path"] for item in usable})
    if len(unique_paths) == 1:
        source = next(item for item in usable if item["image_path"] == unique_paths[0])
        for field in METADATA_FIELDS:
            if not enriched.get(field):
                enriched[field] = source.get(field, "")
    path_text = enriched.get("resolved_image_path") or enriched.get("image_path") or ""
    if not path_text:
        raise ValueError(f"Cannot uniquely resolve rendered PNG for {key}; candidates={len(unique_paths)}")
    enriched["image_path"] = path_text
    enriched["resolved_image_path"] = str(resolve_image_path(path_text))
    return enriched


def load_g07_candidates(input_csv: Path) -> list[dict[str, Any]]:
    metadata = load_page_metadata()
    rows = []
    for row in read_csv(input_csv):
        if row.get("predicted_page_sst") != "G07":
            continue
        try:
            record = resolve_page(row, metadata)
            record.update({"candidate_status": "success", "needs_human_review": False, "error": ""})
        except Exception as error:
            record = {**row, "candidate_status": "failed", "needs_human_review": True, "error": str(error)}
        rows.append(record)
    return rows


# How much one OCR line looks like an MRZ line: uppercase, close to a canonical width,
# and carrying filler. Lowercase rules out ordinary letter text and addresses. The raw
# text must also show a real filler glyph or land almost exactly on a canonical width;
# carpet, wood grain and portrait texture satisfy neither, so they can never be cropped.
def line_score(text: str) -> float:
    letters = [char for char in text if char.isalpha()]
    if letters and sum(char.islower() for char in letters) / len(letters) > 0.2:
        return 0.0
    normalized = normalize_line(text)
    near_exact = min(abs(len(normalized) - width) for width in config.MRZ_CANONICAL_WIDTHS) <= 3
    if not any(char in "<>" for char in text) and not near_exact:
        return 0.0
    line = normalize_line(text)
    length = len(line)
    if length < 20:
        return 0.0
    filler = line.count("<") / length
    if filler < 0.05:
        return 0.0
    gap = min(abs(length - width) for width in config.MRZ_CANONICAL_WIDTHS)
    return max(0.0, 1.0 - gap / 12.0) + min(filler / 0.20, 1.0)


# A band scores as its weakest-to-average member; one non-MRZ line disqualifies the whole group.
def band_score(group: list[dict[str, Any]]) -> float:
    scores = [line_score(line["text"]) for line in group]
    return 0.0 if min(scores) <= 0 else sum(scores) / len(scores)


# Runs of two or three stacked lines that all look like MRZ lines, best first.
def band_candidates(lines: list[dict[str, Any]], width: int, height: int) -> list[dict[str, Any]]:
    candidates = []
    for size in (3, 2):
        for start in range(len(lines) - size + 1):
            group = lines[start : start + size]
            median_height = float(np.median([line["y1"] - line["y0"] for line in group]))
            gaps = [group[index + 1]["y0"] - group[index]["y1"] for index in range(size - 1)]
            if max(gaps) > median_height * 2.0:
                continue
            score = band_score(group)
            if score < config.MRZ_MIN_GROUP_SCORE:
                continue
            pad_x = max(12, int(width * 0.01))
            # Generous vertical margin: if one MRZ line scored too low to join the band,
            # it still lands inside the crop and stage 02 can recover it.
            # Generous vertical margin: when one MRZ line is too faint to join the group,
            # it still lands inside the crop and stage 02 can recover it.
            pad_y = max(12, int(median_height * 2.2))
            candidates.append({
                "x0": max(0, int(min(line["x0"] for line in group)) - pad_x),
                "y0": max(0, int(min(line["y0"] for line in group)) - pad_y),
                "x1": min(width, int(max(line["x1"] for line in group)) + pad_x),
                "y1": min(height, int(max(line["y1"] for line in group)) + pad_y),
                "line_count": size,
                "score": round(score, 3),
                "mrz_detector_text": "\n".join(line["text"] for line in group),
            })
    # Prefer the longer band: a three-line group only survives if all three lines look like MRZ.
    return sorted(candidates, key=lambda item: (item["line_count"], item["score"]), reverse=True)


# Morphology proposes bands from texture alone; OCR then confirms the crop really reads
# as MRZ. This finds bands the OCR-first detector misses, because it does not need to
# recognize the characters in order to notice the band.
def verified_morph_band(page: Image.Image) -> dict[str, Any] | None:
    for candidate in morphological_candidates(page)[: config.MRZ_MORPH_MAX_CANDIDATES]:
        surface = candidate.pop("surface_image", page)
        crop = surface.crop((candidate["x0"], candidate["y0"], candidate["x1"], candidate["y1"]))
        lines = merge_boxes_into_lines(ocr_boxes(crop))
        scores = [line_score(line["text"]) for line in lines]
        confirmed = [score for score in scores if score > 0]
        if len(confirmed) < 2:
            continue
        score = sum(confirmed) / len(confirmed)
        if score >= config.MRZ_MIN_GROUP_SCORE:
            return {**candidate, "score": round(score, 3), "line_count": len(confirmed),
                    "crop_image": crop, "surface_page": surface,
                    "mrz_detector_text": "\n".join(line["text"] for line in lines)}
    return None


# Locate the MRZ band, trying page rotations until one scores well enough.
def detect_mrz_band(image_path: Path) -> tuple[Image.Image, Image.Image, dict[str, Any]]:
    source = Image.open(image_path).convert("RGB")
    best: tuple[Image.Image, dict[str, Any]] | None = None
    for angle in (0, 90, 180, 270):
        rotated = source.rotate(angle, expand=True) if angle else source
        morph = verified_morph_band(rotated)
        # The whole-frame OCR detector is the one that finds carpet and faces on photographs,
        # so it only runs when the card-first detector found nothing at all.
        candidates = [morph] if morph else band_candidates(merge_boxes_into_lines(ocr_boxes(rotated)), rotated.width, rotated.height)
        rank = (candidates[0]["line_count"], candidates[0]["score"]) if candidates else None
        if rank and (best is None or rank > (best[1]["line_count"], best[1]["score"])):
            best = (rotated, {**candidates[0], "rotation_degrees": angle})
        if best is not None and best[1]["score"] >= config.MRZ_GOOD_SCORE:
            break
    if best is None:
        raise ValueError("No MRZ-like line band detected on any rotation")
    page, meta = best
    # The band coordinates belong to the surface they were found on, which may be a
    # flattened, deskewed card rather than the original page. Later stages must crop from
    # that same surface or they slice the wrong rectangle out of the original image.
    surface = meta.pop("surface_page", None) or page
    crop = meta.pop("crop_image", None) or surface.crop((meta["x0"], meta["y0"], meta["x1"], meta["y1"]))
    return surface, crop, {"detector": "rapidocr_lines", **meta, "detector_version": config.MRZ_DETECTOR_VERSION}


# Save the page with the chosen band outlined, so a failure can be inspected visually.
def save_overlay(page: Image.Image, meta: dict[str, Any], out_path: Path) -> None:
    canvas = page.copy()
    ImageDraw.Draw(canvas).rectangle([meta["x0"], meta["y0"], meta["x1"], meta["y1"]], outline=(255, 0, 0), width=6)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def crop_stem(record: dict[str, Any]) -> str:
    return f"{record.get('image_sha256') or record['masterindex_id']}_{record['page_number']}"


def process_candidate(record: dict[str, Any]) -> dict[str, Any]:
    if record.get("candidate_status") != "success":
        return {**record, "status": "failed", "needs_human_review": True, "error": record.get("error") or "candidate resolution failed", "processed_at_utc": utc_now_iso()}
    page, crop, meta = detect_mrz_band(Path(str(record["resolved_image_path"])))
    out_path = config.MRZ_CROP_DIR / f"{crop_stem(record)}_{config.MRZ_DETECTOR_VERSION}.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    crop.save(out_path)
    surface_path = config.SURFACE_DIR / f"{crop_stem(record)}_surface.png"
    surface_path.parent.mkdir(parents=True, exist_ok=True)
    page.save(surface_path)
    overlay_path = config.MRZ_DEBUG_DIR / f"{crop_stem(record)}_overlay.png"
    save_overlay(page, meta, overlay_path)
    return {**record, **meta, "mrz_crop_path": str(out_path), "mrz_surface_path": str(surface_path), "mrz_overlay_path": str(overlay_path), "status": "success", "needs_human_review": False, "error": "", "processed_at_utc": utc_now_iso()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=config.INPUT_PAGES_CSV)
    parser.add_argument("--candidates-output", type=Path, default=config.CANDIDATES_CSV)
    parser.add_argument("--output", type=Path, default=config.MRZ_CROPS_JSONL)
    args = parser.parse_args()
    candidates = load_g07_candidates(args.input)
    fields = list(dict.fromkeys(field for row in candidates for field in row))
    write_csv(args.candidates_output, candidates, fields)
    latest = read_latest_jsonl(args.output, page_key)
    for record in sorted(candidates, key=lambda item: (item.get("masterindex_id", ""), int_value(item.get("page_number")))):
        previous = latest.get(page_key(record)) if record.get("candidate_status") == "success" else None
        if previous and previous.get("status") == "success" and previous.get("detector_version") == config.MRZ_DETECTOR_VERSION:
            continue
        try:
            output = process_candidate(record)
        except Exception as error:
            output = {**record, "status": "failed", "needs_human_review": True, "error": str(error), "detector_version": config.MRZ_DETECTOR_VERSION, "processed_at_utc": utc_now_iso()}
        append_jsonl(args.output, output)
    print(f"Wrote candidates: {args.candidates_output}")
    print(f"Wrote MRZ crop cache: {args.output}")
    print(f"Band overlays for inspection: {config.MRZ_DEBUG_DIR}")


if __name__ == "__main__":
    main()
