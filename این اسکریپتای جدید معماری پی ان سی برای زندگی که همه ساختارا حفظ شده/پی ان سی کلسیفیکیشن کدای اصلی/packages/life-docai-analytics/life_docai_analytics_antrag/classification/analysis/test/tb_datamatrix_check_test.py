"""Scan the rendered AMB and AMU page images for Data Matrix codes.

Run from the life-docai repository root with:
    pixi run -e dev python packages/life-docai-analytics/life_docai_analytics_antrag/classification/analysis/test/tb_datamatrix_check_test.py

Expected layout next to this script:
    AMB_AMU/Rendered_AMB_AMU/AMB/...
    AMB_AMU/Rendered_AMB_AMU/AMU/...
    AMB_AMU/page_inventory.csv
"""

import base64
import csv
import gzip
import re
import zlib
from collections import defaultdict
from pathlib import Path

import zxingcpp
from PIL import Image, ImageFilter, ImageOps


HERE = Path(__file__).resolve().parent
RENDER_ROOT = HERE / "AMB_AMU" / "Rendered_AMB_AMU"
INPUT_DIRS = {"AMB": RENDER_ROOT / "AMB", "AMU": RENDER_ROOT / "AMU"}
OUTPUT_DIR = HERE / "tb_datamatrix_results"
HEADER = "AXA03"
PARTNER, ZUSATZ, ANTRAG, ANGEBOT = "0000000100", "0000000101", "0000000102", "0000000103"
REQUIRED = {ANTRAG, ANGEBOT}
KNOWN = {PARTNER, ZUSATZ, ANTRAG, ANGEBOT}
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
MID_PATTERN = re.compile(r"MIH\d+", re.IGNORECASE)
PDF_PAGE_PATTERN = re.compile(r"(pdf_\d+_\d+)/(page_\d+\.[^.]+)$", re.IGNORECASE)
GROUP_PATTERN = re.compile(r"(?:^|/)(AMB|AMU)(?:/|$)", re.IGNORECASE)


def inventory_mid_map():
    """Map rendered-image identity to MID using page_inventory.csv.

    MIH identifiers live in pdf_path_in_zip, while rendered PNG paths use
    generated pdf_NNN_... folder names. The inventory connects the two.
    """
    inventory = HERE / "AMB_AMU" / "page_inventory.csv"
    if not inventory.is_file():
        raise FileNotFoundError(f"Page inventory not found: {inventory}")

    mapping = {}
    with inventory.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        required = {"pdf_path_in_zip", "image_path"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"{inventory} must contain columns {sorted(required)}")
        for row in reader:
            mid_match = MID_PATTERN.search(row["pdf_path_in_zip"] or "")
            image_path = row["image_path"] or ""
            identity = PDF_PAGE_PATTERN.search(image_path.replace("\\", "/"))
            group_match = GROUP_PATTERN.search(image_path.replace("\\", "/"))
            if mid_match and identity and group_match:
                key = group_match.group(1).upper(), identity.group(1).lower(), identity.group(2).lower()
                mid = mid_match.group(0)
                if key in mapping and mapping[key] != mid:
                    mapping[key] = "__AMBIGUOUS__"
                else:
                    mapping[key] = mid
    return mapping


def extract_mid(path, group, mid_map):
    """Resolve MID by matching the rendered PDF/page tokens to inventory."""
    match = PDF_PAGE_PATTERN.search(path.as_posix())
    if not match:
        return ""
    return mid_map.get((group.upper(), match.group(1).lower(), match.group(2).lower()), "")


def read_codes(image):
    return [code.text for code in zxingcpp.read_barcodes(
        image, formats=zxingcpp.BarcodeFormat.DataMatrix
    )]


def fallback_codes(image):
    variants = (
        ImageOps.autocontrast(image, cutoff=1),
        image.filter(ImageFilter.UnsharpMask(radius=2, percent=200, threshold=0)),
        image.resize((image.width * 2, image.height * 2), Image.Resampling.LANCZOS),
    )
    return [value for variant in variants for value in read_codes(variant)]


def parse_codes(codes):
    chunks, totals, other, conflicts = {}, {}, [], set()
    for value in codes:
        if not value.startswith(HEADER) or len(value) < 21 or not value[18:21].isdigit():
            other.append(value[:120])
            continue
        container, number, total, content = value[5:15], value[15:18], int(value[18:21]), value[21:]
        if number in chunks.setdefault(container, {}) and chunks[container][number] != content:
            conflicts.add(container)
        chunks[container][number] = content
        totals[container] = total
    complete = {container for container, parts in chunks.items() if len(parts) == totals[container]}
    return chunks, totals, complete, other, conflicts


def decode_payload(parts):
    try:
        joined = "".join(parts[key] for key in sorted(parts))
        return gzip.decompress(base64.b64decode(joined))
    except (ValueError, OSError, zlib.error, KeyError):
        return None


def scan_image(path):
    with Image.open(path) as source:
        image = source.convert("L")
        codes = read_codes(image)
        before_chunks, before_totals, before_complete, _, _ = parse_codes(codes)
        needs_fallback = not REQUIRED.issubset(before_complete) or before_complete != set(before_chunks)
        if needs_fallback:
            codes.extend(fallback_codes(image))
    codes = list(dict.fromkeys(codes))
    chunks, totals, complete, other, conflicts = parse_codes(codes)
    payloads = {c: decode_payload(chunks[c]) for c in complete}
    result = {
        "data_matrix_count": len(codes),
        "axa03_chunk_count": sum(len(parts) for parts in chunks.values()),
        "containers": " ".join(sorted(chunks)),
        "complete_containers": " ".join(sorted(complete)),
        "missing_chunks": " ".join(
            f"{c}:{totals[c] - len(chunks[c])}" for c in sorted(chunks) if c not in complete
        ),
        "undecodable_containers": " ".join(sorted(c for c, data in payloads.items() if data is None)),
        "conflicting_chunks": " ".join(sorted(conflicts)),
        "fallback_used": needs_fallback,
        "broken": not REQUIRED.issubset(complete),
        "other_data_matrix": " | ".join(other),
        "raw_codes": " | ".join(codes),
    }
    return result, payloads, codes


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    mid_map = inventory_mid_map()
    images = []
    for group, directory in INPUT_DIRS.items():
        if not directory.is_dir():
            raise FileNotFoundError(f"Rendered image folder not found: {directory}")
        images.extend((group, path) for path in sorted(directory.rglob("*"))
                      if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)

    if not images:
        raise FileNotFoundError(f"No supported images found under {RENDER_ROOT}")

    rows = []
    codes_by_mid = defaultdict(list)
    payload_dir = OUTPUT_DIR / "payloads"
    for i, (group, path) in enumerate(images, start=1):
        try:
            status, payloads, codes = scan_image(path)
            mid = extract_mid(path, group, mid_map)
            rows.append({"group": group, "mid": mid,
                         "image_path": str(path.relative_to(HERE)), **status})
            if mid and mid != "__AMBIGUOUS__":
                codes_by_mid[mid].extend(codes)
            for container, data in payloads.items():
                if data is not None:
                    payload_dir.mkdir(parents=True, exist_ok=True)
                    safe_rel = path.relative_to(RENDER_ROOT).with_suffix("")
                    out = payload_dir / safe_rel.parent / f"{safe_rel.name}_{container}.bin"
                    out.parent.mkdir(parents=True, exist_ok=True)
                    out.write_bytes(data)
        except Exception as exc:
            rows.append({"group": group, "mid": extract_mid(path, group, mid_map),
                         "image_path": str(path.relative_to(HERE)), "error": repr(exc)})
        if i % 100 == 0 or i == len(images):
            print(f"Scanned {i}/{len(images)} images")

    # Reassemble AXA03 chunks across every rendered page belonging to the same MID.
    document_status = {}
    for mid, codes in codes_by_mid.items():
        chunks, totals, complete, _, conflicts = parse_codes(codes)
        payloads = {c: decode_payload(chunks[c]) for c in complete}
        document_status[mid] = {
            "mid_complete_containers": " ".join(sorted(complete)),
            "mid_missing_chunks": " ".join(
                f"{c}:{totals[c] - len(chunks[c])}" for c in sorted(chunks) if c not in complete
            ),
            "mid_conflicting_chunks": " ".join(sorted(conflicts)),
            "mid_undecodable_containers": " ".join(sorted(c for c, data in payloads.items() if data is None)),
            "mid_broken": not REQUIRED.issubset(complete),
        }
        for container, data in payloads.items():
            if data is not None:
                out = OUTPUT_DIR / "payloads" / mid / f"{container}.bin"
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_bytes(data)
    for row in rows:
        row.update(document_status.get(row.get("mid", ""), {}))

    output_csv = OUTPUT_DIR / "data_matrix_scan.csv"
    columns = ["group", "mid", "image_path"] + sorted(
        {key for row in rows for key in row} - {"group", "mid", "image_path"}
    )
    with output_csv.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    found = sum(bool(row.get("data_matrix_count", 0)) for row in rows)
    print(f"Images scanned: {len(images)}; images with Data Matrix: {found}")
    print(f"Known AXA03 containers: {', '.join(KNOWN)}")
    print(f"Results: {output_csv}")


if __name__ == "__main__":
    main()
