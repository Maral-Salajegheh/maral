from __future__ import annotations

from pathlib import Path
from zipfile import ZipFile

import fitz
import pandas as pd
from PIL import Image, ImageStat

from pipeline_utils import sha256_file, utc_now_iso


INVENTORY_PATH = Path("outputs/document_inventory.csv")
OUTPUT_PATH = Path("outputs/page_inventory.csv")
RENDER_ROOT = Path("Ausweiskopie/RenderedPages")

# Baseline DPI for normal A4-like pages. Confirmed pages should later be
# rendered at 300 DPI for field extraction.
SCREENING_DPI = 200
BASE_ZOOM = SCREENING_DPI / 72.0

# Adaptive zoom targets in pixels.
#
# Pages containing Ausweiskopien are often not A4: a directly scanned ID
# card produces a card-format page (~54x86 mm) that renders far too small
# at a fixed DPI and then fails the minimum-size quality check without
# ever reaching SecureGPT. The zoom is therefore adjusted per page:
# - upscale so the short side reaches at least TARGET_MIN_SHORT_SIDE_PX
# - downscale so the long side does not exceed TARGET_MAX_LONG_SIDE_PX
# Normal A4 pages at 200 DPI (about 1654x2339 px) are unaffected.
TARGET_MIN_SHORT_SIDE_PX = 1024
TARGET_MAX_LONG_SIDE_PX = 2400

MIN_IMAGE_WIDTH = 500
MIN_IMAGE_HEIGHT = 500
BLANK_WHITE_PIXEL_RATIO = 0.995
LOW_CONTRAST_STDDEV = 8.0


def safe_folder_name(value: str) -> str:
    """Create a filesystem-safe folder name from a PDF stem."""
    allowed = set(
        "abcdefghijklmnopqrstuvwxyz"
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        "0123456789-_"
    )

    cleaned = "".join(
        character if character in allowed else "_"
        for character in value
    ).strip("_")

    return cleaned or "document"


def compute_page_zoom(
    page_width_pt: float,
    page_height_pt: float,
) -> float:
    """
    Compute the render zoom for one page.

    Starts from the baseline screening zoom and adjusts it so the
    rendered image has a short side of at least
    TARGET_MIN_SHORT_SIDE_PX and a long side of at most
    TARGET_MAX_LONG_SIDE_PX.
    """
    if page_width_pt <= 0 or page_height_pt <= 0:
        return BASE_ZOOM

    zoom = BASE_ZOOM

    short_side_px = min(page_width_pt, page_height_pt) * zoom
    long_side_px = max(page_width_pt, page_height_pt) * zoom

    if short_side_px < TARGET_MIN_SHORT_SIDE_PX:
        zoom *= TARGET_MIN_SHORT_SIDE_PX / short_side_px
        long_side_px = max(page_width_pt, page_height_pt) * zoom

    if long_side_px > TARGET_MAX_LONG_SIDE_PX:
        zoom *= TARGET_MAX_LONG_SIDE_PX / long_side_px

    return zoom


def check_image_quality(
    image_path: Path,
) -> tuple[str, int | None, int | None, int | None]:
    """
    Catch clearly unusable renders before SecureGPT.

    This is a small technical check, not document classification.
    """
    try:
        with Image.open(image_path) as image:
            image.load()
            width, height = image.size
            grayscale = image.convert("L")

            standard_deviation = float(
                ImageStat.Stat(grayscale).stddev[0]
            )
            histogram = grayscale.histogram()
            pixel_count = width * height
            white_ratio = (
                sum(histogram[250:256]) / pixel_count
                if pixel_count
                else 1.0
            )

        file_size = image_path.stat().st_size

        if width < MIN_IMAGE_WIDTH or height < MIN_IMAGE_HEIGHT:
            return "too_small", width, height, file_size

        if white_ratio >= BLANK_WHITE_PIXEL_RATIO:
            return "likely_blank", width, height, file_size

        if standard_deviation < LOW_CONTRAST_STDDEV:
            return "low_contrast", width, height, file_size

        return "usable", width, height, file_size

    except Exception:
        return "corrupt", None, None, None


def render_pdf(
    pdf_bytes: bytes,
    masterindex_id: str,
    pdf_path_in_zip: str,
    pdf_order: int,
    starting_global_page: int,
) -> tuple[list[dict[str, object]], int]:
    """
    Render pages independently.

    One failed page does not erase successful page rows. The global page
    counter advances even for a failed page.
    """
    rows: list[dict[str, object]] = []
    pdf_stem = safe_folder_name(Path(pdf_path_in_zip).stem)

    pdf_output_dir = (
        RENDER_ROOT
        / masterindex_id
        / f"pdf_{pdf_order:03d}_{pdf_stem}"
    )
    pdf_output_dir.mkdir(parents=True, exist_ok=True)

    global_page_number = starting_global_page

    with fitz.open(stream=pdf_bytes, filetype="pdf") as document:
        for source_page_index in range(document.page_count):
            source_page_number = source_page_index + 1
            image_path = (
                pdf_output_dir
                / f"page_{source_page_number:04d}.png"
            )

            row: dict[str, object] = {
                "masterindex_id": masterindex_id,
                "pdf_path_in_zip": pdf_path_in_zip,
                "pdf_order_in_masterindex": pdf_order,
                "page_number": global_page_number,
                "source_page_number": source_page_number,
                "rendered_at_utc": utc_now_iso(),
            }

            try:
                page = document.load_page(source_page_index)

                page_rect = page.rect
                zoom = compute_page_zoom(
                    page_width_pt=page_rect.width,
                    page_height_pt=page_rect.height,
                )
                effective_dpi = round(zoom * 72.0, 1)

                matrix = fitz.Matrix(zoom, zoom)
                pixmap = page.get_pixmap(matrix=matrix, alpha=False)
                pixmap.save(str(image_path))

                (
                    quality_status,
                    width,
                    height,
                    file_size,
                ) = check_image_quality(image_path)

                row.update({
                    "page_width_pt": round(page_rect.width, 2),
                    "page_height_pt": round(page_rect.height, 2),
                    "render_dpi": effective_dpi,
                    "image_path": str(image_path),
                    "image_sha256": sha256_file(image_path),
                    "image_width": width,
                    "image_height": height,
                    "image_size_bytes": file_size,
                    "quality_status": quality_status,
                    "render_status": "success",
                    "error": None,
                })

            except Exception as error:
                image_path.unlink(missing_ok=True)

                row.update({
                    "page_width_pt": None,
                    "page_height_pt": None,
                    "render_dpi": None,
                    "image_path": None,
                    "image_sha256": None,
                    "image_width": None,
                    "image_height": None,
                    "image_size_bytes": None,
                    "quality_status": "render_failed",
                    "render_status": "failed",
                    "error": str(error),
                })

            rows.append(row)
            global_page_number += 1

    return rows, global_page_number


def main() -> None:
    if not INVENTORY_PATH.exists():
        raise FileNotFoundError(
            f"Inventory not found: {INVENTORY_PATH}. "
            "Run 01_scan_downloaded_documents.py first."
        )

    inventory = pd.read_csv(INVENTORY_PATH, dtype=object)
    ready_inventory = inventory[
        inventory["status"] == "ready"
    ].copy()

    ready_inventory["pdf_order_in_masterindex"] = (
        ready_inventory["pdf_order_in_masterindex"].astype(int)
    )

    ready_inventory = ready_inventory.sort_values(
        [
            "masterindex_id",
            "pdf_order_in_masterindex",
            "pdf_path_in_zip",
        ]
    )

    all_rows: list[dict[str, object]] = []

    for masterindex_id, group in ready_inventory.groupby(
        "masterindex_id",
        sort=True,
    ):
        global_page_number = 1

        for _, inventory_row in group.iterrows():
            zip_path = Path(str(inventory_row["zip_path"]))
            pdf_path_in_zip = str(
                inventory_row["pdf_path_in_zip"]
            )
            pdf_order = int(
                inventory_row["pdf_order_in_masterindex"]
            )

            try:
                with ZipFile(zip_path, "r") as zip_file:
                    pdf_bytes = zip_file.read(pdf_path_in_zip)

                page_rows, global_page_number = render_pdf(
                    pdf_bytes=pdf_bytes,
                    masterindex_id=str(masterindex_id),
                    pdf_path_in_zip=pdf_path_in_zip,
                    pdf_order=pdf_order,
                    starting_global_page=global_page_number,
                )
                all_rows.extend(page_rows)

                successful_pages = sum(
                    row["render_status"] == "success"
                    for row in page_rows
                )
                failed_pages = len(page_rows) - successful_pages

                print(
                    f"Rendered {masterindex_id}: {pdf_path_in_zip} "
                    f"({successful_pages} success, "
                    f"{failed_pages} failed)"
                )

            except Exception as error:
                # The PDF could not be opened, so page count is unknown.
                all_rows.append({
                    "masterindex_id": masterindex_id,
                    "pdf_path_in_zip": pdf_path_in_zip,
                    "pdf_order_in_masterindex": pdf_order,
                    "page_number": None,
                    "source_page_number": None,
                    "rendered_at_utc": utc_now_iso(),
                    "page_width_pt": None,
                    "page_height_pt": None,
                    "render_dpi": None,
                    "image_path": None,
                    "image_sha256": None,
                    "image_width": None,
                    "image_height": None,
                    "image_size_bytes": None,
                    "quality_status": "pdf_open_failed",
                    "render_status": "failed",
                    "error": str(error),
                })

                print(
                    f"PDF open failed for {masterindex_id}: "
                    f"{pdf_path_in_zip}"
                )

    page_inventory = pd.DataFrame(all_rows)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    page_inventory.to_csv(OUTPUT_PATH, index=False)

    successful_pages = int(
        (page_inventory["render_status"] == "success").sum()
    )
    failed_rows = int(
        (page_inventory["render_status"] == "failed").sum()
    )
    unusable_images = int(
        (
            (page_inventory["render_status"] == "success")
            & (page_inventory["quality_status"] != "usable")
        ).sum()
    )

    print(f"Rendered pages: {successful_pages}")
    print(f"Failed page/PDF rows: {failed_rows}")
    print(f"Images blocked by quality check: {unusable_images}")
    print(f"Page inventory written to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()