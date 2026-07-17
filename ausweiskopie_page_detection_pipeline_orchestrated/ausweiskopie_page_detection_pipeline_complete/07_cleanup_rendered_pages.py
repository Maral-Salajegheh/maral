from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from pandas.errors import EmptyDataError

from pipeline_utils import utc_now_iso


PAGE_INVENTORY_PATH = Path("outputs/page_inventory.csv")
GROUND_TRUTH_PATH = Path(
    "outputs/ausweis_classification_ground_truth_pages.csv"
)
EXCLUDED_PATH = Path(
    "outputs/ausweis_classification_excluded_rows.csv"
)
OUTPUT_PATH = Path("outputs/rendered_page_deletion_audit.csv")
RENDER_ROOT = Path("Ausweiskopie/RenderedPages")

# Change only after the AXA retention period is approved.
ALLOW_DELETE = False
RETENTION_DAYS = 30


def parse_timestamp(value: object) -> datetime | None:
    """Parse a render timestamp."""
    try:
        parsed = datetime.fromisoformat(str(value))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed
    except (TypeError, ValueError):
        return None


def review_is_complete() -> bool:
    """Refuse cleanup while review rows remain incomplete."""
    if not GROUND_TRUTH_PATH.exists():
        return False

    if not EXCLUDED_PATH.exists():
        return True

    try:
        excluded = pd.read_csv(
            EXCLUDED_PATH,
            dtype=object,
            keep_default_na=False,
        )
    except EmptyDataError:
        return True

    if (
        excluded.empty
        or "exclusion_reason" not in excluded.columns
    ):
        return True

    return not (
        excluded["exclusion_reason"] == "incomplete_review"
    ).any()


def remove_empty_directories(root: Path) -> None:
    """Remove empty folders below the render root."""
    if not root.exists():
        return

    directories = [
        path for path in root.rglob("*") if path.is_dir()
    ]

    for directory in sorted(directories, reverse=True):
        try:
            directory.rmdir()
        except OSError:
            pass


def main() -> None:
    if not ALLOW_DELETE:
        raise RuntimeError(
            "Deletion is disabled. Set ALLOW_DELETE = True only "
            "after the AXA retention period is approved."
        )

    if not PAGE_INVENTORY_PATH.exists():
        raise FileNotFoundError(
            f"Page inventory not found: {PAGE_INVENTORY_PATH}"
        )

    if not review_is_complete():
        raise RuntimeError(
            "Review is incomplete. No rendered pages were deleted."
        )

    inventory = pd.read_csv(
        PAGE_INVENTORY_PATH,
        dtype=object,
    )
    now = datetime.now(timezone.utc)
    audit_rows: list[dict[str, object]] = []

    for _, row in inventory.iterrows():
        image_value = row.get("image_path")
        if pd.isna(image_value):
            continue

        image_path = Path(str(image_value))
        rendered_at = parse_timestamp(row.get("rendered_at_utc"))

        if rendered_at is None:
            status = "missing_render_timestamp"
        elif (now - rendered_at).days < RETENTION_DAYS:
            status = "retention_period_not_reached"
        elif not image_path.exists():
            status = "already_missing"
        else:
            try:
                image_path.unlink()
                status = "deleted"
            except Exception as error:
                status = f"delete_failed: {error}"

        audit_rows.append({
            "masterindex_id": row.get("masterindex_id"),
            "pdf_path_in_zip": row.get("pdf_path_in_zip"),
            "page_number": row.get("page_number"),
            "source_page_number": row.get("source_page_number"),
            "image_path": str(image_path),
            "image_sha256": row.get("image_sha256"),
            "retention_days": RETENTION_DAYS,
            "deletion_checked_at_utc": utc_now_iso(),
            "deletion_status": status,
        })

    remove_empty_directories(RENDER_ROOT)

    audit = pd.DataFrame(audit_rows)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    audit.to_csv(OUTPUT_PATH, index=False)

    deleted_count = int(
        (audit["deletion_status"] == "deleted").sum()
    )
    print(f"Rendered page images deleted: {deleted_count}")
    print(f"Deletion audit written to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
