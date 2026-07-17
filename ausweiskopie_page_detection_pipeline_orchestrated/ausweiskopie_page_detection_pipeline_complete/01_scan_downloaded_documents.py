from __future__ import annotations

from pathlib import Path
from zipfile import ZipFile

import pandas as pd


ZIP_PATH = Path("Ausweiskopie/Input/Ausweisekopie.zip")
OUTPUT_PATH = Path("outputs/document_inventory.csv")


def get_direct_masterindex_folders(zip_file: ZipFile) -> list[str]:
    """
    Return the direct top-level folders inside the ZIP.

    The folder name is preserved exactly and used as masterindex_id.
    """
    folders: set[str] = set()

    for member in zip_file.infolist():
        clean_path = member.filename.strip("/")
        if not clean_path:
            continue

        parts = clean_path.split("/")

        # A file inside MI123/... identifies MI123 as a direct folder.
        if len(parts) >= 2:
            folders.add(parts[0])
        elif member.is_dir():
            folders.add(parts[0])

    return sorted(folders)


def scan_zip(zip_path: Path) -> pd.DataFrame:
    """
    Create one inventory row per PDF.

    Folders without a PDF are kept as one row with status=missing_pdf.
    No SST filtering or SST lookup is performed.
    """
    if not zip_path.exists():
        raise FileNotFoundError(f"ZIP file not found: {zip_path}")

    rows: list[dict[str, object]] = []

    with ZipFile(zip_path, "r") as zip_file:
        masterindex_folders = get_direct_masterindex_folders(zip_file)
        file_members = [
            member.filename
            for member in zip_file.infolist()
            if not member.is_dir()
        ]

        for masterindex_id in masterindex_folders:
            folder_prefix = f"{masterindex_id}/"

            pdf_paths = sorted(
                path
                for path in file_members
                if path.startswith(folder_prefix)
                and Path(path).suffix.lower() == ".pdf"
            )

            if not pdf_paths:
                rows.append({
                    "masterindex_id": masterindex_id,
                    "zip_path": str(zip_path),
                    "pdf_path_in_zip": None,
                    "pdf_filename": None,
                    "pdf_order_in_masterindex": None,
                    "pdf_count_in_masterindex": 0,
                    "status": "missing_pdf",
                })
                continue

            pdf_count = len(pdf_paths)

            for pdf_order, pdf_path_in_zip in enumerate(
                pdf_paths,
                start=1,
            ):
                rows.append({
                    "masterindex_id": masterindex_id,
                    "zip_path": str(zip_path),
                    "pdf_path_in_zip": pdf_path_in_zip,
                    "pdf_filename": Path(pdf_path_in_zip).name,
                    "pdf_order_in_masterindex": pdf_order,
                    "pdf_count_in_masterindex": pdf_count,
                    "status": "ready",
                })

    columns = [
        "masterindex_id",
        "zip_path",
        "pdf_path_in_zip",
        "pdf_filename",
        "pdf_order_in_masterindex",
        "pdf_count_in_masterindex",
        "status",
    ]

    return pd.DataFrame(rows, columns=columns)


def main() -> None:
    inventory = scan_zip(ZIP_PATH)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    inventory.to_csv(OUTPUT_PATH, index=False)

    ready_pdfs = int((inventory["status"] == "ready").sum())
    missing_pdf_folders = int(
        (inventory["status"] == "missing_pdf").sum()
    )
    masterindex_count = inventory["masterindex_id"].nunique()

    print(f"MasterIndex folders scanned: {masterindex_count}")
    print(f"PDF files found: {ready_pdfs}")
    print(f"Folders without a PDF: {missing_pdf_folders}")
    print(f"Inventory written to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
