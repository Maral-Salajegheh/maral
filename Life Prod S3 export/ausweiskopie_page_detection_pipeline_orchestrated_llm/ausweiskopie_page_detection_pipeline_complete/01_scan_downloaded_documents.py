from __future__ import annotations

from pathlib import Path
from zipfile import ZipFile

import pandas as pd


ZIP_PATH = Path("Ausweiskopie/Input/Ausweisekopie.zip")
OUTPUT_PATH = Path("outputs/document_inventory.csv")


def detect_wrapper_depth(pdf_paths: list[str]) -> int:
    """
    Detect wrapper folders created by compressing a parent folder.

    A leading path component is a wrapper when every PDF shares the
    same component at that depth AND stripping it still leaves every
    PDF inside a folder (masterindex_id/file.pdf). This stops the
    descent exactly at the MasterIndex level, also when the wrapper
    contains only one MasterIndex folder. A ZIP whose MasterIndex
    folders sit directly at the top level yields depth 0.
    """
    depth = 0

    while True:
        components: set[str] = set()

        for pdf_path in pdf_paths:
            parts = Path(pdf_path).parts

            # Stripping another component must leave at least
            # masterindex_id/file.pdf for every PDF.
            if len(parts) < depth + 3:
                return depth

            components.add(parts[depth])

        if len(components) != 1:
            return depth

        depth += 1


def scan_zip(zip_path: Path) -> pd.DataFrame:
    """
    Create one inventory row per PDF.

    masterindex_id is the folder directly containing the MasterIndex
    folders, after any wrapper folders from compression are skipped.
    pdf_path_in_zip always keeps the full original ZIP path.
    Folders without a PDF are kept as one row with status=missing_pdf.
    No SST filtering or SST lookup is performed.
    """
    if not zip_path.exists():
        raise FileNotFoundError(f"ZIP file not found: {zip_path}")

    rows: list[dict[str, object]] = []

    with ZipFile(zip_path, "r") as zip_file:
        file_members = [
            member.filename
            for member in zip_file.infolist()
            if not member.is_dir()
        ]

        pdf_members = [
            path
            for path in file_members
            if Path(path).suffix.lower() == ".pdf"
        ]

        wrapper_depth = (
            detect_wrapper_depth(pdf_members)
            if pdf_members
            else 0
        )

        wrapper_parts: tuple[str, ...] = (
            Path(pdf_members[0]).parts[:wrapper_depth]
            if wrapper_depth
            else ()
        )

        wrapper_prefix = (
            "/".join(wrapper_parts) + "/"
            if wrapper_parts
            else ""
        )

        # Collect MasterIndex folders directly below the wrapper.
        masterindex_folders: set[str] = set()

        for member in zip_file.infolist():
            clean_path = member.filename.strip("/")
            if not clean_path:
                continue

            parts = clean_path.split("/")

            if len(parts) <= wrapper_depth:
                continue

            if tuple(parts[:wrapper_depth]) != wrapper_parts:
                continue

            # A file inside <wrapper>/MI123/... identifies MI123.
            if len(parts) >= wrapper_depth + 2:
                masterindex_folders.add(parts[wrapper_depth])
            elif member.is_dir():
                masterindex_folders.add(parts[wrapper_depth])

        for masterindex_id in sorted(masterindex_folders):
            folder_prefix = (
                f"{wrapper_prefix}{masterindex_id}/"
            )

            pdf_paths = sorted(
                path
                for path in pdf_members
                if path.startswith(folder_prefix)
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

        if wrapper_prefix:
            print(
                f"Wrapper folder(s) skipped: {wrapper_prefix}"
            )

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