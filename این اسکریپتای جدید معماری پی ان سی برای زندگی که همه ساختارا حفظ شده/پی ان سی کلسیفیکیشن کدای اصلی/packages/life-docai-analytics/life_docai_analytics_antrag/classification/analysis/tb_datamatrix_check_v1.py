"""Technikblatt Data Matrix check: Python port of tbDmToStringByHeader / tbCheckDocClassForContainer (AXATechnikblatt.spl)."""

import argparse
import base64
import gzip
from pathlib import Path

import numpy as np
import pandas as pd
import zxingcpp
from PIL import Image

HEADER = "AXA03"
PARTNER, ZUSATZ, ANTRAG, ANGEBOT = "0000000100", "0000000101", "0000000102", "0000000103"
REQUIRED = {ANTRAG, ANGEBOT}


def read_chunks(image_paths):
    """AXA03 chunks of one document: {container: {number: base64}}, {container: total}, Data Matrix count per page."""
    chunks, totals, page_counts = {}, {}, []
    for path in image_paths:
        with Image.open(path) as image:
            gray = np.array(image.convert("L"))
        codes = zxingcpp.read_barcodes(gray, formats=zxingcpp.BarcodeFormat.DataMatrix)
        page_counts.append(len(codes))
        for code in codes:
            text = code.text
            if not text.startswith(HEADER):
                continue  # e.g. the small page-id Data Matrix
            container, number, total, value = text[5:15], text[15:18], text[18:21], text[21:]
            if not total.isdigit():
                continue
            chunks.setdefault(container, {})[number] = value
            totals[container] = int(total)
    return chunks, totals, page_counts


def derived_class(complete):
    """Same order as tbCheckDocClassForContainer."""
    if not REQUIRED <= complete:
        return "Bestand"
    if ZUSATZ not in complete:
        return "Antrag/Technikblatt Partner und Antragsdaten"
    if PARTNER not in complete:
        return "Antrag/Technikblatt Partnerdaten"
    return "Antrag/Technikblatt"


def technikblatt_status(image_paths):
    chunks, totals, page_counts = read_chunks(image_paths)
    complete = {c for c in chunks if len(chunks[c]) == totals[c]}
    payloads = {}
    for container in complete:
        joined = "".join(chunks[container][n] for n in sorted(chunks[container]))
        try:
            payloads[container] = gzip.decompress(base64.b64decode(joined))
        except (ValueError, OSError):
            payloads[container] = None
    return {
        "n_datamatrix": sum(page_counts),
        "page_datamatrix": " ".join(map(str, page_counts)),  # 0 on a page = probably not a Technikblatt image
        "n_axa03": sum(len(v) for v in chunks.values()),
        "complete": " ".join(sorted(complete)),
        "missing_chunks": " ".join(f"{c}:{totals[c] - len(chunks[c])}" for c in sorted(chunks) if c not in complete),
        "undecodable": " ".join(sorted(c for c, p in payloads.items() if p is None)),
        "broken": not REQUIRED <= complete,
        "derived_class": derived_class(complete),
    }, payloads


def documents_from_labels(pages):
    """True document boundaries and true page classes from stack_pages.csv."""
    pages = pages.sort_values(["stack_id", "stack_position"]).copy()
    pages["doc_idx"] = pages.groupby("stack_id")["segmentation_target"].cumsum()
    pages["is_tb"] = pages["page_class"] == "TB0"
    return pages


def documents_from_predictions(predictions, pages):
    """Predicted document boundaries and predicted page classes from a predict_stage1/2 CSV."""
    predictions = predictions.sort_values(["stack_id", "stack_position"]).copy()
    start_column = "document_start_probability" if "document_start_probability" in predictions else "first_page_probability"
    starts = (predictions["stack_position"] == 1) | (predictions[start_column] > 0.5)
    predictions["doc_idx"] = starts.astype(int).groupby(predictions["stack_id"]).cumsum()
    predictions["is_tb"] = predictions["predicted_class"] == "TB0"
    predictions["image_path"] = predictions["page_identity"]
    labels = pages[["image_path", "sfdoc_class"]].drop_duplicates("image_path")
    return predictions.merge(labels, on="image_path", how="left")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pages-csv", required=True, help="stack_pages.csv of the batch")
    parser.add_argument("--render-root", required=True, help="folder that image_path is relative to (config.render_root)")
    parser.add_argument("--predictions", help="predict_stage1/2 CSV; without it true page classes and boundaries are used")
    parser.add_argument("--limit-docs", type=int, help="random sample of Technikblatt documents")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--payload-dir", help="optional: write decoded container payloads here for inspection")
    parser.add_argument("--output", required=True, help="result CSV, one row per Technikblatt document")
    args = parser.parse_args()

    pages = pd.read_csv(args.pages_csv)
    if args.predictions:
        rows = documents_from_predictions(pd.read_csv(args.predictions), pages)
    else:
        rows = documents_from_labels(pages)

    tb_docs = rows[rows["is_tb"]].groupby(["stack_id", "doc_idx"])
    keys = list(tb_docs.groups)
    if args.limit_docs and args.limit_docs < len(keys):
        rng = np.random.default_rng(args.seed)
        keys = [keys[i] for i in rng.choice(len(keys), args.limit_docs, replace=False)]

    render_root = Path(args.render_root)
    payload_dir = Path(args.payload_dir) if args.payload_dir else None
    if payload_dir:
        payload_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for number, (stack_id, doc_idx) in enumerate(keys, start=1):
        doc = tb_docs.get_group((stack_id, doc_idx))
        status, payloads = technikblatt_status([render_root / p for p in doc["image_path"]])
        sfdoc_class = doc["sfdoc_class"].dropna().iloc[0] if doc["sfdoc_class"].notna().any() else ""
        results.append({"stack_id": stack_id, "doc_idx": doc_idx, "tb_pages": len(doc),
                        "sfdoc_class": sfdoc_class, **status,
                        "match": status["derived_class"] == sfdoc_class})
        if payload_dir:
            for container, payload in payloads.items():
                if payload is not None:
                    (payload_dir / f"{stack_id}_{doc_idx}_{container}.bin").write_bytes(payload)
        if number % 50 == 0:
            print(f"{number}/{len(keys)} documents")

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 250)
    result = pd.DataFrame(results)
    result.to_csv(args.output, index=False)
    print(f"documents: {len(result)}  broken: {result['broken'].mean():.3f}  match_sfdoc_class: {result['match'].mean():.3f}")
    print(pd.crosstab(result["sfdoc_class"], result["derived_class"]))

    mismatches = result[~result["match"]]
    columns = ["stack_id", "doc_idx", "tb_pages", "sfdoc_class", "n_datamatrix", "page_datamatrix", "n_axa03",
               "complete", "missing_chunks", "derived_class"]
    print(f"\nmismatches: {len(mismatches)}")
    if len(mismatches):
        print(mismatches[columns].to_string(index=False))
    mismatch_path = Path(args.output).with_name(f"{Path(args.output).stem}_mismatches.csv")
    mismatches.to_csv(mismatch_path, index=False)
    print(args.output)
    print(mismatch_path)


if __name__ == "__main__":
    main()