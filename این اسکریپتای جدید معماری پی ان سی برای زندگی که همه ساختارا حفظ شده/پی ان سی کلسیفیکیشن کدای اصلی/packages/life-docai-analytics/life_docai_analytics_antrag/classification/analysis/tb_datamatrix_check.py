"""Technikblatt Data Matrix check: Python port of tbDmToStringByHeader / tbCheckDocClassForContainer (AXATechnikblatt.spl)."""

import argparse
import base64
import gzip
import zlib

import numpy as np
import pandas as pd
import zxingcpp
from PIL import Image, ImageFilter, ImageOps

from ..training.config import PipelineConfig

HEADER = "AXA03"
PARTNER, ZUSATZ, ANTRAG, ANGEBOT = "0000000100", "0000000101", "0000000102", "0000000103"
REQUIRED = {ANTRAG, ANGEBOT}
KNOWN = {PARTNER, ZUSATZ, ANTRAG, ANGEBOT}
DOCUMENT_KEY = ["stack_id", "masterindex_id", "process_id", "doc_id", "subdoc_idx"]


def load_gray(path):
    with Image.open(path) as image:
        return image.convert("L")


def zxing_texts(image):
    return [code.text for code in zxingcpp.read_barcodes(image, formats=zxingcpp.BarcodeFormat.DataMatrix)]


def fallback_texts(image):
    """zxing on a contrast-stretched, a sharpened and a 2x upscaled copy of the page."""
    variants = (ImageOps.autocontrast(image, cutoff=1),
                image.filter(ImageFilter.UnsharpMask(radius=2, percent=200, threshold=0)),
                image.resize((image.width * 2, image.height * 2), Image.Resampling.LANCZOS))
    return [text for variant in variants for text in zxing_texts(variant)]


def read_chunks(page_texts):
    """AXA03 chunks {container: {number: base64}}, chunk totals, the other Data Matrix texts and containers with conflicting chunks."""
    chunks, totals, other, conflicts = {}, {}, [], set()
    for text in (text for texts in page_texts for text in texts):
        container, number, total, value = text[5:15], text[15:18], text[18:21], text[21:]
        if not text.startswith(HEADER) or not total.isdigit():
            other.append(text[:80])  # e.g. the small page-id Data Matrix
            continue
        if chunks.get(container, {}).get(number, value) != value:
            conflicts.add(container)  # same chunk number with another text, e.g. two Technikblätter in one document
        chunks.setdefault(container, {})[number] = value
        totals[container] = int(total)
    return chunks, totals, other, conflicts


def complete_containers(chunks, totals):
    return {c for c in chunks if len(chunks[c]) == totals[c]}


def decode_document(images):
    """zxing; if a chunk is missing, the fallback texts are added (merged, so the fallback never loses a code)."""
    page_texts = [zxing_texts(image) for image in images]
    chunks, totals, _, _ = read_chunks(page_texts)
    complete = complete_containers(chunks, totals)
    fallback = not REQUIRED <= complete or complete != set(chunks)
    if fallback:
        page_texts = [texts + fallback_texts(image) for texts, image in zip(page_texts, images)]
    return [list(dict.fromkeys(texts)) for texts in page_texts], fallback, not REQUIRED <= complete


def derived_class(complete):
    """Same order as tbCheckDocClassForContainer."""
    if not REQUIRED <= complete:
        return "Bestand"
    if ZUSATZ not in complete:
        return "Antrag/Technikblatt Partner und Antragsdaten"
    if PARTNER not in complete:
        return "Antrag/Technikblatt Partnerdaten"
    return "Antrag/Technikblatt"


def payload(chunks, container):
    joined = "".join(chunks[container][n] for n in sorted(chunks[container]))
    try:
        return gzip.decompress(base64.b64decode(joined))
    except (ValueError, OSError, zlib.error):
        return None


def technikblatt_status(images):
    page_texts, fallback, broken_before = decode_document(images)
    chunks, totals, other, conflicts = read_chunks(page_texts)
    complete = complete_containers(chunks, totals)
    payloads = {container: payload(chunks, container) for container in complete}
    page_counts = [len(texts) for texts in page_texts]
    return {
        "n_datamatrix": sum(page_counts),
        "page_datamatrix": " ".join(map(str, page_counts)),  # 0 on a page = probably not a Technikblatt image
        "n_axa03": sum(len(v) for v in chunks.values()),
        "containers": " ".join(sorted(chunks)),
        "complete": " ".join(sorted(complete)),
        "missing_chunks": " ".join(f"{c}:{totals[c] - len(chunks[c])}" for c in sorted(chunks) if c not in complete),
        "undecodable": " ".join(sorted(c for c, p in payloads.items() if p is None)),
        "conflicting_chunks": " ".join(sorted(conflicts)),
        "fallback": fallback,
        "broken_before_fallback": broken_before,
        "broken": not REQUIRED <= complete,
        "derived_class": derived_class(complete),
        "other_datamatrix": " | ".join(other),
    }, payloads


def documents_from_labels(pages):
    """Documents as the mapping defines them (doc_id), also when their pages are not consecutive in image_id order."""
    pages = pages.sort_values(["stack_id", "stack_position"]).copy()
    pages["doc_idx"] = pages.groupby(DOCUMENT_KEY, sort=False).ngroup()
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


def select_documents(rows, limit_docs, seed):
    tb_docs = rows[rows["is_tb"]].groupby(["stack_id", "doc_idx"])
    keys = list(tb_docs.groups)
    if limit_docs and limit_docs < len(keys):
        keys = [keys[i] for i in np.random.default_rng(seed).choice(len(keys), limit_docs, replace=False)]
    return tb_docs, keys


def check_documents(tb_docs, keys, config, args, payload_dir):
    results = []
    for number, (stack_id, doc_idx) in enumerate(keys, start=1):
        doc = tb_docs.get_group((stack_id, doc_idx))
        images = [load_gray(config.render_root / path) for path in doc["image_path"]]
        status, payloads = technikblatt_status(images)
        sfdoc_class = doc["sfdoc_class"].dropna().iloc[0] if doc["sfdoc_class"].notna().any() else ""
        results.append({"stack_id": stack_id, "doc_idx": doc_idx, "tb_pages": len(doc), "sfdoc_class": sfdoc_class,
                        **status, "match": status["derived_class"] == sfdoc_class})
        for container, data in payloads.items():
            if args.save_payloads and data is not None:
                (payload_dir / f"{stack_id}_{doc_idx}_{container}.bin").write_bytes(data)
        if number % 50 == 0:
            print(f"{number}/{len(keys)} documents")
    return pd.DataFrame(results)


def report(result, output):
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 250)
    result.to_csv(output, index=False)
    rescued = result["broken_before_fallback"] & ~result["broken"]
    print(f"documents: {len(result)}  broken: {result['broken'].mean():.3f}  match_sfdoc_class: {result['match'].mean():.3f}")
    print(f"fallback used: {result['fallback'].sum()}  rescued (broken -> not broken): {rescued.sum()}")
    print(f"undecodable: {(result['undecodable'] != '').sum()}  conflicting_chunks: {(result['conflicting_chunks'] != '').sum()}")
    print(pd.crosstab(result["sfdoc_class"], result["derived_class"]))
    containers = result["containers"].str.split().explode().value_counts()
    print("\ncontainers (number of documents):")
    print(containers.rename(lambda c: c if c in KNOWN else f"{c}  <- unknown").to_string())
    mismatches = result[~result["match"]]
    columns = ["stack_id", "doc_idx", "tb_pages", "sfdoc_class", "n_datamatrix", "page_datamatrix", "n_axa03",
               "complete", "missing_chunks", "fallback", "derived_class"]
    print(f"\nmismatches: {len(mismatches)}")
    if len(mismatches):
        print(mismatches[columns].to_string(index=False))
    mismatch_path = output.with_name(f"{output.stem}_mismatches.csv")
    mismatches.to_csv(mismatch_path, index=False)
    print(output)
    print(mismatch_path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-id", required=True)
    parser.add_argument("--predictions", help="predict_stage1/2 CSV (production view); without it doc_id from the mapping groups the pages")
    parser.add_argument("--limit-docs", type=int, help="random sample of Technikblatt documents")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--save-payloads", action="store_true", help="write decoded container payloads for inspection")
    args = parser.parse_args()

    config = PipelineConfig(batch_id=args.batch_id)
    out_dir = config.results_dir / "tb_check"
    payload_dir = out_dir / "payloads"
    payload_dir.mkdir(parents=True, exist_ok=True)

    pages = pd.read_csv(config.model_pages_csv, dtype={column: str for column in DOCUMENT_KEY})
    rows = documents_from_predictions(pd.read_csv(args.predictions), pages) if args.predictions else documents_from_labels(pages)
    tb_docs, keys = select_documents(rows, args.limit_docs, args.seed)
    result = check_documents(tb_docs, keys, config, args, payload_dir)
    report(result, out_dir / ("tb_check_predictions.csv" if args.predictions else "tb_check_labels.csv"))


if __name__ == "__main__":
    main()