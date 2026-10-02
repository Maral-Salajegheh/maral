"""Parallel OCR cache for text/fusion inputs with RapidOCR-first selection."""

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context

from tqdm import tqdm

from .config import PipelineConfig
from .data import read_csv

_WORKER_ENGINE_NAME = None
_WORKER_ENGINE = None


def _read_cache(path):
    cache = {}
    engines = set()
    malformed = 0
    if not path.exists():
        return cache, engines, malformed
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                malformed += 1
                continue
            cache[row["image_path"]] = row["text"]
            engines.add(row.get("engine", "missing"))
    return cache, engines, malformed


def load_ocr_cache(path):
    cache, _engines, malformed = _read_cache(path)
    if malformed:
        print(f"skipped_corrupt_ocr_cache_lines={malformed}")
    return cache


def selected_engine_name():
    try:
        import rapidocr_onnxruntime  # noqa: F401
        return "rapidocr-onnxruntime"
    except BaseException:
        return "paddleocr"


def build_ocr_engine():
    try:
        from rapidocr_onnxruntime import RapidOCR

        return "rapidocr-onnxruntime", RapidOCR(
            intra_op_num_threads=1,
            inter_op_num_threads=1,
        )
    except BaseException:
        from paddleocr import PaddleOCR

        return (
            "paddleocr",
            PaddleOCR(
                lang="de",
                ocr_version="PP-OCRv4",
                use_doc_orientation_classify=True,
                use_doc_unwarping=True,
            ),
        )


def extract_text(result):
    if not result:
        return ""
    if isinstance(result, dict):
        texts = result.get("rec_texts") or result.get("texts") or []
        return "\n".join(map(str, texts))
    texts = []
    for item in result:
        if isinstance(item, dict):
            texts.extend(item.get("rec_texts") or [])
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            value = item[1]
            if isinstance(value, (list, tuple)) and value:
                texts.append(str(value[0]))
            else:
                texts.append(str(value))
    return "\n".join(texts)


def ocr_page(engine_name, engine, image_path):
    if engine_name == "paddleocr":
        if hasattr(engine, "predict"):
            return extract_text(engine.predict(image_path))
        return extract_text(engine.ocr(image_path))
    result, _elapsed = engine(image_path)
    return extract_text(result)


def _init_worker():
    global _WORKER_ENGINE_NAME, _WORKER_ENGINE
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OMP_THREAD_LIMIT"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    _WORKER_ENGINE_NAME, _WORKER_ENGINE = build_ocr_engine()


def _ocr_worker(task):
    image_path, resolved_image_path = task
    text = ocr_page(_WORKER_ENGINE_NAME, _WORKER_ENGINE, resolved_image_path)
    return image_path, text, _WORKER_ENGINE_NAME


def _ensure_consistent_cache(config, engine_name):
    cache, engines, malformed = _read_cache(config.ocr_cache)
    if malformed or len(engines) > 1 or (engines and engines != {engine_name}):
        print(
            f"Clearing inconsistent OCR cache: "
            f"engines={sorted(engines)}, malformed={malformed}, selected={engine_name}"
        )
        config.ocr_cache.unlink(missing_ok=True)
        return {}
    if engines:
        print(f"OCR cache engine: {next(iter(engines))}")
    return cache


def build_ocr_cache(config, limit=None, workers=None, rows=None):
    rows = read_csv(config.matched_pages_csv) if rows is None else rows
    if limit:
        rows = rows[:limit]
    engine_name = selected_engine_name()
    done = _ensure_consistent_cache(config, engine_name)
    pending = [row for row in rows if row["image_path"] not in done]
    print(f"OCR engine in use: {engine_name}")
    print(f"cached_pages={len(rows) - len(pending)} pending_pages={len(pending)}")
    if not pending:
        return
    workers = workers if workers is not None else max(1, (os.cpu_count() or 1) - 1)
    if workers < 1:
        raise ValueError("--workers must be at least 1")
    print(f"OCR workers: {workers}")
    tasks = [
        (row["image_path"], str(config.render_root / row["image_path"]))
        for row in pending
    ]
    config.ocr_cache.parent.mkdir(parents=True, exist_ok=True)
    with config.ocr_cache.open("a", encoding="utf-8") as handle:
        with ProcessPoolExecutor(
            max_workers=workers,
            mp_context=get_context("spawn"),
            initializer=_init_worker,
        ) as executor:
            for image_path, text, result_engine in tqdm(
                executor.map(_ocr_worker, tasks, chunksize=1),
                total=len(tasks),
                desc="ocr",
            ):
                if result_engine != engine_name:
                    raise RuntimeError(
                        f"Worker OCR engine {result_engine} differs from selected {engine_name}"
                    )
                handle.write(
                    json.dumps(
                        {"image_path": image_path, "text": text, "engine": result_engine},
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                handle.flush()
                os.fsync(handle.fileno())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-id", default=PipelineConfig.batch_id)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 1) - 1))
    args = parser.parse_args()
    build_ocr_cache(
        PipelineConfig(batch_id=args.batch_id),
        args.limit,
        args.workers,
    )


if __name__ == "__main__":
    main()