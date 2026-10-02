"""Build corrected-pipeline OCR and CLIP caches without touching old caches."""

import argparse

from .config import PipelineConfig
from .ocr import build_ocr_cache
from .pnc_data import load_model_pages, select_complete_mids, split_assignments


class ConfigProxy:
    def __init__(self, base, manifest, ocr_cache=None, clip_cache=None):
        self._base = base; self.matched_pages_csv = manifest
        if ocr_cache is not None: self.ocr_cache = ocr_cache
        if clip_cache is not None: self.clip_feature_file = clip_cache
    def __getattr__(self, name): return getattr(self._base, name)


def main():
    parser = argparse.ArgumentParser(); sub = parser.add_subparsers(dest="command", required=True)
    def common(command): command.add_argument("--batch-id", default=PipelineConfig.batch_id); command.add_argument("--limit-mids", type=int); command.add_argument("--training-limit-mids", type=int)
    ocr = sub.add_parser("ocr"); common(ocr); ocr.add_argument("--workers", type=int)
    clip = sub.add_parser("clip"); common(clip); clip.add_argument("--batch-size", type=int, default=32); clip.add_argument("--device")
    args = parser.parse_args(); config = PipelineConfig(batch_id=args.batch_id)
    all_rows = load_model_pages(config)
    if args.limit_mids and args.training_limit_mids:
        raise ValueError("Choose --limit-mids or --training-limit-mids")
    if args.training_limit_mids:
        assignments = split_assignments(config, all_rows, args.seed)
        rows = []
        for split in ["train", "val"]:
            split_rows = [row for row in all_rows if assignments[row["stack_id"]] == split]
            rows.extend(select_complete_mids(split_rows, args.training_limit_mids if split == "train" else max(1, args.training_limit_mids // 4)))
    else:
        rows = select_complete_mids(all_rows, args.limit_mids)
    if args.command == "ocr": build_ocr_cache(ConfigProxy(config, config.model_pages_csv, ocr_cache=config.pnc_ocr_cache), workers=args.workers, rows=rows)
    else:
        from .features import precompute_clip  # imported lazily so OCR needs no torch/CLIP deps
        precompute_clip(ConfigProxy(config, config.model_pages_csv, clip_cache=config.pnc_clip_feature_file), batch_size=args.batch_size, device=args.device, rows=rows)


if __name__ == "__main__": main()