import argparse
import json
from pathlib import Path

from . import workflows
from .config import PipelineConfig
from .data import _file_record, prepare_model_population, read_csv, split_assignments, split_report
from .evaluation import (REVIEW_FIELDS, confidence_report, format_confidence_table, level_rows, read_predictions,
                         review_rows, summary_rows, write_csv_new, write_json_new)

# Paths each stage needs; filled from --run-name/--modality when not given explicitly.
RUN_INPUTS = {
    "evaluate_stage1": ("checkpoint",),
    "precompute_features": ("stage1_checkpoint",),
    "train_stage2": ("stage1_checkpoint", "stage1_features"),
    "evaluate_stage2": ("stage1_checkpoint", "stage1_features", "checkpoint"),
    "predict_stage2": ("stage1_checkpoint", "stage1_features", "checkpoint"),
    "report": ("predictions",),
}


def prepare(args):
    config = PipelineConfig(batch_id=args.batch_id)
    rows, population = prepare_model_population(config, args.seed)
    output = {"population": population, "model_pages": str(config.model_pages_csv),
              "excluded_stacks": str(config.excluded_stacks_report)}
    if not args.no_splits:
        output["splits"] = split_report(rows, split_assignments(config, rows, args.seed))
        write_json_new(config.split_report, output["splits"])
    print(json.dumps(output, indent=2))


def split_rows(config, predictions, split):
    """Prediction rows of one split, using the batch's stack_splits.csv."""
    splits = {row["stack_id"]: row["split"] for row in read_csv(config.split_file)}
    rows = [row for row in read_predictions(predictions) if splits.get(row["stack_id"]) == split]
    if not rows:
        raise ValueError(f"No predictions for split {split} in {predictions}")
    return rows


def report_output(config, args):
    """JSON path of the report; its CSV summary and review list are written next to it."""
    if args.output:
        return Path(args.output)
    rule = f"_p{args.confidence_percentile:g}" if args.confidence_percentile > 0 else ""
    return config.results_dir / f"report_{args.split}_{args.level}{rule}_{Path(args.predictions).stem}.json"


def report_payload(config, args, rows, results):
    """Full report with its parameters and the checksums of its inputs."""
    return {"split": args.split, "level": args.level, "pages": len(rows),
            "confidence_percentile": args.confidence_percentile, "review_threshold": args.review_threshold,
            "results": results, "inputs": {"predictions": _file_record(args.predictions),
                                           "splits": _file_record(config.split_file)}}


def write_review(rows, threshold, output):
    """CSV of the pages a human reviews at the threshold."""
    queue = review_rows(rows, threshold)
    path = output.with_name(f"{output.stem}_review_t{threshold:g}.csv")
    write_csv_new(path, queue, REVIEW_FIELDS)
    print(f"review: {len(queue)} pages -> {path}")


def report(args):
    """Confidence report on an existing predictions CSV: JSON in full, CSV summary, optional review list."""
    config = PipelineConfig(batch_id=args.batch_id)
    rows = level_rows(split_rows(config, args.predictions, args.split), args.level)
    results = confidence_report(rows, args.thresholds, args.confidence_percentile)
    output = report_output(config, args)
    write_json_new(output, report_payload(config, args, rows, results))
    summary = summary_rows(results)
    write_csv_new(output.with_suffix(".csv"), summary, list(summary[0]))
    print(format_confidence_table(summary))
    print(output)
    print(output.with_suffix(".csv"))
    if args.review_threshold is not None:
        write_review(rows, args.review_threshold, output)


def best_checkpoint(config, model_prefix, modality, run_name):
    """Best checkpoint of a run with the highest epoch.

    Training writes a *_best_epoch_NNN.pt file only when validation macro-F1 improves, so the highest
    epoch is the best one. Epochs are zero-padded, so name order is epoch order.
    """
    pattern = f"{model_prefix}_{modality}_{run_name}_best_epoch_*.pt"
    matches = sorted(config.checkpoints_dir.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No {pattern} in {config.checkpoints_dir}")
    return matches[-1]


def predictions_file(config, modality, run_name):
    """CSV that predict_stage2 wrote for the run's best Stage 2 checkpoint."""
    checkpoint = best_checkpoint(config, "model_final", modality, run_name)
    path = config.results_dir / f"predictions_final_{checkpoint.stem}.csv"
    if not path.is_file():
        raise FileNotFoundError(f"No predictions for {checkpoint.name}; run predict_stage2 first: {path}")
    return path


def fill_from_run(args, needed):
    """Fill missing paths from the run's naming conventions."""
    config = PipelineConfig(batch_id=args.batch_id)
    if "stage1_checkpoint" in needed and not args.stage1_checkpoint:
        args.stage1_checkpoint = best_checkpoint(config, "model_stage1", args.modality, args.run_name)
    if "stage1_features" in needed and not args.stage1_features:
        args.stage1_features = workflows.default_features_dir(config, args.stage1_checkpoint, args.modality)
    if "checkpoint" in needed and not args.checkpoint:
        prefix = "model_stage1" if args.stage == "evaluate_stage1" else "model_final"
        args.checkpoint = best_checkpoint(config, prefix, args.modality, args.run_name)
    if "predictions" in needed and not args.predictions:
        args.predictions = predictions_file(config, args.modality, args.run_name)


def resolve_run_inputs(args, parser):
    """Fill the paths a stage needs from --run-name/--modality; explicit paths are kept as given."""
    needed = RUN_INPUTS.get(args.stage)
    if not needed:
        return
    if args.run_name:
        if not args.modality:
            parser.error("--modality is required with --run-name")
        fill_from_run(args, needed)
    missing = [name for name in needed if not getattr(args, name)]
    if missing:
        parser.error("give --run-name and --modality, or " + ", ".join("--" + name.replace("_", "-") for name in missing))
    for name in needed:
        print(f"{name}: {getattr(args, name)}")


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="stage", required=True)

    def stage(name, *, device=True, batch_size=None, limit=True):
        command = sub.add_parser(name)
        command.add_argument("--batch-id", required=True)
        command.add_argument("--seed", type=int, default=13)
        if device:
            command.add_argument("--device")
        if batch_size is not None:
            command.add_argument("--batch-size", type=int, default=batch_size)
        if limit:
            command.add_argument("--limit-stacks", type=int)
        return command

    def subset(command):
        command.add_argument("--training-limit-stacks", type=int, help="N train + N//4 val stacks (smoke runs)")

    def precision(command):
        command.add_argument("--use-fp16", action="store_true")
        command.add_argument("--use-bf16", action="store_true")

    def run_lookup(command, run_name=True):
        """--run-name/--modality as an alternative to explicit checkpoint and feature paths."""
        if run_name:
            command.add_argument("--run-name", help="find the run's best checkpoint and features")
        command.add_argument("--modality", choices=["vision", "text", "fusion"], help="modality of the run (with --run-name)")

    def stage2_inputs(command):
        command.add_argument("--stage1-checkpoint")
        command.add_argument("--stage1-features", help="directory written by precompute_features")
        command.add_argument("--max-pages-per-batch", type=int, default=2048,
                             help="padded pages per batch; stacks of similar length are batched together")

    def balance(command):
        command.add_argument("--balance-classes", action=argparse.BooleanOptionalAction, default=True,
                             help="on by default; --no-balance-classes turns it off")

    def predict_only(command):
        command.add_argument("--predict-only", action="store_true",
                             help="new batch, checkpoints from another batch: skip the population/feature checks")

    command = stage("prepare", device=False, limit=False)
    command.add_argument("--no-splits", action="store_true", help="prediction-only batch: no train/val/test split")
    command = stage("compute_ocr", device=False); subset(command); command.add_argument("--workers", type=int)
    command = stage("compute_visual_embeddings", batch_size=32); subset(command)
    command = stage("migrate_clip_cache", device=False, limit=False)
    command.add_argument("--source", required=True, help="old single-file CLIP cache ({image_path: tensor})")

    command = stage("train_stage1", batch_size=8); precision(command)
    command.add_argument("--modality", choices=["vision", "text", "fusion"], required=True)
    command.add_argument("--epochs", type=int, default=5)
    command.add_argument("--lr", type=float, default=1e-5)
    command.add_argument("--lr-decay-epoch", type=int, help="Epoch from which --lr-decay-to is used (PnC: 2e-6)")
    command.add_argument("--lr-decay-to", type=float, default=2e-6)
    balance(command)
    command.add_argument("--resume"); command.add_argument("--run-name")
    command = stage("evaluate_stage1", batch_size=8); precision(command); run_lookup(command)
    command.add_argument("--checkpoint")
    command.add_argument("--split", choices=["train", "val", "test"], default="test")
    command.add_argument("--output", help="optional; defaults to the batch results folder")
    command = stage("predict_stage1", batch_size=8)
    command.add_argument("--checkpoint", required=True)
    command.add_argument("--output", help="optional; defaults to the batch results folder")

    command = stage("precompute_features", batch_size=8); subset(command); run_lookup(command); predict_only(command)
    command.add_argument("--stage1-checkpoint")
    command.add_argument("--output", help="default: stage1_features_<modality>_<checkpoint name> in the output folder")

    command = stage("train_stage2"); stage2_inputs(command); run_lookup(command, run_name=False)
    command.add_argument("--epochs", type=int, default=5)
    command.add_argument("--lr", type=float, default=1e-4)
    balance(command)
    command.add_argument("--resume")
    command.add_argument("--run-name", help="finds the Stage 1 run and names the Stage 2 outputs")
    command = stage("evaluate_stage2"); stage2_inputs(command); run_lookup(command)
    command.add_argument("--checkpoint")
    command.add_argument("--split", choices=["train", "val", "test"], default="test")
    command.add_argument("--output", help="optional; defaults to the batch results folder")
    command = stage("predict_stage2"); stage2_inputs(command); run_lookup(command); predict_only(command)
    command.add_argument("--checkpoint")
    command.add_argument("--output", help="optional; defaults to the batch results folder")

    command = stage("report", device=False, limit=False); run_lookup(command)
    command.add_argument("--predictions", help="CSV written by predict_stage1 or predict_stage2")
    command.add_argument("--split", choices=["train", "val", "test"], default="val")
    command.add_argument("--level", choices=["page", "document"], default="page",
                         help="document: pages take the mean probabilities of their predicted document (PnC pooling)")
    command.add_argument("--thresholds", type=float, nargs="+", default=[0.0, 0.5, 0.6, 0.7, 0.8, 0.9],
                         help="a page is processed automatically when its highest probability is above the threshold")
    command.add_argument("--confidence-percentile", type=float, default=0,
                         help="as PnC: keep this percent of pages with the lowest entropy instead of --thresholds; "
                              "0 (default) = off")
    command.add_argument("--review-threshold", type=float,
                         help="also write a CSV of the pages that go to human review at this threshold")
    command.add_argument("--output", help="optional; defaults to the batch results folder")
    return parser


STAGES = {
    "prepare": prepare,
    "compute_ocr": workflows.compute_ocr,
    "compute_visual_embeddings": workflows.compute_visual_embeddings,
    "migrate_clip_cache": workflows.migrate_clip_cache,
    "train_stage1": workflows.train_stage1,
    "evaluate_stage1": workflows.evaluate_stage1,
    "predict_stage1": workflows.predict_stage1,
    "precompute_features": workflows.precompute_features,
    "train_stage2": workflows.train_stage2,
    "evaluate_stage2": workflows.evaluate_stage2,
    "predict_stage2": workflows.predict_stage2,
    "report": report,
}


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    resolve_run_inputs(args, parser)
    STAGES[args.stage](args)