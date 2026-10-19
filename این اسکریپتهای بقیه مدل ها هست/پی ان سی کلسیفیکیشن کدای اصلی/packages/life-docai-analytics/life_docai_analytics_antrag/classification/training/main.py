"""Single entry point, as in PnC main.py. Run from the repository root (see README for PYTHONPATH):

  python -m life_docai_analytics_antrag.classification.training.main prepare
  python -m life_docai_analytics_antrag.classification.training.main compute_ocr
  python -m life_docai_analytics_antrag.classification.training.main compute_visual_embeddings --device cuda
  python -m life_docai_analytics_antrag.classification.training.main migrate_clip_cache --source <old single-file CLIP cache .pt>
  python -m life_docai_analytics_antrag.classification.training.main train_stage1 --modality fusion --run-name <run_name>
  python -m life_docai_analytics_antrag.classification.training.main evaluate_stage1 --checkpoint <stage1.pt> --output <json>
  python -m life_docai_analytics_antrag.classification.training.main predict_stage1 --checkpoint <stage1.pt> --output <csv>
  python -m life_docai_analytics_antrag.classification.training.main precompute_features --stage1-checkpoint <stage1.pt>
  python -m life_docai_analytics_antrag.classification.training.main train_stage2 --stage1-checkpoint <stage1.pt> --stage1-features <dir>
  python -m life_docai_analytics_antrag.classification.training.main evaluate_stage2 ... --checkpoint <stage2.pt> --output <json>
  python -m life_docai_analytics_antrag.classification.training.main predict_stage2  ... --checkpoint <stage2.pt> --output <csv>
"""

import argparse
import json

from . import workflows
from .config import PipelineConfig
from .data import prepare_model_population, split_assignments, split_report
from .evaluation import write_json_new


def prepare(args):
    config = PipelineConfig(batch_id=args.batch_id)
    rows, population = prepare_model_population(config)
    report = split_report(rows, split_assignments(config, rows, args.seed))
    write_json_new(config.split_report, report)
    print(json.dumps({"population": population, "splits": report, "model_pages": str(config.model_pages_csv),
                      "excluded_stacks": str(config.excluded_stacks_report)}, indent=2))


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="stage", required=True)

    def stage(name, *, device=True, batch_size=None, limit=True):
        command = sub.add_parser(name)
        command.add_argument("--batch-id", default=PipelineConfig.batch_id)
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

    def stage2_inputs(command):
        command.add_argument("--stage1-checkpoint", required=True)
        command.add_argument("--stage1-features", required=True, help="directory written by precompute_features")
        command.add_argument("--max-pages-per-batch", type=int, default=2048,
                             help="padded pages per batch; stacks of similar length are batched together")

    def balance(command):
        command.add_argument("--balance-classes", action=argparse.BooleanOptionalAction, default=True,
                             help="on by default; --no-balance-classes turns it off")

    stage("prepare", device=False, limit=False)
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
    command = stage("evaluate_stage1", batch_size=8); precision(command)
    command.add_argument("--checkpoint", required=True)
    command.add_argument("--split", choices=["train", "val", "test"], default="test")
    command.add_argument("--output", required=True)
    command = stage("predict_stage1", batch_size=8)
    command.add_argument("--checkpoint", required=True); command.add_argument("--output", required=True)

    command = stage("precompute_features", batch_size=8); subset(command)
    command.add_argument("--stage1-checkpoint", required=True)
    command.add_argument("--output", help="default: stage1_features_<modality>_<checkpoint name> in the output folder")

    command = stage("train_stage2"); stage2_inputs(command)
    command.add_argument("--epochs", type=int, default=5)
    command.add_argument("--lr", type=float, default=1e-4)
    balance(command)
    command.add_argument("--resume"); command.add_argument("--run-name")
    command = stage("evaluate_stage2"); stage2_inputs(command)
    command.add_argument("--checkpoint", required=True)
    command.add_argument("--split", choices=["train", "val", "test"], default="test")
    command.add_argument("--output", required=True)
    command = stage("predict_stage2"); stage2_inputs(command)
    command.add_argument("--checkpoint", required=True); command.add_argument("--output", required=True)
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
}


if __name__ == "__main__":
    args = build_parser().parse_args()
    STAGES[args.stage](args)
