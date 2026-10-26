import argparse
import os

import data
import evaluation
import torch
import workflows
from config import BERT_MODEL_PATH
from models import BertPageModel, SequenceModelRNN


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--multimodal", action="store_true", help="Use multimodal model"
    )
    parser.add_argument(
        "--text", action="store_true", help="Use text-only model (non-multimodal)"
    )
    parser.add_argument(
        "--dataset", type=str, required=True, help="Dataset to use (claims or schaden)"
    )
    parser.add_argument(
        "--stage",
        type=str,
        required=True,
        help="Workflow to run (compute_visual_embeddings, train_stage1, train_stage2, evaluate_stage1, evalaute_stage2)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs",
        help="Output directory where model weights are saved",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use (cpu or cuda) - automatically inferred by default",
    )
    parser.add_argument(
        "--batch_size", type=int, default=16, help="Batch size for model training"
    )
    parser.add_argument(
        "--dummy_bert_features",
        action="store_true",
        help="Use only visual features for training sequence model (debug only)",
    )

    args = parser.parse_args()
    assert args.stage in [
        "compute_visual_embeddings",
        "train_stage1",
        "train_stage2",
        "evaluate_stage1",
        "evaluate_stage2",
    ], "Invalid stage specified"
    if not args.multimodal and not args.text:
        raise ValueError("One of --multimodal or --text must be specified")
    if args.multimodal and args.text:
        raise ValueError("Only one of --multimodal or --text can be specified")
    return args


if __name__ == "__main__":
    args = parse_args()

    if args.device == "auto":
        if torch.cuda.is_available():
            print("GPU available")
            args.device = "cuda"
        else:
            print("GPU not available")
            args.device = "cpu"
    print("Using device", args.device)

    dataset, id_to_class, class_to_id, id_remapper_names, id_remapper = (
        data.prepare_dataset()
    )

    if args.stage == "compute_visual_embeddings":
        workflows.compute_visual_embeddings(dataset, args)
    else:
        if args.multimodal:
            dataset = workflows.load_visual_embeddings(dataset, args)
        dataset = data.build_validation_set(dataset)

        bert_model = BertPageModel(
            BERT_MODEL_PATH, multimodal=args.multimodal, num_classes=len(id_to_class)
        ).to(args.device)
        if args.stage in ["train_stage2", "evaluate_stage1", "evaluate_stage2"]:
            # Load pretrained
            if args.multimodal:
                model_file_stage1 = f"model_{args.dataset}_stage1_multimodal.pth"
            else:
                model_file_stage1 = f"model_{args.dataset}_stage1_textonly.pth"
            model_file_stage1 = os.path.join(args.output_dir, model_file_stage1)
            bert_model.load_state_dict(
                torch.load(model_file_stage1, map_location="cpu")["bert_model"]
            )
            print("Loaded Stage 1 weights")
            bert_model.requires_grad_(False).eval()
        model = None

        prefetched_features = None
        if args.stage == "train_stage1":
            workflows.train_stage1(bert_model, dataset, args)
        elif args.stage == "train_stage2":
            # Instantiate sequence model
            model = SequenceModelRNN(
                multimodal=args.multimodal, num_classes=len(id_to_class)
            ).to(args.device)
            prefetched_features = workflows.precompute_bert_features(
                bert_model, dataset, args
            )
            workflows.train_stage2(
                bert_model, model, prefetched_features, dataset, args
            )

        # Evaluation
        if args.stage in ["train_stage1", "evaluate_stage1"]:
            bert_model.requires_grad_(False).eval()
            results = evaluation.evaluate_stage1(bert_model, dataset, args)
            evaluation.generate_report(
                dataset, id_remapper, id_remapper_names, *results
            )

        elif args.stage in ["train_stage2", "evaluate_stage2"]:
            if model is None:
                # Instantiate sequence model
                model = SequenceModelRNN(
                    multimodal=args.multimodal, num_classes=len(id_to_class)
                ).to(args.device)
                if args.multimodal:
                    model_file_stage2 = f"model_{args.dataset}_final_multimodal.pth"
                else:
                    model_file_stage2 = f"model_{args.dataset}_final_textonly.pth"
                model_file_stage2 = os.path.join(args.output_dir, model_file_stage2)
                model.load_state_dict(
                    torch.load(model_file_stage2, map_location="cpu")["model"]
                )
                print("Loaded Stage 2 weights")
            model.requires_grad_(False).eval()
            if prefetched_features is None:
                prefetched_features = workflows.precompute_bert_features(
                    bert_model, dataset, args, no_train=True
                )
            results = evaluation.evaluate_stage2(
                model, prefetched_features, dataset, args
            )
            evaluation.generate_report(
                dataset, id_remapper, id_remapper_names, *results
            )
