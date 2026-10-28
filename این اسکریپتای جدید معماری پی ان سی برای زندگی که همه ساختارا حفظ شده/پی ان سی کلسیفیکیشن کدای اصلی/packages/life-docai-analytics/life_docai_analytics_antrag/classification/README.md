# Page Classification and Document Segmentation

This pipeline classifies the pages of Life Antrag stacks. It takes an incoming
stack of scanned pages, in its original order, and for every page predicts:

- its class: `A00`, `G07`, `MAD` or `TB0` (Technikblatt pages inside `A00` documents), and
- whether a new document starts on that page.

From these two predictions the stack is split into its documents.

## How it works

The pipeline has two stages, following the PnC architecture.

- **Stage 1: page model.** 
- **Stage 2: sequence model.**


## Structure

```
classification/
├── training/
│   ├── config.py       paths, classes, model locations
│   ├── data.py         building the stacks, labels and train/val/test splits
│   ├── models.py       page model and sequence model
│   ├── ocr.py          OCR of the page images
│   ├── storage.py      on-disk storage of image vectors and page features
│   ├── workflows.py    the individual steps
│   ├── evaluation.py   metrics
│   └── main.py         command-line entry point
├── gwg/                optional: finding GWG identification pages from the OCR text
├── output/             intermediate files and model checkpoints, per batch
└── results/            metrics and predictions, per batch
```

## Dependencies

- Python 3.10 or newer, with `torch`, `transformers`, `open_clip_torch`, `safetensors`, `rapidocr-onnxruntime`, `numpy`, `pyarrow`, `pillow` and `tqdm`
- Pretrained weights: GBERT-large and OpenCLIP ViT-H-14. Their locations are set in `training/config.py` and can be overridden with environment variables.
- A GPU for training and for computing the image vectors
- Upstream outputs: the page labels from the mapping step, and the rendered page images from the ingestion step

## Running

Run each command directly from the repository root. Every command below is
self-contained and uses the Pixi `dev` environment.

Every step needs `--batch-id`: it selects the input data and the output folder
of that batch, so nothing depends on a built-in default. The dataset report
records the batch, the split seed and a checksum of every input file, so a run
can always be traced back to its exact data. Add `--help` to any command to see
its options. The pipeline has nine executable commands: seven main
processing/training commands and two optional evaluation commands.

From step 4 on, every command takes the same `--run-name` and `--modality`. Step 4 uses
them to name the run; the later steps use them to find the best checkpoint of that run
and its page features, and write their results to `results/<batch_id>/`.

1. **Build the dataset**: groups the pages into stacks in their original order, sets the labels, and makes the train/validation/test split.
   ```bash
   PYTHONPATH=packages/life-docai-analytics pixi run -e dev python -m life_docai_analytics_antrag.classification.training.main prepare --batch-id <batch_id>
   ```
2. **OCR the pages**:
   ```bash
   PYTHONPATH=packages/life-docai-analytics pixi run -e dev python -m life_docai_analytics_antrag.classification.training.main compute_ocr --batch-id <batch_id> --workers 8
   ```
3. **Compute the image vectors** (needed for `fusion` and `vision`):
   ```bash
   PYTHONPATH=packages/life-docai-analytics pixi run -e dev python -m life_docai_analytics_antrag.classification.training.main compute_visual_embeddings --batch-id <batch_id> --device cuda
   ```
4. **Train Stage 1**:
   ```bash
   PYTHONPATH=packages/life-docai-analytics pixi run -e dev python -m life_docai_analytics_antrag.classification.training.main train_stage1 --batch-id <batch_id> --run-name <run_name> --modality fusion --device cuda
   ```
5. **Evaluate Stage 1** (optional):
   ```bash
   PYTHONPATH=packages/life-docai-analytics pixi run -e dev python -m life_docai_analytics_antrag.classification.training.main evaluate_stage1 --batch-id <batch_id> --run-name <run_name> --modality fusion
   ```
6. **Extract the page features for Stage 2**:
   ```bash
   PYTHONPATH=packages/life-docai-analytics pixi run -e dev python -m life_docai_analytics_antrag.classification.training.main precompute_features --batch-id <batch_id> --run-name <run_name> --modality fusion --device cuda
   ```
7. **Train Stage 2**:
   ```bash
   PYTHONPATH=packages/life-docai-analytics pixi run -e dev python -m life_docai_analytics_antrag.classification.training.main train_stage2 --batch-id <batch_id> --run-name <run_name> --modality fusion --device cuda
   ```
8. **Evaluate Stage 2** (optional):
   ```bash
   PYTHONPATH=packages/life-docai-analytics pixi run -e dev python -m life_docai_analytics_antrag.classification.training.main evaluate_stage2 --batch-id <batch_id> --run-name <run_name> --modality fusion
   ```
9. **Run the final prediction**:
   ```bash
   PYTHONPATH=packages/life-docai-analytics pixi run -e dev python -m life_docai_analytics_antrag.classification.training.main predict_stage2 --batch-id <batch_id> --run-name <run_name> --modality fusion
   ```

Steps 2 and 3 skip pages that are already done, so they can be stopped and
resumed. Training saves a checkpoint after every epoch and keeps the best one
by validation score. A run can be continued with `--resume <checkpoint>`.
Without a GPU, use `--device cpu`. A specific checkpoint or features folder can still
be given with `--checkpoint`, `--stage1-checkpoint` or `--stage1-features`.

## Outputs

- **`output/<batch_id>/`**: the dataset, OCR text, image vectors, Stage 1 features and model checkpoints.
- **`results/<batch_id>/`**: metrics for every epoch, evaluation results, and predictions with one row per page.

Existing files are never overwritten. Each run needs a new `--run-name` or a new output path.

## GWG pages (optional)

Uses the existing OCR cache; does not run OCR, CLIP or an LLM.
Writes gwg_page_scores.csv, gwg_review_sheet.csv and gwg_manifest.json to **`output/<batch_id>/gwg/<run_name>/`**.

`gwg/` finds pages with a GWG identification table, using the OCR text from step 2:

```bash
PYTHONPATH=packages/life-docai-analytics pixi run -e dev python -m life_docai_analytics_antrag.classification.gwg.gwg_candidate --batch-id <batch_id> --run-name <run_name>
```