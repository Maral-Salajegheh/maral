# Page Classification and Document Segmentation

This pipeline takes an incoming stack of scanned pages, in its original order,
and for every page predicts:

- which document class it belongs to, and
- whether a new document starts on that page.

From these two predictions the stack is split into its documents.

## How it works

The pipeline has two stages, following the PnC architecture.

**Stage 1: page model.** Each page is classified on its own. The text of the
page (from OCR) is read by a fine-tuned **GBERT-large** model. In the fusion
setting, an image vector of the page from **OpenCLIP ViT-H-14** is added to the
text input. For every page the model predicts the class, whether it is the
first page of a document, and whether it is the last page.

**Stage 2: sequence model.** The trained Stage 1 model is frozen and turns every
page into a feature vector. A two-layer **GRU** then reads the pages of a whole
stack in order. Using the context of the pages around it, it predicts each
page's class and whether a document starts there.

Three input settings are available:

| Setting | Uses | Note |
|---|---|---|
| `fusion` | text and image | main setting |
| `text` | text only | |
| `vision` | image only | baseline; Stage 1 training does not reach Stage 2 |

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
├── gwg/                optional: labelling of identification pages with an LLM
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
   PYTHONPATH=packages/life-docai-analytics pixi run -e dev python -m life_docai_analytics_antrag.classification.training.main train_stage1 --batch-id <batch_id> --modality fusion --run-name <run_name> --device cuda
   ```
5. **Evaluate Stage 1** (optional):
   ```bash
   PYTHONPATH=packages/life-docai-analytics pixi run -e dev python -m life_docai_analytics_antrag.classification.training.main evaluate_stage1 --batch-id <batch_id> --checkpoint <stage1_checkpoint>
   ```
6. **Extract the page features for Stage 2**:
   ```bash
   PYTHONPATH=packages/life-docai-analytics pixi run -e dev python -m life_docai_analytics_antrag.classification.training.main precompute_features --batch-id <batch_id> --stage1-checkpoint <stage1_checkpoint> --device cuda
   ```
7. **Train Stage 2**:
   ```bash
   PYTHONPATH=packages/life-docai-analytics pixi run -e dev python -m life_docai_analytics_antrag.classification.training.main train_stage2 --batch-id <batch_id> --stage1-checkpoint <stage1_checkpoint> --stage1-features <features_folder> --run-name <run_name> --device cuda
   ```
8. **Evaluate Stage 2** (optional):
   ```bash
   PYTHONPATH=packages/life-docai-analytics pixi run -e dev python -m life_docai_analytics_antrag.classification.training.main evaluate_stage2 --batch-id <batch_id> --stage1-checkpoint <stage1_checkpoint> --stage1-features <features_folder> --checkpoint <stage2_checkpoint>
   ```
9. **Run the final prediction**:
   ```bash
   PYTHONPATH=packages/life-docai-analytics pixi run -e dev python -m life_docai_analytics_antrag.classification.training.main predict_stage2 --batch-id <batch_id> --stage1-checkpoint <stage1_checkpoint> --stage1-features <features_folder> --checkpoint <stage2_checkpoint>
   ```

Steps 2 and 3 skip pages that are already done, so they can be stopped and
resumed. Training saves a checkpoint after every epoch and keeps the best one
by validation score. A run can be continued with `--resume <checkpoint>`.

## Outputs

- **`output/<batch_id>/`**: the dataset, OCR text, image vectors, Stage 1 features and model checkpoints.
- **`results/<batch_id>/`**: metrics for every epoch, evaluation results, and predictions with one row per page.

Existing files are never overwritten. Each run needs a new `--run-name` or a new output path.

## GWG labelling (optional)

`gwg/` finds pages with an identification table and labels them with an LLM:

```bash
python -m life_docai_analytics_antrag.classification.gwg.candidates --batch-id <batch_id> --run-name <name>
python -m life_docai_analytics_antrag.classification.gwg.llm_label --batch-id <batch_id> --run-name <name>
```

The LLM endpoint is configured with environment variables. Use only an
approved endpoint, because the pages contain personal data.
