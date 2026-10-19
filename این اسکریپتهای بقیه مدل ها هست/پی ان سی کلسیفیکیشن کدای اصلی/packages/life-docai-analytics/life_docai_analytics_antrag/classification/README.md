# Page Classification and Document Segmentation

Classifies every page of an incoming stack and finds the document boundaries
inside it. The design follows the two-stage PnC architecture:

```
raw stack (pages in original order)
        │
        ▼
Stage 1  page model      one page at a time  → class, first-page, last-page
        │  frozen page representations
        ▼
Stage 2  sequence model  the whole stack     → class and document start per page
```

The unit of work is the **stack**: the complete incoming delivery, with its pages
in their original order before any manual or automatic separation. Training,
evaluation and inference all use this order, so the model sees the same input
it will get in production.

## Contents

- [How it works](#how-it-works)
- [Package layout](#package-layout)
- [Input data](#input-data)
- [Setup](#setup)
- [Running the pipeline](#running-the-pipeline)
- [Outputs](#outputs)
- [Modalities](#modalities)
- [Safeguards](#safeguards)
- [Troubleshooting](#troubleshooting)
- [GwG labelling (optional)](#gwg-labelling-optional)

## How it works

**Stage 1 (page model).** A pretrained BERT-type encoder is fine-tuned on each
page's OCR text. In fusion mode, a frozen CLIP image vector is projected and
added to the `[CLS]` token embedding. Three heads sit on the `[CLS]` output:
page class, first page of a document, last page of a document.

**Stage 1 → Stage 2.** The trained Stage 1 model is frozen and run over every
page. Its `[CLS]` vector is saved as that page's feature (in fusion mode,
together with the raw CLIP vector). Stage 2 never retrains Stage 1.

**Stage 2 (sequence model).** A two-layer GRU reads all pages of a stack in
order and predicts, per page, the class and whether a new document starts
there. The first page of a stack is always a document start: it is masked in
training and set to 1 at inference.

**Document boundaries.** A page starts a new document when its document key
differs from the previous page in raw order.

| Key | Fields | Used when |
|---|---|---|
| Primary | `stack_id + doc_id + sfdoc_class` | normally |
| Fallback | `stack_id + masterindex_id + process_id + doc_id + subdoc_idx + sfdoc_class` | the primary key is ambiguous (covers several fallback keys), incomplete (an empty field) or not one contiguous run in raw order |

Each page records which key was used (`document_key_source`), and the
population report counts the fallback cases by reason.

## Package layout

The code is part of the `life_docai_analytics_antrag` package:

```
packages/life-docai-analytics/
└── life_docai_analytics_antrag/
    ├── classification/          this pipeline (below)
    ├── datasets/
    │   ├── aws_download_ingestion/   page inventory and rendered images
    │   └── mapping/                  page labels and document table
    └── __init__.py
```

All paths are derived from the package folder in `training/config.py`, so the
pipeline does not depend on the current working directory.

```
classification/
├── training/
│   ├── config.py       paths, classes, model locations (env-overridable)
│   ├── data.py         stack population, document key, splits, sequences
│   ├── models.py       Stage 1 page model, Stage 2 sequence model
│   ├── ocr.py          OCR text per page (JSONL keyed by image path)
│   ├── storage.py      memory-mapped stores: CLIP shards, Stage 1 features
│   ├── workflows.py    caches, Stage 1, feature extraction, Stage 2
│   ├── evaluation.py   metrics, write-once result files
│   └── main.py         single command-line entry point
├── gwg/                optional: candidate ranking and LLM labelling
├── output/<batch_id>/  stack pages, page caches, Stage 1 features, checkpoints
├── results/<batch_id>/ metrics, evaluations and prediction files
└── README.md
```

## Input data

Three files per batch, produced upstream (mapping and ingestion). Their paths
are defined in `training/config.py`.

**1. Page labels** (one row per labelled page; CSV)

| Column | Meaning |
|---|---|
| `stack_id` | the incoming stack the page belongs to |
| `image_id` | numeric; the page's original position in the stack |
| `masterindex_id`, `pdf_page_number` | where the rendered image of the page is (file + page) |
| `process_id`, `doc_id`, `subdoc_idx` | document identity |
| `sfdoc_class` | fine document class (part of the document key) |
| `page_class` | target class; must be one of `CLASSES` in `config.py` |
| `is_first_page` | original first-page flag (only used for comparison) |
| `label_tier`, `training_label_quality` | label provenance, reported per quality |

**2. Document table** (one row per mapped document; CSV) with `stack_id`,
`masterindex_id` and `is_usable`. A stack with any unusable document is
excluded, because its pages would be missing from the sequence.

**3. Page inventory** (Parquet, from rendering) with `masterindex_id`,
`source_page_number`, `image_path`, `image_sha256` and `status`. Only rows
with `status == "success"` are used. `image_path` is relative to the
ingestion folder and is the key of the OCR text and CLIP features.

**Model weights**: a BERT-type text encoder (hidden size 1024) and OpenCLIP
ViT-H-14 weights. Set their locations with the environment variables below.

## Setup

Run every command from the repository root, inside the project environment.
If the package is installed in the environment (for example as an editable
install), no `PYTHONPATH` is needed; otherwise point it at the package's parent
folder:

```bash
cd <repository root>
export PYTHONPATH=packages/life-docai-analytics
```

| Variable | Default | Purpose |
|---|---|---|
| `PNC_ROOT_MODELS_DIR` | shared pretrained-models folder | base folder for model weights |
| `BERT_MODEL_PATH` | `<models>/gbert-large` | text encoder |
| `CLIP_MODEL_PATH` | `<models>/CLIP-ViT-H-14-…/open_clip_pytorch_model.safetensors` | CLIP weights |

Every command accepts `--batch-id <batch_id>` (the default is set in
`config.py`), `--seed` (split seed, default 13) and `--device cuda|cpu`.
`--limit-stacks N` restricts a step to the first N stacks.

`python -m life_docai_analytics_antrag.classification.training.main --help`
lists all steps; add a step name for its options, e.g. `... main train_stage1 --help`.

## Running the pipeline

Set a shortcut once:

```bash
MAIN="python -m life_docai_analytics_antrag.classification.training.main"
```

Run the steps in this order. Each one checks its inputs and stops with a clear
message if something is missing.

### 1. Build the stack population

```bash
$MAIN prepare --batch-id <batch_id>
```

- **Reads:** page labels, document table, page inventory.
- **Writes:** `stack_pages.csv` (pages in stack order with all targets), `stack_population_report.json`, `excluded_stacks.json` (reasons), `stack_splits.csv` and `stack_split_report.json`.
- **Check:** in the population report, look at `stacks_excluded`, `primary_keys_resolved_by_fallback` and `first_page_labels_changed_vs_mapping`.

Splits are made per stack (80/10/10 by default). Once assigned, a stack keeps
its split in every later run.

### 2. Build the page caches

```bash
$MAIN compute_ocr --batch-id <batch_id> --workers 8                         # text, fusion
$MAIN compute_visual_embeddings --batch-id <batch_id> --device cuda         # vision, fusion
```

Both only process pages that are not stored yet, so they can be interrupted
and resumed.

- **OCR text** goes to `ocr_text.jsonl`, one line per page. It is never deleted: an engine mismatch stops the run instead.
- **CLIP vectors** go to `clip_features/`, as write-once shards of 4,096 pages plus `index.jsonl`. Adding pages never rewrites existing shards. Reading memory-maps them, so the full set is never loaded into RAM.

For a quick smoke run, add `--training-limit-stacks N`, which covers N train
stacks and N/4 validation stacks.

**Existing single-file CLIP cache** (`{image_path: tensor}` in one `.pt`):
convert it once instead of recomputing. The vectors stay identical, and the
step can be repeated safely.

```bash
$MAIN migrate_clip_cache --batch-id <batch_id> --source <old_clip_cache.pt>
```

### 3. Train Stage 1

```bash
$MAIN train_stage1 --batch-id <batch_id> --modality fusion --epochs 5 \
  --run-name <run_name> --device cuda [--no-balance-classes] [--lr-decay-epoch 3] [--use-bf16]
```

- `--modality vision|text|fusion` (see [Modalities](#modalities)).
- Class balancing is **on by default**: pages are drawn with probability ∝ 1 / class frequency. `--no-balance-classes` turns it off.
- `--lr-decay-epoch E` lowers the learning rate to `--lr-decay-to` (default 2e-6) from epoch E on.
- `--resume <checkpoint>` continues a run.

Per epoch it writes `model_stage1_<modality>_<run>_epoch_NNN.pt` and
`metrics_stage1_<modality>_<run>_epoch_NNN.json`, plus
`model_stage1_<modality>_<run>_best_epoch_NNN.pt` whenever validation macro-F1
improves.

### 4. Evaluate and inspect Stage 1 (optional)

```bash
$MAIN evaluate_stage1 --checkpoint <stage1_best.pt> --split test --output <results>/evaluation_stage1_test.json
$MAIN predict_stage1  --checkpoint <stage1_best.pt> --output <results>/predictions_stage1.csv
```

### 5. Extract Stage 2 features

```bash
$MAIN precompute_features --stage1-checkpoint <stage1_best.pt> --device cuda
```

This loads the trained Stage 1 model, freezes it and writes one feature row per
page in stack order, to `stage1_features_<modality>_<checkpoint name>/`
(`--output` sets another folder):

- `features.npy`: the feature matrix, memory-mapped when read;
- `pages.csv`: one row per page, with its row number and targets;
- `metadata.json`: written last, with the hashes of the checkpoint, the population and `features.npy`.

Each stack is one contiguous block of rows, so Stage 2 reads only the stacks
of the current batch from disk.

### 6. Train Stage 2

```bash
$MAIN train_stage2 --stage1-checkpoint <stage1_best.pt> --stage1-features <features dir> \
  --epochs 5 --run-name <run_name> --device cuda [--no-balance-classes] [--max-pages-per-batch 2048]
```

- Class balancing is **on by default**: stacks are sampled by the rarest class they contain. `--no-balance-classes` turns it off.
- **Batches group stacks of similar length.** `--max-pages-per-batch` limits the padded size of a batch (stacks × longest stack, default 2,048), so a very long stack gets a batch of its own instead of padding the others. Predictions are not affected, because padding sits at the end of a unidirectional GRU.
- `--resume <checkpoint>` continues a run.

Checkpoints and metrics follow the Stage 1 pattern, with `final` for Stage 2:
`model_final_<modality>_<run>_epoch_NNN.pt`, `…_best_epoch_NNN.pt` and
`metrics_final_<modality>_<run>_epoch_NNN.json`.

### 7. Evaluate and predict with Stage 2

```bash
$MAIN evaluate_stage2 --stage1-checkpoint <stage1_best.pt> --stage1-features <features dir> \
  --checkpoint <final_best.pt> --split test --output <results>/evaluation_final_test.json
$MAIN predict_stage2  --stage1-checkpoint <stage1_best.pt> --stage1-features <features dir> \
  --checkpoint <final_best.pt> --output <results>/predictions_final.csv
```

`--stage1-checkpoint` and `--stage1-features` must be the ones used for
training. Their hashes are checked. Prediction rows are written in stack
order.

### When to re-run what

| You changed | Re-run from |
|---|---|
| page labels, document table, page inventory | step 1 (delete the old population files first) |
| new pages, or a cache missing pages | step 2 |
| Stage 1 settings | step 3 |
| Stage 2 settings only | step 6 |

## Outputs

| File | Where | Contents |
|---|---|---|
| `stack_pages.csv` | classification output | one row per page: stack, position, image, targets, key source |
| `stack_population_report.json`, `excluded_stacks.json`, `stack_split_report.json` | classification output | counts and reasons |
| `stack_splits.csv` | classification output | `stack_id → train/val/test`; reused by every run |
| `ocr_text.jsonl` | classification output | OCR text per page, keyed by `image_path` |
| `clip_features/` | classification output | CLIP vector per page: shards + `index.jsonl` |
| `stage1_features_<modality>_<checkpoint>/` | classification output | `features.npy`, `pages.csv`, `metadata.json` |
| `model_stage1_…`, `model_final_…` | `…/checkpoints/` | `<modality>_<run>_epoch_NNN.pt` and `…_best_epoch_NNN.pt` |
| `metrics_stage1_…`, `metrics_final_…` | results | per-epoch train and validation metrics |
| evaluation / predictions | results | the paths given with `--output` |

**Metrics.** Per-class precision, recall and F1; macro-F1 (over the classes
that occur); weighted F1; confusion matrix; and, for Stage 1, first-page and
last-page scores, also broken down by label quality. Stage 2 segmentation
scores exclude the first page of each stack.

**Prediction files.** One row per page in stack order, with the true and
predicted class, the class probabilities, and the document-start probability.
The document-start probability is always 1.0 for the first page of a stack.

## Modalities

| Modality | Stage 1 input | Feature passed to Stage 2 | Stage 1 training reaches Stage 2 |
|---|---|---|---|
| `text` | OCR text | fine-tuned `[CLS]` (1024) | yes |
| `fusion` | OCR text + CLIP vector | fine-tuned `[CLS]` + raw CLIP (2048) | yes, through the `[CLS]` part |
| `vision` | CLIP vector | raw CLIP vector (1024) | no; only the Stage 1 heads are trained |

`fusion` is the reference configuration. `vision` is a baseline: its Stage 2
learns from the frozen CLIP vectors alone, and the Stage 1 checkpoint does not
change its features.

## Safeguards

- **Nothing is overwritten.** Every output (population, checkpoints, metrics, predictions, features) refuses to replace an existing file. Use a new `--run-name` or `--output`.
- **Inputs are bound to each other.** Checkpoints and feature files carry the population hash and the hash of the Stage 1 checkpoint. A mismatch stops the run.
- **Stacks are complete and ordered.** Every step checks that each stack has positions `1..n` in original order and no duplicate pages.
- **Splits are stable** and made per stack, so no stack appears in two splits.

## Troubleshooting

| Message | Cause and fix |
|---|---|
| `Refusing to overwrite existing file` / `Output already exists` | choose a new `--run-name` or `--output`, or delete the old file on purpose |
| `Missing OCR text` / `Missing CLIP feature(s)` | run step 2 for the same population |
| `OCR cache … was built with …, but this environment selects …` | run in the environment with the original OCR engine; the cache is kept |
| `Stage 1 checkpoint population is incompatible` | the population changed after Stage 1 was trained: retrain Stage 1 |
| `Stage 1 features were built for another population` / `… does not match the Stage 1 features` | rebuild the features (step 5) with the right Stage 1 checkpoint |
| `Not a Stage 1 feature directory` | the folder is incomplete (no `metadata.json`) or from an older format: rebuild it |
| `Incompatible Stage 1 checkpoint (needs a stack-level checkpoint)` | the checkpoint is from an older pipeline: retrain |
| GPU out of memory in Stage 2 | lower `--max-pages-per-batch` |
| `Validation/test need at least 10 TB0 stacks` | too few stacks of that class for the split; use more data or adjust the check in `data.py` |
| `ModuleNotFoundError: life_docai_analytics_antrag` | run from the repository root with `PYTHONPATH=packages/life-docai-analytics`, or install the package in the environment |

## GwG labelling (optional)

`gwg/` builds training labels for pages with an identification table, in two
steps:

1. `python -m life_docai_analytics_antrag.classification.gwg.candidates --run-name <name>` ranks pages by table fields and other signals. It writes the LLM input, a hand-review sheet and a manifest.
2. `python -m life_docai_analytics_antrag.classification.gwg.llm_label --run-name <name> [--dry-run]` labels the selected pages with an LLM through an OpenAI-compatible endpoint, configured with `LLM_BASE_URL`, `LLM_API_KEY`, `LLM_MODEL` and `LLM_API_VERSION`. Use only an approved endpoint: the pages contain personal data.

Both scripts read `stack_pages.csv` and `ocr_text.jsonl` from steps 1 and 2
(`--ocr-text` selects another OCR file).
