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

```
classification/
├── training/
│   ├── config.py       paths, classes, model locations (env-overridable)
│   ├── data.py         stack population, document key, splits, sequences
│   ├── models.py       Stage 1 page model, Stage 2 sequence model
│   ├── ocr.py          OCR cache (JSONL keyed by image path)
│   ├── workflows.py    caches, Stage 1, feature extraction, Stage 2
│   ├── evaluation.py   metrics, write-once result files
│   └── main.py         single command-line entry point
├── gwg/                optional: candidate ranking and LLM labelling
├── output/<batch_id>/  population, caches, features, checkpoints
└── README.md
results/<batch_id>/     metrics and prediction files (next to classification/)
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
ingestion folder and is the key of the OCR and CLIP caches.

**Model weights**: a BERT-type text encoder (hidden size 1024) and OpenCLIP
ViT-H-14 weights. Set their locations with the environment variables below.

## Setup

Run every command from the repository root, inside the project environment:

```bash
cd <repository root>
export PYTHONPATH=.
```

| Variable | Default | Purpose |
|---|---|---|
| `PNC_ROOT_MODELS_DIR` | shared pretrained-models folder | base folder for model weights |
| `BERT_MODEL_PATH` | `<models>/gbert-large` | text encoder |
| `CLIP_MODEL_PATH` | `<models>/CLIP-ViT-H-14-…/open_clip_pytorch_model.safetensors` | CLIP weights |

Every command accepts `--batch-id <batch_id>` (the default is set in
`config.py`), `--seed` (split seed, default 13) and `--device cuda|cpu`.
`--limit-stacks N` restricts a step to the first N stacks.

`python -m antrag.classification.training.main --help` lists all steps; add a
step name for its options, e.g. `... main train_stage1 --help`. If the package
lives elsewhere, adjust the module prefix to its path from the repository root.

## Running the pipeline

Set a shortcut once:

```bash
MAIN="python -m antrag.classification.training.main"
```

Run the steps in this order. Each one checks its inputs and stops with a clear
message if something is missing.

### 1. Build the stack population

```bash
$MAIN prepare --batch-id <batch_id>
```

- **Reads:** page labels, document table, page inventory.
- **Writes:** the page table in stack order, a population report, a list of excluded stacks with reasons, the stack split file, and a split report.
- **Check:** in the population report, look at `stacks_excluded`, `primary_keys_resolved_by_fallback` and `first_page_labels_changed_vs_mapping`.

Splits are made per stack (80/10/10 by default). Once assigned, a stack keeps
its split in every later run.

### 2. Build the page caches

```bash
$MAIN compute_ocr --batch-id <batch_id> --workers 8                         # text, fusion
$MAIN compute_visual_embeddings --batch-id <batch_id> --device cuda         # vision, fusion
```

Both only process pages that are not cached yet, so they can be interrupted
and resumed. The OCR cache is never deleted: an engine mismatch stops the run
instead. For a quick smoke run, add `--training-limit-stacks N`, which covers
N train stacks and N/4 validation stacks.

### 3. Train Stage 1

```bash
$MAIN train_stage1 --batch-id <batch_id> --modality fusion --epochs 5 \
  --run-name <run_name> --device cuda [--balance-classes] [--lr-decay-epoch 3] [--use-bf16]
```

- `--modality vision|text|fusion` (see [Modalities](#modalities)).
- `--balance-classes` draws pages with probability ∝ 1 / class frequency.
- `--lr-decay-epoch E` lowers the learning rate to `--lr-decay-to` (default 2e-6) from epoch E on.
- `--resume <checkpoint>` continues a run.

It writes one checkpoint and one metrics file per epoch, plus a `best` checkpoint
selected by validation macro-F1.

### 4. Evaluate and inspect Stage 1 (optional)

```bash
$MAIN evaluate_stage1 --checkpoint <stage1_best.pt> --split test --output <results>/stage1_test.json
$MAIN predict_stage1  --checkpoint <stage1_best.pt> --output <results>/stage1_predictions.csv
```

### 5. Extract Stage 2 features

```bash
$MAIN precompute_features --stage1-checkpoint <stage1_best.pt> \
  --output <classification output>/stage1_features_<run_name>.pt --device cuda
```

This loads the trained Stage 1 model, freezes it and saves one feature vector
per page in stack order. The file records the hash of the checkpoint and of
the population it was built from.

### 6. Train Stage 2

```bash
$MAIN train_stage2 --stage1-checkpoint <stage1_best.pt> --feature-cache <features.pt> \
  --epochs 5 --run-name <run_name> --device cuda [--balance-classes]
```

`--balance-classes` samples stacks by the rarest class they contain.
`--resume <checkpoint>` continues a run.

### 7. Evaluate and predict with Stage 2

```bash
$MAIN evaluate_stage2 --stage1-checkpoint <stage1_best.pt> --feature-cache <features.pt> \
  --checkpoint <stage2_best.pt> --split test --output <results>/stage2_test.json
$MAIN predict_stage2  --stage1-checkpoint <stage1_best.pt> --feature-cache <features.pt> \
  --checkpoint <stage2_best.pt> --output <results>/stage2_predictions.csv
```

`--stage1-checkpoint` and `--feature-cache` must be the exact files used for
training. Their hashes are checked.

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
| stack page table | classification output | one row per page: stack, position, image, targets, key source |
| population report, excluded stacks, split report | classification output | counts and reasons |
| stack split file | classification output | `stack_id → train/val/test`; reused by every run |
| OCR cache, CLIP cache | classification output | keyed by `image_path`; shared across runs |
| Stage 1 features | classification output | frozen page features + page metadata |
| checkpoints | `…/checkpoints/` | `pnc_stage{1,2}_<run>_<modality>_epoch_NNN.pt` and `…_best_epoch_NNN.pt` |
| per-epoch metrics | results | `pnc_stage{1,2}_<run>_<modality>_epoch_NNN.json` |
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
| `Missing OCR text` / `Missing CLIP feature` | run step 2 for the same population |
| `OCR cache … was built with …, but this environment selects …` | run in the environment with the original OCR engine; the cache is kept |
| `Stage 1 checkpoint population is incompatible` | the population changed after Stage 1 was trained: retrain Stage 1 |
| `Feature cache population is stale` / `… does not match feature cache` | rebuild features (step 5) with the right Stage 1 checkpoint |
| `Incompatible Stage 1 checkpoint (needs a stack_v3 checkpoint)` | the checkpoint is from an older, non-stack pipeline: retrain |
| `Validation/test need at least 10 TB0 stacks` | too few stacks of that class for the split; use more data or adjust the check in `data.py` |
| `ModuleNotFoundError` | run from the repository root with `PYTHONPATH=.` inside the project environment |

## GwG labelling (optional)

`gwg/` builds training labels for pages with an identification table, in two
steps:

1. `python -m antrag.classification.gwg.candidates --run-name <name>` ranks pages by table fields and other signals. It writes the LLM input, a hand-review sheet and a manifest.
2. `python -m antrag.classification.gwg.llm_label --run-name <name> [--dry-run]` labels the selected pages with an LLM through an OpenAI-compatible endpoint, configured with `LLM_BASE_URL`, `LLM_API_KEY`, `LLM_MODEL` and `LLM_API_VERSION`. Use only an approved endpoint: the pages contain personal data.

Both scripts read the stack page table and the OCR cache from steps 1 and 2.
