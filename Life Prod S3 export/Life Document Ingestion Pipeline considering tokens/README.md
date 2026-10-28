# Life Document Ingestion Pipeline

Downloads a ZIP of Life documents from the export bucket, keeps the original
ZIP in the project bucket, and prepares a local, reproducible page corpus
(inventories + rendered page images) for downstream tasks such as page
classification or Ausweiskopie extraction. The pipeline is task-agnostic: it
records facts about what arrived and contains no task-specific logic.

## Credentials: two steps

The source bucket and the project bucket need different credentials, so a new
batch always runs in two steps.

| Step | Stages | Credentials |
|------|--------|-------------|
| 1 | 01 (download) | Temporary AWS tokens in `~/.aws/credentials` |
| 2 | 01b, 02, 03 | **No tokens** (environment role) |

Stage 01 is the only stage that reads the source bucket. The orchestrator
stops after it and prints a reminder to remove the tokens before continuing.

## Stages

| Stage | Script | Does | Writes |
|-------|--------|------|--------|
| 01 | `01_download_batch_zip.py` | Downloads the ZIP, checks size and CRC, computes SHA-256 | Local `source.zip` + `ingestion_manifest.json` |
| 01b | `01b_register_raw_zip.py` | Uploads the ZIP and its manifest to the immutable raw layer | `s3://<PROJECT_BUCKET>/<RAW_DATA_PREFIX>/<batch_id>/` |
| 02 | `02_build_document_inventory.py` | Scans the ZIP without extracting; one row per MasterIndex folder (its PDF and CSV), with page count and readability | Local `document_inventory.parquet` |
| 03 | `03_render_pdf_pages.py` | Renders every PDF page to an image with adaptive zoom, in parallel worker processes (`--workers`, default: CPU count - 1); each worker reads one PDF at a time from the ZIP; resumes documents already rendered | Local `page_inventory.parquet` + page images |

Only the raw ZIP and `ingestion_manifest.json` are stored in S3. Inventories
and images stay on the local server.

## Running a new batch

Run from the pipeline directory; output paths are relative to it. From the
repository root:

```bash
cd packages/life-docai-analytics/life_docai_analytics_antrag/datasets/aws_download_ingestion
```

```bash
# Step 1 - WITH tokens: download only. Prints the derived batch_id.
python run_pipeline.py --source-s3-uri <zip_filename>

# Remove the tokens
rm ~/.aws/credentials

# Step 2 - WITHOUT tokens: register the raw ZIP, build inventory, render
python run_pipeline.py --batch-id <batch_id> --from-stage 01b

# Step 2 with a chosen number of render workers
python run_pipeline.py --batch-id <batch_id> --from-stage 01b --extra-args-03 "--workers 64"
```

`<zip_filename>` is the name of the ZIP in `SOURCE_ZIP_PREFIX`
(`aws s3 ls s3://itecmcm-prod-prod-flexporter-life-prod/02_OUT/`).
`--source-s3-uri` accepts that bare filename or a full `s3://` URI.

The `batch_id` names one delivery; every path is derived from it. When
omitted in step 1 it is built as `life_<YYYYMMDD>_<zip filename>` and printed.
From stage 01b on it must be passed explicitly.

### Parallel rendering

Stage 03 renders documents in parallel. `--extra-args-03` is passed to stage 03
whenever stage 03 is part of the run, so `--workers` can be set with any
`--from-stage` (01b, 02 or 03). Without it, CPU count - 1 workers are used.

Every worker holds one PDF and one rendered page in memory, and all workers
write images to the same disk. On a shared server, start with a moderate
number (for example 64) and check `free -h`; more workers stop helping once the
disk is the bottleneck. The inventory keeps the document order of
`document_inventory.parquet`, so the output does not depend on the worker count.

### Local ZIP cleanup

After a successful run that includes stage 03, the orchestrator deletes the
local `source.zip` when every page in `page_inventory.parquet` has status
`success` and the raw copy in S3 has the same size. Otherwise the ZIP is kept
and the reason is printed. The result is recorded under `zip_cleanup` in
`run_summary.json`. Pass `--keep-zip` to keep it.

To run stage 02 or 03 again after the ZIP was deleted, restore it from the raw
layer first (the command is printed after deletion):

```bash
aws s3 cp s3://<PROJECT_BUCKET>/<RAW_DATA_PREFIX>/<batch_id>/source.zip <LOCAL_TEMP_ROOT>/<batch_id>/input/source.zip
```

### Common variations

```bash
# Re-run only rendering
python run_pipeline.py --batch-id <batch_id> --from-stage 03

# Re-run only rendering with a chosen number of workers
python run_pipeline.py --batch-id <batch_id> --from-stage 03 --extra-args-03 "--workers 64"

# Keep the local ZIP after rendering
python run_pipeline.py --batch-id <batch_id> --from-stage 03 --keep-zip

# A range of stages
python run_pipeline.py --batch-id <batch_id> --from-stage 02 --to-stage 02

# Options for stage 03 (higher DPI, force a full re-render)
python run_pipeline.py --batch-id <batch_id> --from-stage 03 \
  --extra-args-03 "--dpi 300 --no-resume --workers 64"

# Use a different staging location for the ZIP
LOCAL_TEMP_ROOT=<path> python run_pipeline.py --batch-id <batch_id> --from-stage 03

# List available source ZIPs / registered raw batches
aws s3 ls s3://itecmcm-prod-prod-flexporter-life-prod/02_OUT/
aws s3 ls s3://ap-mlops-life-ai/life_prod_raw_data/
```

Each stage can also be run on its own with `--batch-id <batch_id>`. Run
directly, stage 01 needs a full `s3://` URI; stage 01b accepts
`--overwrite-raw` to replace an existing raw ZIP; stage 03 accepts
`--workers N`.

## Where things are written

| What | Location |
|------|----------|
| Downloaded ZIP (deleted after a clean render) | `<LOCAL_TEMP_ROOT>/<batch_id>/input/source.zip` |
| `ingestion_manifest.json`, `run_summary.json` | `<LOCAL_TEMP_ROOT>/<batch_id>/manifests/` |
| `document_inventory.parquet`, `page_inventory.parquet` | `<LOCAL_OUTPUT_ROOT>/<batch_id>/` |
| Page images | `<RENDER_ROOT>/<batch_id>/<masterindex_id>/<render_version>/page_NNNN.png` |
| Raw ZIP + manifest | `s3://<PROJECT_BUCKET>/<RAW_DATA_PREFIX>/<batch_id>/` |

The inventories are the contract for downstream tasks. Every MasterIndex and
every page appears with a `status`, including errors (`missing_pdf`,
`invalid_pdf`, `encrypted_pdf`, `error`); nothing is silently dropped. Rendered
pages also carry a `quality_status` (`ok`, `too_small`, `likely_blank`,
`low_contrast`).

## Configuration

All values in `config.py` can be overridden with environment variables.

| Variable | Default | Meaning |
|----------|---------|---------|
| `LOCAL_TEMP_ROOT` | `/home/shared_folders/life_ai/life-document-ai-temp` | Staging for the ZIP and manifests |
| `LOCAL_OUTPUT_ROOT` | `output` | Inventories |
| `RENDER_ROOT` | `RenderedPages` | Page images |
| `SOURCE_ZIP_PREFIX` | `s3://itecmcm-prod-prod-flexporter-life-prod/02_OUT` | Where source ZIPs are looked up |
| `PROJECT_BUCKET` / `RAW_DATA_PREFIX` | `ap-mlops-life-ai` / `life_prod_raw_data` | Raw layer |
| `RENDER_DPI` / `RENDER_FORMAT` / `RENDER_VERSION` | `200` / `png` / `v1` | Rendering |

Use the same `LOCAL_TEMP_ROOT` for every step of one batch; later stages look
for the ZIP there.

## Files

```
config.py                        Central configuration (env-overridable)
common.py                        Shared utilities (IDs, hashing, Parquet, S3 helpers)
run_pipeline.py                  Orchestrator (deletes the local ZIP after a clean render)
01_download_batch_zip.py         Stage 01  (needs tokens)
01b_register_raw_zip.py          Stage 01b (no tokens)
02_build_document_inventory.py   Stage 02
03_render_pdf_pages.py           Stage 03  (parallel)
```

## Dependencies

Python 3.10+, with `boto3`, `pyarrow`, `pypdf`, `PyMuPDF` (imported as
`fitz`), `Pillow`, and `numpy`.