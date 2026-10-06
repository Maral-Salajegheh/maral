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
| 03 | `03_render_pdf_pages.py` | Renders every PDF page to an image with adaptive zoom; resumes documents already rendered | Local `page_inventory.parquet` + page images |

Only the raw ZIP and `ingestion_manifest.json` are stored in S3. Inventories
and images stay on the local server.

## Running a new batch

Run from this directory (output paths are relative to it).

```bash
# Step 1 - WITH tokens: download only. Prints the derived batch_id.
python run_pipeline.py --source-s3-uri Leben_2026_08_10.zip

# Remove the tokens
rm ~/.aws/credentials

# Step 2 - WITHOUT tokens: register the raw ZIP, build inventory, render
python run_pipeline.py --batch-id <batch_id> --from-stage 01b
```

`--source-s3-uri` accepts a bare filename (resolved against
`SOURCE_ZIP_PREFIX`) or a full `s3://` URI.

The `batch_id` names one delivery; every path is derived from it. When
omitted in step 1 it is built as `life_<YYYYMMDD>_<zip filename>` and printed.
From stage 01b on it must be passed explicitly.

### Common variations

```bash
# Re-run only rendering
python run_pipeline.py --batch-id <batch_id> --from-stage 03

# A range of stages
python run_pipeline.py --batch-id <batch_id> --from-stage 02 --to-stage 02

# Options for stage 03 (higher DPI, force a full re-render)
python run_pipeline.py --batch-id <batch_id> --from-stage 03 \
  --extra-args-03 "--dpi 300 --no-resume"

# Use a different staging location for the ZIP
LOCAL_TEMP_ROOT=/home/shared_folders/life_ai/life-document-ai-temp \
  python run_pipeline.py --batch-id <batch_id> --from-stage 03

# List available source ZIPs / registered raw batches
aws s3 ls s3://itecmcm-prod-prod-flexporter-life-prod/02_OUT/
aws s3 ls s3://ap-mlops-life-ai/life_prod_raw_data/
```

Each stage can also be run on its own with `--batch-id <batch_id>`. Run
directly, stage 01 needs a full `s3://` URI; stage 01b accepts
`--overwrite-raw` to replace an existing raw ZIP.

## Where things are written

| What | Location |
|------|----------|
| Downloaded ZIP | `<LOCAL_TEMP_ROOT>/<batch_id>/input/source.zip` |
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
| `LOCAL_TEMP_ROOT` | `/tmp/life-document-ai` | Staging for the ZIP and manifests |
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
run_pipeline.py                  Orchestrator
01_download_batch_zip.py         Stage 01  (needs tokens)
01b_register_raw_zip.py          Stage 01b (no tokens)
02_build_document_inventory.py   Stage 02
03_render_pdf_pages.py           Stage 03
```

## Dependencies

Python 3.10+, with `boto3`, `pyarrow`, `pypdf`, `PyMuPDF` (imported as
`fitz`), `Pillow`, and `numpy`.