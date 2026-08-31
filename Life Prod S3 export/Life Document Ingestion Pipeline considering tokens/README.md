# Life Document Ingestion Pipeline

Downloads a ZIP of documents from AWS, unpacks it safely, and prepares a
reusable, reproducible document corpus in S3 for any downstream task
(Ausweiskopie detection, GWG/identity extraction, SST classification, or
future work). The ingestion layer is task-agnostic: it records facts about
what arrived and never contains task-specific logic.

## What it produces

For one batch, the pipeline writes four things to S3, all under
`s3://<PROJECT_BUCKET>/<CORPUS_ROOT_PREFIX>/`:

- `raw/<batch_id>/source.zip` — the original ZIP, kept immutable, plus
  `ingestion_manifest.json`.
- `unpacked/<batch_id>/<masterindex_id>/...` — every PDF and CSV as an
  individual object.
- `manifests/<batch_id>/` — the three manifests that downstream tasks read:
  `unpack_manifest.parquet` (one row per ZIP member),
  `document_inventory.parquet` (one row per PDF), and
  `page_inventory.parquet` (one row per rendered page).
- `pages/<batch_id>/<document_id>/<render_version>/page_NNNN.png` — one image
  per PDF page.

The manifests are the contract: downstream tasks read them instead of scanning
S3, and every input file appears in a manifest with a status, including
errors. Nothing is silently dropped.

## The four stages

| Stage | Script | Does | Reads | Writes |
|-------|--------|------|-------|--------|
| 01 | `01_download_batch_zip.py` | Register the ZIP in the immutable raw layer, download it, validate it | Source ZIP in S3 | Raw ZIP + `ingestion_manifest.json` |
| 02 | `02_unpack_and_upload_documents.py` | Safely unpack (incl. nested ZIPs), upload PDFs/CSVs | Local ZIP | `unpack_manifest.parquet` + unpacked objects |
| 03 | `03_build_document_inventory.py` | One canonical row per PDF, with stable `document_id`, page count, readability | `unpack_manifest.parquet` | `document_inventory.parquet` |
| 04 | `04_render_pdf_pages.py` | Render pages to images with adaptive zoom | `document_inventory.parquet` | `page_inventory.parquet` + page images |

Stages 01–03 are the ingestion layer (facts about what arrived). Stage 04 is
the first derived-asset layer (images created under a versioned render config);
it is task-agnostic because page geometry is a fact, not a business rule.

## Running the pipeline

The normal way is the orchestrator, which runs the stages in order and stops at
the first failure.

```bash
# New batch: just the ZIP filename (resolved against SOURCE_ZIP_PREFIX)
python run_pipeline.py --source-s3-url Leben_2026_08_10.zip #the name of zip file depends on what you want to download
```

You type only the filename. It is resolved against `SOURCE_ZIP_PREFIX`
(the Life Prod export location) into a full `s3://...` URI. A full `s3://` URI
is also accepted if the ZIP is somewhere else.

The `batch_id` is the name for one delivery; every S3 path and manifest row is
derived from it. When omitted, it is generated from today's date and the ZIP
filename (e.g. `life_20260810_Leben_2026_08_10`) and printed at the start of
the run. Re-running the same ZIP on the same day reuses the same batch_id, so
re-runs are safe.

### Common variations

```bash
# to see the batch file add this command aws s3 ls s3://ap-mlops-life-ai/life_prod_raw_data/ there you will see the name of the files 
# how to see the raw files aws s3 ls s3://itecmcm-prod-prod-flexporter-life-prod/02_OUT/
# Re-run only rendering (e.g. after fixing an error), naming the batch explicitly
python run_pipeline.py --batch-id life_20260810_Leben_2026_08_10 --from-stage 04

# Run a range of stages
python run_pipeline.py --batch-id <id> --from-stage 02 --to-stage 03

# Pass options to stage 04 (e.g. higher DPI, force full re-render)
python run_pipeline.py --source-s3-url Leben.zip --extra-args-04 "--dpi 300 --no-resume"
```

The orchestrator writes `run_summary.json` (per-stage status and duration)
locally and to S3.

### Running a single stage

Each stage is independently runnable and takes an explicit `--batch-id`. When
run directly, stage 01 needs a full `s3://` URI (the bare-filename convenience
lives only in the orchestrator):

```bash
python 01_download_batch_zip.py --batch-id <id> --source-s3-uri s3://.../Leben.zip
python 02_unpack_and_upload_documents.py --batch-id <id>
python 03_build_document_inventory.py --batch-id <id>
python 04_render_pdf_pages.py --batch-id <id>
```

## Files

```
config.py                          Central configuration (env-overridable)
common.py                          Shared utilities (IDs, hashing, S3 helpers)
run_pipeline.py                    Orchestrator
01_download_batch_zip.py           Stage 01
02_unpack_and_upload_documents.py  Stage 02
03_build_document_inventory.py     Stage 03
04_render_pdf_pages.py             Stage 04
```

## Dependencies

Python 3.10+, with `boto3`, `pyarrow`, `pypdf`, and `PyMuPDF` (imported as
`fitz`).
