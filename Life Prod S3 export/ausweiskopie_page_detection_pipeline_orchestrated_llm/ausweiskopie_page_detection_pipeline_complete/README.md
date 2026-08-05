# Ausweiskopie Page-Detection Pipeline

This pipeline detects possible Ausweiskopie pages without using SST.

## Main output

```json
{
  "masterindex_id": "MI123",
  "expected_page_count": 4,
  "processed_page_count": 4,
  "successful_screening_count": 4,
  "technical_failure_count": 0,
  "failed_pdf_count": 0,
  "is_complete": true,
  "pages": [
    {"page": 1, "label": "not_ausweiskopie"},
    {"page": 2, "label": "ausweiskopie"},
    {"page": 3, "label": "ausweiskopie"},
    {"page": 4, "label": "not_ausweiskopie"}
  ],
  "ausweis_segments": [
    {"start_page": 2, "end_page": 3}
  ]
}
```

## Input

Place the archive at:

```text
Ausweiskopie/Input/Ausweisekopie.zip
```

Each direct folder inside the ZIP is one `masterindex_id`.

SST is not required and is not used.

## Install

```bash
pip install -r requirements.txt
```


## SecureGPT vision connection

The image connection is implemented in:

```text
securegpt_vision.py
```

The code performs:

```text
local PNG/JPEG
→ read bytes
→ Base64 encode
→ data:image/...;base64,... URL
→ SecureGPT.new_chat(user_image=...)
→ Pydantic-validated label
```

Set:

```bash
export SECUREGPT_MODEL_NAME="approved-model-name"
export SECUREGPT_MODEL_VERSION="approved-model-version"
```

Before the full pipeline, test one non-customer image:

```bash
python 00_test_securegpt_vision.py
```

Place the test image at:

```text
test_page.png
```

The implementation uses `response_model=AusweisPageResponse`. If the installed
wrapper version does not support structured output together with `user_image`,
the code falls back to strict JSON and validates it locally with Pydantic.


## Run order

### 1. Scan all PDFs

```bash
python 01_scan_downloaded_documents.py
```

Output:

```text
outputs/document_inventory.csv
```

Every PDF is inventoried. Metadata and SST are not used as filters.

### 2. Render every page

```bash
python 02_render_pdf_pages.py
```

Outputs:

```text
Ausweiskopie/RenderedPages/
outputs/page_inventory.csv
```

The render script now:

- catches errors per page;
- preserves successful pages when another page fails;
- advances page numbering even after a failure;
- renders screening images at 200 DPI;
- records image dimensions, size, timestamp and SHA-256;
- performs a small blank, size, contrast and corruption check.

Confirmed Ausweis pages should later be rendered again at 300 DPI for field
extraction. Rendering the complete corpus at extraction resolution is avoided
because it creates unnecessary PII-heavy files.

### 3. Connect SecureGPT and screen pages

Set model metadata:

```bash
export SECUREGPT_MODEL_NAME="approved-model-name"
export SECUREGPT_MODEL_VERSION="approved-model-version"
```

Connect the approved internal client inside:

```python
run_securegpt_page_screening(...)
```

Use these settings when supported:

```text
temperature = 0
top_p = 1
seed = 42
```

The SecureGPT result must be:

```python
{
    "label": "ausweiskopie"
             | "not_ausweiskopie"
             | "unclear",
    "evidence_code": "one approved coarse evidence code"
}
```

The prompt must not allow names, dates, addresses, document numbers, MRZ text
or other personal values in the response.

Run:

```bash
python 03_screen_ausweis_pages.py
```

Outputs:

```text
outputs/ausweis_page_screening.jsonl
outputs/ausweis_screening_predictions.jsonl
```

Checkpoint reuse requires the same:

```text
masterindex_id
pdf_path_in_zip
source_page_number
image_sha256
```

A changed image is therefore reprocessed instead of silently receiving an old
result.

Render failures, corrupt images and clearly unusable images become `unclear`.
They never become negative.

The nested output includes page counts and `is_complete`, so an interrupted
run cannot look like a complete result.

Segments do not cross PDF boundaries.

### 4. Prepare human review

```bash
python 04_prepare_detection_review.py
```

Output:

```text
outputs/ausweis_detection_review.xlsx
```

The workbook contains:

- all predicted positives;
- all unclear pages;
- a reproducible global random sample of predicted negatives.

Global page sampling is retained because it supports a corpus-level page miss
rate. The workbook also stores:

```text
sampling_probability
sampling_weight
```

and reports MasterIndex audit coverage.

### 5. Build classification Ground Truth

```bash
python 05_build_ausweis_classification_ground_truth.py
```

Outputs:

```text
outputs/ausweis_classification_ground_truth_pages.csv
outputs/ausweis_classification_ground_truth.jsonl
outputs/ausweis_classification_excluded_rows.csv
```

Only reviewed rows become Ground Truth.

The nested file explicitly states:

```text
ground_truth_scope = reviewed_pages_only
```

### 6. Evaluate detection and the negative audit

```bash
python 06_evaluate_detection_audit.py
```

Outputs:

```text
outputs/ausweis_detection_audit_summary.csv
outputs/ausweis_detection_review_errors.csv
```

The misleading flat `review_accuracy` metric has been removed.

The evaluation reports:

- agreement inside reviewed rows, clearly caveated;
- predicted-positive confirmation rate;
- positive review coverage and exclusions;
- negative-audit missed-positive rate;
- 95% Wilson intervals;
- full predicted-negative pool size;
- estimated missed-positive pages;
- approximate page-level detection recall;
- unclear-page rate;
- MasterIndex coverage of the page audit.

The extrapolated recall is an estimate, not a directly observed corpus metric.

### 7. Delete rendered images after the approved retention period

```bash
python 07_cleanup_rendered_pages.py
```

Deletion is disabled by default:

```python
ALLOW_DELETE = False
RETENTION_DAYS = 30
```

Change these only after the AXA retention period is approved.

The script:

- refuses deletion if review rows remain incomplete;
- deletes only images old enough for the configured retention period;
- writes a deletion audit to:

```text
outputs/rendered_page_deletion_audit.csv
```

## Shared utilities

The duplicated JSONL and segment logic is now located in:

```text
pipeline_utils.py
```

No framework or configuration package was added.

## Privacy

- Use only AXA-approved SecureGPT.
- Do not send images to external services.
- Evidence is restricted to coarse codes.
- Rendered images have creation, checksum and deletion metadata.
- Cleanup stays disabled until retention policy is confirmed.

## Run with the orchestrator

Test the SecureGPT image connection:

```bash
python run_pipeline.py test
```

Run all automatic steps before human review:

```bash
python run_pipeline.py pre-review
```

Complete:

```text
outputs/ausweis_detection_review.xlsx
```

Then run:

```bash
python run_pipeline.py post-review
```

Cleanup remains separate:

```bash
python run_pipeline.py cleanup
```
