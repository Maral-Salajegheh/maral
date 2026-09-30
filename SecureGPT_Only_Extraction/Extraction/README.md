# Identity Document Extraction

This pipeline uses SecureGPT Vision to extract information from identity-document images on pages classified as `G07`.

It locates each page image using the input CSV and page metadata, identifies the visible documents, and extracts six fields: place of birth, document number, document type, expiry date, issuing authority, and nationality. A structured response model validates the returned fields and document categories.

When an image contains multiple cards or document sides, each is extracted separately. Front and back observations are combined using the document number within the same MasterIndex ID. Results are saved as CSV and JSON files, with incomplete or conflicting records flagged for review.

## Setup

Place `Extraction/` in your project root, alongside `pixi.toml`.

Use a Linux environment with Python 3.10+, Pillow, Pydantic 2.x, and the internal AXA `axallm` package. Configure AXA authentication and these environment variables:

- `SECUREGPT_MODEL_NAME`
- `SECUREGPT_MODEL_VERSION`

Check the input, metadata, and image paths in `Extraction/config.py`.

The default input is:

`Extraction/Input/variant_a_vision_clip_sample_all_test_pages.csv`

It must contain `masterindex_id`, `page_number`, and `predicted_page_sst`. Only pages predicted as `G07` are processed. Rendered page images and the configured metadata files must already be available.

## Run

Run commands from the project root.

Start with five pages:

```bash
pixi run python Extraction/extract.py --limit 5
```

Process all selected pages:

```bash
pixi run python Extraction/extract.py
```

To resume after an interruption, run the same command again. Completed responses are reused from `Extraction/cache/`; failed requests are retried.

To process pages again, including previously completed ones:

```bash
pixi run python Extraction/extract.py --refresh
```

## Results

The script prints the output folder under `Extraction/outputs/`.

| File | Contents |
|---|---|
| `documents.csv` | Extracted documents, including those requiring review |
| `partner_candidates.csv` | Documents marked `ready` |
| `documents.jsonl` | Document results with field sources and page references |
| `unresolved_pages.jsonl` | Failed pages and observations that could not be grouped |
| `excluded_cards.jsonl` | Unsupported document types |
| `pages.jsonl` | Detailed results for each page |
| `summary.json` | Processing counts and model settings |

Check both the `review` rows in `documents.csv` and `unresolved_pages.jsonl`.

`ready` means all six fields are present and the pipeline found no conflicts. It does not guarantee extraction accuracy or that the document is unexpired. Missing values are left empty. Document type codes are `P` for Personalausweis, `R` for Reisepass/Dienstpass/Diplomatenpass, and `S` for Aufenthaltstitel als Passersatz.

## Check processing status

```bash
pixi run python Extraction/Diagnose.py
```

This reports statuses and counts from the most recently updated output folder.
