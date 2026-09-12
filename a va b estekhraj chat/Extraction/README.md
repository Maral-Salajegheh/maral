# Identity document extraction

Run from the project root, next to `pixi.toml`:

```bash
pixi run python Extraction/extract.py
```

No input-path argument or separate grouping run is required. Use this bundle's
files together. There is no dependency on `securegpt_vision.py`.

## Input and image lookup

Default input: `Extraction/Input/variant_a_vision_clip_sample_all_test_pages.csv`.
Required columns: `masterindex_id`, `page_number`, `predicted_page_sst`.
Only G07 rows are selected. Image paths are not required in this CSV.

Configured metadata sources:

- `data/*_page_labels.jsonl`
- `data/ab1_pseudo_documents.csv`
- Equivalent files under `Extraction/data`
- `outputs/page_inventory.csv` under the project root
- `Extraction/Input/page_inventory.csv`

Lookup uses MID and page number; PDF path and image hash, when supplied in the
prediction row, narrow the match. `source_page_number` is not an automatic
substitute for `page_number`: metadata and predictions must use consistent page
numbering. Available metadata is retained. Ambiguous or missing image matches
are recorded for review. Malformed metadata JSONL raises an explicit error.

All corpus paths are in `config.py`. `PROJECT_ROOT` is the parent of Extraction.
Adjust `DATA_DIRS`, `INVENTORY_FILES`, and `IMAGE_ROOTS` to the actual server layout.
Legacy image roots from the supplied configuration remain included; their presence
on your server has not been verified here.

## Processing

1. Resolve images using metadata; remove duplicate image/PDF rows.
2. Locate card regions, correct perspective, and deskew.
3. Propose MRZ crops, upscale them, and run Tesseract.
4. Accept exact-width TD1/TD2/TD3 text with structural and checksum validation.
5. Preserve document number, nationality, and expiry from accepted MRZ results.
6. Ask AXA LLM to read missing fields and identify document kind from the full image.
7. If MRZ detection, OCR, or validation fails, request all six fields from the LLM.
8. Group extracted pages by matching document number within the same MID,
   without repeating OCR or LLM calls.

If crops fail, full-page OCR is attempted. Rotations of 0, 90, 180, and 270 degrees
are tried until an accepted result or ambiguity is found. OCR text and errors are
audited. Missing Tesseract/OpenCV can increase LLM fallback usage; LLM extraction
does not depend on successful MRZ reading.

## Fields

| Output field | Meaning | Primary source |
|---|---|---|
| `geburtsort` | Place of birth | LLM |
| `ausweisnummer` | Document number | MRZ, otherwise LLM |
| `ausweistyp` | Partner document type P/R/S | Visual LLM kind and fixed mapping |
| `gueltigkeitsdatum` | Expiry date | MRZ, otherwise LLM |
| `ausstellende_behoerde` | Issuing authority | LLM |
| `nationalitaet` | Nationality | MRZ, otherwise LLM |

LLM values do not overwrite accepted MRZ values on the same page. Each field
retains its source and image path. Names and sex are not requested output fields,
although they may occur in raw OCR. Nationality has no MRZ checksum protection.

## OCR characters and German text

The Tesseract whitelist applies only to MRZ OCR: `ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789<`.
It does not filter images sent to the LLM or returned printed fields. German
characters such as `ä`, `ö`, `ü`, and `ß` can remain in printed field values.
JSON and CSV use UTF-8; CSV includes a BOM.

The old `MRZ_CHAR_FIXES` and `OCR_CONFUSIONS` dictionaries are not used. They
substituted symbols with `<` or explored alternatives such as O/0 and I/1;
they were not German alphabet settings. This parser does not pad, truncate,
substitute glyphs, or repair document numbers. Strict parsing can reject
recoverable OCR text and increase LLM fallback usage.

## Combining document sides

Grouping uses `(masterindex_id, extracted_ausweisnummer)`. Case and whitespace
are normalized. One-character differences are not repaired, and numbers with
punctuation are not used as anchors. Grouping never crosses MIDs. Matching numbers
can combine sides from different PDFs in one MID; source PDF paths are retained.
A shared MID, shared PDF, or `multiple_id_sides` screening label alone does not link pages.

Conflicts trigger review. MRZ values take precedence without hiding conflicts.
German nationality variants such as D, DEU, DEUTSCH, and GERMAN compare as equivalent;
no complete partner country-code mapping is assumed.

LLM-read numbers can be wrong. Number matching is a practical rule, not proof of
identity. Different documents with the same number can merge if no contradictory
fields are extracted. Evaluate grouping quality on actual data.

Pages without a usable number, unresolved images, or images containing multiple
distinct documents go to `unresolved_pages.jsonl`, retaining extracted values.
They are not emitted as confidently identified documents. No second visual
matching pass is implemented. If only one side has a readable number, the other
side remains unresolved until its association is established.

## Accepted kinds

Regular and provisional versions use this mapping:

| Document kind | Partner code |
|---|---|
| Personalausweis | P |
| Reisepass | R |
| Dienstpass | R |
| Diplomatenpass | R |
| Aufenthaltstitel with explicit Passersatz evidence | S |

An ordinary residence permit is not automatically a passport substitute. S does
not accept every other document. Service/diplomatic passports map to R based on
the passport-family interpretation; align `TYPE_CODES` in `llm.py` with the partner
definition if different. Visual kind classification does not authenticate documents.

## Outputs and debugging

Each run creates `Extraction/outputs/<timestamp>`. Earlier outputs are retained.
The pipeline does not consume old grouped CSVs or resume from earlier output files.
The AXA client retains `cache_prompts=True` from the supplied connection setup.

- `documents.csv` and `documents.jsonl`: one row/record per grouped document.
- `partner_candidates.csv`: documents with status ready; nothing is sent externally.
- `pages.jsonl`: page audit, written progressively during execution.
- `unresolved_pages.jsonl`: pages with unresolved document associations.
- `mrz_crops/`: upscaled crops, separated by processing-page index.
- `summary.json`: counts and model metadata.

Compare each crop with its `mrz.attempts` entry in `pages.jsonl`, which records the
view label, region, and OCR text. `mrz.parsed` contains the accepted parse;
`mrz.errors` records failures. `llm_requested_fields` and field sources show fallback.
Colored overlays are not generated. Crops are saved before adding Tesseract's
20-pixel white border. Full-page fallback images are not saved, but their OCR text
is recorded. Pages with no proposed crops may have no crop images.

| Status | Meaning |
|---|---|
| `ready` | Accepted kind, all six fields present, no detected issues |
| `review` | Missing fields, unknown kind, conflicts, or extraction issues |
| `rejected` | Non-accepted document kind |

Ready does not guarantee correctness. MRZ fields survive LLM failure in the audit.
Interrupted runs do not resume automatically; completed page audit records remain.

Dates use YYMMDD or explicitly printed UNBEFRISTET. Century is not inferred;
date-format validation uses 2000+YY. Expiration relative to today is not an acceptance
gate. MRZ nationality is a code without filler; LLM nationality may be printed text.
Confirm partner date and nationality formats before operational integration.

## Files and environment

| File | Responsibility |
|---|---|
| `extract.py` | Extraction entry point |
| `config.py` | Paths and image-localization settings |
| `pages.py` | Metadata and image resolution |
| `mrz_morph.py` | Localization and deskew |
| `mrz.py` | OCR and strict MRZ parsing |
| `securegpt_client.py` | AXA connection, image preparation, model/retry settings |
| `llm.py` | German extraction prompts, response validation, retry loop |
| `documents.py` | Grouping and field merging |
| `test_pipeline.py` | Offline tests |

The morphology module is adapted from the supplied code, with corrections for
deskew rotation direction and blank images. No `Life.Extraction` import is used.
Use the existing Pixi environment with Pillow, numpy, OpenCV, and axallm.
Tesseract is needed for OCR; set `TESSERACT_CMD` if its executable is elsewhere.
No credentials or dependencies on your machine are changed automatically.

`securegpt_client.py` belongs inside Extraction and contains no screening prompt
or screening schema. Both extraction prompts in `llm.py` are German; fixed JSON
keys and enum values are unchanged. Images are EXIF-oriented before sending.
Keep SECUREGPT_MODEL_NAME and SECUREGPT_MODEL_VERSION in the existing environment.
As in the supplied wrapper, MODEL_VERSION is validated as an environment setting;
no new SDK constructor argument is assumed.

```bash
pixi run python -c "import axallm, cv2; print('imports OK')"
pixi run python -m unittest discover -s Extraction -p 'test_*.py' -v
pixi run python Extraction/extract.py
```

Use the existing `-e NAME` option if your Pixi environment is named.

## Validation and limitations

The 18 offline tests cover path-free predictions, ambiguous metadata, full LLM
fallback, MRZ preservation, front/back grouping, distinct document numbers,
unresolved pages, and JSON/CSV consistency. AXA calls are mocked; live AXA access
and real-document accuracy have not been tested. The standalone client was also
checked for image payload generation and independence from the screening module.

In a synthetic check, deskew reduced a 10-degree skew to approximately zero and
MRZ crops were produced. Tesseract text still failed strict parsing, requiring
fallback. This does not establish a real-data MRZ success rate; inspect saved crops
and evaluate representative scans.
