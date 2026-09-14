# Corrected first extraction pipeline

This bundle is the first pipeline (`extract.py`), not the six-stage pipeline.
It preserves full-page LLM extraction of fields outside the MRZ and full-page
LLM fallback for missing MRZ fields. No live AXA calls were made during testing.

## Install

Keep a backup of your existing scripts. Copy the replacement files into your
existing Extraction folder. Do not replace your Input, outputs, or cache folders.

This version extends the already corrected bundle with multiple-card extraction.
Changed existing scripts: `llm.py`, `extract.py`, `mrz.py`, `documents.py`.
Updated tests: `test_pipeline.py`, `test_mrz_regressions.py`.
New tests: `test_multicard.py`.
`Diagnose.py`, `mrz_morph.py`, `config.py`, `pages.py`, and `securegpt_client.py`
retain the earlier corrected bundle's exact contents.

Keep any server-specific paths already configured in your
`config.py`. The runtime settings required by the corrected code already existed
in the uploaded first configuration, including `MRZ_MAX_MISSING_FILLER = 8`.
`MRZ_MAX_EXTRA_CHARS`, if still present on your server, is no longer used.
Do not install any files from the six-stage pipeline into this bundle.

## First: diagnose the existing run without paying for another LLM call

Run from the project root:

```bash
pixi run python Extraction/Diagnose.py
```

This reads the newest saved `pages.jsonl`. To examine just its second record:

```bash
pixi run python Extraction/Diagnose.py --row 2
```

`--row` means the one-based nonempty JSONL record number, not a PDF page number.
You can optionally supply a particular run folder or the JSONL path as the
positional argument. For the run shown in the screenshots:

```bash
pixi run python Extraction/Diagnose.py Extraction/outputs/20260914_083544_698131 --row 2
```

It makes no OCR/LLM calls and does not change the saved run. It reports:
- Crop/view labels and OCR line lengths, excluding whitespace.
- Invalid-character counts.
- Which checksum or structural rule failed, identified by field name only.

It never prints MID, image paths, document numbers, names, dates, OCR text, or
raw exception contents. You can share this diagnostic output to investigate a
failure. The underlying `pages.jsonl` and debug PNGs still contain personal data;
they have not been anonymized.

A saved failure that now parses is reported as such. That does not update the old
output and does not establish that other OCR attempts are unambiguous.

## Then: test and rerun extraction

```bash
pixi run python -m unittest discover -s Extraction -p 'test_*.py' -v
pixi run python Extraction/extract.py --limit 2
```

`--limit 2` processes the first two selected G07 records and does make normal LLM
calls. Omit the limit for the full dataset. Each extraction run creates a new
output directory; existing runs are not modified or resumed.

## How the corrected MRZ path works

1. Propose morphological crops, suppress near-identical boxes on the same surface,
   and retain at most the configured crop count.
2. Read each crop with Tesseract PSM 6. If it does not parse, try PSM 11 once.
3. Ignore blank OCR lines and whitespace within lines. Never replace O/0 or other
   glyphs, erase optional data, or trim line edges.
4. Require ordinary TD1/TD2/TD3 structures and all applicable check digits.
5. Limited right-padding is permitted only on TD1 upper/name lines and TD2/TD3
   name lines already ending in at least three `<` characters. All such accepted
   restorations require review. Lower data lines are never padded, including
   padding before their final check digit.
6. Check all shortlisted crops at the current orientation. Preserve every valid
   reading in `mrz.results`, with its view and crop/surface locations. Deduplicate
   identical readings; retain differing readings for card association. Multiple
   valid windows in one OCR attempt are also retained. If no crop parses, try
   the full page; then try subsequent quarter-turn orientations.
7. Record OCR failures per attempt and continue. Keep the full-page LLM for
   printed fields, card classification and fallback.

The scan stops after a successful orientation. It is a bounded search, not proof
that the full page contains only one identity document. The full-page LLM independently enumerates all visible cards and extracts their fields. Overlapping candidates on differently transformed
surfaces are not assumed to have comparable coordinates.

## Interpreting results

`mrz_status` is `success` when at least one valid reading exists, including multi-card pages. `mrz.parsed` remains null when there is no single unambiguous page-wide result; use `mrz.results` for multiple readings. The new `mrz_reason`
and per-attempt diagnostics provide the explanation:

| Reason | Meaning |
|---|---|
| `parsed` | At least one accepted MRZ reading with no detected conflicting identity. |
| `line_length_mismatch` | Plausible lines exist, but their widths cannot be accepted under the restricted rules. |
| `validation_failed` | A fitted candidate failed checksums or structural/date checks. |
| `invalid_characters` | A plausible window contains unsupported OCR characters. |
| `no_mrz_shaped_text` | OCR did not provide a plausible group of MRZ lines. |
| `empty_ocr` | OCR returned no usable text. |
| `ocr_or_image_error` | No usable attempt completed. Inspect technical details locally. |
| `multiple_valid_mrz` | More than one distinct valid reading. See `mrz.results` and per-card associations; this alone does not require review. |

The page-level reason summarizes the attempts. Inspect the specific crop's
attempt to understand that crop. A failed MRZ result on the card front can be
expected. The back page has its own independent MRZ result. Finding a complete
MRZ crop does not guarantee that Tesseract read the characters correctly.

Checksums do not protect nationality, issuer, document code, or names. Even the
protected fields are not authenticated by a check digit. Extended document
numbers and incomplete birth dates are not added in this release; unsupported
readings continue to the full-page LLM fallback. Calendar checking retains the
existing YYMMDD convention and does not reject documents solely for expiry.

## LLM and document output

Six output fields remain: `geburtsort`, `ausweisnummer`, `ausweistyp`,
`gueltigkeitsdatum`, `ausstellende_behoerde`, and `nationalitaet`.
The LLM reads the full image once per page under normal operation (existing
provider retries still apply), returning a `documents` list. Each visible card
or side is separate. It reads all six fields for each card, including the number
needed to associate MRZ readings. It also returns visible issuer code, holder
name and birth date as internal matching evidence, not new business-output fields.
An MRZ success does not eliminate the LLM call. Multi-card extraction can increase
the response length, but does not add a per-card LLM call.

Insurance cards and driving licences must be `not_accepted`, as must other kinds
outside the existing allowlist. Accepted kinds and P/R/S mapping are unchanged:
personalausweis, reisepass, dienstpass, diplomatenpass, and explicitly identified
aufenthaltstitel_als_passersatz. Excluded entries are saved for audit and removed
before grouping; they never donate fields. `unknown` stays subject to review
unless another matching side establishes the accepted kind.

### Association and grouping

- Match MRZ to a card by exact normalized number (case/whitespace only), unique
  among both the page's cards and its distinct MRZ readings. Do not assign by
  list order, visual position or crop coordinates.
- Do not attach when known issuer, full name or birth date contradicts. Same-number
  MRZ alternatives remain unresolved. MRZ/LLM field disagreements require review.
- An unmatched MRZ is preserved as a separate unresolved observation; it is not
  attached to another card. Other non-excluded cards on that page also require
  review because an unmatched reading may mean a missed card or an incorrect number.
- Group card observations only within the same MID and matching document number,
  including matching sides in different PDFs. One MID may produce many documents.
  Two different numbers remain separate even for the same person.
- Same-number observations with contradictory identity or document kinds stay
  separate for review and do not exchange fields. Otherwise field conflicts still
  block `ready`. Missing identity evidence is not proof of a match; exact-number
  grouping retains a residual collision risk when no contradicting evidence is visible.
- Without a readable matching number, front/back association remains unresolved.
  Name, proximity and position are never sufficient. Name comparisons are deliberately
  exact after whitespace/case normalization; spelling/transliteration differences
  can therefore cause review.

### Output schema

- `pages.jsonl`: one audit record per input page, with `cards`, `mrz.results`,
  `unmatched_mrz`, original attempts and private diagnostics. `cards` has one entry
  per detected side, plus any unmatched MRZ observations marked unresolved.
  Each card has a local `card_index`; each field retains its image and card index.
  A single-card page also has legacy summary fields. Never use page-wide fields
  for a multi-card image.
- `documents.jsonl` / `documents.csv`: one record per grouped accepted/unknown
  document, with six business fields. JSON `card_references` identifies each
  contributing image/card. Identity collisions produce separate review records.
- `partner_candidates.csv`: only complete accepted records with no review issues.
  Nothing is transmitted automatically. Fully LLM-derived records can still become
  ready under the original pipeline policy; this is not checksum verification.
- `excluded_cards.jsonl`: insurance/licence/other excluded observations; no donation.
- `unresolved_pages.jsonl`: legacy filename, now containing unresolved card
  observations (or a failed/empty page). It may have several rows for one image.
- `summary.json`: physical page counts remain separate from card, excluded and
  unresolved counts. `cards` includes unmatched MRZ observations; `unresolved_pages`
  counts distinct pages, while `unresolved_cards` counts unresolved records,
  including failed/empty page records.

## Validation and limits

53 offline regression tests pass, including updated earlier tests and 15 new
multi-card tests. The former expectation that MRZ success requests only missing
fields changed deliberately: card association now needs the independently read
number. On LLM failure, MRZ-only observations remain unresolved for review.
New tests cover two fronts/two backs in reversed order, several IDs for one
person, exclusion without field donation, missing anchors, same-number identity
collisions, MRZ association/conflicts, per-card outputs and strict response validation.
All previous parser/diagnostic safeguards remain tested.

No live AXA calls or real multi-card images were used in this update. OpenCV is
unavailable here; actual card separation depends on the LLM's visual extraction.
The unchanged bounded MRZ candidate search can miss regions or differently rotated
cards; full-page LLM fallback remains available. False crop candidates remain
possible and must still pass strict parsing before providing MRZ values.
The previous synthetic Tesseract test and your real back-page parsing failure do
not establish recovery accuracy. Run the private diagnostics for the saved failure,
and inspect a small new multi-card run before deploying to the full dataset.
