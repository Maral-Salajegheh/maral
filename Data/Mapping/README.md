# Life MasterIndex Mapping

Maps the delivered MasterIndex file (MID to stack and document) to the final
Analyse-DB page labels in Snowflake (`<SCHEMA>.PROC_LIFE_FINAL_PAGE_LABELS`).
The result is one row per page with its class and its page number inside the
MasterIndex PDF. The classification pipeline reads its labels from here.

- **Page class:** TB0 is not detected with OCR. The final page table already
  carries the verified document class in `sfdoc_class`. A page is labelled
  `TB0` when its `sfdoc_class` contains "Technikblatt"; otherwise its existing
  SST is retained.
- **PDF page number:** computed with the rule verified on real documents: a
  document starts right after all pages of the documents delivered before it in
  the stack, and `seqno` gives the position inside the document. Documents are
  ordered by where their first row appears in the Snowflake result, so the rule
  runs before any sorting. The count restarts for every MasterIndex PDF.
- **Validation:** every document gets a `mapping_status` (`OK`,
  `NO_ANALYSE_DB_PAGES`, `MISSING_SST`, `DOCUMENT_MULTIPLE_SST`, `SST_MISMATCH`,
  `AMBIGUOUS_PROCESS_ID`, `DUPLICATE_IMAGE_ID`). Only `OK` documents produce
  page rows. `is_training_eligible` additionally requires GOLD or SILVER labels.

## Running

```bash
cd ~/Projects/life-docai/packages/life-docai-analytics/life_docai_analytics_antrag/datasets/mapping

# Mapping only (Analyse-DB tables already built)
pixi run -e dev python 00_map_mid_to_adb.py --schema <SCHEMA>

# Analyse-DB build, then mapping
pixi run -e dev python run_life_data_pipeline.py <SCHEMA> --analyse-db-runner <path to Analyse-DB run_pipeline.py>

# Publish usable documents to <SCHEMA>.PROC_LIFE_MID_ADB (after reviewing the summary)
pixi run -e dev python 00_map_mid_to_adb.py --schema <SCHEMA> --publish-snowflake
```

`--mapping-file` selects the MasterIndex file (CSV or Excel); default:
`/home/shared_folders/life_ai/mid_mapping/Antrag_A00_G07_MAD.csv`.

## Outputs (`output/`)

| File | Content |
|------|---------|
| `life_mid_documents.csv` / `.parquet` | One row per mapped document with its `mapping_status` |
| `life_mid_pages.csv` / `.parquet` | One row per page of the usable documents |
| `life_mid_summary.csv` | Documents, MIDs, stacks and pages per `mapping_status` |