# Corrected Life AnalyseDB Pipeline

This pipeline is based on the validated behaviour of the Life delivery rather
than a direct substitution of PnC `sfdoc_class` with Life SST.

## Execution order

1. `00_completed_stacks.sql`
2. `01_final_document_labels.sql`
3. `02_final_page_labels.sql`
4. `03_page_grouping_changes.sql`
5. `04_stack_aggregation.sql`

Run the complete dependency-ordered pipeline through the Python orchestrator:

```bash
python run_pipeline.py D131_D2D
```

The runner validates the required inputs, executes only the five listed SQL
stages, verifies the main output after each stage, enforces output-grain
uniqueness, checks semantic uniqueness and page process-ID stability, and stops
on the first error.
Use `--yes` only for non-interactive/automated execution.

## Outputs

| Output | Grain | Purpose |
|---|---|---|
| `PROC_LIFE_COMPLETED_STACKS` | stack | Complete history scope for workflow analytics |
| `PROC_LIFE_FINAL_DOCUMENT_LABELS` | final document | Final SST labels, semantics, and evidence tier |
| `TRAINING_LIFE_DOCUMENT_LABELS` | labelled document | Gold/Silver document training view |
| `PROC_LIFE_FINAL_PAGE_LABELS` | export page mapping | Final SST attached to pages |
| `TRAINING_LIFE_PAGE_LABELS` | unambiguous labelled page | Gold/Silver page training view |
| `PROC_LIFE_PAGE_GROUPING_CHANGES` | page comparison | Analyser-to-Export grouping corrections |
| `PROC_LIFE_STACK_AGG` | completed stack | Workflow, label, and grouping metrics |

## Deliberately removed PnC logic

- SST comparisons across workflow entries: Life SST exists only at AfterExport.
- Hard-coded entry IDs `e1` through `e10`.
- PreQA-vs-Export SST comparison: PreQA is rare and contains no SST.
- Numeric conversion of `image_id`.
- PnC-specific system-agnostic stack-ID parsing.
- Placeholder routing-class logic.
- `altered_by <> 'System'` labelled as manual effort or FTE.
- Snowflake primary-key declarations used as a substitute for validation.


22 stacks have non-standard category (10 AXAPnC/Default, 1 leben, 11 multi-category). Kept in PROC_LIFE_STACK_AGG, flagged via has_unambiguous_category. Training extraction must filter: has_unambiguous_category AND category IN ('Antrag','Bestand').

## Label policy

- `GOLD`: QA worker verified.
- `SILVER`: named non-QA verifier.
- `WEAK`: System verified.
- `EXCLUDE`: missing SST, service account, or missing verifier.

The underlying `label_tier` is retained so that training and evaluation can be
reported separately for Gold and Silver labels.
