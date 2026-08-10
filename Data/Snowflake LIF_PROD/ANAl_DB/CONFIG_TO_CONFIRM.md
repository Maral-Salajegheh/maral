# Life pipeline — values to confirm before first run  (SST version)

The classification target is the **SST** (AD_FIELD where field_name = 'SST'),
not sfdoc_class. Every Life-specific assumption is a SET variable in the CONFIG
block at the top of each file. Confirm the values below, then run
`00 → 01 → 02 → 03 → 03b → 04` in order.

## One query resolves most of the state config
Run this against LIF_PROD.D131_D2D and map the results into the STATE_*/COMPONENT_* vars:

    SELECT DISTINCT state, component FROM ad_stack ORDER BY 1, 2;

## State / component vocabulary (the main blocker)
PnC defaults shown -- replace with the Life strings the query above returns.

| Variable            | PnC default          | Used in            |
|---------------------|----------------------|--------------------|
| STATE_IMPORT        | ImportLogicModule    | 00, 04             |
| STATE_EXPORT        | AfterExport          | 00, 03, 03b, 04    |
| STATE_DOCAI_PULL    | DocAiPull            | 01                 |
| ALTERED_BY_DOCAI    | DocAi                | 01                 |
| COMPONENT_ANALYSER  | Analyser             | 03                 |
| STATE_PREQA         | PreQAExporter        | 03b                |
| STATE_ANALYSER1     | Analyser1            | 04                 |
| STATE_ANALYSER2     | Analyser2            | 04                 |
| STATE_SUPERVISOR    | Supervisor           | 04                 |
| STATE_VERIFIER      | Verifier             | 04                 |
| STATE_QA            | PreQAExporter        | 04                 |
| STATE_QA_WORKER     | Supervisor2          | 04                 |
| SYSTEM_AGNOSTIC     | System               | 04 (altered_by)    |

## SST-specific items
- **SST_FIELD_NAME** = 'SST' — confirm this is the AD_FIELD.field_name that
  holds the SST (used in 02, 03, 03b, 04).
- **Grain** = (stack_id, process_id, doc_id, subdoc_idx), matching
  find_field_holding_true_class.sql.
- **Within-entry dedup** — sst_per_doc_entry keeps the latest fill per entry
  via `ORDER BY entry_time DESC`. Confirm there's no better tiebreaker (try_id?).
- **SST_Semantik enrichment** — not joined yet. Kategorie/Klasse/Aktion etc. can
  be layered onto the page/doc tables later (as find_field does) if needed for
  labels or analysis.
- **No '/' split** — PnC's sfdoc_class mandant/type/detail SPLIT_PART logic is
  gone; SST is a single code.

## Other Life-specific items
- **stack_id parsing (04)** — PnC positional system-counter logic disabled;
  fallback `stack_id_sa = stack_id`, `system_counter = 1` active. Replace only if
  Life encodes multiple systems in stack_id.
- **ATTR_DOC_SEP (02)** — 'DocumentSeparation'. Confirm the Life attr_name.
- **ROUTING_CLASS_STR (04)** — currently a placeholder that yields routing_class = 0.
  Define the Life "routed elsewhere" concept (likely an SST or SST_Semantik
  category) or leave as-is (last_system then ignores routing).
- **10-entry cap (02)** — stacks with >10 entry states lose later entries.
- **AD_DOC_ATTR copy** — 02 needs LIF_PROD.analyse_db_full.AD_DOC_ATTR copied
  (see copy_ad_doc_attr in utils.py).

## Notes
- Ingestion is already done by the Life utils.py + upload*.py (shared-folder .txt
  -> typed polars chunks -> LIF_PROD.D131_D2D). These SQL files run on top of it.
- 00 and 01 are unchanged from the first port (they don't touch the class value).
- Files reconstructed from the PnC originals -- diff against repo copies before commit.
- Extraction (field_value correctness on ad_field) is NOT in scope here; these six
  are classification only.
