/*
Life DocAI pipeline validation queries
======================================

Purpose
-------
Run these read-only queries before modifying or executing the PnC-to-Life port.
They validate workflow states, data grain, SST snapshot behaviour, QA evidence,
and whether the Export SST can be treated as a final classification label.

Run the sections in order. Do not interpret NULL or empty results as proof that
a stage does not exist; first confirm that the correct database and schema are
selected.
*/

-- Adjust these two statements if the delivery uses another schema.
USE DATABASE LIF_PROD;
USE SCHEMA D131_D2D;


/* -------------------------------------------------------------------------
01. Inspect the available tables and important table schemas

What this checks:
  - Confirms that all required raw/reference tables exist.
  - Reveals the actual QA_ID column names before attempting a QA join.
--------------------------------------------------------------------------- */

SHOW TABLES;

DESC TABLE AD_STACK;
DESC TABLE AD_PROCESS;
DESC TABLE AD_DOCUMENT;
DESC TABLE AD_FIELD;
DESC TABLE AD_IMAGE2DOCUMENT;
DESC TABLE QA_ID;
DESC TABLE SST_SEMANTIK;


/* -------------------------------------------------------------------------
02. Discover the real Life workflow vocabulary

What this checks:
  - Lists every observed state/component/actor combination.
  - Prevents unverified PnC state values from being used as Life configuration.
--------------------------------------------------------------------------- */

SELECT
    state,
    component,
    altered_by,
    COUNT(*) AS n_rows,
    COUNT(DISTINCT stack_id) AS n_stacks,
    MIN(entry_id) AS min_entry_id,
    MAX(entry_id) AS max_entry_id,
    MIN(entry_time) AS first_seen,
    MAX(entry_time) AS last_seen
FROM AD_STACK
GROUP BY state, component, altered_by
ORDER BY n_stacks DESC, state, component, altered_by;


/* -------------------------------------------------------------------------
03. Inspect real workflow paths for a sample of stacks

What this checks:
  - Shows the actual order of states, components, and actors.
  - Helps identify Import, Analyser, Supervisor, Verifier, QA, and Export.
--------------------------------------------------------------------------- */

WITH sampled_stacks AS (
    SELECT stack_id
    FROM AD_STACK
    GROUP BY stack_id
    QUALIFY ROW_NUMBER() OVER (ORDER BY stack_id) <= 100
),
ordered_events AS (
    SELECT DISTINCT
        s.stack_id,
        s.entry_id,
        s.entry_time,
        s.state,
        s.component,
        s.altered_by
    FROM AD_STACK AS s
    INNER JOIN sampled_stacks AS x
        ON s.stack_id = x.stack_id
)
SELECT
    stack_id,
    LISTAGG(
        entry_id || ': ' ||
        COALESCE(state, '<NULL>') || ' / ' ||
        COALESCE(component, '<NULL>') || ' / ' ||
        COALESCE(altered_by, '<NULL>'),
        ' -> '
    ) WITHIN GROUP (ORDER BY entry_id, entry_time) AS workflow_path
FROM ordered_events
GROUP BY stack_id
ORDER BY stack_id;


/* -------------------------------------------------------------------------
04. Measure entry counts and test the hard-coded e1...e10 assumption

What this checks:
  - Shows how many stacks have entries above 10.
  - Any non-zero result proves that the hard-coded 1..10 change logic loses data.
--------------------------------------------------------------------------- */

WITH per_stack AS (
    SELECT
        stack_id,
        COUNT(DISTINCT entry_id) AS n_distinct_entries,
        MIN(entry_id) AS min_entry_id,
        MAX(entry_id) AS max_entry_id
    FROM AD_STACK
    GROUP BY stack_id
)
SELECT
    COUNT(*) AS n_stacks,
    COUNT_IF(max_entry_id > 10) AS stacks_with_entry_above_10,
    ROUND(100 * COUNT_IF(max_entry_id > 10) / NULLIF(COUNT(*), 0), 2)
        AS pct_stacks_with_entry_above_10,
    MIN(min_entry_id) AS global_min_entry_id,
    MAX(max_entry_id) AS global_max_entry_id,
    APPROX_PERCENTILE(n_distinct_entries, 0.50) AS median_entries,
    APPROX_PERCENTILE(n_distinct_entries, 0.95) AS p95_entries,
    MAX(n_distinct_entries) AS max_distinct_entries
FROM per_stack;


/* -------------------------------------------------------------------------
05. Validate candidate Import and Export states

What this checks:
  - Measures coverage of the current PnC-derived state names.
  - Verifies that Export occurs after Import.
  - Does not yet prove that these are the correct Life business states.
--------------------------------------------------------------------------- */

WITH state_times AS (
    SELECT
        stack_id,
        MIN(IFF(state = 'ImportLogicModule', entry_time, NULL)) AS first_import_time,
        MAX(IFF(state = 'AfterExport', entry_time, NULL)) AS last_export_time,
        COUNT_IF(state = 'ImportLogicModule') AS n_import_events,
        COUNT_IF(state = 'AfterExport') AS n_export_events
    FROM AD_STACK
    GROUP BY stack_id
)
SELECT
    COUNT(*) AS n_all_stacks,
    COUNT_IF(n_import_events > 0) AS stacks_with_import,
    COUNT_IF(n_export_events > 0) AS stacks_with_export,
    COUNT_IF(n_import_events > 0 AND n_export_events > 0) AS stacks_with_both,
    COUNT_IF(
        n_import_events > 0
        AND n_export_events > 0
        AND last_export_time >= first_import_time
    ) AS chronologically_completed_stacks,
    COUNT_IF(
        n_import_events > 0
        AND n_export_events > 0
        AND last_export_time < first_import_time
    ) AS invalid_export_before_import
FROM state_times;


/* -------------------------------------------------------------------------
06. Validate uniqueness of SST inside one document-entry snapshot

Expected business grain:
  one SST per (stack_id, process_id, doc_id, subdoc_idx, entry_id)

What this checks:
  - Finds duplicate SST rows and conflicting SST values within the same entry.
  - A conflicting_sst_values result above zero is a critical blocker.
--------------------------------------------------------------------------- */

WITH sst_grain AS (
    SELECT
        stack_id,
        process_id,
        doc_id,
        subdoc_idx,
        entry_id,
        COUNT(*) AS n_sst_rows,
        COUNT(DISTINCT field_value) AS n_distinct_sst_values
    FROM AD_FIELD
    WHERE UPPER(TRIM(field_name)) = 'SST'
    GROUP BY stack_id, process_id, doc_id, subdoc_idx, entry_id
)
SELECT
    COUNT(*) AS n_document_entries_with_sst,
    COUNT_IF(n_sst_rows > 1) AS document_entries_with_duplicate_sst_rows,
    COUNT_IF(n_distinct_sst_values > 1) AS document_entries_with_conflicting_sst_values,
    MAX(n_sst_rows) AS max_sst_rows_per_document_entry,
    MAX(n_distinct_sst_values) AS max_distinct_sst_values_per_document_entry
FROM sst_grain;

-- Detailed conflicts, if the previous query reports any.
SELECT
    stack_id,
    process_id,
    doc_id,
    subdoc_idx,
    entry_id,
    COUNT(*) AS n_sst_rows,
    COUNT(DISTINCT field_value) AS n_distinct_sst_values,
    ARRAY_AGG(DISTINCT field_value) AS observed_sst_values
FROM AD_FIELD
WHERE UPPER(TRIM(field_name)) = 'SST'
GROUP BY stack_id, process_id, doc_id, subdoc_idx, entry_id
HAVING COUNT(*) > 1 OR COUNT(DISTINCT field_value) > 1
ORDER BY n_distinct_sst_values DESC, n_sst_rows DESC
LIMIT 500;


/* -------------------------------------------------------------------------
07. Validate LAST_ENTRY document uniqueness

What this checks:
  - Tests whether LAST_ENTRY = 1 produces one final AD_DOCUMENT row per document.
  - Duplicate results invalidate the original Life semantic-table query grain.
--------------------------------------------------------------------------- */

WITH final_document_grain AS (
    SELECT
        stack_id,
        process_id,
        doc_id,
        subdoc_idx,
        COUNT(*) AS n_rows
    FROM AD_DOCUMENT
    WHERE last_entry = 1
    GROUP BY stack_id, process_id, doc_id, subdoc_idx
)
SELECT
    COUNT(*) AS n_final_document_keys,
    COUNT_IF(n_rows > 1) AS duplicated_final_document_keys,
    MAX(n_rows) AS max_rows_per_final_document_key
FROM final_document_grain;


/* -------------------------------------------------------------------------
08. Validate SST semantic-table uniqueness and match quality

What this checks:
  - Detects duplicate semantic definitions after TRIM/UPPER normalization.
  - Measures how many observed SST codes have no semantic definition.
--------------------------------------------------------------------------- */

SELECT
    UPPER(TRIM(sst)) AS normalized_sst,
    COUNT(*) AS n_semantic_rows,
    ARRAY_AGG(DISTINCT kategorie) AS categories,
    ARRAY_AGG(DISTINCT klasse) AS classes
FROM SST_SEMANTIK
GROUP BY UPPER(TRIM(sst))
HAVING COUNT(*) > 1
ORDER BY n_semantic_rows DESC, normalized_sst;

WITH observed_sst AS (
    SELECT DISTINCT UPPER(TRIM(field_value)) AS normalized_sst
    FROM AD_FIELD
    WHERE UPPER(TRIM(field_name)) = 'SST'
      AND NULLIF(TRIM(field_value), '') IS NOT NULL
),
semantic_sst AS (
    SELECT DISTINCT UPPER(TRIM(sst)) AS normalized_sst
    FROM SST_SEMANTIK
    WHERE NULLIF(TRIM(sst), '') IS NOT NULL
)
SELECT
    COUNT(*) AS n_observed_sst_codes,
    COUNT_IF(sem.normalized_sst IS NOT NULL) AS matched_semantic_codes,
    COUNT_IF(sem.normalized_sst IS NULL) AS unmatched_semantic_codes
FROM observed_sst AS obs
LEFT JOIN semantic_sst AS sem
    ON obs.normalized_sst = sem.normalized_sst;


/* -------------------------------------------------------------------------
09. Measure exact-entry page-to-SST coverage by workflow state

What this checks:
  - Tests whether pages and SST values coexist at the same entry_id.
  - Low coverage means INNER JOIN would silently remove many pages.
--------------------------------------------------------------------------- */

WITH stack_entries AS (
    SELECT DISTINCT stack_id, entry_id, state, component
    FROM AD_STACK
),
page_documents AS (
    SELECT DISTINCT
        stack_id, process_id, doc_id, subdoc_idx, entry_id
    FROM AD_IMAGE2DOCUMENT
),
sst_documents AS (
    SELECT DISTINCT
        stack_id, process_id, doc_id, subdoc_idx, entry_id
    FROM AD_FIELD
    WHERE UPPER(TRIM(field_name)) = 'SST'
      AND NULLIF(TRIM(field_value), '') IS NOT NULL
),
coverage AS (
    SELECT
        se.state,
        se.component,
        p.stack_id,
        p.entry_id,
        p.process_id,
        p.doc_id,
        p.subdoc_idx,
        IFF(s.stack_id IS NOT NULL, 1, 0) AS has_sst
    FROM page_documents AS p
    INNER JOIN stack_entries AS se
        ON p.stack_id = se.stack_id
       AND p.entry_id = se.entry_id
    LEFT JOIN sst_documents AS s
        ON p.stack_id = s.stack_id
       AND p.entry_id = s.entry_id
       AND p.process_id = s.process_id
       AND p.doc_id = s.doc_id
       AND p.subdoc_idx = s.subdoc_idx
)
SELECT
    state,
    component,
    COUNT(*) AS n_page_document_snapshots,
    SUM(has_sst) AS snapshots_with_sst,
    COUNT(*) - SUM(has_sst) AS snapshots_without_sst,
    ROUND(100 * SUM(has_sst) / NULLIF(COUNT(*), 0), 2) AS sst_coverage_pct
FROM coverage
GROUP BY state, component
ORDER BY n_page_document_snapshots DESC, state, component;


/* -------------------------------------------------------------------------
10. Test whether AD_FIELD behaves like a full snapshot or a delta/event table

What this checks:
  - Looks only at document keys that exist in AD_IMAGE2DOCUMENT in two
    consecutive entries of the same stack/process.
  - Measures whether their SST is repeated in both entries.

Interpretation:
  - High both_entries_have_sst: snapshot-like behaviour.
  - High current_only/next_only: delta-like or stage-specific behaviour.
--------------------------------------------------------------------------- */

WITH process_entries AS (
    SELECT DISTINCT stack_id, process_id, entry_id
    FROM AD_PROCESS
),
entry_pairs AS (
    SELECT
        stack_id,
        process_id,
        entry_id AS current_entry_id,
        LEAD(entry_id) OVER (
            PARTITION BY stack_id, process_id
            ORDER BY entry_id
        ) AS next_entry_id
    FROM process_entries
),
page_documents AS (
    SELECT DISTINCT
        stack_id, process_id, entry_id, doc_id, subdoc_idx
    FROM AD_IMAGE2DOCUMENT
),
stable_documents AS (
    SELECT
        ep.stack_id,
        ep.process_id,
        ep.current_entry_id,
        ep.next_entry_id,
        cur.doc_id,
        cur.subdoc_idx
    FROM entry_pairs AS ep
    INNER JOIN page_documents AS cur
        ON ep.stack_id = cur.stack_id
       AND ep.process_id = cur.process_id
       AND ep.current_entry_id = cur.entry_id
    INNER JOIN page_documents AS nxt
        ON ep.stack_id = nxt.stack_id
       AND ep.process_id = nxt.process_id
       AND ep.next_entry_id = nxt.entry_id
       AND cur.doc_id = nxt.doc_id
       AND cur.subdoc_idx = nxt.subdoc_idx
    WHERE ep.next_entry_id IS NOT NULL
),
sst_documents AS (
    SELECT DISTINCT
        stack_id, process_id, entry_id, doc_id, subdoc_idx
    FROM AD_FIELD
    WHERE UPPER(TRIM(field_name)) = 'SST'
      AND NULLIF(TRIM(field_value), '') IS NOT NULL
),
comparison AS (
    SELECT
        d.*,
        IFF(cur.stack_id IS NOT NULL, 1, 0) AS current_has_sst,
        IFF(nxt.stack_id IS NOT NULL, 1, 0) AS next_has_sst
    FROM stable_documents AS d
    LEFT JOIN sst_documents AS cur
        ON d.stack_id = cur.stack_id
       AND d.process_id = cur.process_id
       AND d.current_entry_id = cur.entry_id
       AND d.doc_id = cur.doc_id
       AND d.subdoc_idx = cur.subdoc_idx
    LEFT JOIN sst_documents AS nxt
        ON d.stack_id = nxt.stack_id
       AND d.process_id = nxt.process_id
       AND d.next_entry_id = nxt.entry_id
       AND d.doc_id = nxt.doc_id
       AND d.subdoc_idx = nxt.subdoc_idx
)
SELECT
    COUNT(*) AS stable_document_entry_pairs,
    COUNT_IF(current_has_sst = 1 AND next_has_sst = 1) AS both_entries_have_sst,
    COUNT_IF(current_has_sst = 1 AND next_has_sst = 0) AS current_only_has_sst,
    COUNT_IF(current_has_sst = 0 AND next_has_sst = 1) AS next_only_has_sst,
    COUNT_IF(current_has_sst = 0 AND next_has_sst = 0) AS neither_entry_has_sst,
    ROUND(
        100 * COUNT_IF(current_has_sst = 1 AND next_has_sst = 1)
        / NULLIF(COUNT(*), 0),
        2
    ) AS repeated_sst_pct
FROM comparison;


/* -------------------------------------------------------------------------
11. Inspect QA workers and verifier values

What this checks:
  - Shows the QA_ID structure and the actual VERIFIED_BY vocabulary.
  - After DESC TABLE QA_ID, replace qa_id below if the identifier column has
    another name.
--------------------------------------------------------------------------- */

SELECT * FROM QA_ID LIMIT 100;

SELECT
    verified_by,
    COUNT(*) AS n_document_rows,
    COUNT(DISTINCT stack_id) AS n_stacks,
    MIN(entry_time) AS first_seen,
    MAX(entry_time) AS last_seen
FROM AD_DOCUMENT
GROUP BY verified_by
ORDER BY n_document_rows DESC;

-- Replace q.qa_id only if DESC TABLE QA_ID shows a different column name.
SELECT
    COUNT(*) AS n_document_rows,
    COUNT_IF(NULLIF(TRIM(d.verified_by), '') IS NOT NULL) AS rows_with_verifier,
    COUNT_IF(q.qa_id IS NOT NULL) AS rows_verified_by_known_qa,
    COUNT_IF(
        NULLIF(TRIM(d.verified_by), '') IS NOT NULL
        AND q.qa_id IS NULL
    ) AS rows_verified_by_non_qa_or_unmatched_actor
FROM AD_DOCUMENT AS d
LEFT JOIN QA_ID AS q
    ON UPPER(TRIM(d.verified_by)) = UPPER(TRIM(q.qa_id));


/* -------------------------------------------------------------------------
12. Profile SST creation/correction metadata

What this checks:
  - Shows which actors create or alter SST values.
  - Determines whether SHOW_IN_VERIFIER, ACTION, QUALITY, and RATING can help
    distinguish machine output from human-confirmed values.
--------------------------------------------------------------------------- */

SELECT
    altered_by,
    show_in_verifier,
    action,
    quality,
    rating,
    COUNT(*) AS n_sst_rows,
    COUNT(DISTINCT stack_id) AS n_stacks
FROM AD_FIELD
WHERE UPPER(TRIM(field_name)) = 'SST'
GROUP BY altered_by, show_in_verifier, action, quality, rating
ORDER BY n_sst_rows DESC;


/* -------------------------------------------------------------------------
13. Validate exact Export-snapshot coverage

What this checks:
  - Selects the last AfterExport entry for every stack.
  - Measures how many exported page-document keys have an SST and an exact
    AD_DOCUMENT row at the same entry.
  - Low coverage means Export cannot be used through a simple exact-entry join.
--------------------------------------------------------------------------- */

WITH last_export AS (
    SELECT stack_id, entry_id, entry_time
    FROM AD_STACK
    WHERE state = 'AfterExport'
    QUALIFY ROW_NUMBER() OVER (
        PARTITION BY stack_id
        ORDER BY entry_time DESC, entry_id DESC
    ) = 1
),
export_documents AS (
    SELECT DISTINCT
        p.stack_id,
        p.process_id,
        p.doc_id,
        p.subdoc_idx,
        p.entry_id
    FROM AD_IMAGE2DOCUMENT AS p
    INNER JOIN last_export AS e
        ON p.stack_id = e.stack_id
       AND p.entry_id = e.entry_id
),
export_sst AS (
    SELECT
        stack_id,
        process_id,
        doc_id,
        subdoc_idx,
        entry_id,
        field_value AS sst,
        altered_by AS sst_altered_by,
        show_in_verifier,
        entry_time,
        ROW_NUMBER() OVER (
            PARTITION BY stack_id, process_id, doc_id, subdoc_idx, entry_id
            ORDER BY entry_time DESC
        ) AS rn
    FROM AD_FIELD
    WHERE UPPER(TRIM(field_name)) = 'SST'
),
assessment AS (
    SELECT
        x.*,
        s.sst,
        s.sst_altered_by,
        s.show_in_verifier,
        d.verified_by,
        d.altered_by AS document_altered_by,
        IFF(s.sst IS NOT NULL, 1, 0) AS has_export_sst,
        IFF(d.stack_id IS NOT NULL, 1, 0) AS has_exact_export_document_row
    FROM export_documents AS x
    LEFT JOIN export_sst AS s
        ON x.stack_id = s.stack_id
       AND x.process_id = s.process_id
       AND x.doc_id = s.doc_id
       AND x.subdoc_idx = s.subdoc_idx
       AND x.entry_id = s.entry_id
       AND s.rn = 1
    LEFT JOIN AD_DOCUMENT AS d
        ON x.stack_id = d.stack_id
       AND x.process_id = d.process_id
       AND x.doc_id = d.doc_id
       AND x.subdoc_idx = d.subdoc_idx
       AND x.entry_id = d.entry_id
)
SELECT
    COUNT(*) AS n_export_document_keys,
    SUM(has_export_sst) AS export_keys_with_sst,
    COUNT(*) - SUM(has_export_sst) AS export_keys_without_sst,
    ROUND(100 * SUM(has_export_sst) / NULLIF(COUNT(*), 0), 2)
        AS export_sst_coverage_pct,
    SUM(has_exact_export_document_row) AS keys_with_exact_document_row,
    COUNT_IF(NULLIF(TRIM(verified_by), '') IS NOT NULL) AS keys_with_verified_by,
    COUNT_IF(show_in_verifier = 1) AS keys_shown_in_verifier
FROM assessment;


/* -------------------------------------------------------------------------
14. Compare Export SST with the latest observed SST in the full history

What this checks:
  - Tests whether Export SST equals the latest SST for the same document key.
  - Differences mean Export is not automatically the final stored SST.
--------------------------------------------------------------------------- */

WITH last_export AS (
    SELECT stack_id, entry_id, entry_time
    FROM AD_STACK
    WHERE state = 'AfterExport'
    QUALIFY ROW_NUMBER() OVER (
        PARTITION BY stack_id
        ORDER BY entry_time DESC, entry_id DESC
    ) = 1
),
export_sst AS (
    SELECT
        f.stack_id,
        f.process_id,
        f.doc_id,
        f.subdoc_idx,
        f.field_value AS export_sst
    FROM AD_FIELD AS f
    INNER JOIN last_export AS e
        ON f.stack_id = e.stack_id
       AND f.entry_id = e.entry_id
    WHERE UPPER(TRIM(f.field_name)) = 'SST'
    QUALIFY ROW_NUMBER() OVER (
        PARTITION BY f.stack_id, f.process_id, f.doc_id, f.subdoc_idx
        ORDER BY f.entry_time DESC
    ) = 1
),
latest_sst AS (
    SELECT
        stack_id,
        process_id,
        doc_id,
        subdoc_idx,
        field_value AS latest_sst,
        entry_id AS latest_sst_entry_id,
        entry_time AS latest_sst_entry_time,
        altered_by AS latest_sst_altered_by
    FROM AD_FIELD
    WHERE UPPER(TRIM(field_name)) = 'SST'
    QUALIFY ROW_NUMBER() OVER (
        PARTITION BY stack_id, process_id, doc_id, subdoc_idx
        ORDER BY entry_time DESC, entry_id DESC
    ) = 1
)
SELECT
    COUNT(*) AS n_export_sst_document_keys,
    COUNT_IF(l.latest_sst IS NULL) AS export_keys_without_latest_sst,
    COUNT_IF(UPPER(TRIM(e.export_sst)) = UPPER(TRIM(l.latest_sst)))
        AS export_equals_latest_sst,
    COUNT_IF(UPPER(TRIM(e.export_sst)) <> UPPER(TRIM(l.latest_sst)))
        AS export_differs_from_latest_sst
FROM export_sst AS e
LEFT JOIN latest_sst AS l
    ON e.stack_id = l.stack_id
   AND e.process_id = l.process_id
   AND e.doc_id = l.doc_id
   AND e.subdoc_idx = l.subdoc_idx;


/* -------------------------------------------------------------------------
15. Build evidence categories for a possible ground-truth definition

What this checks:
  - Does not declare ground truth automatically.
  - Separates QA-verified, other verified, and unverified Export SST values.
  - Replace q.qa_id if QA_ID uses a different identifier column.
--------------------------------------------------------------------------- */

WITH last_export AS (
    SELECT stack_id, entry_id, entry_time
    FROM AD_STACK
    WHERE state = 'AfterExport'
    QUALIFY ROW_NUMBER() OVER (
        PARTITION BY stack_id
        ORDER BY entry_time DESC, entry_id DESC
    ) = 1
),
export_sst AS (
    SELECT
        f.stack_id,
        f.process_id,
        f.doc_id,
        f.subdoc_idx,
        f.entry_id,
        f.field_value AS sst,
        f.altered_by AS sst_altered_by,
        f.show_in_verifier
    FROM AD_FIELD AS f
    INNER JOIN last_export AS e
        ON f.stack_id = e.stack_id
       AND f.entry_id = e.entry_id
    WHERE UPPER(TRIM(f.field_name)) = 'SST'
    QUALIFY ROW_NUMBER() OVER (
        PARTITION BY f.stack_id, f.process_id, f.doc_id, f.subdoc_idx
        ORDER BY f.entry_time DESC
    ) = 1
),
evidence AS (
    SELECT
        s.*,
        d.verified_by,
        IFF(q.qa_id IS NOT NULL, 1, 0) AS verified_by_known_qa
    FROM export_sst AS s
    LEFT JOIN AD_DOCUMENT AS d
        ON s.stack_id = d.stack_id
       AND s.process_id = d.process_id
       AND s.doc_id = d.doc_id
       AND s.subdoc_idx = d.subdoc_idx
       AND s.entry_id = d.entry_id
    LEFT JOIN QA_ID AS q
        ON UPPER(TRIM(d.verified_by)) = UPPER(TRIM(q.qa_id))
)
SELECT
    CASE
        WHEN verified_by_known_qa = 1 THEN 'QA_VERIFIED_CANDIDATE'
        WHEN NULLIF(TRIM(verified_by), '') IS NOT NULL THEN 'OTHER_VERIFIED_CANDIDATE'
        WHEN show_in_verifier = 1 THEN 'SHOWN_BUT_NOT_PROVEN_VERIFIED'
        ELSE 'UNVERIFIED_EXPORT_SST'
    END AS evidence_category,
    COUNT(*) AS n_document_keys,
    COUNT(DISTINCT stack_id) AS n_stacks
FROM evidence
GROUP BY evidence_category
ORDER BY n_document_keys DESC;


/* -------------------------------------------------------------------------
16. Measure non-numeric and normalization risks in IMAGE_ID

What this checks:
  - Quantifies rows that TRY_TO_NUMBER would turn into NULL.
  - Detects whether distinct string IDs collapse to the same number.
--------------------------------------------------------------------------- */

SELECT
    COUNT(*) AS n_page_rows,
    COUNT_IF(TRY_TO_NUMBER(image_id) IS NULL AND image_id IS NOT NULL)
        AS non_numeric_image_id_rows,
    COUNT(DISTINCT image_id) AS distinct_string_image_ids,
    COUNT(DISTINCT TRY_TO_NUMBER(image_id)) AS distinct_numeric_image_ids
FROM AD_IMAGE2DOCUMENT;


/* -------------------------------------------------------------------------
17. Profile repeated candidate stages per stack

What this checks:
  - Reveals whether selecting the first Analyser/PreQA or first/last Export is
    meaningful, or whether stacks revisit these stages repeatedly.
--------------------------------------------------------------------------- */

SELECT
    state,
    component,
    COUNT(*) AS n_stack_stage_groups,
    COUNT_IF(n_occurrences > 1) AS stack_stage_groups_repeated,
    MAX(n_occurrences) AS max_occurrences_per_stack
FROM (
    SELECT
        stack_id,
        state,
        component,
        COUNT(DISTINCT entry_id) AS n_occurrences
    FROM AD_STACK
    GROUP BY stack_id, state, component
)
GROUP BY state, component
ORDER BY stack_stage_groups_repeated DESC, n_stack_stage_groups DESC;


/* -------------------------------------------------------------------------
18. Optional checks after the ported output tables have been created

Run only if these tables already exist. Snowflake primary-key declarations do
not enforce uniqueness, so the output grain must be checked explicitly.
--------------------------------------------------------------------------- */

-- Expected grain: one row per stack/process/image.
SELECT
    stack_id,
    process_id,
    image_id,
    COUNT(*) AS n_rows
FROM PROC_DOC_CHANGES_PAGE_LEVEL
GROUP BY stack_id, process_id, image_id
HAVING COUNT(*) > 1
ORDER BY n_rows DESC
LIMIT 500;

-- Expected grain: one row per stack/process/entry.
SELECT
    stack_id,
    process_id,
    entry_id,
    COUNT(*) AS n_rows
FROM PROC_DOC_CHANGES
GROUP BY stack_id, process_id, entry_id
HAVING COUNT(*) > 1
ORDER BY n_rows DESC
LIMIT 500;

-- Expected grain: one row per stack.
SELECT
    stack_id,
    COUNT(*) AS n_rows
FROM PROC_STACK_AGG
GROUP BY stack_id
HAVING COUNT(*) > 1
ORDER BY n_rows DESC
LIMIT 500;



/*
Classify final Export SST labels by verifier type.

Label tiers:
- QA_VERIFIED: verified by one of the known QA workers.
- SYSTEM_VERIFIED: verified_by is System.
- SERVICE_ACCOUNT: verified by a technical/service account.
- NON_QA_HUMAN: another named verifier, likely human but not in QA_ID.
*/

WITH last_export AS (
    SELECT
        stack_id,
        entry_id,
        entry_time
    FROM AD_STACK
    WHERE state = 'AfterExport'
    QUALIFY ROW_NUMBER() OVER (
        PARTITION BY stack_id
        ORDER BY entry_time DESC, entry_id DESC
    ) = 1
),

export_sst AS (
    SELECT
        f.stack_id,
        f.process_id,
        f.doc_id,
        f.subdoc_idx,
        f.entry_id,
        f.field_value AS sst
    FROM AD_FIELD AS f
    INNER JOIN last_export AS e
        ON f.stack_id = e.stack_id
        AND f.entry_id = e.entry_id
    WHERE UPPER(TRIM(f.field_name)) = 'SST'
      AND NULLIF(TRIM(f.field_value), '') IS NOT NULL
),

label_evidence AS (
    SELECT
        s.stack_id,
        s.process_id,
        s.doc_id,
        s.subdoc_idx,
        s.sst,
        d.verified_by,
        CASE
            WHEN q.id IS NOT NULL
                THEN 'QA_VERIFIED'
            WHEN UPPER(TRIM(d.verified_by)) = 'SYSTEM'
                THEN 'SYSTEM_VERIFIED'
            WHEN UPPER(TRIM(d.verified_by)) LIKE 'SVC%'
                THEN 'SERVICE_ACCOUNT'
            WHEN NULLIF(TRIM(d.verified_by), '') IS NOT NULL
                THEN 'NON_QA_HUMAN'
            ELSE 'MISSING_VERIFIER'
        END AS label_tier
    FROM export_sst AS s
    LEFT JOIN AD_DOCUMENT AS d
        ON s.stack_id = d.stack_id
        AND s.process_id = d.process_id
        AND s.doc_id = d.doc_id
        AND s.subdoc_idx = d.subdoc_idx
        AND s.entry_id = d.entry_id
    LEFT JOIN QA_ID AS q
        ON UPPER(TRIM(d.verified_by)) = UPPER(TRIM(q.id))
)

SELECT
    label_tier,
    COUNT(*) AS n_document_labels,
    COUNT(DISTINCT stack_id) AS n_stacks,
    ROUND(
        100 * COUNT(*) / SUM(COUNT(*)) OVER (),
        2
    ) AS label_percentage
FROM label_evidence
GROUP BY label_tier
ORDER BY n_document_labels DESC;
