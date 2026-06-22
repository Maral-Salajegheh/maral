/*
Build the final Life SST label table at document grain.

Validated facts used here:
  - SST is present at AfterExport, not at intermediate workflow entries.
  - One SST exists per document-entry.
  - The last AfterExport SST equals the latest stored SST for all matchable rows.
  - IMAGE_ID remains a string and is handled in the page-label script.

Expected grain:
  one row per (stack_id, process_id, doc_id, subdoc_idx) at the last AfterExport.
*/

CREATE OR REPLACE TABLE PROC_LIFE_FINAL_DOCUMENT_LABELS AS
WITH last_export AS (
    SELECT
        stack_id,
        entry_id,
        entry_time AS export_entry_time
    FROM AD_STACK
    WHERE state = 'AfterExport'
    QUALIFY ROW_NUMBER() OVER (
        PARTITION BY stack_id
        ORDER BY entry_time DESC, entry_id DESC
    ) = 1
),

export_documents AS (
    SELECT
        d.stack_id,
        d.process_id,
        d.doc_id,
        d.subdoc_idx,
        d.entry_id,
        e.export_entry_time,
        d.status,
        d.altered_by AS document_altered_by,
        d.remark,
        d.sfdoc_class,
        d.exportname,
        d.entry_time AS document_entry_time,
        d.verified_by,
        d.last_entry,
        d.doc_state,
        d.doc_substate
    FROM AD_DOCUMENT AS d
    INNER JOIN last_export AS e
        ON d.stack_id = e.stack_id
       AND d.entry_id = e.entry_id
),

export_sst AS (
    SELECT
        f.stack_id,
        f.process_id,
        f.doc_id,
        f.subdoc_idx,
        f.entry_id,
        NULLIF(TRIM(f.field_value), '') AS sst,
        f.altered_by AS sst_altered_by,
        f.show_in_verifier,
        f.action AS sst_action,
        f.quality AS sst_quality,
        f.rating AS sst_rating,
        f.entry_time AS sst_entry_time
    FROM AD_FIELD AS f
    INNER JOIN last_export AS e
        ON f.stack_id = e.stack_id
       AND f.entry_id = e.entry_id
    WHERE UPPER(TRIM(f.field_name)) = 'SST'
    QUALIFY ROW_NUMBER() OVER (
        PARTITION BY
            f.stack_id,
            f.process_id,
            f.doc_id,
            f.subdoc_idx,
            f.entry_id
        ORDER BY f.entry_time DESC
    ) = 1
),

semantic_lookup AS (
    SELECT
        UPPER(TRIM(sst)) AS normalized_sst,
        kategorie,
        klasse,
        aktion,
        detail,
        zusatz,
        alte_bezeichnung,
        kette,
        laenge,
        kette_kurz,
        laenge_2
    FROM SST_SEMANTIK
    WHERE NULLIF(TRIM(sst), '') IS NOT NULL
    QUALIFY ROW_NUMBER() OVER (
        PARTITION BY UPPER(TRIM(sst))
        ORDER BY UPPER(TRIM(sst))
    ) = 1
),

qa_workers AS (
    SELECT DISTINCT UPPER(TRIM(id)) AS worker_id
    FROM QA_ID
    WHERE NULLIF(TRIM(id), '') IS NOT NULL
)

SELECT
    d.stack_id,
    d.process_id,
    d.doc_id,
    d.subdoc_idx,
    d.entry_id AS export_entry_id,
    d.export_entry_time,
    d.status,
    d.document_altered_by,
    d.remark,
    d.sfdoc_class,
    d.exportname,
    d.document_entry_time,
    d.verified_by,
    d.last_entry,
    d.doc_state,
    d.doc_substate,

    s.sst,
    s.sst_altered_by,
    s.show_in_verifier,
    s.sst_action,
    s.sst_quality,
    s.sst_rating,
    s.sst_entry_time,

    sem.kategorie,
    sem.klasse,
    sem.aktion,
    sem.detail,
    sem.zusatz,
    sem.alte_bezeichnung,
    sem.kette,
    sem.laenge,
    sem.kette_kurz,
    sem.laenge_2,

    (s.sst IS NOT NULL) AS has_sst,
    (sem.normalized_sst IS NOT NULL) AS has_sst_semantics,
    (cs.stack_id IS NOT NULL) AS has_complete_workflow_history,

    CASE
        WHEN s.sst IS NULL THEN 'MISSING_SST'
        WHEN qa.worker_id IS NOT NULL THEN 'QA_VERIFIED'
        WHEN UPPER(TRIM(d.verified_by)) = 'SYSTEM' THEN 'SYSTEM_VERIFIED'
        WHEN UPPER(TRIM(d.verified_by)) LIKE 'SVC%' THEN 'SERVICE_ACCOUNT'
        WHEN NULLIF(TRIM(d.verified_by), '') IS NOT NULL THEN 'NON_QA_VERIFIER'
        ELSE 'MISSING_VERIFIER'
    END AS label_tier,

    CASE
        WHEN s.sst IS NULL THEN 'EXCLUDE'
        WHEN qa.worker_id IS NOT NULL THEN 'GOLD'
        WHEN NULLIF(TRIM(d.verified_by), '') IS NOT NULL
             AND UPPER(TRIM(d.verified_by)) <> 'SYSTEM'
             AND UPPER(TRIM(d.verified_by)) NOT LIKE 'SVC%'
            THEN 'SILVER'
        WHEN UPPER(TRIM(d.verified_by)) = 'SYSTEM' THEN 'WEAK'
        ELSE 'EXCLUDE'
    END AS training_label_quality

FROM export_documents AS d
LEFT JOIN export_sst AS s
    ON d.stack_id = s.stack_id
   AND d.process_id = s.process_id
   AND d.doc_id = s.doc_id
   AND d.subdoc_idx = s.subdoc_idx
   AND d.entry_id = s.entry_id
LEFT JOIN semantic_lookup AS sem
    ON UPPER(TRIM(s.sst)) = sem.normalized_sst
LEFT JOIN qa_workers AS qa
    ON UPPER(TRIM(d.verified_by)) = qa.worker_id
LEFT JOIN PROC_LIFE_COMPLETED_STACKS AS cs
    ON d.stack_id = cs.stack_id;


/*
Training-ready document labels. Gold and Silver remain distinguishable through
training_label_quality so evaluation can always be reported separately.
*/
CREATE OR REPLACE VIEW TRAINING_LIFE_DOCUMENT_LABELS AS
SELECT *
FROM PROC_LIFE_FINAL_DOCUMENT_LABELS
WHERE has_sst
  AND training_label_quality IN ('GOLD', 'SILVER');

