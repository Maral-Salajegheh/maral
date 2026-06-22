/*
Attach final Export SST labels to final Export pages.

IMAGE_ID deliberately remains VARCHAR. The mapping count exposes ambiguous
page-to-document assignments instead of silently selecting one mapping.

Expected training grain after filtering:
  one row per (stack_id, process_id, image_id).
*/

CREATE OR REPLACE TABLE PROC_LIFE_FINAL_PAGE_LABELS AS
WITH export_page_mappings AS (
    SELECT
        p.stack_id,
        p.process_id,
        p.doc_id,
        p.subdoc_idx,
        p.entry_id,
        p.image_id,
        p.seqno,
        p.relation,
        p.rotation,
        p.entry_time AS page_entry_time,
        COUNT(*) OVER (
            PARTITION BY p.stack_id, p.process_id, p.entry_id, p.image_id
        ) AS n_document_mappings_for_page
    FROM AD_IMAGE2DOCUMENT AS p
    INNER JOIN (
        SELECT DISTINCT stack_id, export_entry_id
        FROM PROC_LIFE_FINAL_DOCUMENT_LABELS
    ) AS e
        ON p.stack_id = e.stack_id
       AND p.entry_id = e.export_entry_id
)
SELECT
    p.stack_id,
    p.process_id,
    p.doc_id,
    p.subdoc_idx,
    p.entry_id AS export_entry_id,
    p.image_id,
    p.seqno,
    p.relation,
    p.rotation,
    p.page_entry_time,
    p.n_document_mappings_for_page,
    (p.n_document_mappings_for_page = 1) AS has_unambiguous_document_mapping,

    d.sst,
    d.label_tier,
    d.training_label_quality,
    d.has_sst,
    d.has_sst_semantics,
    d.has_complete_workflow_history,
    d.sfdoc_class,
    d.verified_by,
    d.sst_altered_by,
    d.show_in_verifier,
    d.sst_action,
    d.sst_quality,
    d.sst_rating,
    d.kategorie,
    d.klasse,
    d.aktion,
    d.detail,
    d.zusatz
FROM export_page_mappings AS p
LEFT JOIN PROC_LIFE_FINAL_DOCUMENT_LABELS AS d
    ON p.stack_id = d.stack_id
   AND p.process_id = d.process_id
   AND p.doc_id = d.doc_id
   AND p.subdoc_idx = d.subdoc_idx
   AND p.entry_id = d.export_entry_id;


CREATE OR REPLACE VIEW TRAINING_LIFE_PAGE_LABELS AS
SELECT *
FROM PROC_LIFE_FINAL_PAGE_LABELS
WHERE has_unambiguous_document_mapping
  AND has_sst
  AND training_label_quality IN ('GOLD', 'SILVER');

