/*
Compare page grouping between the first Analyser1 snapshot and the last Export.

This table measures segmentation/grouping corrections only. It does not compare
SST values because the validated Life data contains SST only at AfterExport.
All pages are preserved through a FULL OUTER JOIN, including pages present on
only one side. IMAGE_ID remains VARCHAR.
*/

CREATE OR REPLACE TABLE PROC_LIFE_PAGE_GROUPING_CHANGES AS
WITH first_analyser AS (
    SELECT
        stack_id,
        entry_id,
        entry_time AS analyser_entry_time,
        component AS analyser_component
    FROM AD_STACK
    WHERE state = 'Analyser1'
    QUALIFY ROW_NUMBER() OVER (
        PARTITION BY stack_id
        ORDER BY entry_time, entry_id
    ) = 1
),

last_export AS (
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

analyser_pages AS (
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
        a.analyser_entry_time,
        a.analyser_component
    FROM AD_IMAGE2DOCUMENT AS p
    INNER JOIN first_analyser AS a
        ON p.stack_id = a.stack_id
       AND p.entry_id = a.entry_id
    INNER JOIN PROC_LIFE_COMPLETED_STACKS AS cs
        ON p.stack_id = cs.stack_id
),

export_pages AS (
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
        e.export_entry_time
    FROM AD_IMAGE2DOCUMENT AS p
    INNER JOIN last_export AS e
        ON p.stack_id = e.stack_id
       AND p.entry_id = e.entry_id
    INNER JOIN PROC_LIFE_COMPLETED_STACKS AS cs
        ON p.stack_id = cs.stack_id
)

SELECT
    COALESCE(a.stack_id, e.stack_id) AS stack_id,
    COALESCE(a.process_id, e.process_id) AS process_id,
    COALESCE(a.image_id, e.image_id) AS image_id,

    a.entry_id AS analyser_entry_id,
    e.entry_id AS export_entry_id,
    a.analyser_entry_time,
    e.export_entry_time,
    a.analyser_component,

    a.doc_id AS analyser_doc_id,
    a.subdoc_idx AS analyser_subdoc_idx,
    e.doc_id AS export_doc_id,
    e.subdoc_idx AS export_subdoc_idx,
    a.seqno AS analyser_seqno,
    e.seqno AS export_seqno,
    a.rotation AS analyser_rotation,
    e.rotation AS export_rotation,

    CASE
        WHEN a.image_id IS NOT NULL AND e.image_id IS NOT NULL THEN 'BOTH'
        WHEN a.image_id IS NOT NULL THEN 'ANALYSER_ONLY'
        ELSE 'EXPORT_ONLY'
    END AS page_presence,

    CASE
        WHEN a.image_id IS NOT NULL AND e.image_id IS NOT NULL
        THEN EQUAL_NULL(a.doc_id, e.doc_id)
             AND EQUAL_NULL(a.subdoc_idx, e.subdoc_idx)
        ELSE NULL
    END AS same_grouping,

    CASE
        WHEN a.image_id IS NULL THEN 'ADDED_AT_EXPORT'
        WHEN e.image_id IS NULL THEN 'REMOVED_BEFORE_EXPORT'
        WHEN EQUAL_NULL(a.doc_id, e.doc_id)
             AND EQUAL_NULL(a.subdoc_idx, e.subdoc_idx)
            THEN 'UNCHANGED_GROUPING'
        ELSE 'REGROUPED'
    END AS grouping_change_type,

    labels.sst AS final_sst,
    labels.label_tier,
    labels.training_label_quality

FROM analyser_pages AS a
FULL OUTER JOIN export_pages AS e
    ON a.stack_id = e.stack_id
   AND a.process_id = e.process_id
   AND a.image_id = e.image_id
LEFT JOIN PROC_LIFE_FINAL_DOCUMENT_LABELS AS labels
    ON e.stack_id = labels.stack_id
   AND e.process_id = labels.process_id
   AND e.doc_id = labels.doc_id
   AND e.subdoc_idx = labels.subdoc_idx
   AND e.entry_id = labels.export_entry_id;

