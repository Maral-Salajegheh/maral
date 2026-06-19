/*
02_document_changes.sql  (LIFE / SST version)

Tracks how the SST (Schriftstuecktyp) changed across entry states, and folds in
document-separation events. Builds PROC_DOC_CHANGES.

DIFFERENCE FROM PnC: PnC tracked sfdoc_class (a column on AD_DOCUMENT). Life
tracks the SST, which is a FIELD VALUE in AD_FIELD where field_name = 'SST'.
The sst_per_doc_entry table below resolves it to one value per (stack, process,
doc, subdoc, entry); everything downstream is the same change/add/delete machinery.

Confirm for Life:
  - SST_FIELD_NAME ('SST') is the correct field_name in AD_FIELD.
  - the within-entry dedup tiebreaker (ORDER BY entry_time DESC) is right.
  - stacks rarely exceed 10 entry states (e1..e10 hardcoded).
  - ATTR_DOC_SEP is the correct document-separation attribute name.
  - grain (stack, process, doc, subdoc) matches find_field_holding_true_class.
*/

-- ===== LIFE CONFIG =================================================== START
SET AD_STACK        = 'AD_STACK';
SET AD_PROCESS      = 'AD_PROCESS';
SET AD_DOCUMENT     = 'AD_DOCUMENT';
SET AD_FIELD        = 'AD_FIELD';
SET AD_DOC2DOC_ATTR = 'AD_DOC2DOC_ATTR';
SET AD_DOC_ATTR     = 'AD_DOC_ATTR';
SET SST_FIELD_NAME  = 'SST';                  -- AD_FIELD.field_name holding the SST
SET ATTR_DOC_SEP    = 'DocumentSeparation';   -- TODO confirm Life value
-- ===== LIFE CONFIG ===================================================== END

-- ---------------------------------------------------------------------------
-- SST per (stack, process, doc, subdoc, entry): one value per document per entry
-- ---------------------------------------------------------------------------
CREATE OR REPLACE TEMPORARY TABLE sst_per_doc_entry AS
SELECT
    stack_id,
    process_id,
    doc_id,
    subdoc_idx,
    entry_id,
    field_value AS sst
FROM IDENTIFIER($AD_FIELD)
WHERE field_name = $SST_FIELD_NAME
QUALIFY ROW_NUMBER() OVER (
    PARTITION BY stack_id, process_id, doc_id, subdoc_idx, entry_id
    ORDER BY entry_time DESC
) = 1;

-- ---------------------------------------------------------------------------
-- Track the SST across the first 10 entry states (one CTE per entry_id)
-- ---------------------------------------------------------------------------
CREATE OR REPLACE TEMPORARY TABLE doc_changes_raw AS
WITH
e1  AS (SELECT stack_id, process_id, doc_id, subdoc_idx, sst FROM sst_per_doc_entry WHERE entry_id = 1),
e2  AS (SELECT stack_id, process_id, doc_id, subdoc_idx, sst FROM sst_per_doc_entry WHERE entry_id = 2),
e3  AS (SELECT stack_id, process_id, doc_id, subdoc_idx, sst FROM sst_per_doc_entry WHERE entry_id = 3),
e4  AS (SELECT stack_id, process_id, doc_id, subdoc_idx, sst FROM sst_per_doc_entry WHERE entry_id = 4),
e5  AS (SELECT stack_id, process_id, doc_id, subdoc_idx, sst FROM sst_per_doc_entry WHERE entry_id = 5),
e6  AS (SELECT stack_id, process_id, doc_id, subdoc_idx, sst FROM sst_per_doc_entry WHERE entry_id = 6),
e7  AS (SELECT stack_id, process_id, doc_id, subdoc_idx, sst FROM sst_per_doc_entry WHERE entry_id = 7),
e8  AS (SELECT stack_id, process_id, doc_id, subdoc_idx, sst FROM sst_per_doc_entry WHERE entry_id = 8),
e9  AS (SELECT stack_id, process_id, doc_id, subdoc_idx, sst FROM sst_per_doc_entry WHERE entry_id = 9),
e10 AS (SELECT stack_id, process_id, doc_id, subdoc_idx, sst FROM sst_per_doc_entry WHERE entry_id = 10),
e_all AS (
    SELECT DISTINCT stack_id, process_id, doc_id, subdoc_idx FROM IDENTIFIER($AD_DOCUMENT)
)
SELECT
    e_all.stack_id,
    e_all.process_id,
    e_all.doc_id,
    e_all.subdoc_idx,
    COALESCE(e1.sst,  '') AS e1sst,
    COALESCE(e2.sst,  '') AS e2sst,
    COALESCE(e3.sst,  '') AS e3sst,
    COALESCE(e4.sst,  '') AS e4sst,
    COALESCE(e5.sst,  '') AS e5sst,
    COALESCE(e6.sst,  '') AS e6sst,
    COALESCE(e7.sst,  '') AS e7sst,
    COALESCE(e8.sst,  '') AS e8sst,
    COALESCE(e9.sst,  '') AS e9sst,
    COALESCE(e10.sst, '') AS e10sst
FROM e_all
FULL OUTER JOIN e1  ON e_all.stack_id=e1.stack_id  AND e_all.process_id=e1.process_id  AND e_all.doc_id=e1.doc_id  AND e_all.subdoc_idx=e1.subdoc_idx
FULL OUTER JOIN e2  ON e_all.stack_id=e2.stack_id  AND e_all.process_id=e2.process_id  AND e_all.doc_id=e2.doc_id  AND e_all.subdoc_idx=e2.subdoc_idx
FULL OUTER JOIN e3  ON e_all.stack_id=e3.stack_id  AND e_all.process_id=e3.process_id  AND e_all.doc_id=e3.doc_id  AND e_all.subdoc_idx=e3.subdoc_idx
FULL OUTER JOIN e4  ON e_all.stack_id=e4.stack_id  AND e_all.process_id=e4.process_id  AND e_all.doc_id=e4.doc_id  AND e_all.subdoc_idx=e4.subdoc_idx
FULL OUTER JOIN e5  ON e_all.stack_id=e5.stack_id  AND e_all.process_id=e5.process_id  AND e_all.doc_id=e5.doc_id  AND e_all.subdoc_idx=e5.subdoc_idx
FULL OUTER JOIN e6  ON e_all.stack_id=e6.stack_id  AND e_all.process_id=e6.process_id  AND e_all.doc_id=e6.doc_id  AND e_all.subdoc_idx=e6.subdoc_idx
FULL OUTER JOIN e7  ON e_all.stack_id=e7.stack_id  AND e_all.process_id=e7.process_id  AND e_all.doc_id=e7.doc_id  AND e_all.subdoc_idx=e7.subdoc_idx
FULL OUTER JOIN e8  ON e_all.stack_id=e8.stack_id  AND e_all.process_id=e8.process_id  AND e_all.doc_id=e8.doc_id  AND e_all.subdoc_idx=e8.subdoc_idx
FULL OUTER JOIN e9  ON e_all.stack_id=e9.stack_id  AND e_all.process_id=e9.process_id  AND e_all.doc_id=e9.doc_id  AND e_all.subdoc_idx=e9.subdoc_idx
FULL OUTER JOIN e10 ON e_all.stack_id=e10.stack_id AND e_all.process_id=e10.process_id AND e_all.doc_id=e10.doc_id AND e_all.subdoc_idx=e10.subdoc_idx
ORDER BY stack_id, process_id, doc_id, subdoc_idx;

-- ---------------------------------------------------------------------------
-- Per adjacent pair: SST changed / deleted / added, plus existence flags.
-- SUM(exists)>0 guard prevents phantom changes from missing entry states.
-- ---------------------------------------------------------------------------
CREATE OR REPLACE TEMPORARY TABLE doc_changes_pivoted AS
WITH changes AS (
    SELECT
        stack_id,
        process_id,
        IFF(e1sst <> e2sst  AND e1sst <> '' AND e2sst  <> '', 1, 0) AS change_1_2,
        IFF(e2sst <> e3sst  AND e2sst <> '' AND e3sst  <> '', 1, 0) AS change_2_3,
        IFF(e3sst <> e4sst  AND e3sst <> '' AND e4sst  <> '', 1, 0) AS change_3_4,
        IFF(e4sst <> e5sst  AND e4sst <> '' AND e5sst  <> '', 1, 0) AS change_4_5,
        IFF(e5sst <> e6sst  AND e5sst <> '' AND e6sst  <> '', 1, 0) AS change_5_6,
        IFF(e6sst <> e7sst  AND e6sst <> '' AND e7sst  <> '', 1, 0) AS change_6_7,
        IFF(e7sst <> e8sst  AND e7sst <> '' AND e8sst  <> '', 1, 0) AS change_7_8,
        IFF(e8sst <> e9sst  AND e8sst <> '' AND e9sst  <> '', 1, 0) AS change_8_9,
        IFF(e9sst <> e10sst AND e9sst <> '' AND e10sst <> '', 1, 0) AS change_9_10,

        IFF(e1sst <> e2sst  AND e2sst  = '', 1, 0) AS delete_1_2,
        IFF(e2sst <> e3sst  AND e3sst  = '', 1, 0) AS delete_2_3,
        IFF(e3sst <> e4sst  AND e4sst  = '', 1, 0) AS delete_3_4,
        IFF(e4sst <> e5sst  AND e5sst  = '', 1, 0) AS delete_4_5,
        IFF(e5sst <> e6sst  AND e6sst  = '', 1, 0) AS delete_5_6,
        IFF(e6sst <> e7sst  AND e7sst  = '', 1, 0) AS delete_6_7,
        IFF(e7sst <> e8sst  AND e8sst  = '', 1, 0) AS delete_7_8,
        IFF(e8sst <> e9sst  AND e9sst  = '', 1, 0) AS delete_8_9,
        IFF(e9sst <> e10sst AND e10sst = '', 1, 0) AS delete_9_10,

        IFF(e1sst <> e2sst  AND e1sst  = '', 1, 0) AS add_1_2,
        IFF(e2sst <> e3sst  AND e2sst  = '', 1, 0) AS add_2_3,
        IFF(e3sst <> e4sst  AND e3sst  = '', 1, 0) AS add_3_4,
        IFF(e4sst <> e5sst  AND e4sst  = '', 1, 0) AS add_4_5,
        IFF(e5sst <> e6sst  AND e5sst  = '', 1, 0) AS add_5_6,
        IFF(e6sst <> e7sst  AND e6sst  = '', 1, 0) AS add_6_7,
        IFF(e7sst <> e8sst  AND e7sst  = '', 1, 0) AS add_7_8,
        IFF(e8sst <> e9sst  AND e8sst  = '', 1, 0) AS add_8_9,
        IFF(e9sst <> e10sst AND e9sst  = '', 1, 0) AS add_9_10,

        IFF(e1sst  <> '', 1, 0) AS e1_exists,
        IFF(e2sst  <> '', 1, 0) AS e2_exists,
        IFF(e3sst  <> '', 1, 0) AS e3_exists,
        IFF(e4sst  <> '', 1, 0) AS e4_exists,
        IFF(e5sst  <> '', 1, 0) AS e5_exists,
        IFF(e6sst  <> '', 1, 0) AS e6_exists,
        IFF(e7sst  <> '', 1, 0) AS e7_exists,
        IFF(e8sst  <> '', 1, 0) AS e8_exists,
        IFF(e9sst  <> '', 1, 0) AS e9_exists,
        IFF(e10sst <> '', 1, 0) AS e10_exists
    FROM doc_changes_raw
)
SELECT
    stack_id,
    process_id,
    IFF(SUM(e1_exists) > 0 AND SUM(e2_exists)  > 0, SUM(change_1_2),  NULL) AS c12,
    IFF(SUM(e2_exists) > 0 AND SUM(e3_exists)  > 0, SUM(change_2_3),  NULL) AS c23,
    IFF(SUM(e3_exists) > 0 AND SUM(e4_exists)  > 0, SUM(change_3_4),  NULL) AS c34,
    IFF(SUM(e4_exists) > 0 AND SUM(e5_exists)  > 0, SUM(change_4_5),  NULL) AS c45,
    IFF(SUM(e5_exists) > 0 AND SUM(e6_exists)  > 0, SUM(change_5_6),  NULL) AS c56,
    IFF(SUM(e6_exists) > 0 AND SUM(e7_exists)  > 0, SUM(change_6_7),  NULL) AS c67,
    IFF(SUM(e7_exists) > 0 AND SUM(e8_exists)  > 0, SUM(change_7_8),  NULL) AS c78,
    IFF(SUM(e8_exists) > 0 AND SUM(e9_exists)  > 0, SUM(change_8_9),  NULL) AS c89,
    IFF(SUM(e9_exists) > 0 AND SUM(e10_exists) > 0, SUM(change_9_10), NULL) AS c910,

    IFF(SUM(e1_exists) > 0 AND SUM(e2_exists)  > 0, SUM(delete_1_2),  NULL) AS d12,
    IFF(SUM(e2_exists) > 0 AND SUM(e3_exists)  > 0, SUM(delete_2_3),  NULL) AS d23,
    IFF(SUM(e3_exists) > 0 AND SUM(e4_exists)  > 0, SUM(delete_3_4),  NULL) AS d34,
    IFF(SUM(e4_exists) > 0 AND SUM(e5_exists)  > 0, SUM(delete_4_5),  NULL) AS d45,
    IFF(SUM(e5_exists) > 0 AND SUM(e6_exists)  > 0, SUM(delete_5_6),  NULL) AS d56,
    IFF(SUM(e6_exists) > 0 AND SUM(e7_exists)  > 0, SUM(delete_6_7),  NULL) AS d67,
    IFF(SUM(e7_exists) > 0 AND SUM(e8_exists)  > 0, SUM(delete_7_8),  NULL) AS d78,
    IFF(SUM(e8_exists) > 0 AND SUM(e9_exists)  > 0, SUM(delete_8_9),  NULL) AS d89,
    IFF(SUM(e9_exists) > 0 AND SUM(e10_exists) > 0, SUM(delete_9_10), NULL) AS d910,

    IFF(SUM(e1_exists) > 0 AND SUM(e2_exists)  > 0, SUM(add_1_2),  NULL) AS a12,
    IFF(SUM(e2_exists) > 0 AND SUM(e3_exists)  > 0, SUM(add_2_3),  NULL) AS a23,
    IFF(SUM(e3_exists) > 0 AND SUM(e4_exists)  > 0, SUM(add_3_4),  NULL) AS a34,
    IFF(SUM(e4_exists) > 0 AND SUM(e5_exists)  > 0, SUM(add_4_5),  NULL) AS a45,
    IFF(SUM(e5_exists) > 0 AND SUM(e6_exists)  > 0, SUM(add_5_6),  NULL) AS a56,
    IFF(SUM(e6_exists) > 0 AND SUM(e7_exists)  > 0, SUM(add_6_7),  NULL) AS a67,
    IFF(SUM(e7_exists) > 0 AND SUM(e8_exists)  > 0, SUM(add_7_8),  NULL) AS a78,
    IFF(SUM(e8_exists) > 0 AND SUM(e9_exists)  > 0, SUM(add_8_9),  NULL) AS a89,
    IFF(SUM(e9_exists) > 0 AND SUM(e10_exists) > 0, SUM(add_9_10), NULL) AS a910
FROM changes
GROUP BY stack_id, process_id;

-- ---------------------------------------------------------------------------
-- Unpivot: one row per (stack, process, entry) with changed/deleted/added.
-- ---------------------------------------------------------------------------
CREATE OR REPLACE TEMPORARY TABLE doc_changes_unpivoted AS
SELECT stack_id, process_id, 2  AS entry_id, c12  AS changed, d12  AS deleted, a12  AS added FROM doc_changes_pivoted
UNION ALL
SELECT stack_id, process_id, 3  AS entry_id, c23  AS changed, d23  AS deleted, a23  AS added FROM doc_changes_pivoted
UNION ALL
SELECT stack_id, process_id, 4  AS entry_id, c34  AS changed, d34  AS deleted, a34  AS added FROM doc_changes_pivoted
UNION ALL
SELECT stack_id, process_id, 5  AS entry_id, c45  AS changed, d45  AS deleted, a45  AS added FROM doc_changes_pivoted
UNION ALL
SELECT stack_id, process_id, 6  AS entry_id, c56  AS changed, d56  AS deleted, a56  AS added FROM doc_changes_pivoted
UNION ALL
SELECT stack_id, process_id, 7  AS entry_id, c67  AS changed, d67  AS deleted, a67  AS added FROM doc_changes_pivoted
UNION ALL
SELECT stack_id, process_id, 8  AS entry_id, c78  AS changed, d78  AS deleted, a78  AS added FROM doc_changes_pivoted
UNION ALL
SELECT stack_id, process_id, 9  AS entry_id, c89  AS changed, d89  AS deleted, a89  AS added FROM doc_changes_pivoted
UNION ALL
SELECT stack_id, process_id, 10 AS entry_id, c910 AS changed, d910 AS deleted, a910 AS added FROM doc_changes_pivoted;

-- ---------------------------------------------------------------------------
-- Final: join stack + process + SST-change counts + document-separation events
-- ---------------------------------------------------------------------------
CREATE OR REPLACE TABLE proc_doc_changes AS
WITH separate_document_info AS (
    SELECT
        stack_id,
        entry_id,
        COUNT(*) AS n_first_document_separation
    FROM (
        SELECT
            stack_id,
            doc_id,
            subdoc_idx,
            MIN(entry_id) AS entry_id
        FROM IDENTIFIER($AD_DOC2DOC_ATTR) AS d2d
        LEFT JOIN IDENTIFIER($AD_DOC_ATTR) AS dattr
            ON d2d.id_doc_attr = dattr.id_doc_attr
        WHERE attr_name = $ATTR_DOC_SEP
        GROUP BY stack_id, doc_id, subdoc_idx
    )
    GROUP BY stack_id, entry_id
)
SELECT
    stack.stack_id,
    stack.entry_time,
    process.process_id,
    stack.entry_id,
    stack.altered_by,
    stack.subsystem,
    stack.category,
    stack.state,
    dc.changed,
    dc.deleted,
    dc.added,
    sdi.n_first_document_separation AS doc_sep,
    (
        COALESCE(dc.changed, 0)
        + COALESCE(dc.deleted, 0)
        + COALESCE(dc.added, 0)
        + COALESCE(sdi.n_first_document_separation, 0)
    ) > 0 AS any_change
FROM IDENTIFIER($AD_STACK) AS stack
INNER JOIN IDENTIFIER($AD_PROCESS) AS process
    ON stack.stack_id = process.stack_id AND stack.entry_id = process.entry_id
LEFT JOIN doc_changes_unpivoted AS dc
    ON stack.stack_id = dc.stack_id AND process.process_id = dc.process_id AND stack.entry_id = dc.entry_id
LEFT JOIN separate_document_info AS sdi
    ON stack.stack_id = sdi.stack_id AND stack.entry_id = sdi.entry_id;

ALTER TABLE proc_doc_changes ADD PRIMARY KEY (stack_id, process_id, entry_id);
