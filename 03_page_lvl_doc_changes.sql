/*
03_page_lvl_doc_changes.sql  (LIFE / SST version)

Page-level classification label: for every page (image_id), compare the SST
assigned at the Analyser stage (_i) vs at Export (_e). Builds
PROC_DOC_CHANGES_PAGE_LEVEL.

DIFFERENCE FROM PnC: the tracked value is the SST (from AD_FIELD where
field_name = 'SST'), not sfdoc_class. The SST is a single business code, so the
transition string is just  sst_i -> sst_e  (no '/' SPLIT_PART). Optional
SST_Semantik enrichment can be layered on later, as in find_field_holding_true_class.

Confirm for Life: COMPONENT_ANALYSER, STATE_EXPORT, SST_FIELD_NAME.
*/

-- ===== LIFE CONFIG =================================================== START
SET AD_STACK          = 'AD_STACK';
SET AD_FIELD          = 'AD_FIELD';
SET AD_IMAGE2DOCUMENT = 'AD_IMAGE2DOCUMENT';
SET SST_FIELD_NAME     = 'SST';
SET COMPONENT_ANALYSER = 'Analyser';      -- TODO confirm Life value (ad_stack.component)
SET STATE_EXPORT       = 'AfterExport';   -- TODO confirm Life value (ad_stack.state)
-- ===== LIFE CONFIG ===================================================== END

-- SST per (stack, process, doc, subdoc, entry)
CREATE OR REPLACE TEMPORARY TABLE sst_per_doc_entry AS
SELECT
    stack_id, process_id, doc_id, subdoc_idx, entry_id,
    field_value AS sst
FROM IDENTIFIER($AD_FIELD)
WHERE field_name = $SST_FIELD_NAME
QUALIFY ROW_NUMBER() OVER (
    PARTITION BY stack_id, process_id, doc_id, subdoc_idx, entry_id
    ORDER BY entry_time DESC
) = 1;

-- Pages as seen at the ANALYSER stage -----------------------------------------
CREATE OR REPLACE TEMPORARY TABLE analyser1_pages AS
WITH ad_stack_processed AS (
    SELECT
        stack_id, entry_id,
        ROW_NUMBER() OVER (PARTITION BY stack_id, component ORDER BY entry_id) AS rn,
        subsystem, category, state, component, entry_time
    FROM IDENTIFIER($AD_STACK)
)
SELECT
    pages.entry_id,
    pages.stack_id,
    pages.process_id,
    pages.doc_id,
    pages.subdoc_idx,
    pages.image_id,
    sst.sst,
    stack.subsystem,
    stack.category,
    stack.state,
    stack.entry_time
FROM IDENTIFIER($AD_IMAGE2DOCUMENT) AS pages
INNER JOIN sst_per_doc_entry AS sst
    ON  pages.stack_id   = sst.stack_id
    AND pages.entry_id   = sst.entry_id
    AND pages.process_id = sst.process_id
    AND pages.doc_id     = sst.doc_id
    AND pages.subdoc_idx = sst.subdoc_idx
INNER JOIN ad_stack_processed AS stack
    ON  pages.stack_id = stack.stack_id
    AND pages.entry_id = stack.entry_id
WHERE stack.rn = 1 AND stack.component = $COMPONENT_ANALYSER;

-- Pages as seen at EXPORT ------------------------------------------------------
CREATE OR REPLACE TEMPORARY TABLE export_pages AS
WITH ad_stack_processed AS (
    SELECT
        stack_id, entry_id,
        ROW_NUMBER() OVER (PARTITION BY stack_id, state ORDER BY entry_id) AS rn,
        subsystem, category, state, component, entry_time
    FROM IDENTIFIER($AD_STACK)
)
SELECT
    pages.entry_id,
    pages.stack_id,
    pages.process_id,
    pages.doc_id,
    pages.subdoc_idx,
    pages.image_id,
    sst.sst,
    stack.subsystem,
    stack.category,
    stack.state,
    stack.entry_time
FROM IDENTIFIER($AD_IMAGE2DOCUMENT) AS pages
INNER JOIN sst_per_doc_entry AS sst
    ON  pages.stack_id   = sst.stack_id
    AND pages.entry_id   = sst.entry_id
    AND pages.process_id = sst.process_id
    AND pages.doc_id     = sst.doc_id
    AND pages.subdoc_idx = sst.subdoc_idx
INNER JOIN ad_stack_processed AS stack
    ON  pages.stack_id = stack.stack_id
    AND pages.entry_id = stack.entry_id
WHERE stack.rn = 1 AND stack.state = $STATE_EXPORT;

-- Page-level comparison: analyser (_i) vs export (_e) --------------------------
CREATE OR REPLACE TABLE proc_doc_changes_page_level AS
SELECT
    COALESCE(a.stack_id, b.stack_id)                          AS stack_id,
    COALESCE(c.docai_pull = 'docai_pull', FALSE)              AS docai_pull,
    COALESCE(c.altered_by_docai = 'altered_by_docai', FALSE)  AS altered_by_docai,
    COALESCE(a.process_id, b.process_id)                      AS process_id,
    TRY_TO_NUMBER(COALESCE(a.image_id, b.image_id))           AS image_id,  -- skip non-numeric image_ids
    a.entry_id                                                AS entry_id_i,
    b.entry_id                                                AS entry_id_e,
    a.entry_time                                              AS time_analyser,
    b.entry_time                                              AS time_export,
    a.doc_id                                                  AS doc_id_i,
    b.doc_id                                                  AS doc_id_e,
    a.subdoc_idx                                              AS subdoc_idx_i,
    b.subdoc_idx                                              AS subdoc_idx_e,
    a.sst                                                     AS sst_i,
    b.sst                                                     AS sst_e,
    a.subsystem,
    a.category,
    (a.doc_id = b.doc_id) AND (a.subdoc_idx = b.subdoc_idx)   AS same_grouping,
    (a.sst = b.sst)                                           AS same_doc_class,
    (same_grouping AND same_doc_class)                        AS no_change,
    (a.sst || ' -> ' || b.sst)                                AS change
FROM analyser1_pages AS a
FULL OUTER JOIN export_pages AS b
    ON  a.stack_id   = b.stack_id
    AND a.process_id = b.process_id
    AND a.image_id   = b.image_id
LEFT JOIN docai_stack AS c
    ON a.stack_id = c.stack_id;

ALTER TABLE proc_doc_changes_page_level ADD PRIMARY KEY (stack_id, process_id, image_id);
