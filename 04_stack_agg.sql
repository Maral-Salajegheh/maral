/*
04_stack_agg.sql  (LIFE / SST version)  --> PROC_STACK_AGG

Master stack-level feature table. Joins four temp tables:
  - temp_stack_agg_stack_level(_workers) : per-state timing, throughput, manual
                                           processing counts, worker arrays
  - temp_doc_agg_stack_level             : SST counts/lists at analyser vs export
  - temp_doc_changes_stack_level         : SST changes (from proc_doc_changes, 02)
  - temp_page_agg_stack_level            : page counts + page-change percentages

DIFFERENCES FROM PnC:
  * Tracked classification value is the SST (AD_FIELD field_name = 'SST'), not
    sfdoc_class. The doc-agg block reads SST per (stack,process,doc,subdoc,entry)
    and there is no '/' mandant/type/detail split (SST is a single code; its
    semantics live in SST_Semantik).
  * stack_id positional parsing is PnC-specific and CONFIRMED WRONG for Life.
    It is replaced by the safe fallback stack_id_sa = stack_id, system_counter = 1.

State-dependent: confirm every STATE_* / COMPONENT_* value against Life ad_stack.
*/

-- ===== LIFE CONFIG =================================================== START
SET AD_STACK          = 'AD_STACK';
SET AD_FIELD          = 'AD_FIELD';
SET AD_IMAGE2DOCUMENT = 'AD_IMAGE2DOCUMENT';
SET SST_FIELD_NAME    = 'SST';

SET STATE_IMPORT      = 'ImportLogicModule';
SET STATE_EXPORT      = 'AfterExport';
SET STATE_ANALYSER1   = 'Analyser1';
SET STATE_ANALYSER2   = 'Analyser2';
SET STATE_SUPERVISOR  = 'Supervisor';
SET STATE_VERIFIER    = 'Verifier';
SET STATE_QA          = 'PreQAExporter';
SET STATE_QA_WORKER   = 'Supervisor2';
SET SYSTEM_AGNOSTIC   = 'System';

-- routing_class: PnC flagged "routed to another department". The Life analogue
-- is unknown -- likely an SST or SST_Semantik category. Default below yields 0
-- (no routing) until a real Life definition is supplied.
SET ROUTING_CLASS_STR = '<<LIFE_ROUTING_SST_OR_CATEGORY>>';  -- TODO define for Life
-- ===== LIFE CONFIG ===================================================== END


-- SST per (stack, process, doc, subdoc, entry) ------------------------------
CREATE OR REPLACE TEMPORARY TABLE sst_per_doc_entry AS
SELECT
    stack_id, process_id, doc_id, subdoc_idx, entry_id,
    field_value AS sst,
    IFF(CONTAINS(field_value, $ROUTING_CLASS_STR), 1, 0) AS routing_class
FROM IDENTIFIER($AD_FIELD)
WHERE field_name = $SST_FIELD_NAME
QUALIFY ROW_NUMBER() OVER (
    PARTITION BY stack_id, process_id, doc_id, subdoc_idx, entry_id
    ORDER BY entry_time DESC
) = 1;


/* ************************** STACK TABLES ***************************** */

CREATE OR REPLACE TEMPORARY TABLE temp_stack_agg_stack_level AS
WITH ad_stack_preprocced AS (
    SELECT
        raw.*,
        CASE WHEN raw.state = $STATE_IMPORT     THEN raw.datetime END AS import_time,
        CASE WHEN raw.state = $STATE_EXPORT     THEN raw.datetime END AS export_time,
        CASE WHEN raw.state = $STATE_ANALYSER1  THEN raw.datetime END AS analyser1_time,
        CASE WHEN raw.state = $STATE_ANALYSER2  THEN raw.datetime END AS analyser2_time,
        CASE WHEN raw.state = $STATE_SUPERVISOR THEN raw.datetime END AS supervisor_time,
        CASE WHEN raw.state = $STATE_VERIFIER   THEN raw.datetime END AS verifier_time,
        CASE WHEN raw.state = $STATE_QA         THEN raw.datetime END AS qa_time,
        CASE WHEN raw.state = $STATE_ANALYSER1  THEN raw.priority END AS analyser1_priority,
        CASE WHEN raw.state = $STATE_ANALYSER2  THEN raw.priority END AS analyser2_priority,
        CASE WHEN raw.state = $STATE_EXPORT     THEN raw.priority END AS export_priority,
        IFF(raw.altered_by <> $SYSTEM_AGNOSTIC, 1, 0) AS manual_processing,
        IFF((raw.altered_by <> $SYSTEM_AGNOSTIC) AND (raw.state = $STATE_SUPERVISOR), 1, 0) AS manual_processing_supervisor,
        IFF((raw.altered_by <> $SYSTEM_AGNOSTIC) AND (raw.state = $STATE_VERIFIER),   1, 0) AS manual_processing_verifier,

        -- vvvvv LIFE: PnC system-agnostic derivation removed (wrong for Life). vvvvv
        -- PnC was: SUBSTRING(stack_id,1,5)||SUBSTRING(stack_id,8) AS stack_id_sa,
        --          CAST(SUBSTRING(stack_id,6,2) AS INTEGER)       AS system_counter
        raw.stack_id AS stack_id_sa,
        1            AS system_counter
        -- ^^^^^ replace only if Life encodes multiple systems in stack_id ^^^^^
    FROM IDENTIFIER($AD_STACK) AS raw
),
system_changes AS (
    SELECT stack_id_sa, MAX(system_counter) AS max_system_counter
    FROM ad_stack_preprocced GROUP BY stack_id_sa
)
SELECT
    asp.stack_id,
    asp.stack_id_sa,
    MAX(asp.system_counter)                                  AS system_counter,
    MAX(sc.max_system_counter)                               AS max_system_counter,
    (MAX(asp.system_counter) = MAX(sc.max_system_counter))   AS last_system,
    MAX(asp.subsystem)                                       AS subsystem,
    MAX(asp.category)                                        AS category,
    MIN(asp.datetime)                                        AS min_date,
    MAX(asp.datetime)                                        AS max_date,
    MIN(asp.import_time)                                     AS import_time,
    MAX(asp.export_time)                                     AS export_time,
    MIN(asp.analyser1_time)                                  AS analyser1_time,
    MIN(asp.analyser2_time)                                  AS analyser2_time,
    MIN(asp.supervisor_time)                                 AS supervisor_time,
    MIN(asp.verifier_time)                                   AS verifier1_time,
    MIN(asp.qa_time)                                         AS qa_time,
    MIN(asp.analyser1_priority)                              AS analyser1_priority,
    MIN(asp.analyser2_priority)                              AS analyser2_priority,
    MIN(asp.export_priority)                                 AS export_priority,
    TIMEDIFF('minutes', MIN(asp.import_time), MAX(asp.export_time))     / 60 AS tt_export,
    TIMEDIFF('minutes', MIN(asp.import_time), MIN(asp.analyser1_time))  / 60 AS tt_analyser1,
    TIMEDIFF('minutes', MIN(asp.import_time), MIN(asp.supervisor_time)) / 60 AS tt_supervisor,
    TIMEDIFF('minutes', MIN(asp.import_time), MIN(asp.verifier_time))   / 60 AS tt_verifier,
    TIMEDIFF('minutes', MIN(asp.import_time), MIN(asp.qa_time))         / 60 AS tt_qa,
    SUM(asp.manual_processing)                               AS n_manual_processing_steps,
    SUM(asp.manual_processing_supervisor)                    AS n_manual_processing_steps_supervisor,
    SUM(asp.manual_processing_verifier)                      AS n_manual_processing_steps_verifier
FROM ad_stack_preprocced AS asp
INNER JOIN system_changes AS sc ON asp.stack_id_sa = sc.stack_id_sa
GROUP BY asp.stack_id, asp.stack_id_sa;


CREATE OR REPLACE TEMPORARY TABLE temp_stack_agg_stack_level_workers AS
WITH supervisor_workers AS (
    SELECT stack_id, ARRAY_AGG(altered_by) WITHIN GROUP (ORDER BY entry_id) AS worker_ls
    FROM IDENTIFIER($AD_STACK) WHERE state = $STATE_SUPERVISOR GROUP BY stack_id
),
verifier_workers AS (
    SELECT stack_id, ARRAY_AGG(altered_by) WITHIN GROUP (ORDER BY entry_id) AS worker_ls
    FROM IDENTIFIER($AD_STACK) WHERE state = $STATE_VERIFIER GROUP BY stack_id
),
qa_workers AS (
    SELECT stack_id, ARRAY_AGG(altered_by) WITHIN GROUP (ORDER BY entry_id) AS worker_ls
    FROM IDENTIFIER($AD_STACK) WHERE state = $STATE_QA_WORKER GROUP BY stack_id
)
SELECT
    t.*,
    sw.worker_ls                                               AS supervisor_workers,
    vw.worker_ls                                               AS verifier_workers,
    qw.worker_ls                                               AS qa_workers,
    MIN(t.min_date) OVER (PARTITION BY t.stack_id_sa)          AS min_date_sa,
    MAX(t.max_date) OVER (PARTITION BY t.stack_id_sa)          AS max_date_sa,
    TIMEDIFF('minutes', min_date_sa, max_date_sa) / (60 * 24)  AS dlz_h
FROM temp_stack_agg_stack_level AS t
LEFT JOIN supervisor_workers AS sw ON t.stack_id = sw.stack_id
LEFT JOIN verifier_workers   AS vw ON t.stack_id = vw.stack_id
LEFT JOIN qa_workers         AS qw ON t.stack_id = qw.stack_id;


/* ************************* DOCUMENT (SST) TABLES ******************** */
-- SST counts / lists at Analyser1 vs Export
CREATE OR REPLACE TEMPORARY TABLE temp_doc_agg_stack_level AS
WITH ad_stack_processed AS (
    SELECT
        raw.*,
        ROW_NUMBER() OVER (PARTITION BY raw.stack_id, raw.state ORDER BY raw.entry_time) AS rn
    FROM IDENTIFIER($AD_STACK) AS raw
),
analyser1_docs AS (  -- SST set at state = Analyser1
    SELECT
        sst.stack_id,
        COUNT(DISTINCT sst.doc_id, sst.subdoc_idx) AS n_docs,
        ARRAY_AGG(sst.sst)                         AS doc_list,
        SUM(sst.routing_class)                     AS routing_class_docs
    FROM sst_per_doc_entry AS sst
    INNER JOIN ad_stack_processed AS stack
        ON sst.stack_id = stack.stack_id AND sst.entry_id = stack.entry_id
    WHERE stack.state = $STATE_ANALYSER1 AND stack.rn = 1
    GROUP BY sst.stack_id
),
export_docs AS (     -- SST set at state = AfterExport
    SELECT
        sst.stack_id,
        sst.entry_id,
        COUNT(DISTINCT sst.doc_id, sst.subdoc_idx) AS n_docs,
        ARRAY_AGG(sst.sst)                         AS doc_list,
        SUM(sst.routing_class)                     AS routing_class_docs
    FROM sst_per_doc_entry AS sst
    INNER JOIN ad_stack_processed AS stack
        ON sst.stack_id = stack.stack_id AND sst.entry_id = stack.entry_id
    WHERE stack.state = $STATE_EXPORT AND stack.rn = 1
    GROUP BY sst.stack_id, sst.entry_id
)
SELECT
    COALESCE(a1.stack_id, exp.stack_id)        AS stack_id,
    a1.n_docs                                  AS n_docs_analyser_1,
    exp.n_docs                                 AS n_docs_export,
    a1.doc_list                                AS doc_list_analyser1,
    exp.doc_list                               AS doc_list_export,
    a1.routing_class_docs                      AS routing_class_docs_analyser1,
    exp.routing_class_docs                     AS routing_class_docs_export
FROM analyser1_docs AS a1
FULL OUTER JOIN export_docs AS exp ON a1.stack_id = exp.stack_id;


-- aggregate proc_doc_changes (from 02, SST-based) to stack level
CREATE OR REPLACE TEMPORARY TABLE temp_doc_changes_stack_level AS
WITH stacks AS (
    SELECT DISTINCT stack_id FROM IDENTIFIER($AD_STACK)
),
supervisor_changes AS (
    SELECT stack_id,
        SUM(COALESCE(changed,0)) AS changed, SUM(COALESCE(deleted,0)) AS deleted,
        SUM(COALESCE(added,0))   AS added,   SUM(COALESCE(doc_sep,0)) AS doc_sep,
        BOOLOR_AGG(any_change)   AS any_changes
    FROM proc_doc_changes WHERE state = $STATE_SUPERVISOR GROUP BY stack_id
),
verifier_changes AS (
    SELECT stack_id,
        SUM(COALESCE(changed,0)) AS changed, SUM(COALESCE(deleted,0)) AS deleted,
        SUM(COALESCE(added,0))   AS added,   SUM(COALESCE(doc_sep,0)) AS doc_sep,
        BOOLOR_AGG(any_change)   AS any_changes
    FROM proc_doc_changes WHERE state LIKE $STATE_VERIFIER || '%' GROUP BY stack_id
),
qa_changes AS (
    SELECT stack_id,
        SUM(COALESCE(changed,0)) AS changed, SUM(COALESCE(deleted,0)) AS deleted,
        SUM(COALESCE(added,0))   AS added,   SUM(COALESCE(doc_sep,0)) AS doc_sep,
        BOOLOR_AGG(any_change)   AS any_changes
    FROM proc_doc_changes WHERE state = $STATE_QA_WORKER GROUP BY stack_id
)
SELECT
    stack.stack_id,
    s.changed AS supervisor_changed, s.deleted AS supervisor_deleted, s.added AS supervisor_added, s.any_changes AS supervisor_any_changes,
    v.changed AS verifier_changed,   v.deleted AS verifier_deleted,   v.added AS verifier_added,   v.any_changes AS verifier_any_changes,
    q.changed AS qa_changed,         q.deleted AS qa_deleted,         q.added AS qa_added,         q.any_changes AS qa_any_changes
FROM stacks AS stack
LEFT JOIN supervisor_changes AS s ON stack.stack_id = s.stack_id
LEFT JOIN verifier_changes   AS v ON stack.stack_id = v.stack_id
LEFT JOIN qa_changes         AS q ON stack.stack_id = q.stack_id;


/* *************************** PAGE TABLES *************************** */

CREATE OR REPLACE TEMPORARY TABLE temp_page_agg_doc_level AS
WITH ad_stack_processed AS (
    SELECT raw.*,
        ROW_NUMBER() OVER (PARTITION BY raw.stack_id, raw.state ORDER BY raw.entry_time) AS rn
    FROM IDENTIFIER($AD_STACK) AS raw
),
analyser1_pages AS (
    SELECT page.stack_id, page.process_id, page.doc_id, page.subdoc_idx,
        COUNT(DISTINCT page.image_id) AS n_pages
    FROM IDENTIFIER($AD_IMAGE2DOCUMENT) AS page
    INNER JOIN ad_stack_processed AS stack
        ON page.stack_id = stack.stack_id AND page.entry_id = stack.entry_id
    WHERE stack.state = $STATE_ANALYSER1 AND stack.rn = 1
    GROUP BY page.stack_id, page.process_id, page.doc_id, page.subdoc_idx
),
export_pages AS (
    SELECT page.stack_id, page.process_id, page.doc_id, page.subdoc_idx,
        COUNT(DISTINCT page.image_id) AS n_pages
    FROM IDENTIFIER($AD_IMAGE2DOCUMENT) AS page
    INNER JOIN ad_stack_processed AS stack
        ON page.stack_id = stack.stack_id AND page.entry_id = stack.entry_id
    WHERE stack.state = $STATE_EXPORT AND stack.rn = 1
    GROUP BY page.stack_id, page.process_id, page.doc_id, page.subdoc_idx
)
SELECT
    COALESCE(a1.stack_id,   exp.stack_id)   AS stack_id,
    COALESCE(a1.process_id, exp.process_id) AS process_id,
    COALESCE(a1.doc_id,     exp.doc_id)     AS doc_id,
    COALESCE(a1.subdoc_idx, exp.subdoc_idx) AS subdoc_idx,
    a1.n_pages                              AS n_pages_analyser1,
    exp.n_pages                             AS n_pages_export
FROM analyser1_pages AS a1
FULL OUTER JOIN export_pages AS exp
    ON  a1.stack_id   = exp.stack_id
    AND a1.process_id = exp.process_id
    AND a1.doc_id     = exp.doc_id
    AND a1.subdoc_idx = exp.subdoc_idx;


CREATE OR REPLACE TEMPORARY TABLE temp_page_agg_stack_level AS
WITH page_table AS (
    SELECT stack_id,
        SUM(n_pages_analyser1) AS n_pages_analyser1,
        SUM(n_pages_export)    AS n_pages_export
    FROM temp_page_agg_doc_level GROUP BY stack_id
),
page_change_table AS (
    SELECT stack_id,
        COUNT(DISTINCT image_id)       AS n_pages,
        AVG(IFF(same_grouping,  1, 0)) AS perc_pages_same_grouping,
        AVG(IFF(same_doc_class, 1, 0)) AS perc_pages_same_doc_class,
        AVG(IFF(no_change,      1, 0)) AS perc_pages_no_change
    FROM proc_doc_changes_page_level GROUP BY stack_id
)
SELECT
    pt.stack_id, pt.n_pages_analyser1, pt.n_pages_export,
    pt_change.n_pages,
    pt_change.perc_pages_same_grouping,
    pt_change.perc_pages_same_doc_class,
    pt_change.perc_pages_no_change
FROM page_table AS pt
INNER JOIN page_change_table AS pt_change ON pt.stack_id = pt_change.stack_id;


/* ***************** COMBINE STACK-LEVEL TABLES ********************* */

CREATE OR REPLACE TABLE proc_stack_agg AS
SELECT
    stack.stack_id,
    stack.stack_id_sa,
    stack.system_counter,
    stack.max_system_counter,
    stack.last_system AND (COALESCE(doc.routing_class_docs_export, 0) = 0) AS last_system,
    stack.subsystem,
    stack.category,
    stack.min_date_sa,
    stack.max_date_sa,
    stack.min_date,
    stack.max_date,
    stack.import_time,
    stack.export_time,
    stack.analyser1_time,
    stack.analyser2_time,
    stack.supervisor_time,
    stack.verifier1_time,
    stack.qa_time,
    stack.analyser1_priority,
    stack.analyser2_priority,
    stack.export_priority,
    stack.tt_export,
    stack.tt_analyser1,
    stack.tt_supervisor,
    stack.tt_verifier,
    stack.tt_qa,
    stack.tt_verifier - stack.tt_supervisor                  AS supervisor_to_verifier,
    stack.n_manual_processing_steps,
    stack.n_manual_processing_steps_supervisor,
    stack.n_manual_processing_steps_verifier,
    stack.supervisor_workers,
    stack.verifier_workers,
    stack.qa_workers,
    doc.n_docs_analyser_1,
    doc.n_docs_export,
    COALESCE(doc.doc_list_analyser1, ARRAY_CONSTRUCT())      AS doc_list_analyser1,
    COALESCE(doc.doc_list_export,    ARRAY_CONSTRUCT())      AS doc_list_export,
    dc.supervisor_changed,
    dc.supervisor_added,
    dc.supervisor_deleted,
    dc.supervisor_any_changes,
    dc.verifier_changed,
    dc.verifier_added,
    dc.verifier_deleted,
    dc.verifier_any_changes,
    dc.qa_changed,
    dc.qa_added,
    dc.qa_deleted,
    dc.qa_any_changes,
    page.n_pages_analyser1,
    page.n_pages_export,
    page.perc_pages_same_doc_class,
    page.perc_pages_same_grouping,
    page.perc_pages_no_change
FROM temp_stack_agg_stack_level_workers AS stack
LEFT JOIN temp_doc_agg_stack_level     AS doc  ON stack.stack_id = doc.stack_id
LEFT JOIN temp_page_agg_stack_level    AS page ON stack.stack_id = page.stack_id
LEFT JOIN temp_doc_changes_stack_level AS dc   ON stack.stack_id = dc.stack_id;

ALTER TABLE proc_stack_agg ADD PRIMARY KEY (stack_id);
