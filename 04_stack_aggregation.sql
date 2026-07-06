/*
Build a Life-native stack-level analytical table.

This replaces PnC-specific stack parsing, routing placeholders, hard-coded
entry transitions, and misleading manual-step counts. Event counts are named as
events; they are not interpreted as effort or FTE.
*/

CREATE OR REPLACE TABLE PROC_LIFE_STACK_AGG AS
WITH qa_workers AS (
    SELECT DISTINCT UPPER(TRIM(id)) AS worker_id
    FROM QA_ID
    WHERE NULLIF(TRIM(id), '') IS NOT NULL
),

workflow AS (
    SELECT
        s.stack_id,
        MIN(IFF(s.state = 'ImportLogicModule', s.entry_time, NULL)) AS first_import_time,
        MIN(IFF(s.state = 'Analyser1', s.entry_time, NULL)) AS first_analyser_time,
        MIN(IFF(s.state = 'Supervisor', s.entry_time, NULL)) AS first_supervisor_time,
        MIN(IFF(s.state = 'Verifier', s.entry_time, NULL)) AS first_verifier_time,
        MIN(IFF(s.state = 'Supervisor2', s.entry_time, NULL)) AS first_supervisor2_time,
        MAX(IFF(s.state = 'AfterExport', s.entry_time, NULL)) AS last_export_time,

        COUNT_IF(s.state = 'Analyser1') AS n_analyser_events,
        COUNT_IF(s.state = 'Supervisor') AS n_supervisor_events,
        COUNT_IF(s.state = 'Verifier') AS n_verifier_events,
        COUNT_IF(s.state = 'Supervisor2') AS n_supervisor2_events,
        COUNT_IF(s.state = 'AfterExport') AS n_export_events,

        COUNT_IF(UPPER(TRIM(s.altered_by)) = 'SYSTEM') AS n_system_events,
        COUNT_IF(UPPER(TRIM(s.altered_by)) LIKE 'SVC%') AS n_service_account_events,
        COUNT_IF(
            NULLIF(TRIM(s.altered_by), '') IS NOT NULL
            AND UPPER(TRIM(s.altered_by)) <> 'SYSTEM'
            AND UPPER(TRIM(s.altered_by)) NOT LIKE 'SVC%'
        ) AS n_named_non_system_events,
        COUNT_IF(qa.worker_id IS NOT NULL) AS n_known_qa_events,

        COUNT(DISTINCT IFF(
            NULLIF(TRIM(s.altered_by), '') IS NOT NULL
            AND UPPER(TRIM(s.altered_by)) <> 'SYSTEM'
            AND UPPER(TRIM(s.altered_by)) NOT LIKE 'SVC%',
            s.altered_by,
            NULL
        )) AS n_distinct_named_non_system_actors,
         MAX(s.category) AS category,

        ARRAY_AGG(DISTINCT s.subsystem)
            WITHIN GROUP (ORDER BY s.subsystem) AS subsystem_list,
        ARRAY_AGG(DISTINCT s.category)
            WITHIN GROUP (ORDER BY s.category) AS category_list

    FROM AD_STACK AS s
    INNER JOIN PROC_LIFE_COMPLETED_STACKS AS cs
        ON s.stack_id = cs.stack_id
    LEFT JOIN qa_workers AS qa
        ON UPPER(TRIM(s.altered_by)) = qa.worker_id
    GROUP BY s.stack_id
),

label_summary AS (
    SELECT
        stack_id,
        COUNT(*) AS n_final_documents,
        COUNT_IF(has_sst) AS n_final_documents_with_sst,
        COUNT_IF(NOT has_sst) AS n_final_documents_without_sst,
        COUNT_IF(label_tier = 'QA_VERIFIED') AS n_qa_verified_labels,
        COUNT_IF(label_tier = 'NON_QA_VERIFIER') AS n_non_qa_verifier_labels,
        COUNT_IF(label_tier = 'SYSTEM_VERIFIED') AS n_system_verified_labels,
        COUNT_IF(label_tier = 'SERVICE_ACCOUNT') AS n_service_account_labels,
        ARRAY_AGG(DISTINCT sst) WITHIN GROUP (ORDER BY sst) AS final_sst_list
    FROM PROC_LIFE_FINAL_DOCUMENT_LABELS
    GROUP BY stack_id
),

grouping_summary AS (
    SELECT
        stack_id,
        COUNT(*) AS n_page_comparison_rows,
        COUNT_IF(page_presence = 'BOTH') AS n_pages_present_in_both,
        COUNT_IF(page_presence = 'ANALYSER_ONLY') AS n_pages_removed_before_export,
        COUNT_IF(page_presence = 'EXPORT_ONLY') AS n_pages_added_at_export,
        COUNT_IF(grouping_change_type = 'REGROUPED') AS n_regrouped_pages,
        AVG(
            CASE
                WHEN page_presence = 'BOTH' THEN IFF(same_grouping, 1, 0)
                ELSE NULL
            END
        ) AS pct_pages_same_grouping_among_matched
    FROM PROC_LIFE_PAGE_GROUPING_CHANGES
    GROUP BY stack_id
)

SELECT
    w.stack_id,
    w.category,
    w.subsystem_list,
    w.category_list,
    w.first_import_time,
    w.first_analyser_time,
    w.first_supervisor_time,
    w.first_verifier_time,
    w.first_supervisor2_time,
    w.last_export_time,

    DATEDIFF('second', w.first_import_time, w.last_export_time) / 3600.0
        AS hours_import_to_export,
    DATEDIFF('second', w.first_import_time, w.first_analyser_time) / 3600.0
        AS hours_import_to_first_analyser,
    DATEDIFF('second', w.first_analyser_time, w.last_export_time) / 3600.0
        AS hours_first_analyser_to_export,
    DATEDIFF('second', w.first_import_time, w.first_supervisor_time) / 3600.0
        AS hours_import_to_first_supervisor,
    DATEDIFF('second', w.first_import_time, w.first_verifier_time) / 3600.0
        AS hours_import_to_first_verifier,
    DATEDIFF('second', w.first_import_time, w.first_supervisor2_time) / 3600.0
        AS hours_import_to_first_supervisor2,

    w.n_analyser_events,
    w.n_supervisor_events,
    w.n_verifier_events,
    w.n_supervisor2_events,
    w.n_export_events,
    w.n_system_events,
    w.n_service_account_events,
    w.n_named_non_system_events,
    w.n_known_qa_events,
    w.n_distinct_named_non_system_actors,

    l.n_final_documents,
    l.n_final_documents_with_sst,
    l.n_final_documents_without_sst,
    l.n_qa_verified_labels,
    l.n_non_qa_verifier_labels,
    l.n_system_verified_labels,
    l.n_service_account_labels,
    COALESCE(l.final_sst_list, ARRAY_CONSTRUCT()) AS final_sst_list,

    g.n_page_comparison_rows,
    g.n_pages_present_in_both,
    g.n_pages_removed_before_export,
    g.n_pages_added_at_export,
    g.n_regrouped_pages,
    g.pct_pages_same_grouping_among_matched

FROM workflow AS w
LEFT JOIN label_summary AS l
    ON w.stack_id = l.stack_id
LEFT JOIN grouping_summary AS g
    ON w.stack_id = g.stack_id;
