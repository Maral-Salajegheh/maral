/*
Build the validated workflow-analysis scope.

A completed stack must contain both ImportLogicModule and AfterExport, and the
last export must not precede the first import. This scope is appropriate for
workflow duration and correction analysis. Final-label training data is built
from all valid exports and does not require a complete imported history.
*/

CREATE OR REPLACE TABLE PROC_LIFE_COMPLETED_STACKS AS
WITH stack_boundaries AS (
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
    stack_id,
    first_import_time,
    last_export_time,
    n_import_events,
    n_export_events
FROM stack_boundaries
WHERE n_import_events > 0
  AND n_export_events > 0
  AND last_export_time >= first_import_time;

