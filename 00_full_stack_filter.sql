-- ============================================================================
-- 00_full_stack_filter.sql  (LIFE port of pnc-claims-classification)
-- Keep only completed cases: stacks that reached both an import state and an
-- export state. This is the AfterExport filter behind the volume collapse.
-- ============================================================================

-- ===== LIFE CONFIG =================================================== START
-- Confirm these two state strings against the Life ad_stack vocabulary.
-- (PnC defaults shown. Get the Life values from Herbert / Martin / SmartFix.)
SET STATE_IMPORT = 'ImportLogicModule';   -- TODO confirm Life value
SET STATE_EXPORT = 'AfterExport';         -- TODO confirm Life value
-- ===== LIFE CONFIG ===================================================== END

CREATE OR REPLACE TABLE completed_smartfix_stack AS
WITH imp AS (
    SELECT DISTINCT stack_id
    FROM ad_stack
    WHERE state = $STATE_IMPORT
),
exp AS (
    SELECT DISTINCT stack_id
    FROM ad_stack
    WHERE state = $STATE_EXPORT
)
SELECT imp.stack_id
FROM imp
INNER JOIN exp ON imp.stack_id = exp.stack_id;
