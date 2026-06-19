-- ============================================================================
-- 01_docai.sql  (LIFE port)
-- Flag which stacks had DocAI involvement, two ways:
--   (a) a DocAI pull stack state, or
--   (b) any document altered_by a DocAI actor.
-- NOTE for Life: if Life DocAI is not yet in production, both inputs will be
-- (near) empty. That is expected -- keep the file, the downstream joins are
-- LEFT joins so empty docai_stack does no harm.
-- ============================================================================

-- ===== LIFE CONFIG =================================================== START
SET STATE_DOCAI_PULL = 'DocAiPull';   -- TODO confirm Life value (ad_stack.state)
SET ALTERED_BY_DOCAI = 'DocAi';       -- TODO confirm Life value (ad_document.altered_by)
-- ===== LIFE CONFIG ===================================================== END

CREATE OR REPLACE TEMPORARY TABLE docai_pull AS
SELECT DISTINCT
    stack_id,
    'docai_pull' AS docai_pull
FROM ad_stack
WHERE state = $STATE_DOCAI_PULL;

CREATE OR REPLACE TEMPORARY TABLE docai_alter AS
SELECT DISTINCT
    stack_id,
    'altered_by_docai' AS altered_by_docai
FROM ad_document
WHERE altered_by = $ALTERED_BY_DOCAI;

CREATE OR REPLACE TABLE docai_stack AS
SELECT
    COALESCE(docai_pull.stack_id, docai_alter.stack_id) AS stack_id,
    docai_pull,
    altered_by_docai
FROM docai_pull
FULL OUTER JOIN docai_alter
    ON docai_pull.stack_id = docai_alter.stack_id;
