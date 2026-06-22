# Copyright (c) QuantCo 2025-2025
# SPDX-License-Identifier: LicenseRef-QuantCo

"""Run the validated Life AnalyseDB SQL pipeline in dependency order.

The runner executes only the explicitly listed SQL files. It does not scan and
execute every file in the directory, so legacy SQL files and documentation can
never be executed accidentally.

Example:
    python run_pipeline.py D131_D2D

Non-interactive execution:
    python run_pipeline.py D131_D2D --yes
"""

from __future__ import annotations

import re
import sys
from argparse import ArgumentParser
from pathlib import Path
from time import perf_counter

import sqlalchemy

from life_docai.utils.snowflake_utils import execute_query, get_engine


RUNNER_DIR = Path(__file__).resolve().parent

# Match the existing repository layout. The fallback keeps the delivered flat
# package runnable when the SQL files are placed next to this script.
SQL_SUBDIRECTORY = RUNNER_DIR / "sql_analyse_db"
SQL_DIR = SQL_SUBDIRECTORY if SQL_SUBDIRECTORY.is_dir() else RUNNER_DIR

# The order is part of the pipeline contract. Do not replace this with iterdir().
PIPELINE_STEPS = [
    ("00_completed_stacks.sql", "PROC_LIFE_COMPLETED_STACKS"),
    ("01_final_document_labels.sql", "PROC_LIFE_FINAL_DOCUMENT_LABELS"),
    ("02_final_page_labels.sql", "PROC_LIFE_FINAL_PAGE_LABELS"),
    ("03_page_grouping_changes.sql", "PROC_LIFE_PAGE_GROUPING_CHANGES"),
    ("04_stack_aggregation.sql", "PROC_LIFE_STACK_AGG"),
]

# Every completed stage must satisfy these data-grain contracts. A stage can
# have more than one contract when it creates both a processed table and a
# training view.
GRAIN_CONTRACTS = {
    "00_completed_stacks.sql": [
        ("PROC_LIFE_COMPLETED_STACKS", ["STACK_ID"]),
    ],
    "01_final_document_labels.sql": [
        (
            "PROC_LIFE_FINAL_DOCUMENT_LABELS",
            ["STACK_ID", "PROCESS_ID", "DOC_ID", "SUBDOC_IDX"],
        ),
        (
            "TRAINING_LIFE_DOCUMENT_LABELS",
            ["STACK_ID", "PROCESS_ID", "DOC_ID", "SUBDOC_IDX"],
        ),
    ],
    "02_final_page_labels.sql": [
        (
            "PROC_LIFE_FINAL_PAGE_LABELS",
            [
                "STACK_ID",
                "PROCESS_ID",
                "EXPORT_ENTRY_ID",
                "IMAGE_ID",
                "DOC_ID",
                "SUBDOC_IDX",
            ],
        ),
        (
            "TRAINING_LIFE_PAGE_LABELS",
            ["STACK_ID", "PROCESS_ID", "IMAGE_ID"],
        ),
    ],
    "03_page_grouping_changes.sql": [
        (
            "PROC_LIFE_PAGE_GROUPING_CHANGES",
            ["STACK_ID", "PROCESS_ID", "IMAGE_ID"],
        ),
    ],
    "04_stack_aggregation.sql": [
        ("PROC_LIFE_STACK_AGG", ["STACK_ID"]),
    ],
}

REQUIRED_INPUT_TABLES = {
    "AD_STACK",
    "AD_DOCUMENT",
    "AD_FIELD",
    "AD_IMAGE2DOCUMENT",
    "QA_ID",
    "SST_SEMANTIK",
}


def validate_identifier(value: str) -> str:
    """Allow only unquoted Snowflake identifier characters."""
    if not re.fullmatch(r"[A-Za-z0-9_]+", value):
        raise ValueError(f"Unsafe Snowflake identifier: {value!r}")
    return value.upper()


def get_schema_tables(engine: sqlalchemy.Engine, schema: str) -> set[str]:
    """Return the upper-case table/view names currently present in the schema."""
    query = sqlalchemy.text(
        """
        SELECT table_name
        FROM information_schema.tables
        WHERE table_schema = :schema
        """
    )
    with engine.connect() as connection:
        rows = connection.execute(query, {"schema": schema}).fetchall()
    return {str(row[0]).upper() for row in rows}


def validate_pipeline_files() -> None:
    """Fail before connecting if a required SQL file is missing."""
    missing_files = [
        filename
        for filename, _ in PIPELINE_STEPS
        if not (SQL_DIR / filename).is_file()
    ]
    if missing_files:
        raise FileNotFoundError(
            "Missing pipeline SQL files: " + ", ".join(missing_files)
        )


def validate_input_tables(engine: sqlalchemy.Engine, schema: str) -> None:
    """Fail before execution if a required raw/reference table is unavailable."""
    existing = get_schema_tables(engine, schema)
    missing = sorted(REQUIRED_INPUT_TABLES - existing)
    if missing:
        raise RuntimeError(
            "The pipeline cannot start because required input tables are missing: "
            + ", ".join(missing)
        )

    print("Input validation passed.")
    print("Required tables:", ", ".join(sorted(REQUIRED_INPUT_TABLES)))


def fetch_scalar(
    engine: sqlalchemy.Engine,
    query: str,
    parameters: dict[str, object] | None = None,
) -> int:
    """Execute a scalar validation query and return its integer result."""
    with engine.connect() as connection:
        value = connection.execute(
            sqlalchemy.text(query), parameters or {}
        ).scalar_one()
    return int(value)


def validate_semantic_uniqueness(
    engine: sqlalchemy.Engine,
    schema: str,
) -> None:
    """Reject real SST codes with multiple semantic definitions.

    Blank semantic rows are intentionally ignored. They are not valid SST codes
    and are already excluded by the label-building SQL.
    """
    duplicate_count = fetch_scalar(
        engine,
        f"""
        SELECT COUNT(*)
        FROM (
            SELECT UPPER(TRIM(sst)) AS normalized_sst
            FROM {schema}.SST_SEMANTIK
            WHERE NULLIF(TRIM(sst), '') IS NOT NULL
            GROUP BY UPPER(TRIM(sst))
            HAVING COUNT(*) > 1
        )
        """,
    )
    if duplicate_count > 0:
        raise RuntimeError(
            "SST_SEMANTIK contains "
            f"{duplicate_count} duplicated non-empty SST codes. "
            "The pipeline will not choose a semantic row arbitrarily."
        )
    print("Semantic uniqueness validation passed.")


def validate_process_id_stability(
    engine: sqlalchemy.Engine,
    schema: str,
) -> None:
    """Ensure a physical page keeps its process_id from Analyser to Export.

    The grouping SQL joins by stack_id, process_id, and image_id. If process_id
    changes for the same stack/image pair, the page would be misclassified as
    removed and added instead of being compared directly.
    """
    mismatch_count = fetch_scalar(
        engine,
        f"""
        WITH first_analyser AS (
            SELECT stack_id, entry_id
            FROM {schema}.AD_STACK
            WHERE state = 'Analyser1'
            QUALIFY ROW_NUMBER() OVER (
                PARTITION BY stack_id
                ORDER BY entry_time, entry_id
            ) = 1
        ),
        last_export AS (
            SELECT stack_id, entry_id
            FROM {schema}.AD_STACK
            WHERE state = 'AfterExport'
            QUALIFY ROW_NUMBER() OVER (
                PARTITION BY stack_id
                ORDER BY entry_time DESC, entry_id DESC
            ) = 1
        ),
        analyser_pages AS (
            SELECT DISTINCT p.stack_id, p.image_id, p.process_id
            FROM {schema}.AD_IMAGE2DOCUMENT AS p
            INNER JOIN first_analyser AS a
                ON p.stack_id = a.stack_id
               AND p.entry_id = a.entry_id
            WHERE p.image_id IS NOT NULL
        ),
        export_pages AS (
            SELECT DISTINCT p.stack_id, p.image_id, p.process_id
            FROM {schema}.AD_IMAGE2DOCUMENT AS p
            INNER JOIN last_export AS e
                ON p.stack_id = e.stack_id
               AND p.entry_id = e.entry_id
            WHERE p.image_id IS NOT NULL
        )
        SELECT COUNT(*)
        FROM analyser_pages AS a
        INNER JOIN export_pages AS e
            ON a.stack_id = e.stack_id
           AND a.image_id = e.image_id
        WHERE NOT EQUAL_NULL(a.process_id, e.process_id)
        """,
    )
    if mismatch_count > 0:
        raise RuntimeError(
            "process_id is not stable for "
            f"{mismatch_count} matched Analyser/Export page pairs. "
            "PROC_LIFE_PAGE_GROUPING_CHANGES would misclassify those pages."
        )
    print("Analyser-to-Export process_id stability validation passed.")


def read_sql_file(path: Path, schema: str) -> str:
    """Read one SQL stage and explicitly select the target schema."""
    sql_contents = path.read_text(encoding="utf-8")
    return f"USE SCHEMA {schema};\n\n{sql_contents}"


def verify_output_table(
    engine: sqlalchemy.Engine,
    schema: str,
    expected_output: str,
) -> None:
    """Confirm that a stage created its primary expected output table."""
    existing = get_schema_tables(engine, schema)
    if expected_output.upper() not in existing:
        raise RuntimeError(
            f"Stage completed without creating expected output "
            f"{schema}.{expected_output}."
        )


def validate_grain_contract(
    engine: sqlalchemy.Engine,
    schema: str,
    object_name: str,
    key_columns: list[str],
) -> None:
    """Require a non-empty output with no duplicate expected-grain keys."""
    column_sql = ", ".join(key_columns)

    row_count = fetch_scalar(
        engine,
        f"SELECT COUNT(*) FROM {schema}.{object_name}",
    )
    if row_count == 0:
        raise RuntimeError(
            f"Postcondition failed: {schema}.{object_name} is empty."
        )

    duplicate_group_count = fetch_scalar(
        engine,
        f"""
        SELECT COUNT(*)
        FROM (
            SELECT {column_sql}
            FROM {schema}.{object_name}
            GROUP BY {column_sql}
            HAVING COUNT(*) > 1
        )
        """,
    )
    if duplicate_group_count > 0:
        raise RuntimeError(
            f"Postcondition failed: {schema}.{object_name} contains "
            f"{duplicate_group_count} duplicated grain keys for "
            f"({column_sql})."
        )

    print(
        f"Grain validation passed for {schema}.{object_name}: "
        f"{row_count:,} rows, key ({column_sql})."
    )


def validate_stage_outputs(
    engine: sqlalchemy.Engine,
    schema: str,
    filename: str,
) -> None:
    """Run every declared grain contract for a completed pipeline stage."""
    for object_name, key_columns in GRAIN_CONTRACTS[filename]:
        verify_output_table(engine, schema, object_name)
        validate_grain_contract(
            engine=engine,
            schema=schema,
            object_name=object_name,
            key_columns=key_columns,
        )


def run_pipeline(schema: str, assume_yes: bool = False) -> None:
    """Validate and execute all Life pipeline stages in dependency order."""
    schema = validate_identifier(schema)
    validate_pipeline_files()

    engine = get_engine(schema=schema)
    validate_input_tables(engine, schema)
    validate_semantic_uniqueness(engine, schema)

    print(f"\nTarget schema: {schema}")
    print("Pipeline stages:")
    for index, (filename, output_table) in enumerate(PIPELINE_STEPS, start=1):
        print(f"  {index}. {filename} -> {output_table}")

    if not assume_yes:
        confirmation = input(
            f"\nRun the complete Life pipeline in schema {schema}? (y/n): "
        ).strip().lower()
        if confirmation != "y":
            print("Cancelled.")
            return

    pipeline_start = perf_counter()

    for index, (filename, expected_output) in enumerate(PIPELINE_STEPS, start=1):
        path = SQL_DIR / filename
        stage_start = perf_counter()

        print(f"\n[{index}/{len(PIPELINE_STEPS)}] Executing {filename}")

        if filename == "03_page_grouping_changes.sql":
            validate_process_id_stability(engine, schema)

        sql = read_sql_file(path, schema)

        try:
            execute_query(engine, sql)
            validate_stage_outputs(engine, schema, filename)
        except Exception as exc:
            elapsed = perf_counter() - stage_start
            print(f"FAILED after {elapsed:.2f} seconds: {filename}")
            print(f"Reason: {exc}")
            print("Pipeline stopped. Later dependent stages were not executed.")
            raise

        elapsed = perf_counter() - stage_start
        print(
            f"Completed {filename} in {elapsed:.2f} seconds; "
            f"verified {schema}.{expected_output}."
        )

    total_elapsed = perf_counter() - pipeline_start
    print(f"\nLife pipeline completed successfully in {total_elapsed:.2f} seconds.")


def main() -> None:
    parser = ArgumentParser(
        description="Run the validated Life AnalyseDB Snowflake pipeline."
    )
    parser.add_argument(
        "schema",
        help="Schema containing raw inputs and receiving processed outputs.",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Skip the interactive confirmation prompt.",
    )
    args = parser.parse_args()

    try:
        run_pipeline(schema=args.schema, assume_yes=args.yes)
    except Exception:
        sys.exit(1)


if __name__ == "__main__":
    main()
