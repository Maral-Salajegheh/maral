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


def run_pipeline(schema: str, assume_yes: bool = False) -> None:
    """Validate and execute all Life pipeline stages in dependency order."""
    schema = validate_identifier(schema)
    validate_pipeline_files()

    engine = get_engine(schema=schema)
    validate_input_tables(engine, schema)

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
        sql = read_sql_file(path, schema)

        try:
            execute_query(engine, sql)
            verify_output_table(engine, schema, expected_output)
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
