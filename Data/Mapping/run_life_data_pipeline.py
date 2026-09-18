#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Run the Life data-preparation steps in order and stop on the first failure.

    1. run_pipeline.py       Analyse-DB SQL tables
    2. 00_map_mid_to_adb.py  MasterIndex document and page tables

Step 2 sits next to this file. Step 1 lives elsewhere in the repo, so pass its
path with --analyse-db-runner, or use --skip-analyse-db when the SQL tables are
already built.

Each step runs as its own process, so running them by hand gives the same
result as running them here.

Publishing to PROC_LIFE_MID_ADB is deliberately left out. Read the status
summary from step 2 first, then publish:

    python 00_map_mid_to_adb.py --schema <SCHEMA> --publish-snowflake

    python run_life_data_pipeline.py D131_D2D --skip-analyse-db
"""

from __future__ import annotations

import re
import subprocess
import sys
from argparse import ArgumentParser, Namespace
from pathlib import Path
from time import perf_counter


SCRIPT_DIR = Path(__file__).resolve().parent
MAPPING_SCRIPT = SCRIPT_DIR / "00_map_mid_to_adb.py"
ANALYSE_DB_RUNNER = SCRIPT_DIR / "run_pipeline.py"


def validate_identifier(value: str) -> str:
    """Allow only unquoted Snowflake identifiers."""
    if not re.fullmatch(r"[A-Za-z0-9_]+", value):
        raise ValueError(f"Unsafe Snowflake identifier: {value!r}")
    return value.upper()


def build_steps(schema: str, analyse_db_runner: Path,
                skip_analyse_db: bool) -> list[tuple[str, Path, list[str]]]:
    """The commands to run, in dependency order."""
    steps = []
    if not skip_analyse_db:
        steps.append(
            ("Analyse-DB pipeline", analyse_db_runner, [schema, "--yes"])
        )
    steps.append(
        ("MasterIndex mapping", MAPPING_SCRIPT, ["--schema", schema])
    )
    return steps


def check_scripts_exist(steps: list[tuple[str, Path, list[str]]]) -> None:
    """Fail before running anything if a script is missing."""
    missing = [str(path) for _, path, _ in steps if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing scripts: " + ", ".join(missing)
            + "\nPass --analyse-db-runner, or use --skip-analyse-db."
        )


def print_plan(steps: list[tuple[str, Path, list[str]]]) -> None:
    """Show what will run before asking to run it."""
    for index, (label, path, _) in enumerate(steps, start=1):
        print(f"  {index}. {label}  ({path})")


def confirm(schema: str) -> bool:
    """Ask before creating or replacing tables."""
    answer = input(f"Run all steps in schema {schema}? (y/n): ")
    return answer.strip().lower() == "y"


def run_step(label: str, path: Path, args: list[str]) -> None:
    """Run one script in its own folder and raise if it fails."""
    print(f"\n{'=' * 70}\n{label}\n{'=' * 70}")
    subprocess.run([sys.executable, str(path), *args], cwd=path.parent, check=True)


def print_next_step(schema: str) -> None:
    """Point at publication, which is a separate deliberate command."""
    print("Review the status summary, then publish with:")
    print(f"  python {MAPPING_SCRIPT.name} --schema {schema} --publish-snowflake")


def run_all(args: Namespace) -> None:
    """Validate, confirm, then run every step in order."""
    schema = validate_identifier(args.schema)
    steps = build_steps(schema, args.analyse_db_runner, args.skip_analyse_db)
    check_scripts_exist(steps)
    print_plan(steps)

    if not args.yes and not confirm(schema):
        print("Cancelled.")
        return

    start = perf_counter()
    for label, path, step_args in steps:
        run_step(label, path, step_args)

    print(f"\nFinished in {perf_counter() - start:.2f} seconds.")
    print_next_step(schema)


def parse_args() -> Namespace:
    parser = ArgumentParser(description="Run the Life data-preparation steps.")
    parser.add_argument("schema", help="Snowflake schema.")
    parser.add_argument("--analyse-db-runner", type=Path, default=ANALYSE_DB_RUNNER,
                        help="Path to run_pipeline.py.")
    parser.add_argument("--skip-analyse-db", action="store_true",
                        help="Run only the mapping step.")
    parser.add_argument("--yes", action="store_true", help="Skip the prompt.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        run_all(args)
    except subprocess.CalledProcessError as error:
        print(f"\nStep failed with exit code {error.returncode}. Stopped.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())