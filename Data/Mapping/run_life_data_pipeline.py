#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Run the Life data-preparation steps in order and stop on the first failure.

    1. run_pipeline.py       Analyse-DB SQL tables
    2. 00_map_mid_to_adb.py  MasterIndex document and page tables

Each step runs as its own process, so running them by hand gives the same
result as running them here.

Publishing to PROC_LIFE_MID_ADB is deliberately left out. Read the status
summary from step 2 first, then publish:

    python 00_map_mid_to_adb.py --schema <SCHEMA> --publish-snowflake

    python run_life_data_pipeline.py D131_D2D
"""

from __future__ import annotations

import re
import subprocess
import sys
from argparse import ArgumentParser
from pathlib import Path
from time import perf_counter


SCRIPT_DIR = Path(__file__).resolve().parent
PUBLISH_SCRIPT = "00_map_mid_to_adb.py"


def validate_identifier(value: str) -> str:
    """Allow only unquoted Snowflake identifiers."""
    if not re.fullmatch(r"[A-Za-z0-9_]+", value):
        raise ValueError(f"Unsafe Snowflake identifier: {value!r}")
    return value.upper()


def build_steps(schema: str) -> list[tuple[str, list[str]]]:
    """The commands to run, in dependency order."""
    return [
        ("Analyse-DB pipeline", ["run_pipeline.py", schema, "--yes"]),
        ("MasterIndex mapping", [PUBLISH_SCRIPT, "--schema", schema]),
    ]


def check_scripts_exist(steps: list[tuple[str, list[str]]]) -> None:
    """Fail before running anything if a script is missing."""
    missing = [args[0] for _, args in steps if not (SCRIPT_DIR / args[0]).is_file()]
    if missing:
        raise FileNotFoundError(
            f"Missing scripts in {SCRIPT_DIR}: {', '.join(missing)}"
        )


def print_plan(steps: list[tuple[str, list[str]]]) -> None:
    """Show what will run before asking to run it."""
    for index, (label, args) in enumerate(steps, start=1):
        print(f"  {index}. {label}  ({args[0]})")


def confirm(schema: str) -> bool:
    """Ask before creating or replacing tables."""
    answer = input(f"Run all steps in schema {schema}? (y/n): ")
    return answer.strip().lower() == "y"


def run_step(label: str, args: list[str]) -> None:
    """Run one script and raise if it fails."""
    print(f"\n{'=' * 70}\n{label}\n{'=' * 70}")
    subprocess.run([sys.executable, *args], cwd=SCRIPT_DIR, check=True)


def print_next_step(schema: str) -> None:
    """Point at publication, which is a separate deliberate command."""
    print("Review the status summary, then publish with:")
    print(f"  python {PUBLISH_SCRIPT} --schema {schema} --publish-snowflake")


def run_all(schema: str, assume_yes: bool) -> None:
    """Validate, confirm, then run every step in order."""
    schema = validate_identifier(schema)
    steps = build_steps(schema)
    check_scripts_exist(steps)
    print_plan(steps)

    if not assume_yes and not confirm(schema):
        print("Cancelled.")
        return

    start = perf_counter()
    for label, args in steps:
        run_step(label, args)

    print(f"\nFinished in {perf_counter() - start:.2f} seconds.")
    print_next_step(schema)


def parse_args() -> ArgumentParser:
    parser = ArgumentParser(description="Run the Life data-preparation steps.")
    parser.add_argument("schema", help="Snowflake schema.")
    parser.add_argument("--yes", action="store_true", help="Skip the prompt.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        run_all(args.schema, args.yes)
    except subprocess.CalledProcessError as error:
        print(f"\nStep failed with exit code {error.returncode}. Stopped.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())