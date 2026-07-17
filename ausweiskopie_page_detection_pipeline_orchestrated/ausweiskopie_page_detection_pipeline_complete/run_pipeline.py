from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent

PRE_REVIEW_STEPS = [
    "01_scan_downloaded_documents.py",
    "02_render_pdf_pages.py",
    "03_screen_ausweis_pages.py",
    "04_prepare_detection_review.py",
]

POST_REVIEW_STEPS = [
    "05_build_ausweis_classification_ground_truth.py",
    "06_evaluate_detection_audit.py",
]


def run_script(script_name: str) -> None:
    """Run one script and stop immediately if it fails."""
    script_path = PROJECT_ROOT / script_name

    if not script_path.exists():
        raise FileNotFoundError(
            f"Pipeline script not found: {script_path}"
        )

    print()
    print("=" * 72)
    print(f"Running: {script_name}")
    print("=" * 72)

    subprocess.run(
        [sys.executable, str(script_path)],
        cwd=PROJECT_ROOT,
        check=True,
    )


def run_steps(script_names: list[str]) -> None:
    for script_name in script_names:
        run_script(script_name)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the Ausweiskopie pipeline in order."
    )
    parser.add_argument(
        "phase",
        choices=["test", "pre-review", "post-review", "cleanup"],
        help=(
            "test: test SecureGPT; pre-review: scan/render/screen/review file; "
            "post-review: build Ground Truth and evaluate; cleanup: delete "
            "eligible rendered images."
        ),
    )
    args = parser.parse_args()

    if args.phase == "test":
        run_script("00_test_securegpt_vision.py")

    elif args.phase == "pre-review":
        run_steps(PRE_REVIEW_STEPS)
        print()
        print("Pre-review phase completed.")
        print("Review: outputs/ausweis_detection_review.xlsx")
        print("Then run: python run_pipeline.py post-review")

    elif args.phase == "post-review":
        run_steps(POST_REVIEW_STEPS)
        print()
        print("Ground Truth and evaluation completed.")

    elif args.phase == "cleanup":
        run_script("07_cleanup_rendered_pages.py")


if __name__ == "__main__":
    main()
