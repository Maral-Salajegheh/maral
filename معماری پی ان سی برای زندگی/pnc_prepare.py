"""Prepare corrected PnC model population and leakage-safe split report."""

import argparse
import json

from .config import PipelineConfig
from .pnc_data import prepare_model_population, split_assignments, split_report
from .pnc_metrics import write_json_new


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-id", default=PipelineConfig.batch_id)
    parser.add_argument("--seed", type=int, default=13)        # reconstructed
    args = parser.parse_args()                                  # reconstructed
    config = PipelineConfig(batch_id=args.batch_id)
    rows, population = prepare_model_population(config)
    assignments = split_assignments(config, rows, args.seed)    # reconstructed after "assignments="
    report = split_report(rows, assignments)                    # reconstructed
    output = config.classification_output / "pnc_split_report_v2.json"
    write_json_new(output, report)
    print(json.dumps({"population": population, "splits": report,
                      "model_pages": str(config.model_pages_csv)}, indent=2))


if __name__ == "__main__":
    main()