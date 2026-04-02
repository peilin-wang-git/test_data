from __future__ import annotations

import argparse

from src.evaluator import MRLINACEvaluator
from src.io_utils import load_yaml_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MRLINAC multi-rater segmentation evaluator")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml_config(args.config)
    evaluator = MRLINACEvaluator(config)
    evaluator.run()


if __name__ == "__main__":
    main()
