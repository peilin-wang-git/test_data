from __future__ import annotations

import argparse

from src.evaluator import SegmentationEvaluator
from src.io_utils import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate MRLINAC segmentation consistency.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    evaluator = SegmentationEvaluator(cfg)
    evaluator.run()


if __name__ == "__main__":
    main()
