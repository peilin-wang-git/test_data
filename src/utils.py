from __future__ import annotations

import logging
from pathlib import Path
from typing import Any


def ensure_dir(path: Path) -> None:
    """Create directory (and parents) if it does not exist."""
    path.mkdir(parents=True, exist_ok=True)


def setup_logger(log_file: Path, name: str = "mrlinac_eval") -> logging.Logger:
    """Configure and return a logger writing to console and file."""
    ensure_dir(log_file.parent)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger


def flatten_dict(d: dict[str, Any], parent_key: str = "", sep: str = ".") -> dict[str, Any]:
    """Flatten nested dictionaries into dot-separated keys."""
    items: dict[str, Any] = {}
    for key, value in d.items():
        new_key = f"{parent_key}{sep}{key}" if parent_key else key
        if isinstance(value, dict):
            items.update(flatten_dict(value, new_key, sep=sep))
        else:
            items[new_key] = value
    return items
