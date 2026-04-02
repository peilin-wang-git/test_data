from __future__ import annotations

from pathlib import Path


def ensure_dir(path: Path) -> None:
    """Create directory if not exists."""
    path.mkdir(parents=True, exist_ok=True)
