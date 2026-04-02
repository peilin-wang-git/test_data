from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import SimpleITK as sitk
import yaml


@dataclass
class MaskRecord:
    """A segmentation file descriptor for one source+organ in one case."""

    case_id: str
    source: str
    organ: str
    path: Path


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Read YAML config file."""
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if not isinstance(cfg, dict):
        raise ValueError("Invalid config: root must be a mapping.")
    return cfg


def list_case_dirs(data_root: Path, case_glob: str, recursive: bool = False) -> list[Path]:
    """List case directories under data root according to glob rule."""
    if not data_root.exists():
        raise FileNotFoundError(f"Data root does not exist: {data_root}")

    iterator = data_root.rglob(case_glob) if recursive else data_root.glob(case_glob)
    return sorted([p for p in iterator if p.is_dir()])


def resolve_mask_path(
    case_dir: Path,
    source: str,
    organ: str,
    file_template: str,
    allowed_extensions: list[str],
) -> Path | None:
    """Resolve mask path based on naming template and allowed extensions."""
    for ext in allowed_extensions:
        candidate = case_dir / file_template.format(source=source, organ=organ, ext=ext)
        if candidate.exists():
            return candidate
    return None


def discover_case_masks(
    case_dir: Path,
    case_id: str,
    sources: list[str],
    organs: list[str],
    file_template: str,
    allowed_extensions: list[str],
) -> dict[tuple[str, str], Path]:
    """Discover mask files for all source+organ pairs in one case directory."""
    result: dict[tuple[str, str], Path] = {}
    for source in sources:
        for organ in organs:
            path = resolve_mask_path(case_dir, source, organ, file_template, allowed_extensions)
            if path is not None:
                result[(source, organ)] = path
    return result


def read_mask(path: Path) -> tuple[sitk.Image, np.ndarray]:
    """Read segmentation file into SimpleITK image and numpy array."""
    img = sitk.ReadImage(str(path))
    arr = sitk.GetArrayFromImage(img)
    return img, arr


def metadata_matches(a: sitk.Image, b: sitk.Image) -> bool:
    """Check spacing/origin/direction consistency."""
    return (
        tuple(a.GetSpacing()) == tuple(b.GetSpacing())
        and tuple(a.GetOrigin()) == tuple(b.GetOrigin())
        and tuple(a.GetDirection()) == tuple(b.GetDirection())
    )
