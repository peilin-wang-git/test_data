from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import SimpleITK as sitk
import yaml


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    """Load YAML config as dict."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {p}")
    with p.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError("YAML root must be a mapping")
    return cfg


def read_image(path: Path) -> tuple[sitk.Image, np.ndarray]:
    """Read image with SimpleITK and convert to numpy array."""
    img = sitk.ReadImage(str(path))
    arr = sitk.GetArrayFromImage(img)
    return img, arr


def save_image(image: sitk.Image, path: Path) -> None:
    """Write image to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(image, str(path))


def save_csv(df: pd.DataFrame, path: Path) -> None:
    """Save DataFrame to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def metadata_compatible(a: sitk.Image, b: sitk.Image) -> bool:
    """Check direction/origin/spacing equality."""
    return (
        tuple(a.GetSpacing()) == tuple(b.GetSpacing())
        and tuple(a.GetOrigin()) == tuple(b.GetOrigin())
        and tuple(a.GetDirection()) == tuple(b.GetDirection())
    )
