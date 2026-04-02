from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = ["destination image path", "destination label path", "index"]


@dataclass
class CaseRecord:
    """Case record resolved from new_path.csv."""

    case_id: str
    index: int
    image_path: Path
    oncologist_path: Path
    radiologist_1_path: Path
    radiologist_2_path: Path
    nnunet_path: Path


def strip_known_extension(filename: str, supported_exts: list[str]) -> str:
    """Strip any extension listed in supported_exts from filename."""
    for ext in sorted(supported_exts, key=len, reverse=True):
        if filename.endswith(ext):
            return filename[: -len(ext)]
    return Path(filename).stem


def _fallback_nnunet_name_from_image(
    image_path: Path,
    image_suffix: str,
    supported_exts: list[str],
) -> str:
    base = strip_known_extension(image_path.name, supported_exts)
    if base.endswith(image_suffix):
        base = base[: -len(image_suffix)]
    return base + ".nii.gz"


def resolve_nnunet_pred_path(
    destination_label_path: Path,
    destination_image_path: Path,
    nnunet_pred_root: Path,
    use_label_basename_first: bool,
    allow_fallback_match_from_image_name: bool,
    image_suffix: str,
    supported_exts: list[str],
) -> Path:
    """Resolve nnUNet prediction path from configured matching rules."""
    candidates: list[Path] = []

    if use_label_basename_first:
        candidates.append(nnunet_pred_root / destination_label_path.name)

    if allow_fallback_match_from_image_name:
        fallback_name = _fallback_nnunet_name_from_image(destination_image_path, image_suffix, supported_exts)
        candidates.append(nnunet_pred_root / fallback_name)

    if not use_label_basename_first:
        candidates.append(nnunet_pred_root / destination_label_path.name)

    for c in candidates:
        if c.exists():
            return c

    return candidates[0] if candidates else nnunet_pred_root / destination_label_path.name


def load_case_table(config: dict) -> list[CaseRecord]:
    """Load new_path.csv and resolve all source paths per case."""
    case_csv = Path(config["case_table_csv"])
    if not case_csv.exists():
        raise FileNotFoundError(f"Case table not found: {case_csv}")

    df = pd.read_csv(case_csv)
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in case table: {missing_cols}")

    rad1_root = Path(config["radiologist1_labels_root"])
    rad2_root = Path(config["radiologist2_labels_root"])
    nnunet_root = Path(config["nnunet_pred_root"])

    use_label_basename_first = bool(config.get("nnunet_match_use_label_basename_first", True))
    allow_fallback = bool(config.get("allow_fallback_match_from_image_name", True))
    image_suffix = str(config.get("image_suffix", "_0000"))
    supported_exts = list(config.get("supported_image_exts", [".nii.gz", ".nii"]))

    cases: list[CaseRecord] = []
    for _, row in df.iterrows():
        idx = int(row["index"])
        image_path = Path(str(row["destination image path"]))
        oncologist_path = Path(str(row["destination label path"]))

        case_id = strip_known_extension(oncologist_path.name, supported_exts)
        case_folder = f"Case_{idx:03d}"

        radiologist_1_path = rad1_root / case_folder / "seg.nii.gz"
        radiologist_2_path = rad2_root / case_folder / "seg.nii.gz"

        nnunet_path = resolve_nnunet_pred_path(
            destination_label_path=oncologist_path,
            destination_image_path=image_path,
            nnunet_pred_root=nnunet_root,
            use_label_basename_first=use_label_basename_first,
            allow_fallback_match_from_image_name=allow_fallback,
            image_suffix=image_suffix,
            supported_exts=supported_exts,
        )

        cases.append(
            CaseRecord(
                case_id=case_id,
                index=idx,
                image_path=image_path,
                oncologist_path=oncologist_path,
                radiologist_1_path=radiologist_1_path,
                radiologist_2_path=radiologist_2_path,
                nnunet_path=nnunet_path,
            )
        )

    return cases
