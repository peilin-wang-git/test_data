from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


@dataclass
class CaseFiles:
    """Matched file set for one case."""

    case_id: str
    image_path: Path
    source_files: dict[str, Path | None]


def _strip_extension(name: str, supported_exts: list[str]) -> str:
    for ext in sorted(supported_exts, key=len, reverse=True):
        if name.endswith(ext):
            return name[: -len(ext)]
    return name


def parse_case_id_from_image_name(
    filename: str,
    image_suffix: str,
    image_caseid_pattern: str,
    supported_exts: list[str],
) -> str:
    """Extract case id from an nnUNet-style image file name."""
    base = _strip_extension(filename, supported_exts)
    match = re.match(image_caseid_pattern, base)
    if match and "case_id" in match.groupdict():
        return match.group("case_id")
    if base.endswith(image_suffix):
        return base[: -len(image_suffix)]
    raise ValueError(f"Cannot parse case id from image filename: {filename}")


def discover_images(images_root: Path, supported_exts: list[str]) -> list[Path]:
    """Recursively discover images under images_root."""
    if not images_root.exists():
        raise FileNotFoundError(f"images_root does not exist: {images_root}")
    files: list[Path] = []
    for p in images_root.rglob("*"):
        if p.is_file() and any(str(p).endswith(ext) for ext in supported_exts):
            files.append(p)
    return sorted(files)


def build_label_path(label_root: Path, case_id: str, label_suffix: str, supported_exts: list[str]) -> Path | None:
    """Build expected label path and return first existing match."""
    for ext in supported_exts:
        candidate = label_root / f"{case_id}{label_suffix}{ext}"
        if candidate.exists():
            return candidate
    return None


def discover_cases(config: dict) -> list[CaseFiles]:
    """Create per-case file mapping from image and source label roots."""
    images_root = Path(config["images_root"])
    supported_exts = list(config["supported_image_exts"])
    image_suffix = str(config.get("image_suffix", "_0000"))
    case_pattern = str(config.get("image_caseid_pattern", r"^(?P<case_id>.+)_0000$"))
    label_suffix = str(config.get("label_suffix", ""))

    source_roots = {
        "oncologist": Path(config["oncologist_labels_root"]),
        "radiologist_1": Path(config["radiologist1_labels_root"]),
        "radiologist_2": Path(config["radiologist2_labels_root"]),
        "nnunet": Path(config["nnunet_pred_root"]),
    }

    image_files = discover_images(images_root, supported_exts)
    cases: list[CaseFiles] = []
    for img_path in image_files:
        case_id = parse_case_id_from_image_name(
            img_path.name,
            image_suffix=image_suffix,
            image_caseid_pattern=case_pattern,
            supported_exts=supported_exts,
        )
        source_files = {
            source: build_label_path(root, case_id, label_suffix, supported_exts)
            for source, root in source_roots.items()
        }
        cases.append(CaseFiles(case_id=case_id, image_path=img_path, source_files=source_files))
    return cases
