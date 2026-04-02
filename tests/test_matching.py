from pathlib import Path

from src.io_utils import discover_case_masks, resolve_mask_path


def test_resolve_mask_path_prefers_existing_extension(tmp_path: Path) -> None:
    case_dir = tmp_path / "case_001"
    case_dir.mkdir()
    expected = case_dir / "oncologist_prostate.nii.gz"
    expected.write_text("x", encoding="utf-8")

    found = resolve_mask_path(
        case_dir=case_dir,
        source="oncologist",
        organ="prostate",
        file_template="{source}_{organ}{ext}",
        allowed_extensions=[".nii.gz", ".nii"],
    )
    assert found == expected


def test_discover_case_masks_returns_only_existing(tmp_path: Path) -> None:
    case_dir = tmp_path / "case_002"
    case_dir.mkdir()
    (case_dir / "oncologist_prostate.nii.gz").write_text("x", encoding="utf-8")
    (case_dir / "nnunet_rectum.nii").write_text("x", encoding="utf-8")

    found = discover_case_masks(
        case_dir=case_dir,
        case_id="case_002",
        sources=["oncologist", "nnunet"],
        organs=["prostate", "rectum"],
        file_template="{source}_{organ}{ext}",
        allowed_extensions=[".nii.gz", ".nii"],
    )

    assert ("oncologist", "prostate") in found
    assert ("nnunet", "rectum") in found
    assert ("oncologist", "rectum") not in found
