from pathlib import Path

from src.case_table import resolve_nnunet_pred_path, strip_known_extension


def test_strip_known_extension() -> None:
    assert strip_known_extension("Case001.nii.gz", [".nii.gz", ".nii"]) == "Case001"


def test_resolve_nnunet_pred_path_prefers_label_basename(tmp_path: Path) -> None:
    nn_root = tmp_path / "nnunet"
    nn_root.mkdir()

    label_file = tmp_path / "Case001.nii.gz"
    image_file = tmp_path / "Case001_0000.nii.gz"

    expected = nn_root / "Case001.nii.gz"
    expected.write_text("x", encoding="utf-8")

    out = resolve_nnunet_pred_path(
        destination_label_path=label_file,
        destination_image_path=image_file,
        nnunet_pred_root=nn_root,
        use_label_basename_first=True,
        allow_fallback_match_from_image_name=True,
        image_suffix="_0000",
        supported_exts=[".nii.gz", ".nii"],
    )
    assert out == expected
