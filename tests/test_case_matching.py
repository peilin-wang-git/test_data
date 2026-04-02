from pathlib import Path

from src.case_discovery import (
    build_label_path,
    parse_case_id_from_image_name,
)


def test_parse_case_id_nnunet_style() -> None:
    cid = parse_case_id_from_image_name(
        filename="Case001_0000.nii.gz",
        image_suffix="_0000",
        image_caseid_pattern=r"^(?P<case_id>.+)_0000$",
        supported_exts=[".nii.gz", ".nii"],
    )
    assert cid == "Case001"


def test_build_label_path(tmp_path: Path) -> None:
    root = tmp_path / "labels"
    root.mkdir()
    f = root / "Case123.nii.gz"
    f.write_text("x", encoding="utf-8")

    out = build_label_path(root, "Case123", "", [".nii.gz", ".nii"])
    assert out == f
