from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import SimpleITK as sitk

from src.case_table import CaseRecord, load_case_table
from src.io_utils import load_yaml_config, metadata_compatible, read_image, save_csv, save_image
from src.label_parser import (
    OrganSpec,
    compose_multiclass_from_binary,
    extract_organ_mask,
    organ_present,
    organ_specs_from_config,
)
from src.logger import setup_logger
from src.metrics import dice_coefficient
from src.staple import generate_staple
from src.utils import ensure_dir
from src.visualization import save_overlay_preview, save_overlay_preview_three_plane


@dataclass
class MetricRow:
    case_id: str
    index: int
    organ: str
    source_a: str
    source_b: str
    dice: float | None
    status: str


class MRLINACEvaluator:
    """Evaluate pairwise and STAPLE consistency for MRLINAC multi-rater labels."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.output_root = Path(config["output_root"])
        self.pairwise_csv = self.output_root / "pairwise_metrics.csv"
        self.summary_csv = self.output_root / "summary_metrics.csv"
        self.preview_root = self.output_root / "previews"
        self.staple_root = self.output_root / "staple_masks"

        ensure_dir(self.output_root)
        ensure_dir(self.preview_root)
        ensure_dir(self.staple_root)

        self.logger = setup_logger(Path(config.get("log_file", "./logs/run.log")))
        self.organ_specs: list[OrganSpec] = organ_specs_from_config(config)
        self.organ_orders = {s.name: s.order for s in self.organ_specs}
        self.organ_rgb = {s.name: s.rgb for s in self.organ_specs}

        label_mapping = config.get("label_mapping", {})
        expected = {"background": 0, "prostate": 1, "rectum": 2, "bladder": 3}
        for k, v in expected.items():
            if int(label_mapping.get(k, v)) != v:
                raise ValueError(f"label_mapping[{k}] must be {v}")

        self.sources = ["oncologist", "radiologist_1", "radiologist_2", "nnunet"]
        self.human_pairs = [
            ("oncologist", "radiologist_1"),
            ("oncologist", "radiologist_2"),
            ("radiologist_1", "radiologist_2"),
        ]
        self.nnunet_pairs = [
            ("nnunet", "oncologist"),
            ("nnunet", "radiologist_1"),
            ("nnunet", "radiologist_2"),
        ]

        self.staple_sources = list(config.get("staple_sources", ["oncologist", "radiologist_1", "radiologist_2"]))
        self.staple_threshold = float(config.get("staple_threshold", 0.5))
        self.save_staple_masks = bool(config.get("save_staple_masks", True))
        self.save_staple_multiclass = bool(config.get("save_staple_multiclass", True))
        self.save_overlay = bool(config.get("save_overlay_previews", True))
        self.preview_slice_mode = str(config.get("preview_slice_mode", "axial"))
        self.overlay_alpha = float(config.get("overlay_alpha", 0.4))
        self.continue_on_case_error = bool(config.get("continue_on_case_error", True))

    def run(self) -> None:
        cases = load_case_table(self.config)
        self.logger.info("Loaded %d rows from case table", len(cases))

        all_rows: list[MetricRow] = []
        for i, case in enumerate(cases, start=1):
            self.logger.info("[%d/%d] Processing case_id=%s index=%d", i, len(cases), case.case_id, case.index)
            try:
                all_rows.extend(self._process_case(case))
            except Exception as exc:
                self.logger.exception("Case failed case_id=%s index=%d err=%s", case.case_id, case.index, exc)
                if not self.continue_on_case_error:
                    raise

        pairwise_df = pd.DataFrame([r.__dict__ for r in all_rows])
        if pairwise_df.empty:
            pairwise_df = pd.DataFrame(
                columns=["case_id", "index", "organ", "source_a", "source_b", "dice", "status"]
            )
        save_csv(pairwise_df, self.pairwise_csv)

        summary_df = self._build_summary(pairwise_df)
        save_csv(summary_df, self.summary_csv)

    def _process_case(self, case: CaseRecord) -> list[MetricRow]:
        rows: list[MetricRow] = []
        source_paths = {
            "oncologist": case.oncologist_path,
            "radiologist_1": case.radiologist_1_path,
            "radiologist_2": case.radiologist_2_path,
            "nnunet": case.nnunet_path,
        }

        if not case.image_path.exists():
            self.logger.warning("Missing image: case_id=%s index=%d path=%s", case.case_id, case.index, case.image_path)
            # continue without overlay; still compute metrics if seg exists
            image_arr = None
            image_sitk = None
        else:
            image_sitk, image_arr = read_image(case.image_path)

        loaded: dict[str, tuple[sitk.Image, Any]] = {}
        for src, p in source_paths.items():
            if not p.exists():
                self.logger.warning("Missing file: case_id=%s index=%d source=%s path=%s", case.case_id, case.index, src, p)
                continue
            loaded[src] = read_image(p)
            if image_sitk is not None and not metadata_compatible(image_sitk, loaded[src][0]):
                self.logger.warning("Metadata mismatch image-vs-%s for case_id=%s index=%d", src, case.case_id, case.index)

        staple_binary_by_organ: dict[str, sitk.Image] = {}

        for spec in self.organ_specs:
            organ = spec.name
            pair_candidates = self.human_pairs + self.nnunet_pairs

            # normal pairwise
            for a, b in pair_candidates:
                row = self._pairwise_row(case, organ, spec.order, a, b, loaded)
                rows.append(row)

            # STAPLE for this organ
            staple_img, staple_status = self._staple_for_organ(case, organ, spec.order, loaded)
            if staple_img is not None:
                staple_binary_by_organ[organ] = staple_img

            for src in self.sources:
                if staple_img is None:
                    rows.append(MetricRow(case.case_id, case.index, organ, src, "STAPLE", None, staple_status))
                    continue

                if src not in loaded:
                    rows.append(MetricRow(case.case_id, case.index, organ, src, "STAPLE", None, "skipped_missing_file"))
                    continue

                if not organ_present(loaded[src][1], spec.order):
                    rows.append(MetricRow(case.case_id, case.index, organ, src, "STAPLE", None, "skipped_missing_organ"))
                    continue

                try:
                    src_mask = extract_organ_mask(loaded[src][1], spec.order)
                    staple_mask = sitk.GetArrayFromImage(staple_img)
                    d = dice_coefficient(src_mask, staple_mask)
                    rows.append(MetricRow(case.case_id, case.index, organ, src, "STAPLE", d, "ok"))
                except Exception as exc:
                    self.logger.exception(
                        "Error STAPLE compare case_id=%s index=%d organ=%s src=%s err=%s",
                        case.case_id,
                        case.index,
                        organ,
                        src,
                        exc,
                    )
                    rows.append(MetricRow(case.case_id, case.index, organ, src, "STAPLE", None, "error"))

        self._save_optional_staple_masks(case, staple_binary_by_organ)
        self._save_optional_previews(case, image_arr, loaded, staple_binary_by_organ)
        return rows

    def _pairwise_row(
        self,
        case: CaseRecord,
        organ: str,
        order: int,
        source_a: str,
        source_b: str,
        loaded: dict[str, tuple[sitk.Image, Any]],
    ) -> MetricRow:
        if source_a not in loaded or source_b not in loaded:
            return MetricRow(case.case_id, case.index, organ, source_a, source_b, None, "skipped_missing_file")

        if not organ_present(loaded[source_a][1], order) or not organ_present(loaded[source_b][1], order):
            self.logger.warning(
                "Missing organ label case_id=%s index=%d organ=%s in pair=%s_vs_%s",
                case.case_id,
                case.index,
                organ,
                source_a,
                source_b,
            )
            return MetricRow(case.case_id, case.index, organ, source_a, source_b, None, "skipped_missing_organ")

        try:
            m_a = extract_organ_mask(loaded[source_a][1], order)
            m_b = extract_organ_mask(loaded[source_b][1], order)
            d = dice_coefficient(m_a, m_b)
            return MetricRow(case.case_id, case.index, organ, source_a, source_b, d, "ok")
        except Exception as exc:
            self.logger.exception(
                "Error pairwise case_id=%s index=%d organ=%s pair=%s_vs_%s err=%s",
                case.case_id,
                case.index,
                organ,
                source_a,
                source_b,
                exc,
            )
            return MetricRow(case.case_id, case.index, organ, source_a, source_b, None, "error")

    def _staple_for_organ(
        self,
        case: CaseRecord,
        organ: str,
        order: int,
        loaded: dict[str, tuple[sitk.Image, Any]],
    ) -> tuple[sitk.Image | None, str]:
        staple_inputs: list[sitk.Image] = []
        for src in self.staple_sources:
            if src not in loaded:
                self.logger.warning(
                    "Skip STAPLE organ due to missing file case_id=%s index=%d organ=%s src=%s",
                    case.case_id,
                    case.index,
                    organ,
                    src,
                )
                return None, "skipped_missing_file"
            if not organ_present(loaded[src][1], order):
                self.logger.warning(
                    "Skip STAPLE organ due to missing organ case_id=%s index=%d organ=%s src=%s",
                    case.case_id,
                    case.index,
                    organ,
                    src,
                )
                return None, "skipped_missing_organ"

            src_img = loaded[src][0]
            src_mask = extract_organ_mask(loaded[src][1], order)
            bin_img = sitk.GetImageFromArray(src_mask)
            bin_img.CopyInformation(src_img)
            staple_inputs.append(bin_img)

        try:
            staple = generate_staple(staple_inputs, threshold=self.staple_threshold)
            return staple, "ok"
        except Exception as exc:
            self.logger.exception(
                "STAPLE error case_id=%s index=%d organ=%s err=%s",
                case.case_id,
                case.index,
                organ,
                exc,
            )
            return None, "error"

    def _save_optional_staple_masks(self, case: CaseRecord, staple_binary_by_organ: dict[str, sitk.Image]) -> None:
        if not self.save_staple_masks or not staple_binary_by_organ:
            return

        out_case_dir = self.staple_root / case.case_id
        ensure_dir(out_case_dir)

        for organ, img in staple_binary_by_organ.items():
            save_image(img, out_case_dir / f"staple_{organ}.nii.gz")

        if self.save_staple_multiclass:
            multi = compose_multiclass_from_binary(staple_binary_by_organ, self.organ_specs)
            save_image(multi, out_case_dir / "staple_multiclass.nii.gz")

    def _save_optional_previews(
        self,
        case: CaseRecord,
        image_arr: Any,
        loaded: dict[str, tuple[sitk.Image, Any]],
        staple_binary_by_organ: dict[str, sitk.Image],
    ) -> None:
        if not self.save_overlay or image_arr is None:
            return

        out_case_dir = self.preview_root / case.case_id
        ensure_dir(out_case_dir)

        for src, (_, seg_arr) in loaded.items():
            self._save_preview(image_arr, seg_arr, out_case_dir / f"{src}_overlay.png")

        if staple_binary_by_organ:
            multi = compose_multiclass_from_binary(staple_binary_by_organ, self.organ_specs)
            self._save_preview(image_arr, sitk.GetArrayFromImage(multi), out_case_dir / "staple_overlay.png")

    def _save_preview(self, image_arr: Any, seg_arr: Any, out_path: Path) -> None:
        if self.preview_slice_mode == "three_plane":
            save_overlay_preview_three_plane(
                image_arr,
                seg_arr,
                organ_rgb=self.organ_rgb,
                organ_orders=self.organ_orders,
                out_path=out_path,
                alpha=self.overlay_alpha,
            )
        else:
            save_overlay_preview(
                image_arr,
                seg_arr,
                organ_rgb=self.organ_rgb,
                organ_orders=self.organ_orders,
                out_path=out_path,
                slice_mode="axial",
                alpha=self.overlay_alpha,
            )

    @staticmethod
    def _build_summary(pairwise_df: pd.DataFrame) -> pd.DataFrame:
        ok = pairwise_df[pairwise_df["status"] == "ok"].copy()
        if ok.empty:
            return pd.DataFrame(columns=["organ", "comparison_pair", "n", "mean", "std", "median", "min", "max"])

        ok["comparison_pair"] = ok["source_a"] + "_vs_" + ok["source_b"]
        return (
            ok.groupby(["organ", "comparison_pair"], dropna=False)["dice"]
            .agg(["count", "mean", "std", "median", "min", "max"])
            .reset_index()
            .rename(columns={"count": "n"})
        )


def run_from_config(config_path: str | Path) -> None:
    cfg = load_yaml_config(config_path)
    MRLINACEvaluator(cfg).run()
