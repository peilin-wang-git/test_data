from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import SimpleITK as sitk

from src.case_discovery import CaseFiles, discover_cases
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
    organ: str
    source_a: str
    source_b: str
    dice: float | None
    status: str


class MRLINACEvaluator:
    """End-to-end evaluator for multi-rater MRLINAC segmentation consistency."""

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
        """Run case discovery, evaluation, and report generation."""
        cases = discover_cases(self.config)
        self.logger.info("Discovered %d cases from images root", len(cases))

        all_rows: list[MetricRow] = []
        for idx, case in enumerate(cases, start=1):
            self.logger.info("[%d/%d] Processing %s", idx, len(cases), case.case_id)
            try:
                all_rows.extend(self._process_case(case))
            except Exception as exc:
                self.logger.exception("Case failed: %s error=%s", case.case_id, exc)
                if not self.continue_on_case_error:
                    raise

        pairwise_df = pd.DataFrame([r.__dict__ for r in all_rows])
        if pairwise_df.empty:
            pairwise_df = pd.DataFrame(
                columns=["case_id", "organ", "source_a", "source_b", "dice", "status"]
            )
        save_csv(pairwise_df, self.pairwise_csv)

        summary = self._build_summary(pairwise_df)
        save_csv(summary, self.summary_csv)
        self.logger.info("Saved pairwise metrics: %s", self.pairwise_csv)
        self.logger.info("Saved summary metrics: %s", self.summary_csv)

    def _process_case(self, case: CaseFiles) -> list[MetricRow]:
        rows: list[MetricRow] = []

        image_sitk, image_arr = read_image(case.image_path)

        loaded: dict[str, tuple[sitk.Image, Any]] = {}
        missing_files = [s for s, p in case.source_files.items() if p is None]
        for source in missing_files:
            self.logger.warning("Missing file: case=%s source=%s", case.case_id, source)

        for source, p in case.source_files.items():
            if p is None:
                continue
            loaded[source] = read_image(p)
            if not metadata_compatible(image_sitk, loaded[source][0]):
                self.logger.warning(
                    "Image/seg metadata mismatch: case=%s source=%s", case.case_id, source
                )

        staple_binary_by_organ: dict[str, sitk.Image] = {}

        for spec in self.organ_specs:
            organ = spec.name
            # Strict rule: any missing source file => this case-organ skip all comparisons
            if missing_files:
                rows.extend(self._emit_all_pairs_skipped(case.case_id, organ, "skipped_missing_file"))
                self.logger.warning(
                    "Skip case-organ due to missing file(s): case=%s organ=%s missing=%s",
                    case.case_id,
                    organ,
                    ",".join(missing_files),
                )
                continue

            # Strict rule: any source lacks this organ => skip all comparisons for this case-organ
            absent_sources = []
            for source in self.sources:
                seg_arr = loaded[source][1]
                if not organ_present(seg_arr, spec.order):
                    absent_sources.append(source)
            if absent_sources:
                rows.extend(self._emit_all_pairs_skipped(case.case_id, organ, "skipped_missing_organ"))
                self.logger.warning(
                    "Skip case-organ due to missing organ label: case=%s organ=%s absent_in=%s",
                    case.case_id,
                    organ,
                    ",".join(absent_sources),
                )
                continue

            # Pairwise comparisons (now all source files and organ labels exist)
            for a, b in self.human_pairs + self.nnunet_pairs:
                try:
                    dice = dice_coefficient(
                        extract_organ_mask(loaded[a][1], spec.order),
                        extract_organ_mask(loaded[b][1], spec.order),
                    )
                    rows.append(MetricRow(case.case_id, organ, a, b, dice, "ok"))
                except Exception as exc:
                    self.logger.exception(
                        "Error in pairwise: case=%s organ=%s pair=%s_vs_%s error=%s",
                        case.case_id,
                        organ,
                        a,
                        b,
                        exc,
                    )
                    rows.append(MetricRow(case.case_id, organ, a, b, None, "error"))

            # STAPLE per organ using configured participants
            try:
                staple_inputs = []
                for src in self.staple_sources:
                    if src not in loaded:
                        raise ValueError(f"STAPLE source missing file: {src}")
                    src_img = loaded[src][0]
                    src_mask = extract_organ_mask(loaded[src][1], spec.order)
                    bin_img = sitk.GetImageFromArray(src_mask)
                    bin_img.CopyInformation(src_img)
                    staple_inputs.append(bin_img)

                staple_bin = generate_staple(staple_inputs, threshold=self.staple_threshold)
                staple_binary_by_organ[organ] = staple_bin

                for src in self.sources:
                    src_mask = extract_organ_mask(loaded[src][1], spec.order)
                    staple_mask = sitk.GetArrayFromImage(staple_bin)
                    dice = dice_coefficient(src_mask, staple_mask)
                    rows.append(MetricRow(case.case_id, organ, src, "STAPLE", dice, "ok"))
            except Exception as exc:
                self.logger.warning(
                    "Skip STAPLE for case-organ: case=%s organ=%s error=%s",
                    case.case_id,
                    organ,
                    exc,
                )
                for src in self.sources:
                    rows.append(
                        MetricRow(case.case_id, organ, src, "STAPLE", None, "error")
                    )

        self._save_optional_staple_masks(case.case_id, staple_binary_by_organ)
        self._save_optional_previews(case.case_id, image_arr, loaded, staple_binary_by_organ)
        return rows

    def _emit_all_pairs_skipped(self, case_id: str, organ: str, status: str) -> list[MetricRow]:
        all_pairs = self.human_pairs + self.nnunet_pairs + [(s, "STAPLE") for s in self.sources]
        return [MetricRow(case_id, organ, a, b, None, status) for a, b in all_pairs]

    def _save_optional_staple_masks(
        self,
        case_id: str,
        staple_binary_by_organ: dict[str, sitk.Image],
    ) -> None:
        if not self.save_staple_masks or not staple_binary_by_organ:
            return

        case_dir = self.staple_root / case_id
        ensure_dir(case_dir)
        for organ, img in staple_binary_by_organ.items():
            save_image(img, case_dir / f"staple_{organ}.nii.gz")

        if self.save_staple_multiclass:
            multi = compose_multiclass_from_binary(staple_binary_by_organ, self.organ_specs)
            save_image(multi, case_dir / "staple_multiclass.nii.gz")

    def _save_optional_previews(
        self,
        case_id: str,
        image_arr: Any,
        loaded: dict[str, tuple[sitk.Image, Any]],
        staple_binary_by_organ: dict[str, sitk.Image],
    ) -> None:
        if not self.save_overlay:
            return

        case_preview_dir = self.preview_root / case_id
        ensure_dir(case_preview_dir)

        # source overlays
        for source, (_, seg_arr) in loaded.items():
            out_path = case_preview_dir / f"{source}_overlay.png"
            self._save_preview(image_arr, seg_arr, out_path)

        # STAPLE overlay only when all three organs produced
        if len(staple_binary_by_organ) == len(self.organ_specs):
            staple_multi = compose_multiclass_from_binary(staple_binary_by_organ, self.organ_specs)
            staple_arr = sitk.GetArrayFromImage(staple_multi)
            out_path = case_preview_dir / "staple_overlay.png"
            self._save_preview(image_arr, staple_arr, out_path)

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
            return pd.DataFrame(
                columns=["organ", "comparison_pair", "n", "mean", "std", "median", "min", "max"]
            )

        ok["comparison_pair"] = ok["source_a"] + "_vs_" + ok["source_b"]
        summary = (
            ok.groupby(["organ", "comparison_pair"], dropna=False)["dice"]
            .agg(["count", "mean", "std", "median", "min", "max"])
            .reset_index()
            .rename(columns={"count": "n"})
        )
        return summary


def run_from_config(config_path: str | Path) -> None:
    """Convenience helper for script usage."""
    cfg = load_yaml_config(config_path)
    MRLINACEvaluator(cfg).run()
