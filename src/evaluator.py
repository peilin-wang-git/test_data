from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from src.io_utils import (
    discover_case_masks,
    list_case_dirs,
    metadata_matches,
    read_mask,
)
from src.metrics import dice_coefficient
from src.staple import generate_staple_image, save_staple_image, sitk_to_numpy
from src.utils import ensure_dir, setup_logger


@dataclass
class EvalRow:
    case_id: str
    organ: str
    source_a: str
    source_b: str
    dice: float


class SegmentationEvaluator:
    """Main orchestration class for MRLINAC segmentation consistency evaluation."""

    def __init__(self, cfg: dict[str, Any]) -> None:
        self.cfg = cfg

        self.data_root = Path(cfg["data"]["data_root"])
        self.case_glob = cfg["data"].get("case_glob", "*")
        self.allowed_extensions = cfg["data"].get("allowed_extensions", [".nii.gz", ".nii"])
        self.file_template = cfg["data"].get("file_template", "{source}_{organ}{ext}")
        self.recursive = bool(cfg["data"].get("recursive", False))

        self.sources_all = cfg["sources"]["all"]
        self.sources_human = cfg["sources"]["human"]
        self.staple_inputs = cfg["sources"].get("staple_inputs", self.sources_human)
        self.organs = cfg["organs"]

        out_cfg = cfg["output"]
        self.output_dir = Path(out_cfg["output_dir"])
        self.detailed_csv = self.output_dir / out_cfg["detailed_csv"]
        self.summary_csv = self.output_dir / out_cfg["summary_csv"]
        self.staple_dir = self.output_dir / out_cfg.get("staple_dir", "staple_masks")
        self.save_staple_masks = bool(out_cfg.get("save_staple_masks", True))

        runtime_cfg = cfg.get("runtime", {})
        self.staple_threshold = float(runtime_cfg.get("staple_threshold", 0.5))
        self.continue_on_error = bool(runtime_cfg.get("continue_on_error", True))
        self.logger = setup_logger(Path(runtime_cfg.get("log_file", "./logs/run.log")))

        ensure_dir(self.output_dir)
        ensure_dir(self.staple_dir)

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

    def run(self) -> None:
        """Execute evaluation pipeline and write CSV outputs."""
        case_dirs = list_case_dirs(self.data_root, self.case_glob, recursive=self.recursive)
        self.logger.info("Found %d case directories under %s", len(case_dirs), self.data_root)

        rows: list[EvalRow] = []
        for idx, case_dir in enumerate(case_dirs, start=1):
            case_id = case_dir.name
            self.logger.info("[%d/%d] Processing case: %s", idx, len(case_dirs), case_id)
            try:
                case_rows = self._process_case(case_dir=case_dir, case_id=case_id)
                rows.extend(case_rows)
            except Exception as exc:
                self.logger.exception("Case failed: %s | error=%s", case_id, exc)
                if not self.continue_on_error:
                    raise

        detailed_df = pd.DataFrame([r.__dict__ for r in rows])
        if detailed_df.empty:
            self.logger.warning("No evaluation rows produced. Writing empty CSVs.")
            detailed_df = pd.DataFrame(columns=["case_id", "organ", "source_a", "source_b", "dice"])

        detailed_df.to_csv(self.detailed_csv, index=False)
        self.logger.info("Detailed results written to %s", self.detailed_csv)

        summary_df = self._make_summary(detailed_df)
        summary_df.to_csv(self.summary_csv, index=False)
        self.logger.info("Summary results written to %s", self.summary_csv)

    def _process_case(self, case_dir: Path, case_id: str) -> list[EvalRow]:
        records = discover_case_masks(
            case_dir=case_dir,
            case_id=case_id,
            sources=self.sources_all,
            organs=self.organs,
            file_template=self.file_template,
            allowed_extensions=self.allowed_extensions,
        )

        rows: list[EvalRow] = []
        for organ in self.organs:
            try:
                rows.extend(self._process_case_organ(case_id, organ, records))
            except Exception as exc:
                self.logger.exception(
                    "Failed case-organ. case=%s organ=%s error=%s", case_id, organ, exc
                )
                if not self.continue_on_error:
                    raise
        return rows

    def _process_case_organ(
        self,
        case_id: str,
        organ: str,
        records: dict[tuple[str, str], Path],
    ) -> list[EvalRow]:
        loaded: dict[str, tuple[Any, Any]] = {}
        for source in self.sources_all:
            p = records.get((source, organ))
            if p is None:
                self.logger.warning("Missing mask. case=%s organ=%s source=%s", case_id, organ, source)
                continue
            loaded[source] = read_mask(p)

        rows: list[EvalRow] = []

        for a, b in self.human_pairs + self.nnunet_pairs:
            if a not in loaded or b not in loaded:
                continue
            row = self._compare_pair(case_id, organ, a, b, loaded[a], loaded[b])
            if row is not None:
                rows.append(row)

        staple_img = self._build_staple(case_id, organ, loaded)
        if staple_img is not None:
            staple_np = sitk_to_numpy(staple_img)
            for source in self.sources_all:
                if source not in loaded:
                    continue
                _, src_np = loaded[source]
                try:
                    dice = dice_coefficient(src_np, staple_np)
                    rows.append(EvalRow(case_id, organ, source, "STAPLE", dice))
                except ValueError as exc:
                    self.logger.warning(
                        "Shape mismatch with STAPLE. case=%s organ=%s source=%s err=%s",
                        case_id,
                        organ,
                        source,
                        exc,
                    )

        return rows

    def _compare_pair(
        self,
        case_id: str,
        organ: str,
        source_a: str,
        source_b: str,
        a_data: tuple[Any, Any],
        b_data: tuple[Any, Any],
    ) -> EvalRow | None:
        img_a, np_a = a_data
        img_b, np_b = b_data

        if not metadata_matches(img_a, img_b):
            self.logger.warning(
                "Metadata mismatch. case=%s organ=%s %s_vs_%s",
                case_id,
                organ,
                source_a,
                source_b,
            )

        try:
            dice = dice_coefficient(np_a, np_b)
            return EvalRow(case_id, organ, source_a, source_b, dice)
        except ValueError as exc:
            self.logger.warning(
                "Skipping due to shape mismatch. case=%s organ=%s pair=%s_vs_%s err=%s",
                case_id,
                organ,
                source_a,
                source_b,
                exc,
            )
            return None

    def _build_staple(
        self,
        case_id: str,
        organ: str,
        loaded: dict[str, tuple[Any, Any]],
    ):
        staple_sources = [s for s in self.staple_inputs if s in loaded]
        if len(staple_sources) == 0:
            self.logger.warning("No STAPLE inputs. case=%s organ=%s", case_id, organ)
            return None

        staple_imgs = [loaded[s][0] for s in staple_sources]
        try:
            staple_img = generate_staple_image(staple_imgs, threshold=self.staple_threshold)
        except Exception as exc:
            self.logger.warning("STAPLE failed. case=%s organ=%s err=%s", case_id, organ, exc)
            return None

        if self.save_staple_masks:
            output_path = self.staple_dir / case_id / f"STAPLE_{organ}.nii.gz"
            save_staple_image(staple_img, output_path)

        return staple_img

    @staticmethod
    def _make_summary(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return pd.DataFrame(
                columns=["organ", "source_a", "source_b", "mean", "std", "median", "min", "max"]
            )

        grouped = (
            df.groupby(["organ", "source_a", "source_b"], dropna=False)["dice"]
            .agg(["mean", "std", "median", "min", "max"])
            .reset_index()
        )
        return grouped
