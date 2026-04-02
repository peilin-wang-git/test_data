# MRLINAC Segmentation Consistency Evaluator

A runnable Python framework for evaluating segmentation consistency on MRLINAC cases across multiple sources (oncologist, radiologists, nnUNet), computing Dice scores, generating STAPLE consensus masks, and exporting detailed/summary CSV reports.

## Features

- Config-driven data discovery (no hardcoded paths)
- Supports `.nii` and `.nii.gz` by default (plus optional `.mha`, `.nrrd`)
- Per-case, per-organ Dice computation
- Pairwise human-vs-human and nnUNet-vs-human comparisons
- STAPLE consensus generation from configurable source list
- Robust handling for missing files/organs, shape mismatch, metadata mismatch, empty masks
- Error-tolerant execution (skip failed case/organ and continue)
- Logging to both console and `logs/run.log`

## Project Structure

```text
.
├── config.yaml
├── main.py
├── README.md
├── requirements.txt
├── scripts/
│   └── run_eval.py
├── src/
│   ├── __init__.py
│   ├── evaluator.py
│   ├── io_utils.py
│   ├── metrics.py
│   ├── staple.py
│   └── utils.py
└── tests/
    ├── test_matching.py
    └── test_metrics.py
```

## Expected Data Layout (default)

The default loader assumes each case has a folder under `data_root`:

```text
data_root/
  case_001/
    oncologist_prostate.nii.gz
    oncologist_bladder.nii.gz
    oncologist_rectum.nii.gz
    radiologist_1_prostate.nii.gz
    ...
    nnunet_rectum.nii.gz
  case_002/
    ...
```

You can change naming conventions via `config.yaml` keys:

- `data.case_glob`
- `data.file_template`
- `data.allowed_extensions`

`file_template` supports `{source}`, `{organ}`, `{ext}` placeholders.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
python main.py --config config.yaml
```

or:

```bash
python scripts/run_eval.py --config config.yaml
```

## Outputs

Configured in `output` section:

- Detailed CSV: one row per `case_id x organ x pair`
  - columns: `case_id, organ, source_a, source_b, dice`
- Summary CSV grouped by `(organ, source_a, source_b)`
  - metrics: `mean, std, median, min, max`
- STAPLE masks per case/organ under `output.staple_dir`
- Logs at `logs/run.log`

## Comparison Sets Implemented

A. Human pairwise:
- oncologist vs radiologist_1
- oncologist vs radiologist_2
- radiologist_1 vs radiologist_2

B. nnUNet vs human:
- nnunet vs oncologist
- nnunet vs radiologist_1
- nnunet vs radiologist_2

C. Each source vs STAPLE:
- oncologist vs STAPLE
- radiologist_1 vs STAPLE
- radiologist_2 vs STAPLE
- nnunet vs STAPLE

## Notes on Robustness

- Missing source/organ files: warning and skip relevant comparison
- Empty masks: handled in Dice (both empty => 1.0, one empty => 0.0)
- Shape mismatch: logged and skipped
- Metadata mismatch (spacing/origin/direction): warning logged; current implementation does not resample automatically
- Per-case/organ errors are captured and processing continues

## Tests

```bash
pytest -q
```

Tests cover Dice behavior and filename matching utilities.
