# MRLINAC Consistency Evaluation Framework

用于评估 MRLINAC 多专家标注（oncologist / radiologist_1 / radiologist_2）与 nnUNet 预测一致性的完整可运行 Python 框架。

## 1. 任务覆盖

本项目实现：

- 递归扫描 `images_root`（nnUNet 风格 `Case001_0000.nii.gz`）
- 自动匹配同 case 的多来源多类别标签（`Case001.nii.gz`）
- 严格按固定 label order 提取器官：
  - prostate = 1
  - rectum = 2
  - bladder = 3
  - background = 0
- 计算 Dice（pairwise + 与 STAPLE 比较）
- 逐器官生成 STAPLE（默认仅人工）并可导出多类别 STAPLE
- 生成叠加预览图（PNG）
- 导出：
  - `pairwise_metrics.csv`
  - `summary_metrics.csv`
- 输出日志 `logs/run.log`

## 2. 严格器官定义

框架默认使用如下映射（可在 `config.yaml` 保持一致）：

```yaml
organs:
  prostate: {order: 1, RGB: [205, 180, 219], note: purple}
  rectum:   {order: 2, RGB: [255, 175, 204], note: blue}
  bladder:  {order: 3, RGB: [162, 210, 255], note: pink}
```

Dice / STAPLE / Overlay 均基于该 label 定义。

## 3. 数据组织（nnUNet 风格）

```text
data/
  images/
    Case001_0000.nii.gz
    Case002_0000.nii.gz
  labels/
    oncologist/
      Case001.nii.gz
      Case002.nii.gz
    radiologist_1/
      Case001.nii.gz
      Case002.nii.gz
    radiologist_2/
      Case001.nii.gz
      Case002.nii.gz
  preds/
    nnunet/
      Case001.nii.gz
      Case002.nii.gz
```

## 4. 缺失器官规则（严格执行）

若某 case 的某 organ 在任意来源中不存在（即 label 值完全缺失），则：

- 该 case-organ **全部比较都跳过**（包括 STAPLE 相关）
- 不会把缺失器官当全 0 参与 Dice
- 在 `pairwise_metrics.csv` 中写入 `status=skipped_missing_organ`
- 在日志明确记录

同理，若某 case 缺任意来源文件，则该 case-organ 全部比较写入 `status=skipped_missing_file`。

## 5. 安装

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 6. 运行

```bash
python main.py --config config.yaml
```

或

```bash
python scripts/run_eval.py --config config.yaml
```

## 7. 输出内容

```text
outputs/
  pairwise_metrics.csv
  summary_metrics.csv
  previews/
    Case001/
      oncologist_overlay.png
      radiologist_1_overlay.png
      radiologist_2_overlay.png
      nnunet_overlay.png
      staple_overlay.png   # 若该 case 三器官 STAPLE 全部成功
  staple_masks/
    Case001/
      staple_prostate.nii.gz
      staple_rectum.nii.gz
      staple_bladder.nii.gz
      staple_multiclass.nii.gz
logs/
  run.log
```

## 8. CSV 字段定义

### 8.1 pairwise_metrics.csv

字段：

- `case_id`
- `organ`
- `source_a`
- `source_b`
- `dice`
- `status`

`status` 取值：

- `ok`
- `skipped_missing_organ`
- `skipped_missing_file`
- `error`

### 8.2 summary_metrics.csv

按 `organ + comparison_pair` 聚合（仅统计 `status=ok`）：

- `n`
- `mean`
- `std`
- `median`
- `min`
- `max`

## 9. config.yaml 说明

必须字段与说明：

- `images_root`: MRI 根目录（递归扫描）
- `oncologist_labels_root`: oncologist 标签目录
- `radiologist1_labels_root`: radiologist_1 标签目录
- `radiologist2_labels_root`: radiologist_2 标签目录
- `nnunet_pred_root`: nnUNet 预测目录
- `output_root`: 输出根目录
- `log_file`: 日志文件路径
- `image_suffix`: 图像 case 后缀（默认 `_0000`）
- `label_suffix`: 标签 case 后缀（默认空）
- `image_caseid_pattern`: 从图像文件名提取 case_id 的正则（需包含命名组 `case_id`）
- `supported_image_exts`: 支持扩展名（如 `.nii.gz/.nii/.mha/.nrrd`）
- `organs`: 器官定义（order + RGB）
- `staple_sources`: STAPLE 参与来源（默认仅人工）
- `staple_threshold`: STAPLE 二值阈值
- `save_staple_masks`: 是否导出分器官 STAPLE
- `save_staple_multiclass`: 是否导出多类别 STAPLE
- `save_overlay_previews`: 是否导出叠加图
- `preview_slice_mode`: `axial` 或 `three_plane`
- `overlay_alpha`: overlay 透明度
- `continue_on_case_error`: 单 case 异常是否继续

## 10. 测试

```bash
pytest -q
```

包含：

- `test_metrics.py`: Dice 行为
- `test_case_matching.py`: nnUNet 风格 case 提取与匹配
- `test_label_parsing.py`: label 提取与器官存在判定
