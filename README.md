# MRLINAC MRI Multi-Rater Consistency Framework

完整可运行 Python 框架：基于 `new_path.csv` 读取 case，评估 oncologist / radiologist_1 / radiologist_2 / nnUNet 在 prostate、rectum、bladder 上的一致性，计算 Dice、生成 STAPLE、导出 CSV 和 overlay 预览图。

## 1) 关键约束（已实现）

- 单个来源是**一个多类别 NIfTI**（不是每器官一个文件）
- 固定 label：
  - background = 0
  - prostate = 1
  - rectum = 2
  - bladder = 3
- Dice / STAPLE / overlay 都基于该定义
- 主入口是 `new_path.csv`，不依赖递归扫描 images 作为主索引

## 2) `new_path.csv` 格式

必须包含列：

- `destination image path`
- `destination label path`
- `index`

每行一个 case。

## 3) 路径解析规则

对于每一行：

- MRI image = `destination image path`
- oncologist = `destination label path`
- radiologist_1 = `radiologist1_labels_root / f"Case_{index:03d}" / "seg.nii.gz"`
- radiologist_2 = `radiologist2_labels_root / f"Case_{index:03d}" / "seg.nii.gz"`
- nnUNet（默认）= `nnunet_pred_root / basename(destination label path)`

若配置允许，可从 image 名回退推导 nnUNet 文件名再尝试匹配。

## 4) case_id 与 index

- `case_id`：由 `destination label path` 的 basename 去扩展名得到
- `index`：直接来自 `new_path.csv`

两个字段都会写入结果 CSV。

## 5) 缺失处理策略

### 缺失文件
- 记录 warning
- 只跳过受影响比较（`status=skipped_missing_file`）
- 流程继续处理其他 case

### 缺失器官（label 不存在或提取 mask 为空）
- 不把空 mask 当有效输入
- 依赖该来源该器官的比较跳过（`status=skipped_missing_organ`）
- 若 STAPLE 输入不完整，该 case-organ 的 STAPLE 跳过

## 6) 功能

### Pairwise Dice
A. 人工两两：
- oncologist vs radiologist_1
- oncologist vs radiologist_2
- radiologist_1 vs radiologist_2

B. nnUNet vs 人工：
- nnunet vs oncologist
- nnunet vs radiologist_1
- nnunet vs radiologist_2

C. 与 STAPLE：
- oncologist vs STAPLE
- radiologist_1 vs STAPLE
- radiologist_2 vs STAPLE
- nnunet vs STAPLE

### STAPLE
- 默认仅人工来源（可配置）
- 按器官独立计算
- 可保存每器官二值 STAPLE 与合成多类别 STAPLE

### Overlay Preview
- 输出 PNG（至少中间 axial 切片）
- 支持 `three_plane`（axial/coronal/sagittal）
- 默认颜色：
  - prostate: [205, 180, 219]
  - rectum: [255, 175, 204]
  - bladder: [162, 210, 255]

## 7) 输出

```text
outputs/
  pairwise_metrics.csv
  summary_metrics.csv
  previews/<case_id>/{oncologist,radiologist_1,radiologist_2,nnunet,staple}_overlay.png
  staple_masks/<case_id>/staple_{organ}.nii.gz
  staple_masks/<case_id>/staple_multiclass.nii.gz
logs/run.log
```

## 8) 配置说明（config.yaml）

最少字段：

- `case_table_csv`
- `nnunet_pred_root`
- `radiologist1_labels_root`
- `radiologist2_labels_root`
- `output_root`
- `staple_sources`
- `save_staple_masks`
- `save_overlay_previews`
- `preview_slice_mode`
- `organs`
- `label_mapping`
- `nnunet_match_use_label_basename_first`
- `allow_fallback_match_from_image_name`

## 9) 运行

```bash
python main.py --config config.yaml
```

或

```bash
python scripts/run_eval.py --config config.yaml
```

## 10) 测试

```bash
pytest -q
```

## 11) 仓库结构

```text
.
├── README.md
├── requirements.txt
├── config.yaml
├── main.py
├── scripts/
│   └── run_eval.py
├── src/
│   ├── __init__.py
│   ├── io_utils.py
│   ├── case_table.py
│   ├── label_parser.py
│   ├── metrics.py
│   ├── staple.py
│   ├── visualization.py
│   ├── evaluator.py
│   ├── utils.py
│   └── logger.py
└── tests/
    ├── test_metrics.py
    ├── test_case_matching.py
    └── test_label_parsing.py
```
