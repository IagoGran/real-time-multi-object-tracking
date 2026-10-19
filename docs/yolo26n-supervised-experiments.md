# YOLO26n supervised follow-up runs

Run commands from the repository root. E1, E2, and E3 each load `yolo26n.pt`; none loads another experiment's checkpoint.

## Prepared E0 diagnostics

Completed public evaluation JSONs:

- `results/evaluation/e0_vs_original_train_640.json`
- `results/evaluation/e0_vs_original_validation_640.json`
- `results/evaluation/e0_vs_original_train_native.json`
- `results/evaluation/e0_vs_original_validation_native.json`

The two native-resolution JSONs were derived from the completed `results/detector_benchmark.json` per-sequence observations after verifying its full seven-sequence split, frame counts, thresholds, matching IoU, no frame limit, and the original and E0 checkpoint SHA-256 values. The fixed-640 evaluations were executed with the updated evaluator. Plot files are under `results/plots/e0_diagnostic/`.

## Training commands

These pin the requested RTX 3060 as device 0. The same 50 epochs, patience 20, batch 16, AdamW, seed 0, split, and augmentations are used in each run. Only the listed factor changes between runs.

```bash
python -m src.training.train_detector --mode supervised --student-weights yolo26n.pt --epochs 50 --patience 20 --batch 16 --imgsz 640 --lr0 0.0001 --optimizer AdamW --seed 0 --multi-scale 0.0 --device 0 --experiment-label E1-low-lr-640
```

```bash
python -m src.training.train_detector --mode supervised --student-weights yolo26n.pt --epochs 50 --patience 20 --batch 16 --imgsz 960 --lr0 0.0001 --optimizer AdamW --seed 0 --multi-scale 0.0 --device 0 --experiment-label E2-low-lr-960
```

```bash
python -m src.training.train_detector --mode supervised --student-weights yolo26n.pt --epochs 50 --patience 20 --batch 16 --imgsz 960 --lr0 0.0001 --optimizer AdamW --seed 0 --multi-scale 0.25 --device 0 --experiment-label E3-low-lr-960-ms025
```

E4 changes only the native Ultralytics `freeze` setting from E2. It starts directly from `yolo26n.pt`. The CLI checks the loaded model's indexed layers and requires an 11-layer declared backbone for `--freeze 11`. At `on_pretrain_routine_end`, it verifies `requires_grad`, confirms trainable parameters are in the optimizer, and checks that frozen parameters have no gradients. Ultralytics may still place frozen parameters in optimizer groups; the CLI snapshots them and checks after each epoch that their values did not change. Layer and parameter counts and verification results are saved to `experiment.json`.

```bash
python -m src.training.train_detector --mode supervised --student-weights yolo26n.pt --epochs 50 --patience 20 --batch 16 --imgsz 960 --lr0 0.0001 --optimizer AdamW --seed 0 --multi-scale 0.0 --freeze 11 --device 0 --experiment-label E4-frozen-backbone-960
```

Omitting `--freeze` keeps the existing behavior; `--freeze 0` is passed explicitly to Ultralytics. Its standard mechanism disables parameter gradients for the first N indexed layers, but does not force those modules into evaluation mode. BatchNorm running statistics in frozen layers can therefore still change during training. This first E4 run keeps that native behavior. Ultralytics also freezes its DFL parameters independently of `freeze`; the verification records these in total frozen parameter counts while checking the requested layer range separately.

Before starting E2 and E3 on the RTX 3060, check batch 16 with short one-epoch runs. These use eight frames per sequence and create clearly labeled preflight directories; they are not E2/E3 training runs.

```bash
python -m src.training.train_detector --mode supervised --student-weights yolo26n.pt --epochs 1 --patience 0 --batch 16 --imgsz 960 --lr0 0.0001 --optimizer AdamW --seed 0 --multi-scale 0.0 --device 0 --max-frames-per-sequence 8 --experiment-label E2-960-b16-preflight
```

```bash
python -m src.training.train_detector --mode supervised --student-weights yolo26n.pt --epochs 1 --patience 0 --batch 16 --imgsz 960 --lr0 0.0001 --optimizer AdamW --seed 0 --multi-scale 0.25 --device 0 --max-frames-per-sequence 8 --experiment-label E3-960-ms025-b16-preflight
```

The two preflights in the current execution environment ran on CPU and cannot establish RTX 3060 VRAM use. If either GPU preflight runs out of memory, stop before the full run. A controlled follow-up is to use batch 8 for E2 and E3 and add an E1 batch-8 reference at 640; record the changed batch in each manifest. Do not silently change batch size.

If that adjustment is needed, the additional E1 reference is:

```bash
python -m src.training.train_detector --mode supervised --student-weights yolo26n.pt --epochs 50 --patience 20 --batch 8 --imgsz 640 --lr0 0.0001 --optimizer AdamW --seed 0 --multi-scale 0.0 --device 0 --experiment-label E1-low-lr-640-b8-reference
```

## Re-run the E0 comparison

The evaluator uses confidence thresholds 0.10–0.60 and match IoU 0.5 by default. These commands use the exact train and validation partitions from the split JSON.

```bash
python -m src.evaluation.evaluate_detectors --split-config configs/mot17_training_split.json --partition train --model Original=yolo26n.pt --model E0=results/training/yolo26n_supervised_20261015_153159_ab7d8745/weights/best.pt --imgsz 640 --evaluation-name e0_vs_original_train_640 --results results/evaluation/checkpoints/e0_vs_original_train_640.json --evaluation-output results/evaluation/e0_vs_original_train_640.json --skip-estimate
```

```bash
python -m src.evaluation.evaluate_detectors --split-config configs/mot17_training_split.json --partition validation --model Original=yolo26n.pt --model E0=results/training/yolo26n_supervised_20261015_153159_ab7d8745/weights/best.pt --imgsz 640 --evaluation-name e0_vs_original_validation_640 --results results/evaluation/checkpoints/e0_vs_original_validation_640.json --evaluation-output results/evaluation/e0_vs_original_validation_640.json --skip-estimate
```

Omit `--imgsz 640` to use the existing native, stride-aligned inference behavior.

## Training curve comparison

After the full runs complete, resolve their uniquely labeled directories and create all eight comparative curves from the CSVs. This reads no model weights and does not use CUDA.

```bash
E0_DIR=results/training/yolo26n_supervised_20261015_153159_ab7d8745
E1_DIR=$(find results/training -maxdepth 1 -type d -name 'yolo26n_supervised_E1-low-lr-640_*' -print -quit)
E2_DIR=$(python -c 'import json; from pathlib import Path; runs=sorted(Path("results/training").glob("yolo26n_supervised_E2-low-lr-960_*/experiment.json")); valid=[p.parent for p in runs if json.loads(p.read_text()).get("status")=="completed" and (p.parent/"weights/best.pt").is_file()]; print(valid[-1] if valid else "")')
E3_DIR=$(find results/training -maxdepth 1 -type d -name 'yolo26n_supervised_E3-low-lr-960-ms025_*' -print -quit)
python -m src.analysis.plot_evaluation --training-csv E0=$E0_DIR/results.csv --training-csv E1=$E1_DIR/results.csv --training-csv E2=$E2_DIR/results.csv --training-csv E3=$E3_DIR/results.csv --output-dir results/plots/yolo26n-supervised-comparison
```

Training loss magnitudes may vary with resolution; interpret their trends within each resolution. Each plot marks the epoch selected by maximum validation mAP50-95.

## Final validation comparison

After E1–E4 complete, resolve their labeled directories as above. For the E2/E4 hypothesis, compare Original, E2, and E4 on identical validation sequences at 640, 960, and native resolution where supported. Compare models only within the same sweep. The operational speed benchmark remains a separate command/tool.

```bash
python -m src.evaluation.evaluate_detectors --split-config configs/mot17_training_split.json --partition validation --model Original=yolo26n.pt --model E0=results/training/yolo26n_supervised_20261015_153159_ab7d8745/weights/best.pt --model E1=$E1_DIR/weights/best.pt --model E2=$E2_DIR/weights/best.pt --model E3=$E3_DIR/weights/best.pt --imgsz 640 --evaluation-name yolo26n_final_validation_640 --results results/evaluation/checkpoints/yolo26n_final_validation_640.json --evaluation-output results/evaluation/yolo26n_final_validation_640.json --skip-estimate
```

```bash
python -m src.evaluation.evaluate_detectors --split-config configs/mot17_training_split.json --partition validation --model Original=yolo26n.pt --model E0=results/training/yolo26n_supervised_20261015_153159_ab7d8745/weights/best.pt --model E1=$E1_DIR/weights/best.pt --model E2=$E2_DIR/weights/best.pt --model E3=$E3_DIR/weights/best.pt --evaluation-name yolo26n_final_validation_native --results results/evaluation/checkpoints/yolo26n_final_validation_native.json --evaluation-output results/evaluation/yolo26n_final_validation_native.json --skip-estimate
```

Both JSONs include aggregate and per-sequence results for MOT17-04-FRCNN and MOT17-11-FRCNN. Do not use the test split during this selection stage.

Generate matching aggregate plots after the evaluations finish:

```bash
python -m src.analysis.plot_evaluation --evaluation results/evaluation/yolo26n_final_validation_640.json --evaluation results/evaluation/yolo26n_final_validation_native.json --output-dir results/plots/yolo26n-final-validation
```

For the focused Original/E2/E4 comparison, reuse the evaluator with these same three models for each resolution. Example at 640 (repeat with `--imgsz 960` and distinct `_960` output names; omit `--imgsz` for native resolution):

```bash
E2_DIR=$(python -c 'import json; from pathlib import Path; runs=sorted(Path("results/training").glob("yolo26n_supervised_E2-low-lr-960_*/experiment.json")); valid=[p.parent for p in runs if json.loads(p.read_text()).get("status")=="completed" and (p.parent/"weights/best.pt").is_file()]; print(valid[-1] if valid else "")')
E4_DIR=$(python -c 'import json; from pathlib import Path; runs=sorted(Path("results/training").glob("yolo26n_supervised_E4-frozen-backbone-960_*/experiment.json")); valid=[p.parent for p in runs if json.loads(p.read_text()).get("status")=="completed" and (p.parent/"weights/best.pt").is_file()]; print(valid[-1] if valid else "")')
test -n "$E2_DIR" -a -n "$E4_DIR" || { echo 'Completed E2/E4 checkpoints were not found'; exit 1; }
python -m src.evaluation.evaluate_detectors --split-config configs/mot17_training_split.json --partition validation --model Original=yolo26n.pt --model E2=$E2_DIR/weights/best.pt --model E4=$E4_DIR/weights/best.pt --imgsz 640 --evaluation-name yolo26n_E2_E4_validation_640 --results results/evaluation/checkpoints/yolo26n_E2_E4_validation_640.json --evaluation-output results/evaluation/yolo26n_E2_E4_validation_640.json --skip-estimate
```

The JSONs include aggregate and per-sequence results, including MOT17-04-FRCNN and MOT17-11-FRCNN. Compare Match%, IoU, geometric errors, unmatched predictions, and existing metrics by sequence; validation mAP alone does not establish better tracking. Keep the speed benchmark separate and do not use the test split for this selection.

## E2/E4 validation results

The completed Original/E2/E4 validation sweeps use the same two validation sequences, match IoU 0.5, and confidence thresholds 0.10–0.60. At confidence 0.50, aggregate results are:

| Inference size | Model | Match% | IoU | Mean edge error | Mean center error | Mean size error | Unmatched predictions/frame |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 640 | Original | 23.19 | 0.841 | 4.27% | 3.37% | 6.99% | 0.49 |
| 640 | E2 | 39.55 | 0.819 | 5.32% | 3.73% | 8.65% | 0.48 |
| 640 | E4 | 37.49 | 0.829 | 4.93% | 3.41% | 7.71% | 0.55 |
| 960 | Original | 34.17 | 0.850 | 4.07% | 2.95% | 6.63% | 0.74 |
| 960 | E2 | 48.27 | 0.821 | 5.26% | 3.82% | 8.41% | 1.56 |
| 960 | E4 | 47.94 | 0.827 | 5.03% | 3.61% | 7.96% | 1.29 |
| Native | Original | 56.59 | 0.824 | 4.91% | 3.77% | 8.04% | 2.30 |
| Native | E2 | 56.54 | 0.816 | 5.48% | 4.19% | 8.82% | 2.47 |
| Native | E4 | 59.90 | 0.823 | 5.22% | 4.05% | 8.40% | 1.37 |

At 640 and 960, E4 improves the listed geometry measures over E2 while Match% is slightly lower; at native size E4 improves both Match% and the listed geometry measures. Match% at confidence 0.50 by sequence:

| Inference size | Sequence | E2 | E4 |
| --- | --- | ---: | ---: |
| 640 | MOT17-04-FRCNN | 36.06% | 33.07% |
| 640 | MOT17-11-FRCNN | 57.10% | 59.75% |
| 960 | MOT17-04-FRCNN | 45.81% | 44.70% |
| 960 | MOT17-11-FRCNN | 60.65% | 64.26% |
| Native | MOT17-04-FRCNN | 55.10% | 58.45% |
| Native | MOT17-11-FRCNN | 63.81% | 67.23% |

This supports further investigation of the freezing hypothesis but does not establish better tracking: these are detector-to-ground-truth matching and geometry metrics, not tracker identity or trajectory metrics. The operational speed benchmark remains separate.

Evaluation JSONs: `results/evaluation/yolo26n_E2_E4_validation_{640,960,native}.json`. Regenerate the existing comparative plots with:

```bash
python -m src.analysis.plot_evaluation --evaluation results/evaluation/yolo26n_E2_E4_validation_640.json --evaluation results/evaluation/yolo26n_E2_E4_validation_960.json --evaluation results/evaluation/yolo26n_E2_E4_validation_native.json --output-dir results/plots/yolo26n-E2-E4-validation
```

The plots are under `results/plots/yolo26n-E2-E4-validation/yolo26n_E2_E4_validation_640/`.
