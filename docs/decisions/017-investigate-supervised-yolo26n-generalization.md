# ADR-017: Investigate supervised YOLO26n generalization with controlled fine-tuning experiments

## Status

Proposed

## Date

2026-10-16

## Context

The first supervised YOLO26n run used the MOT17 sequence split, `imgsz=640`, AdamW, `lr0=0.001`, batch 16, and seed 0. Its best validation checkpoint was selected at epoch 3 and training stopped after epoch 23. The recorded validation mAP50-95 peaked at 0.36233. The subsequent quality benchmark and duplicate-detection diagnostic reported worse localization and more duplicate pairs for the fine-tuned checkpoint than for the original model.

These are observations from the recorded experiment. They do not establish that learning rate, image size, scene domain shift, or any other factor caused the regression.

## Decision

Prepare controlled supervised runs from the original `yolo26n.pt` weights:

- E1 changes only `lr0` from 0.001 to 0.0001 at 640.
- E2 uses E1's configuration and changes `imgsz` to 960.
- E3 uses E2's configuration and sets Ultralytics `multi_scale=0.25`.

Before those runs, compare the original and E0 checkpoint on the configured train and validation sequences at fixed 640 and native inference sizes, using the existing evaluator and matching protocol. Keep the test split reserved. Preserve distinct experiment directories and record checkpoints, configuration, metrics, selected epoch, and environment metadata.

## Alternatives considered

### Change NMS or annotation policy

These would alter the evaluation or training target while the source of the observed regression is unknown. They are outside this experiment plan.

### Change several training variables at once

This would make results harder to attribute. The planned sequence isolates learning rate, then resolution, then multiscale training.

## Consequences

Positive consequences:

- The diagnostic checks whether degradation differs between train and validation and between inference resolutions.
- Later experiments isolate three candidate factors while always starting from the same original weights.
- Evaluation metrics, sequence breakdowns, and curve plots are reproducible from saved JSON and CSV files.

Trade-offs:

- Training at 960 and especially multiscale training may exceed the RTX 3060's 12 GB VRAM at batch 16. Any batch-size adjustment must be recorded and will weaken direct comparability.
- Training loss magnitude may vary with image resolution and should not be compared as an absolute quantity across resolutions.

## Evidence

- E0 manifest: `results/training/yolo26n_supervised_20261015_153159_ab7d8745/experiment.json`.
- E0 history: `results/training/yolo26n_supervised_20261015_153159_ab7d8745/results.csv`; mAP50-95 is highest at epoch 3 (0.36233) and the run records 23 epochs.
- E0 `best.pt` and `last.pt` files exist and their SHA-256 hashes match the manifest.
- Ultralytics 8.4.174 documents `multi_scale` as a fraction of `imgsz`; its detector trainer resizes each batch and rounds dimensions to the model stride.
- Duplicate-detection results are stored under `results/duplicate_diagnostics/` and `results/duplicate_diagnostics_nms_false/`.
- E0 train/validation diagnostics at fixed 640 and native resolution are stored in `results/evaluation/e0_vs_original_{train,validation}_{640,native}.json`; plots are in `results/plots/e0_diagnostic/`.
- At confidence 0.30 and 640, E0 train Match is 63.69% vs 35.71% for the original, while validation Match is 48.13% vs 30.76%; E0 validation IoU is 0.790 vs 0.836, normalized edge error 6.43% vs 4.46%, and FP/frame 3.02 vs 0.82.
- At confidence 0.30 and native resolution, E0 train Match is 63.09% vs 52.03%, while validation Match is 45.22% vs 63.92%; E0 validation IoU is 0.787 vs 0.810, normalized edge error 6.72% vs 5.39%, and FP/frame 6.99 vs 5.31.
- One-epoch 960 preflights with batch 16 completed, including `multi_scale=0.25`, but Ultralytics selected CPU in this session. No RTX 3060 VRAM conclusion can be drawn from them.

## Outcome

The diagnostics verify that E0 has higher matched coverage but more false positives than the original on train; on validation it has worse localization and coverage at native resolution, and worse localization despite higher coverage at 640. This is evidence of a train/validation and inference-resolution interaction in the observed metrics, not proof of its cause. E1-E3 remain pending, and no cause has been established.

## Revisit when

The controlled experiments and the final validation comparison are complete, or if a preflight establishes that the planned resolution/batch settings cannot run on the available GPU.
