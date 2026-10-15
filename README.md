# Real-Time Multi-Object Tracking

A small Computer Vision project focused on **multi-object tracking in video**, object identity persistence, trajectory analysis and tracking evaluation.

The goal is not just to run a pretrained model, but to understand and compare the behaviour of different tracking strategies under realistic conditions such as occlusions, crossings, reappearances, motion blur and changing perspectives.

## Project goals

This project aims to build an end-to-end pipeline for:

- Object detection in video.
- Multi-object tracking with persistent IDs.
- Trajectory visualization.
- Entry/exit or line-crossing counting.
- Comparison between different tracking algorithms.
- Quantitative evaluation using standard tracking metrics.
- Basic inference performance analysis.

## Initial stack

- Python
- PyTorch
- Torchvision
- OpenCV
- YOLO
- ByteTrack
- BoT-SORT
- TrackEval

## Detectors

Detectors implement `predict(frame)` and shared confidence filtering;
`detect(frame, confidence_threshold)` remains available for tracking and
analysis components. Evaluation runs each frame through the model once and
applies every requested confidence threshold to those detections.

Evaluate detector localization on one MOT17 sequence from the repository root:

```bash
python -m src.evaluation.evaluate_detectors --max-frames 8
```

By default, the evaluator sweeps confidence thresholds from 0.10 to 0.60.
Pass `--confidence-threshold 0.3` to run a single threshold. A sweep stores all
thresholds in one checkpoint, `results/detector_benchmark.json`. The evaluator logs to the console and
`results/detector_benchmark.log`, and saves sequence and aggregate metrics
progressively. Use `--all-train` to evaluate one FRCNN variant per scene,
`--resume` to continue compatible checkpoints, and `--skip-estimate` to skip
runtime sampling. Faster R-CNN uses Torchvision's pretrained COCO weights,
which may be downloaded the first time it is initialized.

## Dataset

The initial experiments use **MOT17**, a public benchmark for multi-object pedestrian tracking.

Dataset files are **not included in this repository**.

Official dataset:
https://motchallenge.net/data/MOT17/

The dataset is distributed by MOTChallenge under its corresponding license. This repository only contains code, configuration files and experiment results generated from local dataset copies.

## Planned pipeline

```text
Video
  ↓
Object Detector
  ↓
Detections
  ↓
Multi-Object Tracker
  ↓
Persistent IDs
  ↓
Trajectories / Counting
  ↓
Evaluation
```

## Trackers

The first experiments will compare:

- ByteTrack
- BoT-SORT

The goal is to analyse not only aggregate metrics, but also practical failure modes such as:

- ID switches.
- Long occlusions.
- Crossing trajectories.
- Reappearances.
- Small or partially visible objects.
- Motion blur.
- Dense scenes.

## Evaluation

Planned metrics include:

### Detection
- Precision
- Recall
- mAP

### Tracking
- IDF1
- MOTA
- HOTA
- ID switches

### Performance
- FPS
- Average latency per frame
- GPU / CPU inference characteristics when relevant

## Repository structure

```text
.
├── src/
│   ├── detection/
│   ├── tracking/
│   ├── evaluation/
│   └── visualization/
├── configs/
├── tests/
├── results/
│   ├── metrics/
│   └── plots/
├── scripts/
├── requirements.txt
└── README.md
```

## Milestones

### 1. Detection baseline
Run a pretrained detector over MOT17 sequences and visualize detections.

### 2. ByteTrack integration
Add persistent object identities across frames.

### 3. Trajectories and counting
Visualize object paths and implement line/zone crossing counters.

### 4. Tracker comparison
Add BoT-SORT and compare it against ByteTrack on the same sequences.

### 5. Evaluation
Measure tracking quality and inference performance.

### 6. Failure analysis
Document concrete examples where the system succeeds or fails.

## What I want to learn

The main objective of this project is to gain practical experience with:

- Video-based Computer Vision.
- Multi-object tracking.
- Identity association.
- Occlusion handling.
- Tracking metrics.
- Real-time inference constraints.
- Experiment design and model comparison.
- Debugging Computer Vision systems under real-world conditions.

## Scope

This project is intentionally limited in scope.

It does **not** aim to:

- Identify real people.
- Perform facial recognition.
- Build persistent identity profiles across unrelated videos.
- Re-host the original dataset.

IDs are only used as temporary tracking identifiers inside benchmark sequences.

## Status

Work in progress.

## MOT17 training and evaluation preflight

The split in `configs/mot17_training_split.json` is shared by training, the dataset audit, and evaluation. Supervised training currently uses marked class-1 pedestrian boxes without a visibility cutoff and skips a whole frame when that frame contains an excluded annotation. The audit reports the lost frames and pedestrian boxes before training; it never edits the split or this policy. Review its dropped-frame percentage before training.

Install dependencies and run the tests from the repository root:

```powershell
python -m pip install -r requirements-training.txt -r requirements-test.txt -r requirements-analysis.txt
python -m pytest
```

Audit the local MOT17 copy and prepare a small supervised dataset without training:

```powershell
python -m src.training.inspect_dataset --split-config configs/mot17_training_split.json
python -m src.training.train_detector --mode supervised --split-config configs/mot17_training_split.json --prepare-only --max-frames-per-sequence 8
```

Run one-epoch smoke tests on up to eight frames per sequence. The teacher run generates pseudo-labels for the train frames and records their count and confidence statistics in its experiment manifest.

```powershell
python -m src.training.train_detector --mode supervised --split-config configs/mot17_training_split.json --epochs 1 --max-frames-per-sequence 8 --batch 2 --workers 0 --device 0
python -m src.training.train_detector --mode teacher --teacher-weights yolo26m.pt --teacher-confidence 0.25 --split-config configs/mot17_training_split.json --epochs 1 --max-frames-per-sequence 8 --batch 2 --workers 0 --device 0
```

Each run prints its `results/training/<experiment_id>` directory. Confirm that `experiment.json` and `weights/best.pt` plus `weights/last.pt` exist there. Use those paths for one joint, same-frame validation evaluation; this example uses eight frames per validation sequence and the shared default threshold sweep:

```powershell
Get-ChildItem results/training -Filter experiment.json -Recurse | ForEach-Object { $d = $_.Directory.FullName; [pscustomobject]@{ Experiment = $_.Directory.Name; Manifest = $_.Exists; Best = Test-Path (Join-Path $d 'weights/best.pt'); Last = Test-Path (Join-Path $d 'weights/last.pt') } } | Format-Table
```

```powershell
python -m src.evaluation.evaluate_detectors --split-config configs/mot17_training_split.json --partition validation --max-frames 8 --model "YOLO26n-baseline=yolo26n.pt" --model "YOLO26n-supervised=results/training/<supervised-id>/weights/best.pt" --model "YOLO26n-teacher=results/training/<teacher-id>/weights/best.pt" --model "Teacher-original=yolo26m.pt" --evaluation-name preflight_comparison --skip-estimate
python -m src.analysis.plot_evaluation --evaluation results/evaluation/preflight_comparison.json
```

The evaluation writes a stable JSON under `results/evaluation/` with split, partition, sequence, model, weight signature/hash, threshold, aggregate, and per-sequence data. Plotting reads only that JSON. Training curves read the existing Ultralytics `results.csv` referenced by each manifest:

```powershell
python -m src.analysis.plot_evaluation --training-manifest results/training/<experiment-id>/experiment.json
```

For operational speed, benchmark a detector directly at each deployment confidence threshold. This times `predict()` at that confidence, after warmup and CUDA synchronization, and excludes frame decoding and matching; it is separate from the low-floor quality threshold sweep:

```powershell
python -m src.evaluation.benchmark_detector_speed --split-config configs/mot17_training_split.json --partition validation --max-frames 100 --warmup 5 --model "student=results/training/<experiment-id>/weights/best.pt" --confidence-threshold 0.3 --confidence-threshold 0.5 --output results/evaluation/speed.json
```

The commands above are smoke tests only. For full supervised training, omit `--max-frames-per-sequence` and set the desired epoch count. For teacher training, keep the same split and confidence selected for the experiment.
