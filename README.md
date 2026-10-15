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

Detectors implement a small common interface: `detect(frame, confidence_threshold)`
returns `list[Detection]`. The existing YOLO detector and the pretrained
Torchvision Faster R-CNN detector both follow this contract; tracking and
analysis components consume detections without depending on either model.

Evaluate detector localization on one MOT17 sequence from the repository root:

```bash
python -m src.evaluation.evaluate_detectors --max-frames 8
```

By default, the evaluator sweeps confidence thresholds from 0.10 to 0.60.
Pass `--confidence-threshold 0.3` to run a single threshold. A sweep creates
one checkpoint per threshold (for example,
`results/detector_benchmark_conf_0p30.json`); a single-threshold run keeps the
exact `--results` path. The evaluator logs to the console and
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
