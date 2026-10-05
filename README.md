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
- OpenCV
- YOLO
- ByteTrack
- BoT-SORT
- TrackEval

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
