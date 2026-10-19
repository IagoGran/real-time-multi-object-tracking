# Technical Decision Log

This directory records significant technical and experimental decisions made during the development of **Real-Time Multi-Object Tracking**.

The purpose is not to duplicate the implementation or act as a changelog. Each record captures a decision that involved meaningful alternatives, trade-offs, experimental reasoning, or consequences for later work.

The records distinguish between:

- what was known when the decision was made;
- what was chosen and why;
- alternatives that were considered;
- evidence gathered later;
- conditions under which the decision should be revisited.

Statuses used in this project:

- **Proposed** — planned or under evaluation, not yet validated enough to treat as settled.
- **Accepted** — currently adopted in the project.
- **Superseded** — replaced by a later decision.
- **Rejected** — explicitly considered and not adopted.

## Decision index

| ADR | Decision | Status | Date |
|---|---|---|---|
| [ADR-001](001-use-mot17-as-primary-benchmark-dataset.md) | Use MOT17 as the primary benchmark dataset | Accepted | 2026-10 |
| [ADR-002](002-separate-detection-from-tracking.md) | Separate detection from tracking behind a detector abstraction | Accepted | 2026-10 |
| [ADR-003](003-use-bytetrack-as-first-tracker.md) | Use ByteTrack as the first tracker | Accepted | 2026-10 |
| [ADR-004](004-preserve-source-resolution-for-detection.md) | Preserve source resolution for detection and align to model stride | Accepted | 2026-10 |
| [ADR-005](005-use-bbox-center-for-trajectories.md) | Use bounding-box center as the trajectory point | Accepted | 2026-10 |
| [ADR-006](006-use-normalized-line-crossing.md) | Represent the counting line in normalized frame coordinates | Accepted | 2026-10 |
| [ADR-007](007-use-geometric-detection-metrics.md) | Use geometric localization metrics in addition to coverage metrics | Accepted | 2026-10 |
| [ADR-008](008-evaluate-thresholds-from-one-inference-pass.md) | Evaluate confidence thresholds from one inference pass | Accepted | 2026-10 |
| [ADR-009](009-persist-benchmarks-with-checkpoints-and-json.md) | Persist long benchmarks with logs, checkpoints and JSON results | Accepted | 2026-10 |
| [ADR-010](010-use-one-mot17-detector-variant-per-scene.md) | Use one MOT17 detector variant per scene for detector evaluation | Accepted | 2026-10 |
| [ADR-011](011-split-training-and-validation-by-sequence.md) | Split training and validation by complete sequences | Proposed | 2026-10 |
| [ADR-012](012-use-yolo26n-as-student-candidate.md) | Use YOLO26n as the primary student candidate | Proposed | 2026-10 |
| [ADR-013](013-use-pseudo-labeling-before-true-distillation.md) | Test teacher pseudo-labeling before true knowledge distillation | Proposed | 2026-10 |
| [ADR-014](014-keep-supervised-and-teacher-training-separate.md) | Keep supervised and teacher training as separate experiments | Proposed | 2026-10 |
| [ADR-015](015-separate-final-evaluation-json-from-checkpoint.md) | Separate final evaluation JSON from operational checkpoints | Proposed | 2026-10 |
| [ADR-016](016-retain-mot17-frames-and-label-target-class-only.md) | Retain MOT17 frames and label only target-class GT | Accepted | 2026-10 |
| [ADR-017](017-investigate-supervised-yolo26n-generalization.md) | Investigate supervised YOLO26n generalization with controlled fine-tuning experiments | Proposed | 2026-10 |

## Reading the log

These records were reconstructed from the project's development discussion and current implementation direction. Where a decision is already reflected in working code or benchmark results it is marked **Accepted**. Training-related decisions that have been designed but not yet validated experimentally remain **Proposed**.

When later experiments contradict an earlier assumption, the original decision should remain intact and the new evidence should be added under **Outcome** or captured by a superseding ADR.
