# ADR-001: Use MOT17 as the primary benchmark dataset

## Status

Accepted

## Date

2026-10

## Context

The project needs a reproducible pedestrian-tracking dataset with video sequences, ground-truth bounding boxes and stable frame metadata. The dataset must support both detector evaluation and later multi-object tracking evaluation.

The project initially focused on real-time pedestrian detection and tracking, with experiments already running on MOT17 sequences such as `MOT17-04-FRCNN` and `MOT17-05-FRCNN`.

## Decision

Use MOT17 as the primary benchmark dataset for detector and tracker experiments.

For detector-only evaluation, use the image frames and ground-truth person boxes. Tracking IDs are ignored until the tracking evaluation stage.

## Alternatives considered

### Ad-hoc videos

Useful for visual demos, but unsuitable as the primary benchmark because they lack reproducible ground truth.

### A custom annotated dataset

Potentially useful later for domain-specific experiments, but too expensive to create before the baseline pipeline is established.

## Consequences

Positive:

- reproducible benchmark sequences;
- ground-truth boxes and identities are available;
- supports both detection and tracking evaluation;
- well suited to crowded pedestrian scenes and occlusion.

Trade-offs:

- MOT17 has detector-specific directory variants that require care to avoid duplicate image evaluation;
- official MOT17 evaluation contains ignored/distractor handling that must be treated correctly before claiming official-comparable results.

## Evidence

The current project loads MOT17 sequence metadata, frame rates, image frames and `gt/gt.txt`, and has already completed detector benchmarks across MOT17 train scenes.

## Outcome

MOT17 proved useful for exposing differences that were less visible on a single sequence. In particular, conclusions drawn from `MOT17-04-FRCNN` changed slightly when the evaluation was expanded to all unique training scenes.

## Revisit when

Reconsider if the project shifts toward a different domain, such as vehicle tracking, indoor tracking, or a production dataset whose distribution differs substantially from MOT17.
