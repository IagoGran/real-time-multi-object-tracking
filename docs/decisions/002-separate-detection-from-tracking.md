# ADR-002: Separate detection from tracking behind a detector abstraction

## Status

Accepted

## Date

2026-10

## Context

The project originally used YOLO for pedestrian detection and ByteTrack for identity persistence. Later experiments added Faster R-CNN and required comparing detectors without rewriting the tracking or evaluation pipeline.

Coupling model-specific inference directly to tracking would make detector comparison and later teacher/student experiments unnecessarily difficult.

## Decision

Define a detector abstraction that returns a common `Detection` representation independent of the underlying model.

Keep detector implementations, tracking, trajectory management, analytics and evaluation as separate responsibilities.

## Alternatives considered

### Build the pipeline directly around Ultralytics YOLO output

This would reduce initial code, but would tightly couple downstream components to one library and make Faster R-CNN or future detectors harder to add.

### One end-to-end script per detector

Rejected because it would duplicate matching, visualization, tracking and evaluation logic.

## Consequences

Positive:

- YOLO and Faster R-CNN can be benchmarked through the same evaluation pipeline;
- tracking code does not need to know which detector produced the boxes;
- teacher/student experiments can reuse detector outputs;
- tests can use fake detector implementations.

Trade-offs:

- requires maintaining a small common interface;
- model-specific capabilities must not leak accidentally into generic code.

## Evidence

The project currently has detector-specific implementations under `src/detection/` and a common `Detection` type used downstream.

## Outcome

The abstraction enabled direct YOLO26n / YOLO26s / YOLO26m / Faster R-CNN comparison using the same matching and geometric metrics.

## Revisit when

Revisit if a future detector requires outputs that cannot reasonably be represented by the current detection abstraction without losing essential information.
