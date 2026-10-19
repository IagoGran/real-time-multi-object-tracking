# ADR-008: Evaluate confidence thresholds from one inference pass

## Status

Accepted

## Date

2026-10

## Context

The first confidence-threshold sweep reran the full detector benchmark separately for 0.10, 0.20, 0.30, 0.40, 0.50 and 0.60.

This was unnecessarily expensive because each detection already carries a confidence score.

The initial sweep also exposed that model-internal confidence filtering must be distinguished from evaluation-time filtering.

## Decision

Run detector inference once at a sufficiently low internal confidence floor, retain detection confidences, and evaluate multiple confidence thresholds by filtering the same predictions.

Conceptually:

    frame
      -> detector inference once
      -> raw detections
          -> >= 0.10 -> evaluation
          -> >= 0.20 -> evaluation
          -> ...
          -> >= 0.60 -> evaluation

Separate the detector's minimum inference confidence from evaluation thresholds.

## Alternatives considered

### Rerun inference for every threshold

Functionally valid but multiplies expensive GPU work and can introduce run-to-run timing variation.

### Let each detector use its own default threshold

Rejected for controlled comparison because defaults differ and low-confidence predictions may be discarded before the evaluator can inspect them.

## Consequences

Positive:

- threshold sweeps cost approximately one inference pass plus additional matching/metric computation;
- all thresholds operate on exactly the same underlying predictions;
- thresholds can be added without rerunning model inference if predictions are persisted later;
- detector FPS becomes independent of evaluation threshold.

Trade-offs:

- detector implementations need a clean separation between prediction and confidence filtering;
- keeping a very low inference floor may increase temporary prediction volume.

## Evidence

The first sweep produced identical YOLO results at 0.10 and 0.20, revealing that confidence was being filtered before the intended evaluation stage. The detector API was then redesigned around raw prediction plus post-inference filtering.

## Revisit when

Revisit if a detector's inference algorithm fundamentally changes with its confidence threshold in a way that makes post-filtering non-equivalent.
