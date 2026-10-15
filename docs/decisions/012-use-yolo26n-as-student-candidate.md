# ADR-012: Use YOLO26n as the primary student candidate

## Status

Proposed

## Date

2026-10

## Context

The project aims to test whether a small detector can recover part of the quality of larger models while retaining real-time performance.

The full MOT17-train baseline showed a clear performance hierarchy:

- YOLO26n was substantially faster than the larger variants;
- YOLO26s improved match rate and geometry moderately;
- YOLO26m increased cost sharply while offering only limited additional coverage and no clear geometric advantage over YOLO26s.

## Decision

Use YOLO26n as the primary student candidate for fine-tuning experiments.

The target hypothesis is:

> Can a fine-tuned YOLO26n approach the quality of larger models while preserving a large part of its real-time speed advantage?

## Alternatives considered

### YOLO26s as student

Already a strong quality/performance compromise, but offers less headroom for demonstrating that training can improve a small model.

### YOLO26m as student

Rejected as the primary student because its baseline inference cost is already much higher.

## Consequences

Positive:

- creates a meaningful quality-versus-speed experiment;
- makes improvements easy to interpret relative to a fast baseline;
- aligns with a deployment-oriented portfolio story.

Trade-offs:

- the smallest model may have limited capacity;
- gains may saturate below YOLO26s quality.

## Evidence

At confidence 0.10 on the full MOT17-train benchmark:

- YOLO26n: Match 60.02%, IoU 0.790, edge error 5.97%, roughly 49–50 FPS.
- YOLO26s: Match 63.23%, IoU 0.798, edge error 5.66%, roughly 29 FPS.
- YOLO26m: Match 64.69%, IoU 0.795, edge error 5.70%, roughly 15 FPS.

The threshold sweep continued to show a substantial speed advantage for YOLO26n.

## Revisit when

Revisit if supervised fine-tuning shows that YOLO26n cannot close enough of the quality gap to justify further student experiments.
