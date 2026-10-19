# ADR-011: Split training and validation by complete sequences

## Status

Proposed

## Date

2026-10

## Context

The project is moving from evaluation-only experiments to fine-tuning detectors on MOT17.

Adjacent frames from a video sequence are strongly correlated. Randomly splitting frames from the same sequence between training and validation would allow near-duplicate temporal content to appear on both sides and inflate validation performance.

## Decision

Split training and validation by complete MOT17 sequences.

A sequence must belong entirely to either training or validation.

The split must be explicit, reproducible and stored as experiment configuration.

## Alternatives considered

### Random frame split

Rejected because of temporal leakage between highly similar neighboring frames.

### Train on all MOT17 train sequences and report the same set as validation

Useful only for fitting diagnostics, not for an independent estimate of generalization.

### External validation dataset

Potentially stronger, but adds a new domain before the first controlled training baseline exists.

## Consequences

Positive:

- substantially reduces temporal leakage;
- makes baseline / supervised / teacher comparisons more defensible;
- preserves a genuine hold-out.

Trade-offs:

- fewer sequences remain available for training;
- performance may depend on which complete scenes are held out.

## Evidence

This decision was made during planning of the first supervised and teacher/student training experiments. The exact sequence split has not yet been experimentally finalized.

## Revisit when

Revisit after the first training runs, especially if scene diversity is too limited for a stable hold-out. Cross-validation by sequence may later be appropriate.
