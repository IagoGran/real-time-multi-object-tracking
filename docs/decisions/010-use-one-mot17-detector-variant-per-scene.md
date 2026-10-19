# ADR-010: Use one MOT17 detector variant per scene for detector evaluation

## Status

Accepted

## Date

2026-10

## Context

MOT17 provides DPM, FRCNN and SDP detector variants for the same underlying scene images. For detector evaluation using our own models, evaluating all three variants would repeat the same frames and ground truth three times.

## Decision

When evaluating the full MOT17 training set as an image dataset, select one variant per scene, currently the `*-FRCNN` directories.

The suffix is used only as a convenient unique-scene selector; the project's detector does not consume the provided FRCNN detections.

## Alternatives considered

### Evaluate all DPM/FRCNN/SDP directories

Rejected because it would triple-count identical scene imagery and distort aggregate detector metrics.

### Build a separate deduplicated image index independent of directory variant

Cleaner in theory, but unnecessary while selecting one consistent variant already guarantees one copy per scene.

## Consequences

Positive:

- each scene contributes once;
- aggregate metrics are not artificially weighted by duplicate directories;
- implementation remains simple.

Trade-offs:

- the directory name may misleadingly suggest that Faster R-CNN detections are being used even though only images/ground truth are consumed.

## Evidence

The full-train evaluator discovers `MOT17-*-FRCNN` sequences to represent the seven unique MOT17 train scenes.

## Revisit when

Revisit if the dataset loader is generalized beyond MOT17 or if an explicit canonical scene abstraction replaces directory-based discovery.
