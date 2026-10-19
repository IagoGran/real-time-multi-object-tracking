# ADR-014: Keep supervised and teacher training as separate experiments

## Status

Proposed

## Date

2026-10

## Context

The first training phase will compare standard fine-tuning against teacher-generated pseudo-labels.

Combining ground truth and pseudo-labels immediately would make it difficult to attribute any improvement to one source.

## Decision

Run two clean experiments first:

- **supervised** — train only from MOT17 ground truth;
- **teacher** — train only from teacher pseudo-labels.

Do not automatically merge ground truth and teacher predictions in the first iteration.

## Alternatives considered

### Hybrid GT + pseudo-label training

Potentially stronger, but introduces policy questions about conflicts, duplicate boxes, weighting and trust that would obscure the first experiment.

### Teacher-only experiment without supervised baseline

Rejected because it would be impossible to tell whether any gain came from teacher information or simply from fine-tuning YOLO26n on the target domain.

## Consequences

Positive:

- clear causal comparison;
- easier interpretation of results;
- simpler dataset generation and training code.

Trade-offs:

- the best final training strategy may ultimately be hybrid;
- teacher-only pseudo-label training may intentionally ignore cleaner ground truth that is already available.

## Evidence

The planned experiment matrix explicitly compares baseline, supervised fine-tuned YOLO26n and teacher-trained YOLO26n before attempting hybrid approaches.

## Revisit when

After both experiments have been evaluated on the same held-out sequences. A hybrid training ADR should be created only if the evidence justifies it.
