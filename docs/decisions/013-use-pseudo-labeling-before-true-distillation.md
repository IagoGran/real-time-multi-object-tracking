# ADR-013: Test teacher pseudo-labeling before true knowledge distillation

## Status

Proposed

## Date

2026-10

## Context

A later project milestone aims to transfer useful information from a larger detector to YOLO26n.

There are several possible teacher/student strategies, ranging from simple pseudo-labeling to true knowledge distillation using logits or intermediate features.

## Decision

Test teacher/student learning first through pseudo-labeling:

1. run a configurable teacher over training images once;
2. filter predictions by a configurable teacher confidence threshold;
3. cache the pseudo-labels;
4. fine-tune YOLO26n on those labels;
5. compare the result against both the baseline and supervised fine-tuning.

Do not call this true knowledge distillation.

## Alternatives considered

### Logit distillation

Potentially more expressive, but requires custom loss/integration work before the simpler hypothesis has been tested.

### Feature distillation

Even more invasive and architecture-dependent.

### Teacher ensemble

Potentially stronger pseudo-labels but complicates attribution and experimental interpretation.

## Consequences

Positive:

- simple, reproducible teacher/student baseline;
- teacher inference is performed once rather than every epoch;
- results can be compared cleanly against supervised fine-tuning.

Trade-offs:

- student can inherit teacher errors;
- pseudo-label quality depends strongly on teacher selection and confidence filtering;
- may provide less information than true distillation.

## Evidence

This decision emerged after detector benchmarks showed that larger models are not uniformly superior across all metrics. The teacher therefore remains configurable rather than assuming YOLO26m is automatically best.

## Revisit when

Move to true distillation only if pseudo-labeling provides evidence that teacher information is useful and the expected benefit justifies the additional implementation complexity.
