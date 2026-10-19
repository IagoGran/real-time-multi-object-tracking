# ADR-015: Separate final evaluation JSON from operational checkpoints

## Status

Proposed

## Date

2026-10

## Context

The evaluation checkpoint contains operational state required for resuming long benchmarks. Future plotting and analysis scripts need a stable result format but should not depend on internal checkpoint details.

The project also plans to compare baseline, supervised and teacher-trained models across confidence thresholds and generate plots from those results.

## Decision

Keep the resumable checkpoint as an internal operational format and export a separate final evaluation JSON after successful completion.

The final JSON should expose stable, analysis-oriented data including:

- dataset / sequence configuration;
- evaluated models;
- confidence thresholds;
- matching threshold;
- aggregate metrics per threshold and model;
- per-sequence results;
- start/end timestamps.

Plotting code should consume this final format, not parse logs or checkpoint internals.

## Alternatives considered

### Use the checkpoint directly for plotting

Initially simpler, but tightly couples future analysis to resume implementation details.

### Parse log output

Rejected because logs are human-oriented and much more fragile than structured data.

### CSV only

Convenient for tables but less suitable for nested per-sequence metadata and experiment configuration.

## Consequences

Positive:

- stable contract for plotting and analysis;
- checkpoint schema can evolve independently;
- easier comparison of multiple experiments.

Trade-offs:

- introduces one more persisted artifact;
- export code must ensure it reuses existing aggregates rather than recalculating metrics differently.

## Evidence

The current benchmark already persists machine-readable checkpoints and aggregated metrics. The planned next step is to create plotting scripts for Match%, FP/frame, IoU, edge error, center error, FPS and quality-versus-speed relationships.

## Revisit when

Revisit if a later experiment registry or database replaces file-based result storage.
