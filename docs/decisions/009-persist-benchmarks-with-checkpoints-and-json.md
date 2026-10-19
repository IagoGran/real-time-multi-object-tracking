# ADR-009: Persist long benchmarks with logs, checkpoints and JSON results

## Status

Accepted

## Date

2026-10

## Context

A full MOT17 detector benchmark spans multiple models and sequences and can run long enough that interruption would otherwise waste substantial compute.

The benchmark also needs reproducible outputs that can later feed analysis and plotting.

## Decision

Add:

- structured logging to console and file;
- runtime estimation before long runs;
- progress and ETA logging;
- JSON checkpoints;
- atomic checkpoint writes through a temporary file followed by replacement;
- `--resume` support;
- checkpoint granularity of model × sequence.

## Alternatives considered

### Console output only

Insufficient for unattended runs and later auditing.

### Save only the final table

Too fragile if interrupted and not machine-readable enough for later analysis.

### Frame-level checkpointing

More resilient but adds unnecessary complexity and I/O for the current workload.

## Consequences

Positive:

- interrupted experiments can resume;
- completed model/sequence combinations are not rerun;
- results are machine-readable;
- logs make long experiments auditable.

Trade-offs:

- checkpoint schema becomes part of the experiment infrastructure;
- format changes require explicit versioning.

## Evidence

The evaluation subsystem includes checkpoint format versions, atomic JSON writes, configuration compatibility checks, per-sequence progress and aggregate results.

## Revisit when

Revisit checkpoint granularity if individual sequence runs become long enough that rerunning one sequence is expensive.
