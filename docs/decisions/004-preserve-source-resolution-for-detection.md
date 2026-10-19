# ADR-004: Preserve source resolution for detection and align to model stride

## Status

Accepted

## Date

2026-10

## Context

MOT17 sequences use different source resolutions. Early experiments showed that detector localization quality matters for downstream tracking, so silently forcing a generic square inference size would make the geometry experiment harder to interpret.

YOLO requires dimensions compatible with its stride.

## Decision

Derive inference size from the source frame dimensions and round each dimension upward to the detector stride rather than replacing the source geometry with an arbitrary square size.

For example, a 1920×1080 frame is evaluated at a stride-compatible size of 1920×1088.

## Alternatives considered

### Fixed square inference size

Simpler and common in demos, but changes the scaling regime and makes localization comparisons less directly tied to the original frame geometry.

### Manually configured size per sequence

Works, but duplicates information already available from the frame itself.

## Consequences

Positive:

- consistent treatment of different MOT17 resolutions;
- preserves source aspect ratio;
- makes pixel-level geometric errors easier to interpret.

Trade-offs:

- high-resolution sequences are more expensive;
- runtime differs substantially between sequences.

## Evidence

The detector implementation computes inference dimensions dynamically from each frame and a stride of 32.

## Outcome

Runtime differences between sequences became important enough that benchmark runtime estimation was later made sequence-aware.

## Revisit when

Reconsider if deployment hardware requires a fixed inference size or if a dedicated resize study shows a better quality/performance operating point.
