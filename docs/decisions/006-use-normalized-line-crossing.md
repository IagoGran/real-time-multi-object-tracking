# ADR-006: Represent the counting line in normalized frame coordinates

## Status

Accepted

## Date

2026-10

## Context

MOT17 sequences have different resolutions. A counting line expressed directly in pixels would need to be redefined for every resolution.

## Decision

Represent counting-line endpoints in normalized frame coordinates and convert them to pixels for the current frame.

## Alternatives considered

### Pixel coordinates

Straightforward for a single fixed-resolution video, but not reusable across sequences or resolutions.

### Automatically inferred line placement

Out of scope for the initial analytics implementation and harder to reason about during validation.

## Consequences

Positive:

- resolution-independent configuration;
- easier reuse across sequences;
- clean separation between analytics configuration and rendering resolution.

Trade-offs:

- the same normalized line does not necessarily correspond to the same semantic location in different scenes;
- per-scene configuration may still be necessary.

## Evidence

The current line-crossing implementation uses a normalized `CountingLine` and finite-segment intersection logic.

## Revisit when

Revisit if evaluation moves from a demo line to scene-specific semantic zones or calibrated world coordinates.
