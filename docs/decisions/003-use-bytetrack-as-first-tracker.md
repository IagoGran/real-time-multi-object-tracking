# ADR-003: Use ByteTrack as the first tracker

## Status

Accepted

## Date

2026-10

## Context

After establishing a pedestrian detector, the next milestone required persistent identities across frames with reasonable real-time performance.

The project needed a tracker that could operate on detector outputs without introducing a second heavy neural model into the first implementation.

## Decision

Use ByteTrack as the first tracking algorithm and wrap it behind the project's tracking layer.

## Alternatives considered

### BoT-SORT

Kept as a later comparison target. It may improve association in difficult cases, but is not required to establish the first end-to-end tracking baseline.

### Custom centroid-only tracker

Simpler, but less representative of a serious modern MOT pipeline and more fragile in crowded scenes and short occlusions.

## Consequences

Positive:

- lightweight association relative to adding another deep network;
- integrates naturally with detector confidence scores;
- provides a realistic baseline for later tracker comparison.

Trade-offs:

- tracking quality depends strongly on detector quality and confidence handling;
- long occlusions can still break identity continuity;
- ByteTrack-specific confidence behavior must not be confused with detector-evaluation thresholds.

## Evidence

The project has a ByteTrack wrapper and has verified ID persistence across short gaps.

## Revisit when

Compare with BoT-SORT once the detector benchmark and evaluation infrastructure are stable.
