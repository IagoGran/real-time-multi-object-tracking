# ADR-005: Use bounding-box center as the trajectory point

## Status

Accepted

## Date

2026-10

## Context

The trajectory layer needs one representative point per tracked bounding box. Possible choices include the geometric center, bottom-center / foot point, or another task-specific anchor.

The initial project goal is generic person trajectory visualization and line-crossing, not precise ground-plane reconstruction.

## Decision

Use the geometric center of the bounding box as the trajectory point.

## Alternatives considered

### Bottom-center / foot point

Potentially better for ground-plane movement and crossing physical floor boundaries, but more sensitive to box truncation and assumptions about camera geometry.

### Full-box trajectory representation

More complete but unnecessarily complex for the initial visualization and counting milestone.

## Consequences

Positive:

- simple and deterministic;
- independent of pose estimation;
- works consistently with the existing bounding-box representation.

Trade-offs:

- center motion may not correspond exactly to the person's contact point with the ground;
- perspective-heavy scenes may benefit from a bottom-center representation later.

## Evidence

`TrajectoryManager` stores points derived from the center of each tracked bounding box.

## Revisit when

Revisit if line-crossing errors appear to be caused by the center anchor or if the project adds ground-plane calibration.
