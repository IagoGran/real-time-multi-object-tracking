# ADR-016: Retain MOT17 frames and label only target-class GT

## Status

Accepted for dataset preparation. Teacher ambiguity handling and official-protocol evaluation remain open.

## Context

The previous preparation policy discarded an entire image whenever its GT contained an annotation other than `mark=1, class=1`. The real-data audit reported that this removed 2,857/3,366 train frames (84.88%) and 1,950/1,950 frames from MOT17-04 validation, including 51,141 train and 56,993 validation pedestrian annotations according to the audit supplied for this decision. That policy made the prepared split unusable.

## Decision

- Retain every readable image from each configured sequence, including frames with no target pedestrian annotation.
- Emit YOLO pedestrian labels only from positive-area `mark=1, class=1` boxes, clipped to the source image as before.
- Keep a corresponding empty label file for every retained image with no valid target box.
- Never turn other MOT classes into pedestrian positives. Report their annotation counts separately from target-label losses.
- Fail preparation on unreadable images rather than silently dropping them.
- Version prepared artifacts as `all_frames_target_class_only_v2`; use a separate prepared-data directory and include this version in dataset and experiment manifests and the teacher cache identity.
- Do not add custom ignore-region loss, visual masking, or overlap-based pseudo-label suppression in this change.

## Trade-offs and limits

Ultralytics detection training receives only target pedestrian boxes. It has no ignore-region supervision in this pipeline, so ambiguous objects left unlabeled can be treated implicitly as background. This is a known limitation; no visual masks or custom loss are used.

Teacher pseudo-labeling retains all `person` detections above its configured confidence and does not consult MOT GT. A teacher detection on an ambiguous class can therefore become a noisy positive pseudo-label. Detections overlapping occluder classes 9–11 are not removed. Any future suppression rule needs a separately justified overlap protocol and evaluation.

The project evaluator currently loads only marked class-1 GT and counts every unmatched prediction as a false positive, including detections overlapping distractors or occluders. Its metrics are not equivalent to official MOT17 metrics. A separate future change should integrate or validate against the official MOTChallenge ignore/distractor protocol before making official-comparability claims.

## Validation

Unit tests cover mixed target and non-target annotations, retained frames, empty labels, split non-overlap, and policy-versioned manifests/cache identity. The real-data audit and prepare-only outputs must be compared when the configured MOT17 files are available.

## Revisit when

Revisit teacher pseudo-label filtering after measuring class-specific overlap and noise. Revisit ignored-region training only with a documented objective and a dedicated implementation proposal. Implement official-protocol detector evaluation independently from this preparation policy.
