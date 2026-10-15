# ADR-007: Use geometric localization metrics in addition to coverage metrics

## Status

Accepted

## Date

2026-10

## Context

Standard detector metrics such as precision and recall are useful for measuring detection coverage, but this project is specifically interested in how detector bounding-box quality affects downstream multi-object tracking.

For tracking, two detectors with similar detection coverage may still behave differently if one produces boxes with less stable centers, inaccurate edges, or inconsistent sizes.

IoU is useful but does not fully describe these differences. Two boxes with similar IoU may have different center displacement, edge placement, or size error.

The detector evaluation therefore needs to distinguish between:

- whether a person was detected;
- how accurately the matched bounding box was localized.

## Decision

Evaluate matched detections using geometric localization metrics in addition to coverage metrics.

For matched prediction / ground-truth pairs, record:

- mean and median IoU;
- mean normalized edge error;
- median normalized edge error;
- P90 normalized edge error;
- mean edge error in pixels;
- center error in pixels;
- center error normalized by the ground-truth box diagonal;
- relative size error.

Use the following as coverage/context metrics:

- Match%;
- unmatched predictions / false positives per frame;
- false negatives where required.

Do not use precision/recall or IoU alone as the primary description of detector quality.

The purpose is not to replace standard detector metrics, but to expose geometric differences that may later help explain tracking behaviour.

## Alternatives considered

### Precision / recall only

Precision and recall describe detection coverage well, but do not characterize the geometry of correctly matched bounding boxes.

For this project that is insufficient because localization quality may influence association stability in the tracker.

### IoU only

IoU provides a standard measure of overlap and remains part of the benchmark.

However, a single IoU value does not explain whether disagreement is caused primarily by:

- center displacement;
- inaccurate edges;
- incorrect width or height.

### Pixel errors only

Pixel displacement is intuitive within a single sequence, but difficult to compare across people of different sizes and across sequences with different resolutions.

For this reason the benchmark keeps pixel errors for interpretability while also computing normalized errors.

## Consequences

Positive:

- detector coverage and localization quality can be analyzed separately;
- normalized metrics can be compared across object sizes and sequences;
- center error provides a metric directly relevant to the current trajectory representation;
- edge and size errors expose differences hidden by IoU alone;
- P90 highlights difficult detections that may be hidden by mean or median values;
- the resulting metrics provide a basis for later studying whether localization quality predicts tracking stability.

Trade-offs:

- custom metrics require explicit definitions and tests;
- additional metrics make model comparison more multidimensional;
- these metrics are complementary to, not replacements for, standard detection and MOT metrics;
- current Match% and unmatched-prediction results should not be presented as official MOT17 metrics until ignored regions and distractor handling are aligned with the appropriate MOT17 evaluation rules.

## Evidence

A full evaluation over the unique MOT17 training scenes was performed using the same detector predictions for a confidence-threshold sweep from 0.10 to 0.60.

At confidence threshold 0.10:

| Model | Match% | Mean IoU | Mean edge error | Center error | FP/frame |
|---|---:|---:|---:|---:|---:|
| YOLO26n | 66.92% | 0.780 | 6.32% | 4.61% | 12.52 |
| YOLO26s | 68.30% | 0.790 | 5.91% | 4.35% | 11.28 |
| YOLO26m | 69.06% | 0.790 | 5.90% | 4.42% | 10.33 |
| Faster R-CNN | 67.64% | 0.771 | 6.72% | 4.79% | 20.17 |

The threshold sweep also showed the expected relationship between confidence and localization quality.

For example, YOLO26s evolved as follows:

| Confidence | Match% | Mean IoU | Edge error | Center error | FP/frame |
|---|---:|---:|---:|---:|---:|
| 0.10 | 68.30% | 0.790 | 5.91% | 4.35% | 11.28 |
| 0.20 | 64.72% | 0.795 | 5.73% | 4.24% | 6.12 |
| 0.30 | 61.71% | 0.800 | 5.58% | 4.11% | 4.09 |
| 0.40 | 58.54% | 0.806 | 5.39% | 3.95% | 2.91 |
| 0.50 | 54.82% | 0.814 | 5.16% | 3.76% | 2.06 |
| 0.60 | 50.17% | 0.822 | 4.90% | 3.52% | 1.37 |

As confidence increases:

- Match% decreases;
- unmatched predictions decrease substantially;
- mean IoU increases;
- edge error decreases;
- center error decreases;
- size error also decreases.

This confirms that confidence is associated not only with detection coverage but also with bounding-box localization quality.

The comparison between YOLO26s and YOLO26m also illustrates why coverage and geometry should remain separate dimensions.

YOLO26m consistently achieves somewhat higher Match%, but YOLO26s produces equal or slightly better localization metrics through much of the threshold sweep.

For example, at confidence 0.50:

| Model | Match% | Mean IoU | Edge error | Center error | Size error |
|---|---:|---:|---:|---:|---:|
| YOLO26s | 54.82% | 0.814 | 5.16% | 3.76% | 8.18% |
| YOLO26m | 57.15% | 0.808 | 5.30% | 4.02% | 8.47% |

A ranking based only on detection coverage would therefore hide a meaningful localization difference.

Faster R-CNN provides another example. At low confidence it achieves high Match%, but with substantially more unmatched predictions and poorer geometric localization than the YOLO models.

## Outcome

The experiments support the original decision to treat detector quality as a multidimensional problem rather than reducing it to a single precision, recall, or IoU value.

The threshold sweep revealed three particularly relevant behaviours:

1. Higher-confidence detections are also geometrically better localized, not merely less frequent.
2. A model with higher coverage is not necessarily the model with the best bounding-box geometry.
3. Detector comparisons can change depending on whether the objective is coverage, localization quality, false-positive control, or runtime.

The first implementation of the confidence sweep also exposed an evaluation issue: YOLO's internal confidence filtering was discarding low-confidence predictions before the evaluator could apply thresholds.

The benchmark was subsequently changed so detector inference is performed once at a low confidence floor and the evaluation thresholds are applied afterwards to the same set of predictions.

Therefore, earlier 0.10/0.20 YOLO benchmark values obtained before that correction are considered superseded by the current sweep.

Inference timing from the current sweep should also be interpreted separately from deployment speed: the detector is intentionally run with a low internal confidence floor so that all evaluation thresholds can be derived from one inference pass.

## Revisit when

Revisit the metric set after tracking evaluation is implemented with metrics such as:

- IDF1;
- HOTA;
- MOTA;
- ID switches.

At that point, test whether edge error, center error, size error, or their higher percentiles actually correlate with downstream identity stability.

If they do not provide explanatory value for tracking performance, simplify the detector metric set accordingly.