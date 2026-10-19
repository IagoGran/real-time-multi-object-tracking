"""Diagnose duplicate-like YOLO detections on MOT17 validation sequences.

This is an inference-only report. It does not train or edit model weights.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import statistics
import sys

import cv2
import numpy as np
import torch
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.mot17_dataset import load_sequence


SEQUENCES = [
    ROOT / "datasets/MOT17/train/MOT17-04-FRCNN",
    ROOT / "datasets/MOT17/train/MOT17-11-FRCNN",
]
WEIGHTS = {
    "YOLO26n_original": ROOT / "yolo26n.pt",
    "YOLO26n_finetuned": ROOT / "results/training/yolo26n_supervised_20261015_153159_ab7d8745/weights/best.pt",
}
THRESHOLDS = (0.10, 0.25, 0.30, 0.50)
PAIR_THRESHOLDS = (0.5, 0.7, 0.9)
INFERENCE_CONF = 0.01
NMS_IOU = 0.7
MAX_DET = 300


def confidence_key(value: float) -> str:
    return f"{value:.2f}"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def box_ious(boxes: np.ndarray, other: np.ndarray) -> np.ndarray:
    if not len(boxes) or not len(other):
        return np.zeros((len(boxes), len(other)), dtype=np.float32)
    tl = np.maximum(boxes[:, None, :2], other[None, :, :2])
    br = np.minimum(boxes[:, None, 2:], other[None, :, 2:])
    wh = np.maximum(0, br - tl)
    inter = wh[..., 0] * wh[..., 1]
    area_a = np.maximum(0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0, boxes[:, 3] - boxes[:, 1])
    area_b = np.maximum(0, other[:, 2] - other[:, 0]) * np.maximum(0, other[:, 3] - other[:, 1])
    return inter / np.maximum(area_a[:, None] + area_b[None, :] - inter, 1e-12)


def empty_metric(include_confidence_buckets: bool = True) -> dict:
    metric = {
        "frames": 0, "detections_by_confidence": {confidence_key(x): 0 for x in THRESHOLDS},
        "same_class_prediction_pairs_iou_gt": {str(x): 0 for x in PAIR_THRESHOLDS},
        "pairs_associated_to_same_valid_gt_iou_gt": {str(x): 0 for x in PAIR_THRESHOLDS},
        "pairs_associated_to_different_valid_gt_iou_gt": {str(x): 0 for x in PAIR_THRESHOLDS},
        "valid_gt_boxes": 0, "gt_boxes_with_2plus_predictions_at_iou_0_5": 0,
        "extra_predictions_associated_to_valid_gt_at_iou_0_5": 0,
        "all_predictions_associated_to_repeated_valid_gt": 0,
        "predictions_overlapping_ignored_gt_only_at_iou_0_5": 0,
        "predictions_with_no_valid_or_ignored_gt_iou_0_5": 0,
        "unique_matches": 0, "unmatched_predictions_after_unique_matching": 0,
        "unmatched_valid_gt": 0, "pair_confidences": [], "duplicate_extra_confidences": [],
        "unique_match_ious": [], "unique_match_center_error_px": [],
        "unique_match_edge_error_px": [], "frame_anomalies": [],
    }
    if include_confidence_buckets:
        metric["by_confidence"] = {confidence_key(t): empty_metric(False) for t in THRESHOLDS}
    return metric


def analyze_frame(boxes: np.ndarray, scores: np.ndarray, classes: np.ndarray,
                  valid_gt: list, ignored_gt: list) -> dict:
    valid_arr = np.asarray(valid_gt, dtype=np.float32).reshape(-1, 4)
    ignored_arr = np.asarray(ignored_gt, dtype=np.float32).reshape(-1, 4)
    overlaps = box_ious(boxes, boxes)
    same_class = classes[:, None] == classes[None, :]
    upper = np.triu(np.ones(overlaps.shape, dtype=bool), 1)
    pair_masks = {t: upper & same_class & (overlaps > t) for t in PAIR_THRESHOLDS}

    vg = box_ious(boxes, valid_arr)
    ig = box_ious(boxes, ignored_arr)
    best_valid = vg.argmax(axis=1) if vg.shape[1] else np.full(len(boxes), -1)
    best_valid_iou = vg.max(axis=1) if vg.shape[1] else np.zeros(len(boxes))
    associated_valid = np.where(best_valid_iou >= 0.5, best_valid, -1)
    counts = Counter(int(i) for i in associated_valid if i >= 0)
    repeated = {g for g, n in counts.items() if n >= 2}
    pair_same_gt = {}
    pair_different_gt = {}
    for t, mask in pair_masks.items():
        both_valid = (associated_valid[:, None] >= 0) & (associated_valid[None, :] >= 0)
        pair_same_gt[t] = mask & both_valid & (associated_valid[:, None] == associated_valid[None, :])
        pair_different_gt[t] = mask & both_valid & (associated_valid[:, None] != associated_valid[None, :])

    # Greedy one-to-one match, mirroring the repository benchmark matcher.
    candidates = [(float(vg[p, g]), p, g) for p, g in zip(*np.where(vg >= 0.5))]
    candidates.sort(reverse=True)
    used_p, used_g, matches = set(), set(), []
    for iou, p, g in candidates:
        if p not in used_p and g not in used_g:
            used_p.add(p); used_g.add(g); matches.append((p, g, iou))
    ignored_max = ig.max(axis=1) if ig.shape[1] else np.zeros(len(boxes))
    ignored_only = (best_valid_iou < 0.5) & (ignored_max >= 0.5)
    no_gt = (best_valid_iou < 0.5) & (ignored_max < 0.5)
    match_iou = []
    center_err, edge_err = [], []
    for p, g, iou in matches:
        match_iou.append(iou)
        pb, gb = boxes[p], valid_arr[g]
        pc = (pb[:2] + pb[2:]) / 2
        gc = (gb[:2] + gb[2:]) / 2
        center_err.append(float(np.linalg.norm(pc - gc)))
        edge_err.append(float(np.abs(pb - gb).mean()))
    pair_union = pair_masks[0.5]
    dup_pair_idx = np.where(pair_same_gt[0.5])
    duplicate_scores = [float(scores[i]) for i in set(dup_pair_idx[0]) | set(dup_pair_idx[1])]
    # Extra members of repeated GT groups, retaining confidence-ranked leading box.
    extra_scores = []
    for g in repeated:
        members = np.where(associated_valid == g)[0]
        ordered = sorted(members.tolist(), key=lambda i: float(scores[i]), reverse=True)
        extra_scores.extend(float(scores[i]) for i in ordered[1:])
    return {
        "pair_counts": {str(t): int(mask.sum()) for t, mask in pair_masks.items()},
        "same_gt_pair_counts": {str(t): int(mask.sum()) for t, mask in pair_same_gt.items()},
        "different_gt_pair_counts": {str(t): int(mask.sum()) for t, mask in pair_different_gt.items()},
        "repeated_gt_count": len(repeated),
        "extra_duplicate_count": sum(n - 1 for n in counts.values() if n >= 2),
        "repeated_prediction_count": sum(n for n in counts.values() if n >= 2),
        "ignored_only_count": int(ignored_only.sum()), "no_gt_count": int(no_gt.sum()),
        "unique_match_count": len(matches), "unique_unmatched_prediction_count": len(boxes) - len(used_p),
        "unmatched_gt_count": len(valid_gt) - len(used_g), "pair_confidences": duplicate_scores,
        "duplicate_extra_confidences": extra_scores, "match_ious": match_iou,
        "center_errors": center_err, "edge_errors": edge_err,
        "overlapping_pair_count_05": int(pair_union.sum()),
        "boxes": boxes, "scores": scores, "valid_gt": valid_arr, "ignored_gt": ignored_arr,
        "associated_valid": associated_valid,
    }


def add_frame(acc: dict, frame_info: dict, scores: np.ndarray, threshold_counts: dict,
              threshold_info: dict) -> None:
    acc["frames"] += 1
    acc["valid_gt_boxes"] += len(frame_info["valid_gt"])
    for threshold, count in threshold_counts.items():
        acc["detections_by_confidence"][str(threshold)] += count
    for key, values in (("same_class_prediction_pairs_iou_gt", frame_info["pair_counts"]),
                        ("pairs_associated_to_same_valid_gt_iou_gt", frame_info["same_gt_pair_counts"]),
                        ("pairs_associated_to_different_valid_gt_iou_gt", frame_info["different_gt_pair_counts"])):
        for t, count in values.items():
            acc[key][t] += count
    for key, source in (("gt_boxes_with_2plus_predictions_at_iou_0_5", "repeated_gt_count"),
                        ("extra_predictions_associated_to_valid_gt_at_iou_0_5", "extra_duplicate_count"),
                        ("all_predictions_associated_to_repeated_valid_gt", "repeated_prediction_count"),
                        ("predictions_overlapping_ignored_gt_only_at_iou_0_5", "ignored_only_count"),
                        ("predictions_with_no_valid_or_ignored_gt_iou_0_5", "no_gt_count"),
                        ("unique_matches", "unique_match_count"),
                        ("unmatched_predictions_after_unique_matching", "unique_unmatched_prediction_count"),
                        ("unmatched_valid_gt", "unmatched_gt_count")):
        acc[key] += frame_info[source]
    acc["pair_confidences"].extend(frame_info["pair_confidences"])
    acc["duplicate_extra_confidences"].extend(frame_info["duplicate_extra_confidences"])
    acc["unique_match_ious"].extend(frame_info["match_ious"])
    acc["unique_match_center_error_px"].extend(frame_info["center_errors"])
    acc["unique_match_edge_error_px"].extend(frame_info["edge_errors"])
    for threshold, info in threshold_info.items():
        selected_scores = scores[scores >= float(threshold)]
        nested_counts = {confidence_key(t): int(np.sum(selected_scores >= t)) for t in THRESHOLDS}
        add_frame(acc["by_confidence"][threshold], info, selected_scores, nested_counts, {})


def distribution(values: list[float]) -> dict:
    if not values:
        return {"count": 0, "mean": None, "median": None, "p90": None, "min": None, "max": None}
    return {"count": len(values), "mean": float(statistics.fmean(values)),
            "median": float(statistics.median(values)), "p90": float(np.percentile(values, 90)),
            "min": float(min(values)), "max": float(max(values))}


def finalize(acc: dict) -> dict:
    n = acc["frames"]
    out = {k: v for k, v in acc.items() if k not in {"pair_confidences", "duplicate_extra_confidences", "unique_match_ious", "unique_match_center_error_px", "unique_match_edge_error_px", "frame_anomalies", "by_confidence"}}
    out["detections_per_frame_by_confidence"] = {t: nval / n for t, nval in acc["detections_by_confidence"].items()}
    out["pair_rates_per_frame"] = {k: {t: c / n for t, c in v.items()} for k, v in (
        ("same_class_prediction_pairs_iou_gt", acc["same_class_prediction_pairs_iou_gt"]),
        ("pairs_associated_to_same_valid_gt_iou_gt", acc["pairs_associated_to_same_valid_gt_iou_gt"]),
        ("pairs_associated_to_different_valid_gt_iou_gt", acc["pairs_associated_to_different_valid_gt_iou_gt"]))}
    out["duplicate_candidate_confidence"] = distribution(acc["pair_confidences"])
    out["extra_duplicate_prediction_confidence"] = distribution(acc["duplicate_extra_confidences"])
    out["unique_match_iou"] = distribution(acc["unique_match_ious"])
    out["unique_match_center_error_px"] = distribution(acc["unique_match_center_error_px"])
    out["unique_match_edge_error_px"] = distribution(acc["unique_match_edge_error_px"])
    out["gt_boxes_with_2plus_predictions_rate"] = acc["gt_boxes_with_2plus_predictions_at_iou_0_5"] / max(1, acc["valid_gt_boxes"])
    out["ignored_only_predictions_per_frame"] = acc["predictions_overlapping_ignored_gt_only_at_iou_0_5"] / max(1, n)
    out["no_valid_or_ignored_gt_overlap_predictions_per_frame"] = acc["predictions_with_no_valid_or_ignored_gt_iou_0_5"] / max(1, n)
    out["unmatched_predictions_per_frame"] = acc["unmatched_predictions_after_unique_matching"] / max(1, n)
    if "by_confidence" in acc:
        out["by_confidence"] = {threshold: finalize(values) for threshold, values in acc["by_confidence"].items()}
    return out


def merge_accumulator(target: dict, source: dict) -> None:
    for key, value in source.items():
        if key == "by_confidence":
            for threshold, metric in value.items():
                merge_accumulator(target[key][threshold], metric)
        elif isinstance(value, list):
            target[key].extend(value)
        elif isinstance(value, dict):
            for subkey, subvalue in value.items():
                target[key][subkey] += subvalue
        elif isinstance(value, (int, float)):
            target[key] += value


def draw_example(path: Path, detections: dict, out_path: Path, title: str) -> None:
    img = cv2.imread(str(path))
    valid, ignored = detections["valid_gt"], detections["ignored_gt"]
    associated = detections["associated_valid"]
    scores, boxes = detections["scores"], detections["boxes"]
    counts = Counter(int(x) for x in associated if x >= 0)
    ignored_overlap = box_ious(boxes, ignored) if len(ignored) else np.zeros((len(boxes), 0))
    relevant_ignored = set(np.where((ignored_overlap >= 0.5).any(axis=0))[0]) if len(ignored) else set()
    for i, (x1, y1, x2, y2) in enumerate(ignored):
        if i not in relevant_ignored:
            continue
        p1, p2 = (round(x1), round(y1)), (round(x2), round(y2))
        cv2.rectangle(img, p1, p2, (0, 165, 255), 1)
    for i, (x1, y1, x2, y2) in enumerate(valid):
        p1, p2 = (round(x1), round(y1)), (round(x2), round(y2))
        cv2.rectangle(img, p1, p2, (0, 220, 0), 2)
    for i, (x1, y1, x2, y2) in enumerate(boxes):
        gt = int(associated[i])
        color = (0, 0, 255) if gt >= 0 and counts[gt] > 1 else ((255, 255, 0) if gt >= 0 else (255, 0, 255))
        p1, p2 = (round(x1), round(y1)), (round(x2), round(y2))
        cv2.rectangle(img, p1, p2, color, 2)
        if gt >= 0 and counts[gt] > 1:
            cv2.putText(img, f"dup {scores[i]:.2f}", (p1[0], min(img.shape[0] - 3, p1[1] + 13)), cv2.FONT_HERSHEY_SIMPLEX, .42, color, 1)
    legend = [("GT válido", (0, 220, 0)), ("GT ignorado", (0, 165, 255)),
              ("Pred. duplicada", (0, 0, 255)), ("Pred. asociada", (255, 255, 0)),
              ("Pred. sin match", (255, 0, 255))]
    cv2.rectangle(img, (0, 0), (img.shape[1], 62), (20, 20, 20), -1)
    cv2.putText(img, title, (12, 23), cv2.FONT_HERSHEY_SIMPLEX, .58, (255, 255, 255), 1)
    x = 12
    for label, color in legend:
        cv2.rectangle(img, (x, 39), (x + 16, 54), color, -1)
        cv2.putText(img, label, (x + 21, 52), cv2.FONT_HERSHEY_SIMPLEX, .36, (255, 255, 255), 1)
        x += 150
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), img)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "results/duplicate_diagnostics")
    parser.add_argument("--max-frames", type=int, default=None, help="Optional debugging limit; omit for full validation.")
    parser.add_argument("--nms-mode", choices=("none", "false"), default="none", help="Ultralytics nms argument, explicitly recorded.")
    args = parser.parse_args()
    seqs = [load_sequence(p, max_frames=args.max_frames) for p in SEQUENCES]
    out_dir = args.output.resolve()
    nms_arg = None if args.nms_mode == "none" else False
    images_dir = out_dir / "examples"
    out_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "report_version": 1, "created_at": datetime.now(timezone.utc).isoformat(),
        "scope": {"sequences": [s.name for s in seqs], "frames_by_sequence": {s.name: len(s.frames) for s in seqs},
                  "thresholds": list(THRESHOLDS), "input_confidence_floor": INFERENCE_CONF,
                  "analysis_is_inference_only": True, "training_performed": False, "test_set_accessed": False},
        "runtime": {"ultralytics": importlib.metadata.version("ultralytics"), "torch": torch.__version__,
                    "cuda_available": torch.cuda.is_available(), "device": "cuda:0" if torch.cuda.is_available() else "cpu"},
        "definitions": {
            "class_filter": "class_id=0 (person), applied after Ultralytics postprocessing, matching YoloDetector.class_names=['person']",
            "valid_ground_truth": "MOT rows with mark=1 and class=1; all other GT rows are retained only as ignored annotations",
            "pair_counts": "unordered same-class pairs with strict IoU > threshold; all candidate pairs are counted",
            "same_gt_duplicate": "predictions whose best valid-GT IoU is >=0.5 and whose best GT index is identical; excess count sums n-1 per GT",
            "different_gt_overlap": "overlapping pairs at threshold whose members associate at IoU >=0.5 to different valid GT boxes",
            "unique_match": "greedy one-to-one highest-IoU assignment at IoU >=0.5, as in repository benchmark matcher",
            "ignored_only": "prediction with no valid-GT IoU >=0.5 but ignored-annotation IoU >=0.5; categories retained in report",
            "unmatched_warning": "unique unmatched prediction totals include duplicates and ignored overlaps; they are not false positives",
            "nms_none_route": "Ultralytics DetectionPredictor passes end2end=model.end2end; for these weights head_end2end=false, so standard class-aware NMS uses iou",
            "nms_false_route": "Ultralytics Predictor sets AutoBackend end2end=true; non_max_suppression bypasses IoU NMS and retains confidence-filtered rows up to max_det",
        },
        "inference_configuration": {"nms_argument": nms_arg, "nms_false_control": "compare first validation frame per sequence and model",
                                    "iou": NMS_IOU, "max_det": MAX_DET, "conf": INFERENCE_CONF,
                                    "rect": True, "benchmark_native_size_policy": "(ceil(H/32)*32, ceil(W/32)*32) as in YoloDetector"},
        "models": {}, "nms_control": [], "comparison": {},
    }
    all_configs = {}
    for model_name, weight_path in WEIGHTS.items():
        model = YOLO(str(weight_path))
        if model.names.get(0) != "person":
            raise RuntimeError(f"Expected class_id=0 to be person for {model_name}, got {model.names.get(0)!r}")
        weights_info = {"path": str(weight_path.relative_to(ROOT)), "sha256": sha256(weight_path)}
        model_entry = {"weights": weights_info, "class_id_0_name": model.names[0],
                       "head_end2end": bool(getattr(model.model.model[-1], "end2end", False)), "sizes": {}}
        result["models"][model_name] = model_entry
        # Compare nms=None with explicitly requested nms=False on the same control frames.
        for seq in seqs:
            fr = seq.frames[0]
            image = cv2.imread(str(fr.path))
            hh, ww = image.shape[:2]
            native_size = (math.ceil(hh / 32) * 32, math.ceil(ww / 32) * 32)
            a = model.predict(image, imgsz=native_size, conf=INFERENCE_CONF, iou=NMS_IOU, max_det=MAX_DET,
                              nms=None, verbose=False)[0].boxes.data.detach().cpu().numpy()
            b = model.predict(image, imgsz=native_size, conf=INFERENCE_CONF, iou=NMS_IOU, max_det=MAX_DET,
                              nms=False, verbose=False)[0].boxes.data.detach().cpu().numpy()
            a = a[a[:, 5].astype(np.int32) == 0] if len(a) else a
            b = b[b[:, 5].astype(np.int32) == 0] if len(b) else b
            same = len(a) == len(b) and (not len(a) or np.allclose(a, b, rtol=0, atol=1e-4))
            result["nms_control"].append({"model": model_name, "sequence": seq.name, "frame": fr.number,
                                          "nms_none_detections": int(len(a)), "nms_false_detections": int(len(b)),
                                          "outputs_equal_within_1e-4": bool(same), "imgsz": list(native_size), "iou": NMS_IOU, "max_det": MAX_DET})
        for size_name in ("imgsz_640", "benchmark_native"):
            acc_total = empty_metric()
            seq_reports = {}
            seq_categories = {}
            anomaly_candidates = []
            for seq in seqs:
                acc = empty_metric()
                gt_details = {}
                categories = Counter()
                for frame_ref in seq.frames:
                    img = cv2.imread(str(frame_ref.path))
                    if img is None:
                        raise RuntimeError(f"Cannot read {frame_ref.path}")
                    h, w = img.shape[:2]
                    inference_size = 640 if size_name == "imgsz_640" else (math.ceil(h / 32) * 32, math.ceil(w / 32) * 32)
                    # Ground truth is loaded once per sequence below.
                    if not gt_details:
                        with (seq.path / "gt/gt.txt").open() as f:
                            rows = defaultdict(lambda: ([], [], Counter()))
                            for line in f:
                                row = line.strip().split(",")
                                if len(row) < 9: continue
                                fn = int(float(row[0])); x,y,bw,bh = map(float,row[2:6]); mark,cls = int(float(row[6])),int(float(row[7]))
                                box=(x,y,x+bw,y+bh)
                                if mark == 1 and cls == 1: rows[fn][0].append(box)
                                else: rows[fn][1].append(box); rows[fn][2][f"mark={mark},class={cls}"] += 1
                        gt_details = rows
                    valid_gt, ignored_gt, cat = gt_details[frame_ref.number]
                    categories.update(cat)
                    pred = model.predict(img, imgsz=inference_size, conf=INFERENCE_CONF, iou=NMS_IOU,
                                         max_det=MAX_DET, nms=nms_arg, verbose=False)[0]
                    raw = pred.boxes.data.detach().cpu().numpy()
                    # Match YoloDetector: Ultralytics postprocessing runs first, then class_names filters person.
                    raw = raw[raw[:, 5].astype(np.int32) == 0] if len(raw) else raw
                    boxes = raw[:, :4].astype(np.float32) if len(raw) else np.empty((0,4),np.float32)
                    scores = raw[:, 4].astype(np.float32) if len(raw) else np.empty((0,),np.float32)
                    classes = raw[:, 5].astype(np.int32) if len(raw) else np.empty((0,),np.int32)
                    info = analyze_frame(boxes, scores, classes, valid_gt, ignored_gt)
                    threshold_counts = {confidence_key(t): int(np.sum(scores >= t)) for t in THRESHOLDS}
                    threshold_info = {}
                    for threshold in THRESHOLDS:
                        mask = scores >= threshold
                        threshold_info[confidence_key(threshold)] = analyze_frame(boxes[mask], scores[mask], classes[mask], valid_gt, ignored_gt)
                    add_frame(acc, info, scores, threshold_counts, threshold_info)
                    at_025 = scores >= 0.25
                    info_025 = analyze_frame(boxes[at_025], scores[at_025], classes[at_025], valid_gt, ignored_gt)
                    anomaly_candidates.append((info_025["extra_duplicate_count"], info_025["overlapping_pair_count_05"],
                                               float(scores[at_025].mean()) if at_025.any() else 0.0, seq.name, frame_ref.number,
                                               frame_ref.path, info_025, size_name))
                    if acc["frames"] % 250 == 0:
                        print(f"{model_name} {size_name} {seq.name}: {acc['frames']}/{len(seq.frames)} frames", flush=True)
                seq_categories[seq.name] = dict(categories)
                seq_reports[seq.name] = finalize(acc)
                merge_accumulator(acc_total, acc)
                del gt_details
            # Keep example pictures at the largest anomaly frames, ranked first by same-GT extras then overlap pairs.
            examples = []
            anomaly_candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
            used_frames = set()
            for rank, cand in enumerate(anomaly_candidates):
                extra, pairs, _, seq_name, frame_no, frame_path, info, _ = cand
                if (seq_name, frame_no) in used_frames: continue
                used_frames.add((seq_name, frame_no))
                if rank > 15 or len(examples) >= 4: break
                ep = images_dir / model_name / size_name / f"{seq_name}_frame_{frame_no:06d}.jpg"
                draw_example(frame_path, info, ep, f"{model_name} {size_name} {seq_name} frame {frame_no} | conf>=0.25 extras={extra} pairs>.5={pairs}")
                examples.append({"sequence": seq_name, "frame": frame_no, "path": str(ep.relative_to(ROOT)),
                                 "extra_predictions_same_gt": extra, "overlapping_prediction_pairs_iou_gt_0_5": pairs})
            model_entry["sizes"][size_name] = {"aggregated": finalize(acc_total), "sequences": seq_reports,
                                               "ignored_gt_annotation_categories_by_sequence": seq_categories,
                                               "examples": examples,
                                               "frames_with_gt": sum(seq_reports[s.name]["valid_gt_boxes"] for s in seqs)}
            all_configs[(model_name, size_name)] = model_entry["sizes"][size_name]["aggregated"]
        del model
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    # Explicitly report changes in counts across resolution settings.
    for model_name in WEIGHTS:
        small = all_configs[(model_name, "imgsz_640")]
        native = all_configs[(model_name, "benchmark_native")]
        result["comparison"][model_name] = {
            "native_minus_640_detections_per_frame": {t: native["detections_per_frame_by_confidence"][t] - small["detections_per_frame_by_confidence"][t] for t in small["detections_per_frame_by_confidence"]},
            "native_minus_640_same_gt_extras": native["extra_predictions_associated_to_valid_gt_at_iou_0_5"] - small["extra_predictions_associated_to_valid_gt_at_iou_0_5"],
        }
    result["inference_configuration"]["nms_argument_label"] = args.nms_mode
    output_path = out_dir / "duplicate_diagnostics.json"
    output_path.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
