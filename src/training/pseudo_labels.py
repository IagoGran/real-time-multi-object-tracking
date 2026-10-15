import hashlib
import importlib.metadata
import json
from pathlib import Path
from typing import Any, Iterable

from src.training.mot17_training_dataset import PREPARATION_POLICY_VERSION


TEACHER_LABEL_POLICY_VERSION = "raw_teacher_person_predictions_at_threshold_v1"
TEACHER_LABEL_POLICY_DESCRIPTION = (
    "Keep every teacher person detection at or above the confidence threshold. "
    "Do not suppress detections by overlap with MOT GT classes or occluders. "
    "Teacher boxes on ambiguous objects can therefore become noisy pseudo-labels."
)


def _as_list(values: Any) -> list[Any]:
    if hasattr(values, "tolist"):
        return values.tolist()
    return list(values)


def _class_name(names: Any, class_id: int) -> str:
    if isinstance(names, dict):
        return str(names.get(class_id, names.get(str(class_id), "")))
    if isinstance(names, (list, tuple)) and 0 <= class_id < len(names):
        return str(names[class_id])
    return ""


def _cache_signature(
    teacher_weights: str,
    confidence_threshold: float,
    train_images: Iterable[Path],
    train_sequences: Iterable[str] = (),
    detector_version: str | None = None,
    preparation_policy_version: str = PREPARATION_POLICY_VERSION,
) -> str:
    weight_path = Path(teacher_weights)
    weight_info: dict[str, Any] = {"value": teacher_weights}
    if weight_path.is_file():
        stat = weight_path.stat()
        digest_builder = hashlib.sha256()
        with weight_path.open("rb") as weight_file:
            for chunk in iter(lambda: weight_file.read(1024 * 1024), b""):
                digest_builder.update(chunk)
        digest = digest_builder.hexdigest()
        weight_info.update({"size": stat.st_size, "mtime_ns": stat.st_mtime_ns, "sha256": digest})
    if detector_version is None:
        try:
            detector_version = importlib.metadata.version("ultralytics")
        except importlib.metadata.PackageNotFoundError:
            detector_version = None
    payload = {
        "teacher": weight_info,
        "confidence_threshold": confidence_threshold,
        "train_sequences": list(train_sequences),
        "preparation_policy_version": preparation_policy_version,
        "teacher_label_policy_version": TEACHER_LABEL_POLICY_VERSION,
        "detector": {"ultralytics_version": detector_version, "predict_confidence_floor": min(0.001, confidence_threshold), "class": "person"},
        "images": [
            {
                "path": str(path.resolve()),
                "size": path.stat().st_size if path.exists() else None,
                "mtime_ns": path.stat().st_mtime_ns if path.exists() else None,
            }
            for path in train_images
        ],
    }
    encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def generate_pseudo_labels(
    model: Any,
    train_images: Iterable[Path],
    teacher_weights: str,
    confidence_threshold: float,
    cache_root: Path,
    train_sequences: Iterable[str] = (),
    preparation_policy_version: str = PREPARATION_POLICY_VERSION,
) -> tuple[dict[str, list[tuple[float, float, float, float]]], Path, bool, dict[str, Any]]:
    """Run one streamed teacher prediction pass and cache accepted person boxes."""
    if not 0.0 <= confidence_threshold <= 1.0:
        raise ValueError("teacher confidence must be between 0 and 1")
    images = tuple(Path(path) for path in train_images)
    if not images:
        raise ValueError("Pseudo-label generation requires training images.")

    sequence_names = tuple(train_sequences)
    signature = _cache_signature(teacher_weights, confidence_threshold, images, sequence_names,
                                 preparation_policy_version=preparation_policy_version)
    cache_dir = cache_root / signature
    labels_path = cache_dir / "labels.json"
    manifest_path = cache_dir / "manifest.json"
    if labels_path.is_file() and manifest_path.is_file():
        with labels_path.open(encoding="utf-8") as file:
            raw = json.load(file)
        labels = {
            key: [tuple(map(float, box)) for box in boxes]
            for key, boxes in raw.items()
        }
        stats = json.loads(manifest_path.read_text(encoding="utf-8")).get("stats", _label_statistics(labels, len(images)))
        return labels, cache_dir, True, stats

    # Keep the model's own pre-filter low so the explicitly requested teacher
    # threshold is the single acceptance rule below.
    predictions = model.predict(
        source=[str(path) for path in images],
        conf=min(0.001, confidence_threshold),
        verbose=False,
        stream=True,
    )
    image_by_path = {str(path.resolve()): path for path in images}
    labels: dict[str, list[tuple[float, float, float, float]]] = {}
    accepted_confidences: list[float] = []
    processed_images = 0
    result_iterator = iter(predictions)
    for fallback_image in images:
        try:
            result = next(result_iterator)
        except StopIteration:
            break
        processed_images += 1
        result_path = getattr(result, "path", None)
        image_path = image_by_path.get(str(Path(result_path).resolve())) if result_path else fallback_image
        if image_path is None:
            image_path = fallback_image
        key = str(image_path.resolve())
        labels.setdefault(key, [])

        boxes = getattr(result, "boxes", None)
        if boxes is None:
            continue
        xyxy = _as_list(boxes.xyxy)
        box_confidences = _as_list(boxes.conf)
        class_ids = _as_list(boxes.cls)
        names = getattr(result, "names", getattr(model, "names", {}))
        for box, confidence, raw_class_id in zip(xyxy, box_confidences, class_ids):
            class_id = int(raw_class_id)
            if _class_name(names, class_id) != "person":
                continue
            if float(confidence) < confidence_threshold:
                continue
            labels[key].append(tuple(float(coord) for coord in box))
            accepted_confidences.append(float(confidence))

    stats = _label_statistics(labels, processed_images, accepted_confidences)

    cache_dir.mkdir(parents=True, exist_ok=True)
    labels_path.write_text(json.dumps(labels, indent=2) + "\n", encoding="utf-8")
    manifest_path.write_text(
        json.dumps(
            {
                "format_version": 2,
                "teacher_weights": teacher_weights,
                "confidence_threshold": confidence_threshold,
                "image_count": len(images),
                "train_sequences": list(sequence_names),
                "preparation_policy_version": preparation_policy_version,
                "teacher_label_policy_version": TEACHER_LABEL_POLICY_VERSION,
                "teacher_label_policy": TEACHER_LABEL_POLICY_DESCRIPTION,
                "stats": stats,
                "signature": signature,
            },
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )
    return labels, cache_dir, False, stats


def _label_statistics(labels: dict[str, list[Any]], image_count: int, confidences: list[float] | None = None) -> dict[str, Any]:
    import statistics

    confidence_values = sorted(confidences or [])
    def percentile(fraction: float) -> float | None:
        if not confidence_values:
            return None
        return confidence_values[round((len(confidence_values) - 1) * fraction)]
    label_count = sum(len(boxes) for boxes in labels.values())
    labeled_images = sum(bool(boxes) for boxes in labels.values())
    return {
        "total_pseudo_labels": label_count,
        "images_processed": image_count,
        "images_with_pseudo_labels": labeled_images,
        "images_without_pseudo_labels": image_count - labeled_images,
        "mean_labels_per_image": label_count / image_count if image_count else 0.0,
        "mean_confidence": statistics.fmean(confidence_values) if confidence_values else None,
        "median_confidence": statistics.median(confidence_values) if confidence_values else None,
        "confidence_percentiles": {"p10": percentile(0.10), "p50": percentile(0.50), "p90": percentile(0.90)},
    }
