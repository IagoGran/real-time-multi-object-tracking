import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


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
) -> str:
    weight_path = Path(teacher_weights)
    weight_info: dict[str, Any] = {"value": teacher_weights}
    if weight_path.is_file():
        stat = weight_path.stat()
        weight_info.update({"size": stat.st_size, "mtime_ns": stat.st_mtime_ns})
    payload = {
        "teacher": weight_info,
        "confidence_threshold": confidence_threshold,
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
) -> tuple[dict[str, list[tuple[float, float, float, float]]], Path, bool]:
    """Run one streamed teacher prediction pass and cache accepted person boxes."""
    if not 0.0 <= confidence_threshold <= 1.0:
        raise ValueError("teacher confidence must be between 0 and 1")
    images = tuple(Path(path) for path in train_images)
    if not images:
        raise ValueError("Pseudo-label generation requires training images.")

    signature = _cache_signature(teacher_weights, confidence_threshold, images)
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
        return labels, cache_dir, True

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
    result_iterator = iter(predictions)
    for fallback_image in images:
        try:
            result = next(result_iterator)
        except StopIteration:
            break
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
        confidences = _as_list(boxes.conf)
        class_ids = _as_list(boxes.cls)
        names = getattr(result, "names", getattr(model, "names", {}))
        for box, confidence, raw_class_id in zip(xyxy, confidences, class_ids):
            class_id = int(raw_class_id)
            if _class_name(names, class_id) != "person":
                continue
            if float(confidence) < confidence_threshold:
                continue
            labels[key].append(tuple(float(coord) for coord in box))

    cache_dir.mkdir(parents=True, exist_ok=True)
    labels_path.write_text(json.dumps(labels, indent=2) + "\n", encoding="utf-8")
    manifest_path.write_text(
        json.dumps(
            {
                "format_version": 1,
                "teacher_weights": teacher_weights,
                "confidence_threshold": confidence_threshold,
                "image_count": len(images),
                "labeled_image_count": len(labels),
                "signature": signature,
            },
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )
    return labels, cache_dir, False
