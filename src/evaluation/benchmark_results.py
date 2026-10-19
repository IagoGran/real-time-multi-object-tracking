from datetime import datetime, timezone
from decimal import Decimal
import json
import os
from pathlib import Path
import tempfile
from typing import Any


FORMAT_VERSION = 2


def confidence_threshold_key(threshold: float) -> str:
    """Return the stable JSON representation used for threshold keys."""
    key = format(Decimal(str(threshold)), "f")
    if "." not in key:
        return f"{key}.00"
    integer, fractional = key.split(".", maxsplit=1)
    return f"{integer}.{fractional.ljust(2, '0')}"


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def build_benchmark_config(
    models: list[dict[str, str]],
    sequences: list[dict[str, str | int]],
    confidence_thresholds: list[float],
    match_iou_threshold: float,
    max_frames: int | None,
    inference_imgsz: int | None = None,
) -> dict[str, Any]:
    config = {
        "models": models,
        "sequences": sequences,
        "confidence_thresholds": confidence_thresholds,
        "match_iou_threshold": match_iou_threshold,
        "max_frames": max_frames,
    }
    # Keep historical native-resolution checkpoint configs resume-compatible.
    if inference_imgsz is not None:
        config["inference_imgsz"] = inference_imgsz
    return config


def new_checkpoint(config: dict[str, Any]) -> dict[str, Any]:
    timestamp = utc_timestamp()
    return {
        "format_version": FORMAT_VERSION,
        "config": config,
        "started_at": timestamp,
        "updated_at": timestamp,
        "completed_at": None,
        "status": "pending",
        "results": {},
        "aggregates": {},
    }


def load_checkpoint(
    path: Path,
    expected_config: dict[str, Any],
) -> dict[str, Any]:
    if not path.exists():
        raise RuntimeError(
            f"Cannot resume: checkpoint does not exist: {path}"
        )

    try:
        with path.open(encoding="utf-8") as checkpoint_file:
            checkpoint = json.load(checkpoint_file)
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(
            f"Cannot resume: failed to read checkpoint {path}: {error}"
        ) from error

    if not isinstance(checkpoint, dict):
        raise RuntimeError(
            "Cannot resume: checkpoint root must be a JSON object."
        )

    if checkpoint.get("format_version") != FORMAT_VERSION:
        version = checkpoint.get("format_version")
        if version == 1:
            raise RuntimeError(
                "Cannot resume: checkpoint format version 1 is incompatible "
                "with version 2 confidence-threshold sweep results. Start a new checkpoint."
            )
        raise RuntimeError(
            "Cannot resume: unsupported checkpoint format version "
            f"{checkpoint.get('format_version')!r}; expected "
            f"{FORMAT_VERSION}."
        )

    actual_config = checkpoint.get("config")
    if not isinstance(actual_config, dict):
        raise RuntimeError(
            "Cannot resume: checkpoint configuration must be a JSON object."
        )
    if actual_config != expected_config:
        fields = sorted(
            key
            for key in set(expected_config) | set(actual_config or {})
            if expected_config.get(key) != (actual_config or {}).get(key)
        )
        differences = ", ".join(
            f"{key}: checkpoint={actual_config.get(key)!r}, "
            f"requested={expected_config.get(key)!r}"
            for key in fields
        )
        raise RuntimeError(
            "Cannot resume: checkpoint configuration is incompatible "
            f"({differences})."
        )

    if not isinstance(checkpoint.get("results"), dict):
        raise RuntimeError(
            "Cannot resume: checkpoint results must be a JSON object."
        )

    return checkpoint


def save_checkpoint(path: Path, checkpoint: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint["updated_at"] = utc_timestamp()

    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            json.dump(
                checkpoint,
                temporary_file,
                indent=2,
                allow_nan=False,
            )
            temporary_file.write("\n")
            temporary_file.flush()
            os.fsync(temporary_file.fileno())

        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def combination_completed(
    checkpoint: dict[str, Any],
    model_name: str,
    sequence_name: str,
) -> bool:
    model_results = checkpoint["results"].get(model_name, {})
    sequence_results = model_results.get(sequence_name)
    if not isinstance(sequence_results, dict):
        return False
    thresholds = checkpoint.get("config", {}).get("confidence_thresholds", [])
    return bool(thresholds) and all(
        confidence_threshold_key(threshold) in sequence_results
        for threshold in thresholds
    )
