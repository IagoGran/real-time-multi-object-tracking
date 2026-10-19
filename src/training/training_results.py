from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import tempfile
from typing import Any
import uuid


FORMAT_VERSION = 1


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def make_experiment_id(mode: str, student_weights: str, teacher_weights: str | None = None, experiment_label: str | None = None) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    suffix = uuid.uuid4().hex[:8]
    student = re.sub(r"[^A-Za-z0-9.-]+", "-", Path(student_weights).stem).strip("-")
    label = re.sub(r"[^A-Za-z0-9.-]+", "-", experiment_label or "").strip("-")
    label_part = f"_{label}" if label else ""
    if mode == "teacher" and teacher_weights:
        teacher = re.sub(r"[^A-Za-z0-9.-]+", "-", Path(teacher_weights).stem).strip("-")
        return f"{student}_{mode}_{teacher}{label_part}_{timestamp}_{suffix}"
    return f"{student}_{mode}{label_part}_{timestamp}_{suffix}"


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=2,
        )
        return result.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def environment_metadata(device: str) -> dict[str, Any]:
    try:
        import torch

        torch_version = torch.__version__
        cuda_version = torch.version.cuda
        cuda_available = torch.cuda.is_available()
        effective_device = (
            str(device)
            if device != "auto"
            else "cuda:0" if cuda_available else "cpu"
        )
        gpu_name = torch.cuda.get_device_name(0) if cuda_available else None
    except ImportError:
        torch_version = None
        cuda_version = None
        cuda_available = False
        effective_device = device
        gpu_name = None
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch_version,
        "ultralytics": _package_version("ultralytics"),
        "cuda": cuda_version,
        "cuda_available": cuda_available,
        "requested_device": device,
        "effective_device": effective_device,
        "gpu_name": gpu_name,
        "git_commit": _git_commit(),
    }


def new_experiment_manifest(
    experiment_id: str,
    mode: str,
    initial_weights: str,
    teacher: dict[str, Any] | None,
    dataset: dict[str, Any],
    training: dict[str, Any],
) -> dict[str, Any]:
    started_at = utc_timestamp()
    return {
        "format_version": FORMAT_VERSION,
        "experiment_id": experiment_id,
        "mode": mode,
        "student": {
            "initial_weights": initial_weights,
            "output_weights": None,
        },
        "teacher": teacher,
        "dataset": dataset,
        "training": training,
        "environment": environment_metadata(str(training.get("device", "auto"))),
        "started_at": started_at,
        "completed_at": None,
        "status": "running",
    }


def save_experiment_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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
            json.dump(manifest, temporary_file, indent=2, allow_nan=False)
            temporary_file.write("\n")
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
