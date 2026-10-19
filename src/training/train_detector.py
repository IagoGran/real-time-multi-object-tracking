import argparse
import csv
from datetime import datetime
import hashlib
import json
import logging
from pathlib import Path
import random
import subprocess
import sys
from typing import Any

from src.training.mot17_training_dataset import (
    PREPARATION_POLICY_VERSION,
    PreparedDataset,
    prepare_mot17_dataset,
)
from src.training.pseudo_labels import (
    TEACHER_LABEL_POLICY_DESCRIPTION,
    TEACHER_LABEL_POLICY_VERSION,
    generate_pseudo_labels,
)
from src.training.training_config import (
    DEFAULT_SPLIT_CONFIG,
    Mot17TrainingConfig,
    load_training_config,
)
from src.training.training_results import (
    make_experiment_id,
    new_experiment_manifest,
    save_experiment_manifest,
    utc_timestamp,
)


LOGGER = logging.getLogger("src.training")
DEFAULT_EXPERIMENTS_ROOT = Path("results/training")
DEFAULT_PSEUDO_LABEL_CACHE = Path("datasets/MOT17_pseudo_labels")

# Usual training run settings. Edit these values for the normal workflow;
# command-line options can still override them for one-off runs.
TRAINING_CONFIG: dict[str, Any] = {
    "mode": "supervised",  # "supervised" or "teacher"
    "student_weights": "yolo26n.pt",
    "teacher_weights": None,  # Set a checkpoint here when mode="teacher".
    "teacher_confidence": 0.25,
    "split_config": DEFAULT_SPLIT_CONFIG,
    "dataset_root": None,  # None uses the root in split_config.
    "prepared_dataset_root": None,  # None uses the root in split_config.
    "output_root": DEFAULT_EXPERIMENTS_ROOT,
    "pseudo_label_cache": DEFAULT_PSEUDO_LABEL_CACHE,
    "epochs": 50,
    "batch": 16,
    "imgsz": 640,
    "multi_scale": 0.0,
    "freeze": None,
    "experiment_label": None,
    "lr0": 0.001,
    "optimizer": "AdamW",
    "patience": 20,
    "workers": 8,
    "seed": 0,
    "device": None,  # None lets Ultralytics choose automatically.
    "evaluate_after_training": False,
    "resume_from": None,
    "max_frames_per_sequence": None,  # Set a small number for smoke runs.
    "prepare_only": False,
}


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train YOLO26n on MOT17.")
    parser.add_argument("--mode", choices=("supervised", "teacher"), default=TRAINING_CONFIG["mode"])
    parser.add_argument("--student-weights", default=TRAINING_CONFIG["student_weights"])
    parser.add_argument("--teacher-weights", default=TRAINING_CONFIG["teacher_weights"])
    parser.add_argument("--teacher-confidence", type=float, default=TRAINING_CONFIG["teacher_confidence"])
    parser.add_argument("--split-config", type=Path, default=TRAINING_CONFIG["split_config"])
    parser.add_argument("--dataset-root", type=Path, default=TRAINING_CONFIG["dataset_root"])
    parser.add_argument("--prepared-dataset-root", type=Path, default=TRAINING_CONFIG["prepared_dataset_root"])
    parser.add_argument("--output-root", type=Path, default=TRAINING_CONFIG["output_root"])
    parser.add_argument("--pseudo-label-cache", type=Path, default=TRAINING_CONFIG["pseudo_label_cache"])
    parser.add_argument("--epochs", type=int, default=TRAINING_CONFIG["epochs"])
    parser.add_argument("--batch", type=int, default=TRAINING_CONFIG["batch"])
    parser.add_argument("--imgsz", type=int, default=TRAINING_CONFIG["imgsz"])
    parser.add_argument("--multi-scale", type=float, default=TRAINING_CONFIG["multi_scale"], help="Ultralytics multi-scale range as a fraction of imgsz (0 disables it).")
    parser.add_argument("--freeze", type=int, default=TRAINING_CONFIG["freeze"], help="Freeze the first N Ultralytics model layers; omit to keep the existing behavior.")
    parser.add_argument("--experiment-label", default=TRAINING_CONFIG["experiment_label"], help="Optional readable label included in the unique experiment directory name.")
    parser.add_argument("--lr0", type=float, default=TRAINING_CONFIG["lr0"])
    parser.add_argument("--optimizer", default=TRAINING_CONFIG["optimizer"])
    parser.add_argument("--patience", type=int, default=TRAINING_CONFIG["patience"])
    parser.add_argument("--workers", type=int, default=TRAINING_CONFIG["workers"])
    parser.add_argument("--seed", type=int, default=TRAINING_CONFIG["seed"])
    parser.add_argument("--device", default=TRAINING_CONFIG["device"], help="Ultralytics device, e.g. 0, cpu; default selects automatically.")
    parser.add_argument(
        "--evaluate-after-training",
        action=argparse.BooleanOptionalAction,
        default=TRAINING_CONFIG["evaluate_after_training"],
    )
    parser.add_argument("--resume-from", type=Path, default=TRAINING_CONFIG["resume_from"], help="Resume Ultralytics training from an experiment's last.pt.")
    parser.add_argument("--max-frames-per-sequence", type=int, default=TRAINING_CONFIG["max_frames_per_sequence"], help="Limit every sequence for smoke tests/preflight.")
    parser.add_argument(
        "--prepare-only",
        action=argparse.BooleanOptionalAction,
        default=TRAINING_CONFIG["prepare_only"],
        help="Prepare the configured dataset and exit without training.",
    )
    args = parser.parse_args(argv)
    if args.mode == "teacher" and not args.teacher_weights and not args.resume_from:
        parser.error("--teacher-weights is required with --mode teacher.")
    if args.mode == "supervised" and args.teacher_weights:
        parser.error("--teacher-weights can only be used with --mode teacher.")
    if not 0.0 <= args.teacher_confidence <= 1.0:
        parser.error("--teacher-confidence must be between 0 and 1.")
    if args.epochs <= 0 or args.batch <= 0 or args.imgsz <= 0 or args.workers < 0 or args.patience < 0:
        parser.error("epochs, batch, and imgsz must be positive; workers and patience cannot be negative.")
    if not 0.0 <= args.multi_scale <= 1.0:
        parser.error("--multi-scale must be between 0 and 1 inclusive.")
    if args.freeze is not None and args.freeze < 0:
        parser.error("--freeze cannot be negative.")
    if args.multi_scale > 0 and max(32, int(args.imgsz * (1.0 - args.multi_scale))) // 32 * 32 < 64:
        parser.error("--imgsz and --multi-scale must keep the minimum training size at least 64 pixels (two YOLO strides).")
    if args.max_frames_per_sequence is not None and args.max_frames_per_sequence <= 0:
        parser.error("--max-frames-per-sequence must be positive.")
    return args


def _override_config(args: argparse.Namespace) -> Mot17TrainingConfig:
    config = load_training_config(args.split_config)
    values: dict[str, Any] = {}
    if args.dataset_root is not None:
        values["dataset_root"] = args.dataset_root
    if args.prepared_dataset_root is not None:
        values["prepared_dataset_root"] = args.prepared_dataset_root
    if not values:
        return config
    return Mot17TrainingConfig(**{**config.__dict__, **values})


def _set_seed(seed: int) -> None:
    random.seed(seed)
    import numpy as np

    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def _training_parameters(args: argparse.Namespace, data_yaml: Path) -> dict[str, Any]:
    return {
        "data": str(data_yaml.resolve()),
        "epochs": args.epochs,
        "batch": args.batch,
        "imgsz": args.imgsz,
        "multi_scale": args.multi_scale,
        "freeze": args.freeze,
        "experiment_label": args.experiment_label,
        "lr0": args.lr0,
        "optimizer": args.optimizer,
        "patience": args.patience,
        "workers": args.workers,
        "seed": args.seed,
        "device": args.device or "auto",
        "deterministic": True,
        "max_frames_per_sequence": args.max_frames_per_sequence,
    }


def _ultralytics_training_kwargs(args: argparse.Namespace, data_yaml: Path, output_root: Path, experiment_id: str) -> dict[str, Any]:
    kwargs = {
        "data": str(data_yaml),
        "epochs": args.epochs,
        "batch": args.batch,
        "imgsz": args.imgsz,
        "multi_scale": args.multi_scale,
        "lr0": args.lr0,
        "optimizer": args.optimizer,
        "patience": args.patience,
        "workers": args.workers,
        "seed": args.seed,
        "deterministic": True,
        "project": str(output_root.resolve()),
        "name": experiment_id,
        "exist_ok": True,
    }
    if args.freeze is not None:
        kwargs["freeze"] = args.freeze
    if args.device is not None:
        kwargs["device"] = args.device
    return kwargs


def _resolve_training_run(model: Any, experiment_dir: Path) -> Path:
    trainer = getattr(model, "trainer", None)
    save_dir = getattr(trainer, "save_dir", None)
    return Path(save_dir).resolve() if save_dir else experiment_dir.resolve()


def _evaluate_weights(weights: Path, experiment_id: str, mode: str, split_config: Path) -> None:
    model_name = f"YOLO26n-{mode}"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "src.evaluation.evaluate_detectors",
            "--model",
            f"{model_name}={weights.resolve()}",
            "--evaluation-name",
            experiment_id,
            "--split-config",
            str(split_config),
            "--partition",
            "validation",
        ],
        check=True,
    )


def _prepare_dataset(args: argparse.Namespace, config: Mot17TrainingConfig) -> tuple[PreparedDataset, Path | None, bool | None, dict[str, Any] | None]:
    pseudo_cache_dir = None
    pseudo_cache_hit = None
    if args.mode == "supervised":
        output_dir = config.prepared_dataset_root / f"supervised_{PREPARATION_POLICY_VERSION}"
        prepared = prepare_mot17_dataset(config, output_dir, mode="supervised", max_frames_per_sequence=args.max_frames_per_sequence)
        return prepared, pseudo_cache_dir, pseudo_cache_hit, None

    from ultralytics import YOLO

    train_images = tuple(
        tuple(sorted((config.dataset_root / sequence / "img1").glob("*.jpg"))[:args.max_frames_per_sequence])
        if args.max_frames_per_sequence is not None
        else tuple(sorted((config.dataset_root / sequence / "img1").glob("*.jpg")))
        for sequence in config.train_sequences
    )
    flattened_train_images = tuple(path for group in train_images for path in group)
    teacher = YOLO(args.teacher_weights)
    teacher_path = getattr(teacher, "ckpt_path", None)
    teacher_metadata = _weight_metadata(args.teacher_weights, teacher_path)
    pseudo_labels, pseudo_cache_dir, pseudo_cache_hit, pseudo_stats = generate_pseudo_labels(
        model=teacher,
        train_images=flattened_train_images,
        teacher_weights=(str(Path(teacher_path).resolve()) if teacher_path and Path(teacher_path).is_file() else args.teacher_weights),
        confidence_threshold=args.teacher_confidence,
        cache_root=args.pseudo_label_cache,
        train_sequences=config.train_sequences,
        preparation_policy_version=PREPARATION_POLICY_VERSION,
    )
    dataset_key = pseudo_cache_dir.name
    output_dir = config.prepared_dataset_root / f"teacher_{dataset_key}_{PREPARATION_POLICY_VERSION}"
    prepared = prepare_mot17_dataset(
        config, output_dir, mode="teacher", pseudo_labels=pseudo_labels,
        max_frames_per_sequence=args.max_frames_per_sequence,
    )
    del teacher
    teacher_metadata.update({"pseudo_label_stats": pseudo_stats})
    return prepared, pseudo_cache_dir, pseudo_cache_hit, teacher_metadata


def _weight_metadata(value: str, resolved_path: str | Path | None = None) -> dict[str, Any]:
    path = Path(resolved_path or value).expanduser()
    metadata: dict[str, Any] = {"signature": value, "resolved_path": str(path.resolve()) if path.is_file() else None, "sha256": None}
    if path.is_file():
        digest = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                digest.update(chunk)
        metadata["sha256"] = digest.hexdigest()
    return metadata


def _model_layers(model: Any) -> tuple[Any, list[Any], dict[str, Any]]:
    detection_model = model if hasattr(model, "yaml") else getattr(model, "model", model)
    layers = getattr(detection_model, "model", None)
    yaml_config = getattr(detection_model, "yaml", {}) or {}
    if layers is None or not hasattr(layers, "__len__") or not isinstance(yaml_config, dict):
        raise ValueError("--freeze requires a loaded Ultralytics model with an indexed model and YAML architecture.")
    return detection_model, list(layers), yaml_config


def _validate_freeze_compatibility(model: Any, freeze: int | None) -> dict[str, Any] | None:
    if freeze is None:
        return None
    _, layers, yaml_config = _model_layers(model)
    if freeze > len(layers):
        raise ValueError(f"--freeze {freeze} exceeds this model's {len(layers)} indexed layers.")
    backbone = yaml_config.get("backbone")
    if freeze == 11 and (not isinstance(backbone, list) or len(backbone) != 11):
        raise ValueError(
            "--freeze 11 is intended to freeze the complete YOLO26n backbone, but the loaded model's "
            f"YAML declares {len(backbone) if isinstance(backbone, list) else 'no indexed'} backbone layers."
        )
    return {
        "model_layers": len(layers),
        "backbone_layers": len(backbone) if isinstance(backbone, list) else None,
        "backbone_layer_indices": list(range(len(backbone))) if isinstance(backbone, list) else None,
        "freeze": freeze,
    }


def _verify_freeze_state(trainer: Any, freeze: int, manifest: dict[str, Any], manifest_path: Path) -> None:
    """Verify Ultralytics' post-setup requires_grad flags and optimizer groups."""
    active_model = trainer.model
    while hasattr(active_model, "module"):
        active_model = active_model.module
    _, layers, yaml_config = _model_layers(active_model)
    parameters = dict(active_model.named_parameters())
    optimizer_parameter_ids = {
        id(parameter)
        for group in trainer.optimizer.param_groups
        for parameter in group["params"]
    }
    if not optimizer_parameter_ids:
        raise RuntimeError("Ultralytics initialized an optimizer without parameters.")

    layer_summary = []
    trainable_by_layer: list[int] = []
    for index, layer in enumerate(layers):
        layer_parameters = list(layer.parameters())
        total = sum(parameter.numel() for parameter in layer_parameters)
        frozen = sum(parameter.numel() for parameter in layer_parameters if not parameter.requires_grad)
        trainable = total - frozen
        trainable_by_layer.append(trainable)
        layer_summary.append({"index": index, "type": type(layer).__name__, "total": total, "frozen": frozen, "trainable": trainable})
        if index < freeze and trainable:
            raise RuntimeError(f"Ultralytics left parameters trainable in requested frozen layer {index}.")

    missing_trainable = [name for name, parameter in parameters.items() if parameter.requires_grad and id(parameter) not in optimizer_parameter_ids]
    frozen_parameters = {name: parameter for name, parameter in parameters.items() if not parameter.requires_grad}
    frozen_with_gradients = [name for name, parameter in frozen_parameters.items() if parameter.grad is not None]
    if missing_trainable or frozen_with_gradients:
        raise RuntimeError(
            "Freeze state does not match gradient setup "
            f"(trainable parameters missing from optimizer={missing_trainable[:5]}, frozen parameters with gradients={frozen_with_gradients[:5]})."
        )
    total_count = sum(parameter.numel() for parameter in parameters.values())
    frozen_count = sum(parameter.numel() for parameter in parameters.values() if not parameter.requires_grad)
    trainable_count = total_count - frozen_count
    if trainable_count == 0 or (freeze > 0 and frozen_count == 0):
        raise RuntimeError("Freeze verification found an unexpected all-frozen or all-trainable model state.")
    if sum(trainable_by_layer[freeze:]) == 0:
        raise RuntimeError("Ultralytics left no parameters trainable after the requested frozen layers.")
    backbone = yaml_config.get("backbone")
    if freeze == 11 and isinstance(backbone, list) and len(backbone) == 11:
        neck_trainable = sum(trainable_by_layer[freeze:-1])
        head_trainable = trainable_by_layer[-1]
        if neck_trainable == 0 or head_trainable == 0:
            raise RuntimeError(
                "YOLO26n backbone freeze verification expected trainable parameters in both the neck and detection head."
            )

    summary = {
        "verified_after_ultralytics_setup": True,
        "freeze": freeze,
        "frozen_layer_indices": list(range(freeze)),
        "layer_parameters": layer_summary,
        "parameters": {"total": total_count, "frozen": frozen_count, "trainable": trainable_count},
        "neck_trainable_parameters": sum(trainable_by_layer[freeze:-1]) if freeze == 11 else None,
        "head_trainable_parameters": trainable_by_layer[-1] if freeze == 11 else None,
        "optimizer_includes_frozen_parameters": sum(id(parameter) in optimizer_parameter_ids for parameter in frozen_parameters.values()),
        "optimizer_includes_trainable_parameters": True,
        "frozen_parameters_receive_gradients": False,
        "parameters_unchanged_after_epoch": False,
        "batchnorm_running_statistics_frozen": False,
    }
    trainer._codex_frozen_parameter_snapshot = {
        name: parameter.detach().cpu().clone() for name, parameter in frozen_parameters.items()
    }
    manifest["training"]["freeze_verification"] = summary
    save_experiment_manifest(manifest_path, manifest)
    LOGGER.info("Freeze verification passed: %s", summary["parameters"])


def _verify_frozen_parameters_unchanged(trainer: Any, manifest: dict[str, Any], manifest_path: Path) -> None:
    """Check frozen tensor values after Ultralytics completes an optimizer epoch."""
    import torch

    active_model = trainer.model
    while hasattr(active_model, "module"):
        active_model = active_model.module
    current_parameters = dict(active_model.named_parameters())
    snapshots = trainer._codex_frozen_parameter_snapshot
    changed = [
        name for name, before in snapshots.items()
        if name not in current_parameters
        or not torch.equal(before, current_parameters[name].detach().cpu())
    ]
    if changed:
        raise RuntimeError(f"Frozen parameters changed during optimizer training: {changed[:5]}.")
    verification = manifest["training"]["freeze_verification"]
    verification["parameters_unchanged_after_epoch"] = True
    verification["verified_epochs"] = int(verification.get("verified_epochs", 0)) + 1
    save_experiment_manifest(manifest_path, manifest)


def _record_training_outcome(manifest: dict[str, Any], results_csv: Path, requested_epochs: int) -> None:
    """Record Ultralytics' checkpoint selection metric and observed stop point."""
    selected_metric = "metrics/mAP50-95(B)"
    if results_csv.is_file():
        with results_csv.open(newline="", encoding="utf-8-sig") as history_file:
            rows = list(csv.DictReader(history_file))
        valid_rows = [row for row in rows if row.get(selected_metric)]
        if valid_rows:
            best_row = max(valid_rows, key=lambda row: float(row[selected_metric]))
            manifest["training"]["best_epoch"] = int(float(best_row["epoch"]))
            manifest["training"]["best_checkpoint_selection_metric"] = selected_metric
            manifest["training"]["best_checkpoint_selection_value"] = float(best_row[selected_metric])
        if rows:
            manifest["training"]["stopped_after_epoch"] = int(float(rows[-1]["epoch"]))
            manifest["training"]["finish_reason"] = "early_stopping" if len(rows) < requested_epochs else "requested_epochs_completed"


def _run_training(args: argparse.Namespace) -> Path:
    if args.resume_from:
        checkpoint = args.resume_from.resolve()
        if not checkpoint.is_file() or checkpoint.name != "last.pt":
            raise FileNotFoundError("--resume-from must point to an existing last.pt checkpoint.")
        experiment_dir = checkpoint.parent.parent
        manifest_path = experiment_dir / "experiment.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Training manifest not found: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("mode") != args.mode:
            raise ValueError("--mode does not match the saved experiment manifest.")
        manifest["status"] = "running"
        manifest["completed_at"] = None
        save_experiment_manifest(manifest_path, manifest)
        from ultralytics import YOLO

        model = YOLO(str(checkpoint))
        try:
            model.train(resume=True)
        except Exception:
            manifest["status"] = "failed"
            manifest["completed_at"] = utc_timestamp()
            save_experiment_manifest(manifest_path, manifest)
            raise
    else:
        config = _override_config(args)
        if not (args.prepare_only and args.mode == "supervised"):
            _set_seed(args.seed)
        prepared, pseudo_cache_dir, pseudo_cache_hit, resolved_teacher_metadata = _prepare_dataset(args, config)
        if args.prepare_only:
            if args.mode == "supervised" and args.max_frames_per_sequence is None:
                from src.training.inspect_dataset import audit_dataset, assert_preparation_matches_audit

                source_audit = audit_dataset(args.split_config, dataset_root=config.dataset_root)
                prepared_manifest = json.loads((prepared.root / "dataset-manifest.json").read_text(encoding="utf-8"))
                assert_preparation_matches_audit(source_audit, prepared_manifest)
                LOGGER.info("Prepared dataset matches source audit.")
            LOGGER.info("Prepared dataset: %s (%s)", prepared.root, prepared.stats)
            return prepared.root
        experiment_id = make_experiment_id(
            args.mode, args.student_weights,
            args.teacher_weights if args.mode == "teacher" else None,
            args.experiment_label,
        )
        experiment_dir = args.output_root / experiment_id
        experiment_dir.mkdir(parents=True, exist_ok=False)
        teacher_metadata = (
            {
                **(resolved_teacher_metadata or {"weights": args.teacher_weights}),
                "confidence_threshold": args.teacher_confidence,
                "pseudo_label_cache": str(pseudo_cache_dir.resolve()),
                "pseudo_label_cache_hit": pseudo_cache_hit,
                "teacher_label_policy_version": TEACHER_LABEL_POLICY_VERSION,
                "teacher_label_policy": TEACHER_LABEL_POLICY_DESCRIPTION,
            }
            if args.mode == "teacher"
            else None
        )
        manifest = new_experiment_manifest(
            experiment_id=experiment_id,
            mode=args.mode,
            initial_weights=args.student_weights,
            teacher=teacher_metadata,
            dataset={
                "name": "MOT17",
                "source_root": str(config.dataset_root.resolve()),
                "prepared_root": str(prepared.root),
                "dataset_yaml": str(prepared.yaml_path),
                "split_config": str(args.split_config.resolve()),
                "train_sequences": list(config.train_sequences),
                "validation_sequences": list(config.validation_sequences),
                "preparation_policy": config.preparation_policy,
                "preparation_policy_version": PREPARATION_POLICY_VERSION,
                "preparation_stats": prepared.stats,
            },
            training=_training_parameters(args, prepared.yaml_path),
        )
        manifest_path = experiment_dir / "experiment.json"
        if args.experiment_label:
            manifest["training"]["experiment_label"] = args.experiment_label
        save_experiment_manifest(manifest_path, manifest)

        from ultralytics import YOLO

        model = YOLO(args.student_weights)
        student_weight_metadata = _weight_metadata(args.student_weights, getattr(model, "ckpt_path", None))
        manifest["student"]["initial_weights_metadata"] = student_weight_metadata
        architecture = _validate_freeze_compatibility(model.model, args.freeze)
        if architecture is not None:
            manifest["training"]["freeze_architecture"] = architecture
        save_experiment_manifest(manifest_path, manifest)
        kwargs = _ultralytics_training_kwargs(args, prepared.yaml_path, args.output_root, experiment_id)
        if args.freeze is not None:
            model.add_callback(
                "on_pretrain_routine_end",
                lambda trainer: _verify_freeze_state(trainer, args.freeze, manifest, manifest_path),
            )
            model.add_callback(
                "on_train_epoch_end",
                lambda trainer: _verify_frozen_parameters_unchanged(trainer, manifest, manifest_path),
            )
        try:
            model.train(**kwargs)
        except Exception:
            manifest["status"] = "failed"
            manifest["completed_at"] = utc_timestamp()
            save_experiment_manifest(manifest_path, manifest)
            raise

    run_dir = _resolve_training_run(model, experiment_dir)
    weights_dir = run_dir / "weights"
    best_weights = weights_dir / "best.pt"
    last_weights = weights_dir / "last.pt"
    if not best_weights.is_file() or not last_weights.is_file():
        manifest["status"] = "failed"
        manifest["completed_at"] = utc_timestamp()
        save_experiment_manifest(manifest_path, manifest)
        raise FileNotFoundError(
            f"Ultralytics training output is missing best.pt or last.pt under {weights_dir}"
        )

    manifest["student"]["output_weights"] = str(best_weights.resolve())
    manifest["student"]["best_weights"] = _weight_metadata(str(best_weights), best_weights)
    manifest["student"]["last_weights"] = _weight_metadata(str(last_weights), last_weights)
    manifest["training"]["ultralytics_run_dir"] = str(run_dir)
    results_csv = run_dir / "results.csv"
    if results_csv.is_file():
        manifest["training"]["history"] = str(results_csv)
    _record_training_outcome(manifest, results_csv, int(manifest["training"].get("epochs", args.epochs)))
    manifest["status"] = "completed"
    manifest["completed_at"] = utc_timestamp()
    save_experiment_manifest(manifest_path, manifest)
    LOGGER.info("Training complete: %s", experiment_dir)
    LOGGER.info("Best weights: %s", best_weights)
    if args.evaluate_after_training:
        _evaluate_weights(best_weights, manifest["experiment_id"], args.mode, args.split_config)
    return best_weights


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    args = _parse_args(argv)
    try:
        _run_training(args)
    except Exception:
        LOGGER.exception("Detector training failed")
        raise


if __name__ == "__main__":
    main()
