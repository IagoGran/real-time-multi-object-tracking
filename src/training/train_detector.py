import argparse
from datetime import datetime
import json
import logging
from pathlib import Path
import random
import subprocess
import sys
from typing import Any

from src.training.mot17_training_dataset import PreparedDataset, prepare_mot17_dataset
from src.training.pseudo_labels import generate_pseudo_labels
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


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train YOLO26n on MOT17.")
    parser.add_argument("--mode", choices=("supervised", "teacher"), required=True)
    parser.add_argument("--student-weights", default="yolo26n.pt")
    parser.add_argument("--teacher-weights")
    parser.add_argument("--teacher-confidence", type=float, default=0.25)
    parser.add_argument("--split-config", type=Path, default=DEFAULT_SPLIT_CONFIG)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--prepared-dataset-root", type=Path)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_EXPERIMENTS_ROOT)
    parser.add_argument("--pseudo-label-cache", type=Path, default=DEFAULT_PSEUDO_LABEL_CACHE)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--lr0", type=float, default=0.001)
    parser.add_argument("--optimizer", default="AdamW")
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=None, help="Ultralytics device, e.g. 0, cpu; default selects automatically.")
    parser.add_argument("--evaluate-after-training", action="store_true")
    parser.add_argument("--resume-from", type=Path, help="Resume Ultralytics training from an experiment's last.pt.")
    args = parser.parse_args(argv)
    if args.mode == "teacher" and not args.teacher_weights and not args.resume_from:
        parser.error("--teacher-weights is required with --mode teacher.")
    if args.mode == "supervised" and args.teacher_weights:
        parser.error("--teacher-weights can only be used with --mode teacher.")
    if not 0.0 <= args.teacher_confidence <= 1.0:
        parser.error("--teacher-confidence must be between 0 and 1.")
    if args.epochs <= 0 or args.batch <= 0 or args.imgsz <= 0 or args.workers < 0 or args.patience < 0:
        parser.error("epochs, batch, and imgsz must be positive; workers and patience cannot be negative.")
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
        "lr0": args.lr0,
        "optimizer": args.optimizer,
        "patience": args.patience,
        "workers": args.workers,
        "seed": args.seed,
        "device": args.device or "auto",
        "deterministic": True,
    }


def _resolve_training_run(model: Any, experiment_dir: Path) -> Path:
    trainer = getattr(model, "trainer", None)
    save_dir = getattr(trainer, "save_dir", None)
    return Path(save_dir).resolve() if save_dir else experiment_dir.resolve()


def _evaluate_weights(weights: Path, experiment_id: str, mode: str) -> None:
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
        ],
        check=True,
    )


def _prepare_dataset(args: argparse.Namespace, config: Mot17TrainingConfig) -> tuple[PreparedDataset, Path | None, bool | None]:
    pseudo_cache_dir = None
    pseudo_cache_hit = None
    if args.mode == "supervised":
        output_dir = config.prepared_dataset_root / "supervised"
        prepared = prepare_mot17_dataset(config, output_dir, mode="supervised")
        return prepared, pseudo_cache_dir, pseudo_cache_hit

    from ultralytics import YOLO

    train_images = tuple(
        sorted(
            (config.dataset_root / sequence / "img1").glob("*.jpg")
        )
        for sequence in config.train_sequences
    )
    flattened_train_images = tuple(path for group in train_images for path in group)
    teacher = YOLO(args.teacher_weights)
    pseudo_labels, pseudo_cache_dir, pseudo_cache_hit = generate_pseudo_labels(
        model=teacher,
        train_images=flattened_train_images,
        teacher_weights=args.teacher_weights,
        confidence_threshold=args.teacher_confidence,
        cache_root=args.pseudo_label_cache,
    )
    dataset_key = pseudo_cache_dir.name
    output_dir = config.prepared_dataset_root / f"teacher_{dataset_key}"
    prepared = prepare_mot17_dataset(
        config, output_dir, mode="teacher", pseudo_labels=pseudo_labels
    )
    del teacher
    return prepared, pseudo_cache_dir, pseudo_cache_hit


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
        _set_seed(args.seed)
        prepared, pseudo_cache_dir, pseudo_cache_hit = _prepare_dataset(args, config)
        experiment_id = make_experiment_id(
            args.mode, args.student_weights,
            args.teacher_weights if args.mode == "teacher" else None,
        )
        experiment_dir = args.output_root / experiment_id
        experiment_dir.mkdir(parents=True, exist_ok=False)
        teacher_metadata = (
            {
                "weights": args.teacher_weights,
                "confidence_threshold": args.teacher_confidence,
                "pseudo_label_cache": str(pseudo_cache_dir.resolve()),
                "pseudo_label_cache_hit": pseudo_cache_hit,
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
                "ignored_frame_policy": config.ignored_frame_policy,
                "preparation_stats": prepared.stats,
            },
            training=_training_parameters(args, prepared.yaml_path),
        )
        manifest_path = experiment_dir / "experiment.json"
        save_experiment_manifest(manifest_path, manifest)

        from ultralytics import YOLO

        model = YOLO(args.student_weights)
        kwargs = {
            "data": str(prepared.yaml_path),
            "epochs": args.epochs,
            "batch": args.batch,
            "imgsz": args.imgsz,
            "lr0": args.lr0,
            "optimizer": args.optimizer,
            "patience": args.patience,
            "workers": args.workers,
            "seed": args.seed,
            "deterministic": True,
            "project": str(args.output_root.resolve()),
            "name": experiment_id,
            "exist_ok": True,
        }
        if args.device is not None:
            kwargs["device"] = args.device
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
    manifest["training"]["ultralytics_run_dir"] = str(run_dir)
    results_csv = run_dir / "results.csv"
    if results_csv.is_file():
        manifest["training"]["history"] = str(results_csv)
    manifest["status"] = "completed"
    manifest["completed_at"] = utc_timestamp()
    save_experiment_manifest(manifest_path, manifest)
    LOGGER.info("Training complete: %s", experiment_dir)
    LOGGER.info("Best weights: %s", best_weights)
    if args.evaluate_after_training:
        _evaluate_weights(best_weights, manifest["experiment_id"], args.mode)
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
