from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any


DEFAULT_SPLIT_CONFIG = Path("configs/mot17_training_split.json")
SUPPORTED_IGNORED_FRAME_POLICY = "skip_frames_with_excluded_annotations"


@dataclass(frozen=True)
class Mot17TrainingConfig:
    name: str
    dataset_root: Path
    prepared_dataset_root: Path
    variant: str
    train_sequences: tuple[str, ...]
    validation_sequences: tuple[str, ...]
    ignored_frame_policy: str
    format_version: int = 1

    def __post_init__(self) -> None:
        validate_sequence_split(self.train_sequences, self.validation_sequences)
        if self.variant != "FRCNN":
            raise ValueError("MOT17 training uses one FRCNN variant per scene.")
        if self.ignored_frame_policy != SUPPORTED_IGNORED_FRAME_POLICY:
            raise ValueError(
                "Unsupported ignored_frame_policy: "
                f"{self.ignored_frame_policy!r}."
            )


def validate_sequence_split(
    train_sequences: tuple[str, ...] | list[str],
    validation_sequences: tuple[str, ...] | list[str],
) -> None:
    if len(set(train_sequences)) != len(train_sequences):
        raise ValueError("Train sequence list contains duplicates.")
    if len(set(validation_sequences)) != len(validation_sequences):
        raise ValueError("Validation sequence list contains duplicates.")
    train = set(train_sequences)
    validation = set(validation_sequences)
    if not train or not validation:
        raise ValueError("Train and validation sequence lists must both be non-empty.")
    overlap = train & validation
    if overlap:
        raise ValueError(
            "Train/validation sequence leakage: " + ", ".join(sorted(overlap))
        )
    for sequence in train | validation:
        if not sequence.endswith("-FRCNN"):
            raise ValueError(
                f"Expected one FRCNN variant per scene, got {sequence!r}."
            )
    scene_names = ["-".join(sequence.split("-")[:2]) for sequence in train | validation]
    if len(scene_names) != len(set(scene_names)):
        raise ValueError("The split must use exactly one MOT17 variant per scene.")


def load_training_config(path: Path = DEFAULT_SPLIT_CONFIG) -> Mot17TrainingConfig:
    with path.open(encoding="utf-8") as config_file:
        raw: dict[str, Any] = json.load(config_file)
    if raw.get("format_version") != 1:
        raise ValueError("Unsupported MOT17 training split config version.")
    config_root = path.resolve().parent.parent

    def configured_path(value: str) -> Path:
        result = Path(value)
        return result if result.is_absolute() else config_root / result

    return Mot17TrainingConfig(
        name=str(raw["name"]),
        dataset_root=configured_path(raw["dataset_root"]),
        prepared_dataset_root=configured_path(raw["prepared_dataset_root"]),
        variant=str(raw["variant"]),
        train_sequences=tuple(raw["train_sequences"]),
        validation_sequences=tuple(raw["validation_sequences"]),
        ignored_frame_policy=str(raw["ignored_frame_policy"]),
    )
