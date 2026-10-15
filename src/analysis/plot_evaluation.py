"""Plot persisted evaluation JSON and Ultralytics CSV histories."""
import argparse
import csv
import json
from pathlib import Path
from typing import Any

EVALUATION_METRICS = {
    "match_rate": "Match%",
    "false_positives_per_frame": "FP/frame",
    "mean_iou": "Mean IoU",
    "mean_edge_error": "Normalized edge error",
    "mean_center_error": "Normalized center error",
    "mean_size_error": "Size error",
}
HISTORY_METRICS = (
    "train/box_loss", "train/cls_loss", "train/dfl_loss",
    "val/box_loss", "val/cls_loss", "val/dfl_loss",
    "metrics/precision(B)", "metrics/recall(B)",
    "metrics/mAP50(B)", "metrics/mAP50-95(B)",
)


def flatten_evaluations(paths: list[Path]) -> list[dict[str, Any]]:
    rows = []
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        name = payload.get("evaluation_name") or payload.get("evaluation_id") or path.stem
        for threshold, models in payload["results"].items():
            for model, metrics in models.items():
                rows.append({"evaluation": name, "model": model, "threshold": float(threshold), **metrics})
    return rows


def _save_plot(rows: list[dict[str, Any]], metric: str, label: str, directory: Path, prefix: str) -> None:
    import matplotlib.pyplot as plt

    fig, axis = plt.subplots()
    for evaluation in sorted({row["evaluation"] for row in rows}):
        for model in sorted({row["model"] for row in rows if row["evaluation"] == evaluation}):
            points = sorted((row["threshold"], row.get(metric)) for row in rows if row["evaluation"] == evaluation and row["model"] == model and row.get(metric) is not None)
            if points:
                x, y = zip(*points)
                if metric == "match_rate":
                    y = tuple(value * 100 for value in y)
                axis.plot(x, y, marker="o", label=f"{evaluation} / {model}")
    axis.set(xlabel="Confidence threshold", ylabel=label, title=f"{label} vs confidence threshold")
    axis.grid(True, alpha=0.3)
    axis.legend()
    fig.tight_layout()
    fig.savefig(directory / f"{prefix}.png", dpi=160)
    fig.savefig(directory / f"{prefix}.svg")
    plt.close(fig)


def plot_evaluations(paths: list[Path], output_dir: Path) -> list[dict[str, Any]]:
    rows = flatten_evaluations(paths)
    if not rows:
        raise ValueError("No evaluation metric rows found.")
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "evaluation_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(dict.fromkeys(key for row in rows for key in row)))
        writer.writeheader()
        writer.writerows(rows)
    for key, label in EVALUATION_METRICS.items():
        _save_plot(rows, key, label, output_dir, key)
    return rows


def plot_training_history(manifest_paths: list[Path], output_dir: Path) -> int:
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    plotted = 0
    for manifest_path in manifest_paths:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        history_value = manifest.get("training", {}).get("history")
        if not history_value:
            continue
        history_path = Path(history_value)
        if not history_path.is_absolute():
            history_path = manifest_path.parent / history_path
        with history_path.open(newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            rows = [{key.strip(): value.strip() for key, value in row.items() if key} for row in reader]
        columns = set(rows[0]) if rows else set()
        available = [column for column in HISTORY_METRICS if column in columns]
        if not available:
            continue
        fig, axis = plt.subplots()
        epochs = [float(row.get("epoch", index)) for index, row in enumerate(rows)]
        for column in available:
            values = [(x, float(row[column])) for x, row in zip(epochs, rows) if row.get(column)]
            if values:
                x, y = zip(*values)
                axis.plot(x, y, marker=".", label=column)
        axis.set(xlabel="Epoch", ylabel="Metric", title=manifest.get("experiment_id", manifest_path.stem))
        axis.grid(True, alpha=0.3)
        axis.legend()
        fig.tight_layout()
        stem = manifest.get("experiment_id", manifest_path.stem)
        fig.savefig(output_dir / f"training_{stem}.png", dpi=160)
        fig.savefig(output_dir / f"training_{stem}.svg")
        plt.close(fig)
        plotted += 1
    return plotted


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", type=Path, action="append", default=[])
    parser.add_argument("--training-manifest", type=Path, action="append", default=[])
    parser.add_argument("--output-dir", type=Path, default=Path("results/plots"))
    args = parser.parse_args(argv)
    if not args.evaluation and not args.training_manifest:
        parser.error("Provide --evaluation and/or --training-manifest.")
    if args.evaluation:
        name = json.loads(args.evaluation[0].read_text(encoding="utf-8")).get("evaluation_name", args.evaluation[0].stem)
        plot_evaluations(args.evaluation, args.output_dir / name)
    if args.training_manifest:
        plot_training_history(args.training_manifest, args.output_dir)


if __name__ == "__main__":
    main()
