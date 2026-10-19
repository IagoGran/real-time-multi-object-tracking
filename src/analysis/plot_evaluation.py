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
    "mean_edge_error_px": "Mean edge error (px)",
    "p90_edge_error": "P90 edge error",
    "mean_center_error": "Normalized center error",
    "mean_center_error_px": "Mean center error (px)",
    "mean_size_error": "Size error",
    "fps": "Quality sweep FPS",
}
HISTORY_METRICS = (
    "train/box_loss", "train/cls_loss", "train/dfl_loss",
    "val/box_loss", "val/cls_loss", "val/dfl_loss",
    "metrics/precision(B)", "metrics/recall(B)",
    "metrics/mAP50(B)", "metrics/mAP50-95(B)",
)
COMPARISON_HISTORY_METRICS = {
    "train/box_loss": ("Training box loss", True),
    "val/box_loss": ("Validation box loss", False),
    "train/cls_loss": ("Training classification loss", True),
    "val/cls_loss": ("Validation classification loss", False),
    "metrics/mAP50(B)": ("Validation mAP50", False),
    "metrics/mAP50-95(B)": ("Validation mAP50-95", False),
    "metrics/precision(B)": ("Validation precision", False),
    "metrics/recall(B)": ("Validation recall", False),
}


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


def plot_training_comparison(csvs: dict[str, Path], output_dir: Path) -> int:
    """Plot shared training/validation histories from existing Ultralytics CSVs."""
    import matplotlib.pyplot as plt

    histories: dict[str, list[dict[str, str]]] = {}
    for label, path in csvs.items():
        with path.open(newline="", encoding="utf-8-sig") as history_file:
            histories[label] = [
                {key.strip(): value.strip() for key, value in row.items() if key}
                for row in csv.DictReader(history_file)
            ]
    if not histories or not any(histories.values()):
        raise ValueError("No training history rows found in the supplied CSV files.")

    output_dir.mkdir(parents=True, exist_ok=True)
    for column, (title, training_loss) in COMPARISON_HISTORY_METRICS.items():
        fig, axis = plt.subplots()
        plotted = False
        for label, rows in histories.items():
            if not rows or column not in rows[0]:
                continue
            points = [(float(row.get("epoch", index + 1)), float(row[column]))
                      for index, row in enumerate(rows) if row.get(column)]
            if not points:
                continue
            x, y = zip(*points)
            axis.plot(x, y, marker=".", label=label)
            plotted = True
            map_column = "metrics/mAP50-95(B)"
            if map_column in rows[0]:
                best = max((row for row in rows if row.get(map_column)), key=lambda row: float(row[map_column]))
                best_epoch = float(best.get("epoch", 0))
                if best_epoch:
                    axis.axvline(best_epoch, color=axis.lines[-1].get_color(), alpha=0.28, linestyle="--")
        if not plotted:
            plt.close(fig)
            continue
        subtitle = "Training loss magnitude can vary with imgsz; compare trends." if training_loss else None
        axis.set(xlabel="Epoch", ylabel=title, title=title)
        if subtitle:
            axis.text(0.5, 1.02, subtitle, transform=axis.transAxes, ha="center", va="bottom", fontsize=8)
        axis.grid(True, alpha=0.3)
        axis.legend()
        fig.tight_layout()
        fig.savefig(output_dir / f"{column.replace('/', '_').replace('(', '').replace(')', '').replace('-', '_')}.png", dpi=160)
        fig.savefig(output_dir / f"{column.replace('/', '_').replace('(', '').replace(')', '').replace('-', '_')}.svg")
        plt.close(fig)
    return sum(1 for column in COMPARISON_HISTORY_METRICS if (output_dir / f"{column.replace('/', '_').replace('(', '').replace(')', '').replace('-', '_')}.png").is_file())


def _parse_training_csvs(values: list[str]) -> dict[str, Path]:
    runs: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Invalid --training-csv {value!r}; expected LABEL=PATH")
        label, path_text = value.split("=", 1)
        if not label.strip() or not path_text.strip() or label.strip() in runs:
            raise ValueError(f"Invalid or duplicate --training-csv: {value!r}")
        runs[label.strip()] = Path(path_text.strip())
    return runs


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", type=Path, action="append", default=[])
    parser.add_argument("--training-manifest", type=Path, action="append", default=[])
    parser.add_argument("--training-csv", action="append", default=[], metavar="LABEL=CSV", help="Add a run to the comparative training plots; repeat for each experiment.")
    parser.add_argument("--output-dir", type=Path, default=Path("results/plots"))
    args = parser.parse_args(argv)
    if not args.evaluation and not args.training_manifest and not args.training_csv:
        parser.error("Provide --evaluation, --training-manifest, and/or --training-csv.")
    if args.evaluation:
        name = json.loads(args.evaluation[0].read_text(encoding="utf-8")).get("evaluation_name", args.evaluation[0].stem)
        plot_evaluations(args.evaluation, args.output_dir / name)
    if args.training_manifest:
        plot_training_history(args.training_manifest, args.output_dir)
    if args.training_csv:
        plot_training_comparison(_parse_training_csvs(args.training_csv), args.output_dir / "training_comparison")


if __name__ == "__main__":
    main()
