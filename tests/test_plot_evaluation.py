import json

from src.analysis.plot_evaluation import flatten_evaluations, plot_training_comparison


def test_plotting_flattens_public_fixture_without_models_or_matplotlib(tmp_path):
    path = tmp_path / "evaluation.json"
    path.write_text(json.dumps({
        "evaluation_name": "fixture",
        "results": {
            "0.30": {"student": {"match_rate": 0.7, "mean_iou": 0.8}},
            "0.50": {"student": {"match_rate": 0.6, "mean_iou": 0.82}},
        },
    }), encoding="utf-8")
    rows = flatten_evaluations([path])
    assert len(rows) == 2
    assert rows[0]["evaluation"] == "fixture"
    assert rows[1]["threshold"] == 0.5
    assert rows[1]["mean_iou"] == 0.82


def test_training_comparison_plots_synthetic_csvs_without_models(tmp_path):
    columns = ["epoch", "train/box_loss", "val/box_loss", "train/cls_loss", "val/cls_loss",
               "metrics/mAP50(B)", "metrics/mAP50-95(B)", "metrics/precision(B)", "metrics/recall(B)"]
    csv_a = tmp_path / "e0.csv"
    csv_b = tmp_path / "e1.csv"
    header = ",".join(columns)
    csv_a.write_text(header + "\n1,2,3,4,5,.5,.3,.6,.7\n2,1.8,2.8,3.8,4.8,.6,.4,.7,.8\n", encoding="utf-8")
    csv_b.write_text(header + "\n1,2.1,3.1,4.1,5.1,.4,.2,.5,.6\n2,1.7,2.7,3.7,4.7,.7,.5,.8,.9\n", encoding="utf-8")
    output = tmp_path / "plots"
    assert plot_training_comparison({"E0": csv_a, "E1": csv_b}, output) == 8
    assert len(list(output.glob("*.png"))) == 8
    assert len(list(output.glob("*.svg"))) == 8
