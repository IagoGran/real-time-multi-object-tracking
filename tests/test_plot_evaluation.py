import json

from src.analysis.plot_evaluation import flatten_evaluations


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
