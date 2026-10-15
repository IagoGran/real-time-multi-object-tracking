from pathlib import Path
from types import SimpleNamespace

from src.training.pseudo_labels import generate_pseudo_labels


class Values(list):
    def tolist(self):
        return list(self)


class FakeModel:
    names = {0: "person", 1: "bicycle"}

    def __init__(self, results):
        self.results = results
        self.calls = []

    def predict(self, **kwargs):
        self.calls.append(kwargs)
        return iter(self.results)


def test_teacher_threshold_filters_person_boxes_and_cache_prevents_second_pass(tmp_path):
    image = tmp_path / "frame.jpg"
    image.touch()
    result = SimpleNamespace(
        path=str(image),
        names={0: "person", 1: "bicycle"},
        orig_shape=(100, 200),
        boxes=SimpleNamespace(
            xyxy=Values([[0, 0, 20, 40], [5, 5, 25, 45], [2, 2, 12, 12]]),
            conf=Values([0.8, 0.2, 0.95]),
            cls=Values([0, 0, 1]),
        ),
    )
    model = FakeModel([result])

    labels, cache_dir, cache_hit = generate_pseudo_labels(
        model, [image], "teacher.pt", 0.5, tmp_path / "cache"
    )
    assert not cache_hit
    assert labels[str(image.resolve())] == [(0.0, 0.0, 20.0, 40.0)]
    assert model.calls[0]["stream"] is True
    assert model.calls[0]["conf"] == 0.001

    cached, reused_dir, cache_hit = generate_pseudo_labels(
        model, [image], "teacher.pt", 0.5, tmp_path / "cache"
    )
    assert cache_hit and reused_dir == cache_dir
    assert cached == labels
    assert len(model.calls) == 1
