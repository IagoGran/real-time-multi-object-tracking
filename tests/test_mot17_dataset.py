from src.evaluation.mot17_dataset import (
    discover_sequence_paths,
    load_sequence,
)


def test_sequence_keeps_real_frame_numbers_and_applies_limit(tmp_path):
    sequence_path = tmp_path / "MOT17-02-FRCNN"
    frames_path = sequence_path / "img1"
    gt_path = sequence_path / "gt" / "gt.txt"
    frames_path.mkdir(parents=True)
    gt_path.parent.mkdir()

    for name in ("000003.jpg", "000010.jpg", "000025.jpg"):
        (frames_path / name).touch()

    gt_path.write_text(
        "3,1,10,20,30,40,1,1,1.0\n"
        "3,2,1,2,3,4,0,1,1.0\n"
        "10,3,2,3,4,5,1,2,1.0\n"
        "25,4,5,6,7,8,1,1,1.0\n",
        encoding="utf-8",
    )

    sequence = load_sequence(sequence_path, max_frames=2)

    assert [frame.number for frame in sequence.frames] == [3, 10]
    assert sequence.ground_truth == {
        3: [(10.0, 20.0, 40.0, 60.0)],
        25: [(5.0, 6.0, 12.0, 14.0)],
    }


def test_all_train_uses_only_one_variant_per_scene(tmp_path):
    train_path = tmp_path / "train"
    train_path.mkdir()
    for variant in ("DPM", "FRCNN", "SDP"):
        (train_path / f"MOT17-02-{variant}").mkdir()
    (train_path / "MOT17-04-FRCNN").mkdir()

    sequences = discover_sequence_paths(
        all_train=True,
        train_path=train_path,
    )

    assert [sequence.name for sequence in sequences] == [
        "MOT17-02-FRCNN",
        "MOT17-04-FRCNN",
    ]
