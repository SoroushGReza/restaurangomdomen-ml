"""Slututvärdering på syntetiska delmängder, utan projektets originalrecensioner."""
import json

import joblib
import pandas as pd
import pytest

from src.final_evaluation import run_final_evaluation, validate_split_separation
from src.train_baseline import MajorityBaseline, save_baseline
from src.train_model import build_pipeline


def reviews(ids, texts, labels, groups=None):
    return pd.DataFrame(
        {
            "review_id": ids,
            "duplicate_group_id": ids if groups is None else groups,
            "text": texts,
            "label": labels,
        }
    )


def write_splits(directory):
    parts = {
        "train": reviews([1, 2, 3, 4], ["bad cold meal", "awful slow service", "great fresh meal", "excellent friendly service"], [0, 0, 1, 1]),
        "validation": reviews([5, 6], ["bad validation meal", "great validation meal"], [0, 1]),
        "test": reviews([7, 8], ["secret-test-negative-token bad", "secret-test-positive-token great"], [0, 1]),
    }
    for name, frame in parts.items():
        frame.to_csv(directory / f"{name}.csv", index=False)
    return parts


def test_split_separation_rejects_overlap_empty_text_and_invalid_label():
    train = reviews([1, 2], ["bad meal", "great meal"], [0, 1])
    validation = reviews([3], ["ok meal"], [0])
    validate_split_separation({"train": train, "validation": validation})

    overlapping = reviews([1], ["another meal"], [1])
    with pytest.raises(ValueError, match="review_id"):
        validate_split_separation({"train": train, "validation": overlapping})

    shared_group = reviews([9], ["another meal"], [1], groups=[1])
    with pytest.raises(ValueError, match="duplicate_group_id"):
        validate_split_separation({"train": train, "validation": shared_group})

    empty_text = reviews([9], ["   "], [0])
    with pytest.raises(ValueError, match="tomma texter"):
        validate_split_separation({"train": train, "validation": empty_text})

    invalid = reviews([9], ["bad meal"], [2])
    with pytest.raises(ValueError, match="ogiltiga etiketter"):
        validate_split_separation({"train": train, "validation": invalid})


def test_final_evaluation_uses_saved_models_without_refitting(tmp_path):
    parts = write_splits(tmp_path)
    model_path = tmp_path / "model.joblib"
    baseline_path = tmp_path / "baseline.json"
    pipeline = build_pipeline((1, 1))
    pipeline.fit(parts["train"]["text"], parts["train"]["label"])
    joblib.dump(pipeline, model_path)
    save_baseline(MajorityBaseline(majority_class=1, class_counts={0: 1, 1: 2}), baseline_path)
    model_bytes = model_path.read_bytes()
    baseline_bytes = baseline_path.read_bytes()

    metrics = run_final_evaluation(
        tmp_path / "train.csv",
        tmp_path / "validation.csv",
        tmp_path / "test.csv",
        baseline_path,
        model_path,
        tmp_path / "metrics_test.csv",
        tmp_path / "results.md",
        tmp_path / "figures",
    )

    assert model_path.read_bytes() == model_bytes
    assert baseline_path.read_bytes() == baseline_bytes
    assert metrics["split"].eq("test").all()
    assert set(metrics["model"]) == {"majority_baseline", "tfidf_unigram"}
    assert metrics["n"].eq(2).all()
    report = (tmp_path / "results.md").read_text(encoding="utf-8")
    assert "secret-test-negative-token" not in report
    assert "Modellinställningarna ändras inte" in report
    assert "4 möjliga namnträffar" in report
    for figure in ("confusion_baseline.png", "confusion_text_model.png", "metric_comparison.png"):
        assert (tmp_path / "figures" / figure).is_file()
    saved = json.loads(baseline_path.read_text(encoding="utf-8"))
    assert saved["majority_class"] == 1
