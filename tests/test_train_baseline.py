"""Tester med syntetiska data; Yelp-filer behöver inte finnas."""

from pathlib import Path

import pandas as pd
import pytest

from src.train_baseline import fit_baseline, load_baseline, predict_baseline, run_baseline


def _reviews(ids: list[int], labels: list[int]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "review_id": ids,
            "text": ["Syntetisk testtext."] * len(ids),
            "label": pd.Series(labels, dtype="int64"),
            "duplicate_group_id": ids,
        }
    )


def test_majority_comes_from_training_even_when_validation_has_opposite_majority() -> None:
    train = _reviews([1, 2, 3], [0, 1, 1])
    validation = _reviews([10, 11, 12], [0, 0, 1])

    model = fit_baseline(train["label"])
    predictions = predict_baseline(model, validation, "validation")

    assert model.majority_class == 1
    assert model.class_counts == {0: 1, 1: 2}
    assert predictions["y_pred"].tolist() == [1, 1, 1]


def test_tied_training_counts_choose_zero_regardless_of_row_order() -> None:
    for labels in ([1, 0, 1, 0], [0, 0, 1, 1]):
        assert fit_baseline(pd.Series(labels)).majority_class == 0


@pytest.mark.parametrize("labels", [[], [0, 2], [0, None]])
def test_empty_or_invalid_training_labels_are_rejected(labels) -> None:
    with pytest.raises(ValueError):
        fit_baseline(pd.Series(labels))


def test_saved_baseline_can_predict_test_without_training_data(tmp_path: Path) -> None:
    train_path = tmp_path / "train.csv"
    validation_path = tmp_path / "validation.csv"
    model_path = tmp_path / "models" / "baseline.json"
    predictions_path = tmp_path / "processed" / "predictions.csv"
    report_path = tmp_path / "reports" / "baseline.md"
    _reviews([1, 2, 3], [0, 1, 1]).to_csv(train_path, index=False)
    validation = _reviews([12, 10, 11], [0, 0, 1])
    validation.to_csv(validation_path, index=False)

    run_baseline(train_path, validation_path, model_path, predictions_path, report_path)

    saved = pd.read_csv(predictions_path)
    assert list(saved.columns) == ["review_id", "y_true", "y_pred", "model", "split"]
    assert saved["review_id"].tolist() == [12, 10, 11]
    assert saved["y_true"].tolist() == [0, 0, 1]
    assert saved["y_pred"].tolist() == [1, 1, 1]
    assert saved["model"].tolist() == ["majority_baseline"] * 3
    assert saved["split"].tolist() == ["validation"] * 3
    assert report_path.is_file()

    train_path.unlink()
    model_bytes = model_path.read_bytes()
    model = load_baseline(model_path)
    future_test = _reviews([20, 21, 22], [0, 0, 0])
    test_predictions = predict_baseline(model, future_test, "test")
    assert model.class_counts == {0: 1, 1: 2}
    assert test_predictions["y_pred"].tolist() == [1, 1, 1]
    assert test_predictions["split"].tolist() == ["test"] * 3
    assert model_path.read_bytes() == model_bytes


@pytest.mark.parametrize("column", ["review_id", "duplicate_group_id"])
def test_training_validation_overlap_is_rejected(tmp_path: Path, column: str) -> None:
    train_path = tmp_path / "train.csv"
    validation_path = tmp_path / "validation.csv"
    _reviews([1, 2, 3], [0, 1, 1]).to_csv(train_path, index=False)
    validation = _reviews([10, 11, 12], [0, 0, 1])
    validation.loc[0, column] = 1
    validation.to_csv(validation_path, index=False)

    with pytest.raises(ValueError, match=column):
        run_baseline(
            train_path,
            validation_path,
            tmp_path / "baseline.json",
            tmp_path / "predictions.csv",
            tmp_path / "baseline.md",
        )
    assert not (tmp_path / "baseline.json").exists()
