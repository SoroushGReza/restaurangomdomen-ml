"""Syntetiska testfall: inga projektresultat eller originalrecensioner."""
import joblib
import pandas as pd
import pytest
from src.evaluate import evaluate_predictions
from src.train_model import build_pipeline, predict_model, run_training, select_variant


def reviews(ids, texts, labels):
    return pd.DataFrame({"review_id": ids, "duplicate_group_id": ids, "text": texts, "label": labels})


def test_hand_calculated_metrics_and_id_alignment():
    expected = pd.DataFrame({"review_id": [1, 2, 3, 4], "label": [0, 0, 1, 1]})
    prediction = pd.DataFrame({"review_id": [4, 3, 2, 1], "y_true": [1, 1, 0, 0], "y_pred": [1, 0, 0, 0]})
    scores = evaluate_predictions(expected, prediction)
    # F1 klass 0 = 4/5, F1 klass 1 = 2/3, negativ recall = 2/2.
    assert scores["macro_f1"] == pytest.approx((4/5 + 2/3)/2)
    assert scores["negative_recall"] == 1
    prediction["y_pred"] = 1
    scores = evaluate_predictions(expected, prediction)
    assert scores["macro_f1"] == pytest.approx(1/3)
    assert scores["negative_recall"] == 0


@pytest.mark.parametrize("problem", ["missing", "wrong_truth", "duplicate", "invalid"])
def test_invalid_predictions_rejected(problem):
    expected = pd.DataFrame({"review_id": [1, 2], "label": [0, 1]})
    prediction = pd.DataFrame({"review_id": [1, 2], "y_true": [0, 1], "y_pred": [0, 1]})
    if problem == "missing":
        prediction = prediction.iloc[:1]
    elif problem == "wrong_truth":
        prediction.loc[0, "y_true"] = 1
    elif problem == "duplicate":
        prediction.loc[1, "review_id"] = 1
    else:
        prediction.loc[0, "y_pred"] = 3
    with pytest.raises(ValueError):
        evaluate_predictions(expected, prediction)


def test_selection_priority():
    def select(a, b):
        return select_variant({"tfidf_unigram": dict(zip(("macro_f1", "negative_recall"), a)),
                               "tfidf_unigram_bigram": dict(zip(("macro_f1", "negative_recall"), b))})
    assert select((.8, .5), (.7, 1)) == "tfidf_unigram"
    assert select((.8, .5), (.8, .6)) == "tfidf_unigram_bigram"
    assert select((.8, .6), (.8, .6)) == "tfidf_unigram"


def test_pipeline_settings():
    for ngrams in [(1, 1), (1, 2)]:
        model = build_pipeline(ngrams)
        assert model["tfidf"].ngram_range == ngrams
        assert model["tfidf"].stop_words is None
        assert model["classifier"].C == 1.0
        assert model["classifier"].max_iter == 1000
        assert model["classifier"].random_state == 42


def test_training_export_reload_and_no_validation_leakage(tmp_path):
    train = reviews([1, 2, 3, 4], ["bad cold meal", "awful slow service", "great fresh meal", "excellent friendly service"], [0, 0, 1, 1])
    validation = reviews([5, 6], ["bad validationonlytoken", "great delicious meal"], [0, 1])
    train_path, val_path = tmp_path / "train.csv", tmp_path / "validation.csv"
    train.to_csv(train_path, index=False)
    validation.to_csv(val_path, index=False)
    model_path, report_path = tmp_path / "model.joblib", tmp_path / "report.md"
    selected, scores = run_training(train_path, val_path, tmp_path / "predictions", model_path, report_path)
    saved = joblib.load(model_path)
    assert "validationonlytoken" not in saved["tfidf"].vocabulary_
    reference = build_pipeline(saved["tfidf"].ngram_range).fit(train["text"], train["label"])
    assert saved["tfidf"].vocabulary_ == reference["tfidf"].vocabulary_
    assert saved["classifier"].coef_ == pytest.approx(reference["classifier"].coef_)
    exported = pd.read_csv(tmp_path / "predictions/model_validation_predictions.csv")
    assert list(exported) == ["review_id", "y_true", "y_pred", "model", "split"]
    assert exported["review_id"].tolist() == [5, 6]
    assert exported["model"].eq(selected).all()
    assert exported["y_pred"].tolist() == saved.predict(validation["text"]).tolist()
    assert scores[selected] == evaluate_predictions(validation, exported)
    assert "majority_baseline" in report_path.read_text(encoding="utf-8")
    train_path.unlink()  # Återladdning och prediktion kräver inte träning igen.
    before = model_path.read_bytes()
    result = predict_model(joblib.load(model_path), validation, selected, "test")
    assert result["split"].eq("test").all()
    assert model_path.read_bytes() == before


@pytest.mark.parametrize("column", ["review_id", "duplicate_group_id", "text"])
def test_overlap_rejected_before_model_is_saved(tmp_path, column):
    train = reviews([1, 2], ["bad cold meal", "great fresh meal"], [0, 1])
    validation = reviews([3, 4], ["awful food", "excellent food"], [0, 1])
    validation.loc[0, column] = train.loc[0, column]
    train.to_csv(tmp_path / "train.csv", index=False)
    validation.to_csv(tmp_path / "validation.csv", index=False)
    model_path = tmp_path / "model.joblib"
    with pytest.raises(ValueError, match="överlappar"):
        run_training(tmp_path / "train.csv", tmp_path / "validation.csv", tmp_path / "out", model_path, tmp_path / "report.md")
    assert not model_path.exists()
