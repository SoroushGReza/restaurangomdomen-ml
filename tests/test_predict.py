"""Demonstration med en liten pipeline, utan projektets originalrecensioner."""
import joblib
import pytest

from src.predict import predict_review
from src.train_model import build_pipeline


def fitted_model(tmp_path):
    model = build_pipeline((1, 1))
    model.fit(
        ["bad cold meal", "awful slow service", "great fresh meal", "excellent friendly service"],
        [0, 0, 1, 1],
    )
    path = tmp_path / "model.joblib"
    joblib.dump(model, path)
    return joblib.load(path), path


def test_blank_review_is_rejected(tmp_path):
    model, _ = fitted_model(tmp_path)

    with pytest.raises(ValueError, match="Tom eller blank"):
        predict_review(model, "   ")


def test_saved_pipeline_predicts_without_refitting(tmp_path):
    model, path = fitted_model(tmp_path)
    before = path.read_bytes()

    assert predict_review(model, "great fresh meal") == "positiv"
    assert predict_review(model, "bad cold meal") == "negativ"
    assert path.read_bytes() == before
