"""Demonstration med en liten pipeline, utan projektets originalrecensioner."""

import joblib
import pytest

from src.predict import predict_review
from src.train_model import build_pipeline


def fitted_model(tmp_path):
    model = build_pipeline((1, 1))
    model.fit(
        [
            "bad cold meal",
            "awful slow service",
            "great fresh meal",
            "excellent friendly service",
        ],
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


@pytest.mark.parametrize("review", ["", "   "])
def test_cli_rejects_blank_input(monkeypatch, capsys, review):
    from src.predict import main

    monkeypatch.setattr("sys.argv", ["predict.py", review])

    with pytest.raises(SystemExit) as error:
        main()

    assert error.value.code == 2
    assert "Tom eller blank" in capsys.readouterr().err


def test_cli_explains_missing_model(tmp_path, monkeypatch, capsys):
    from src.predict import main

    monkeypatch.setattr(
        "sys.argv",
        [
            "predict.py",
            "Great food",
            "--model",
            str(tmp_path / "missing.joblib"),
        ],
    )

    with pytest.raises(SystemExit) as error:
        main()

    assert error.value.code == 2
    message = capsys.readouterr().err
    assert "Modellfilen saknas" in message
    assert "run_pipeline.py" in message
