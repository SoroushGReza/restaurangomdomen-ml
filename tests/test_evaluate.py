"""Handräknade mått och matchning via review_id."""
import pandas as pd
import pytest

from src.evaluate import evaluate_predictions, explain_missing_predictions


def test_hand_calculated_metrics_follow_review_id_not_row_order():
    expected = pd.DataFrame({"review_id": [1, 2, 3, 4], "label": [0, 0, 1, 1]})
    prediction = pd.DataFrame(
        {"review_id": [4, 3, 2, 1], "y_true": [1, 1, 0, 0], "y_pred": [1, 0, 0, 0]}
    )

    scores = evaluate_predictions(expected, prediction)

    # Ordning efter review_id: sanna [0, 0, 1, 1], prediktioner [0, 0, 0, 1].
    # Klass 0: precision 2/3, recall 1, F1 4/5. Klass 1: precision 1, recall 1/2, F1 2/3.
    assert scores["confusion_matrix"] == [[2, 0], [1, 1]]
    assert scores["accuracy"] == pytest.approx(3 / 4)
    assert scores["precision"][0] == pytest.approx(2 / 3)
    assert scores["recall"][0] == pytest.approx(1)
    assert scores["f1"][0] == pytest.approx(4 / 5)
    assert scores["precision"][1] == pytest.approx(1)
    assert scores["recall"][1] == pytest.approx(1 / 2)
    assert scores["f1"][1] == pytest.approx(2 / 3)
    assert scores["macro_f1"] == pytest.approx((4 / 5 + 2 / 3) / 2)
    assert scores["negative_recall"] == pytest.approx(1)
    assert scores["support"] == {0: 2, 1: 2}
    assert scores["classes_not_predicted"] == []


def test_class_never_predicted_gets_zero_precision():
    expected = pd.DataFrame({"review_id": [1, 2, 3, 4], "label": [0, 0, 1, 1]})
    prediction = pd.DataFrame(
        {"review_id": [1, 2, 3, 4], "y_true": [0, 0, 1, 1], "y_pred": [1, 1, 1, 1]}
    )

    scores = evaluate_predictions(expected, prediction)

    assert scores["confusion_matrix"] == [[0, 2], [0, 2]]
    assert scores["accuracy"] == pytest.approx(1 / 2)
    assert scores["precision"][0] == 0
    assert scores["recall"][0] == 0
    assert scores["f1"][0] == 0
    assert scores["f1"][1] == pytest.approx(2 / 3)
    assert scores["macro_f1"] == pytest.approx(1 / 3)
    assert scores["classes_not_predicted"] == [0]
    assert "aldrig klass 0" in explain_missing_predictions(scores["classes_not_predicted"])


def test_missing_prediction_id_is_rejected():
    expected = pd.DataFrame({"review_id": [1, 2], "label": [0, 1]})
    prediction = pd.DataFrame({"review_id": [1], "y_true": [0], "y_pred": [0]})

    with pytest.raises(ValueError, match="review_id"):
        evaluate_predictions(expected, prediction)
