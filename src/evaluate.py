"""Gemensamma mått för baseline och textmodell.

Macro-F1 är huvudmåttet eftersom båda klasserna är lika viktiga.
Negativ recall visar om negativa omdömen hittas. Accuracy kan se hög ut
även när en modell alltid väljer majoritetsklassen. Precision, recall och F1
per klass samt confusion matrix förklarar den skillnaden.
"""
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    recall_score,
)

CLASS_LABELS = [0, 1]


def evaluate_predictions(expected: pd.DataFrame, predictions: pd.DataFrame) -> dict:
    """Matcha prediktioner mot facit via review_id, aldrig via radposition.

    Confusion matrix har klassordningen [0, 1]. Rader är verklig klass och
    kolumner är förutsagd klass. Om en klass aldrig förutsägs blir dess
    precision 0, eftersom zero_division=0 används när nämnaren saknas.
    """
    if expected.empty or predictions.empty:
        raise ValueError("Utvärderingsunderlaget får inte vara tomt.")
    for frame in (expected, predictions):
        if frame["review_id"].isna().any() or frame["review_id"].duplicated().any():
            raise ValueError("review_id måste vara unikt och får inte saknas.")
    if set(expected["review_id"]) != set(predictions["review_id"]):
        raise ValueError("Prediktionernas review_id matchar inte facit.")
    aligned = predictions.set_index("review_id").loc[expected["review_id"]]
    truth = expected["label"].to_numpy()
    predicted = aligned["y_pred"].to_numpy()
    if not (aligned["y_true"].to_numpy() == truth).all():
        raise ValueError("y_true matchar inte facit för respektive review_id.")
    for values in (expected["label"], aligned["y_pred"]):
        if not values.isin(CLASS_LABELS).all():
            raise ValueError("Etiketter och prediktioner måste vara 0 eller 1.")

    precision, recall, f1, support = precision_recall_fscore_support(
        truth,
        predicted,
        labels=CLASS_LABELS,
        average=None,
        zero_division=0,
    )
    matrix = confusion_matrix(truth, predicted, labels=CLASS_LABELS)
    predicted_classes = set(int(value) for value in predicted)
    return {
        "macro_f1": float(f1_score(truth, predicted, labels=CLASS_LABELS, average="macro", zero_division=0)),
        "negative_recall": float(recall_score(truth, predicted, pos_label=0, average="binary", zero_division=0)),
        "accuracy": float(accuracy_score(truth, predicted)),
        "precision": {label: float(precision[index]) for index, label in enumerate(CLASS_LABELS)},
        "recall": {label: float(recall[index]) for index, label in enumerate(CLASS_LABELS)},
        "f1": {label: float(f1[index]) for index, label in enumerate(CLASS_LABELS)},
        "support": {label: int(support[index]) for index, label in enumerate(CLASS_LABELS)},
        "confusion_matrix": [[int(value) for value in row] for row in matrix.tolist()],
        "classes_not_predicted": [label for label in CLASS_LABELS if label not in predicted_classes],
    }


def explain_missing_predictions(classes_not_predicted: list[int]) -> str:
    """Förklara precision 0 när en modell aldrig förutsäger en klass."""
    if not classes_not_predicted:
        return "Modellen förutsäger båda klasserna."
    labels = ", ".join(str(label) for label in classes_not_predicted)
    return (
        f"Modellen förutsäger aldrig klass {labels}. "
        "Precision för den klassen sätts till 0 eftersom det inte finns några sådana prediktioner."
    )
