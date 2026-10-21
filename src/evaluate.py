"""Gemensamma mått för modellval. Byggs vidare i uppgift 07."""
import pandas as pd
from sklearn.metrics import f1_score, recall_score


def evaluate_predictions(expected: pd.DataFrame, predictions: pd.DataFrame) -> dict:
    """Matcha prediktioner mot facit via review_id, aldrig via radposition."""
    if expected.empty or predictions.empty:
        raise ValueError("Utvärderingsunderlaget får inte vara tomt.")
    for frame in (expected, predictions):
        if frame["review_id"].isna().any() or frame["review_id"].duplicated().any():
            raise ValueError("review_id måste vara unikt och får inte saknas.")
    if set(expected["review_id"]) != set(predictions["review_id"]):
        raise ValueError("Prediktionernas review_id matchar inte facit.")
    aligned = predictions.set_index("review_id").loc[expected["review_id"]]
    truth = expected["label"].to_numpy()
    if not (aligned["y_true"].to_numpy() == truth).all():
        raise ValueError("y_true matchar inte facit för respektive review_id.")
    for values in (expected["label"], aligned["y_pred"]):
        if not values.isin([0, 1]).all():
            raise ValueError("Etiketter och prediktioner måste vara 0 eller 1.")
    return {
        "macro_f1": float(f1_score(truth, aligned["y_pred"], labels=[0, 1], average="macro", zero_division=0)),
        "negative_recall": float(recall_score(truth, aligned["y_pred"], pos_label=0, average="binary", zero_division=0)),
    }
