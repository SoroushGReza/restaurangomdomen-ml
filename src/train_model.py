"""Jämför två TF-IDF-pipelines på validering; läser aldrig testdata."""
from __future__ import annotations

import argparse
import hashlib
import warnings
from pathlib import Path

import joblib
import pandas as pd
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

if __package__:
    from .evaluate import evaluate_predictions
    from .train_baseline import fit_baseline, predict_baseline
else:
    from evaluate import evaluate_predictions
    from train_baseline import fit_baseline, predict_baseline

ROOT = Path(__file__).resolve().parents[1]
VARIANTS = {"tfidf_unigram": (1, 1), "tfidf_unigram_bigram": (1, 2)}


def load_reviews(path: Path) -> pd.DataFrame:
    data = pd.read_csv(path, keep_default_na=False, dtype={
        "review_id": "int64", "duplicate_group_id": "int64", "label": "int64", "text": "string"
    })
    required = {"review_id", "duplicate_group_id", "label", "text"}
    if not required.issubset(data.columns):
        raise ValueError(f"Saknade kolumner i {path}: {required - set(data.columns)}")
    if data.empty or not data["label"].isin([0, 1]).all():
        raise ValueError("Data måste innehålla rader med etikett 0 eller 1.")
    if data["review_id"].duplicated().any() or data["text"].str.strip().eq("").any():
        raise ValueError("Duplicerade review_id eller tomma texter upptäcktes.")
    return data


def build_pipeline(ngram_range: tuple[int, int]) -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=ngram_range, stop_words=None)),
        ("classifier", LogisticRegression(C=1.0, max_iter=1000, random_state=42)),
    ])


def predict_model(model: Pipeline, reviews: pd.DataFrame, name: str, split: str) -> pd.DataFrame:
    if split not in ("validation", "test"):
        raise ValueError("split måste vara validation eller test.")
    return pd.DataFrame({
        "review_id": reviews["review_id"].to_numpy(),
        "y_true": reviews["label"].to_numpy(),
        "y_pred": model.predict(reviews["text"]),
        "model": name,
        "split": split,
    })


def select_variant(scores: dict[str, dict]) -> str:
    # Exakta lika värden bryts med negativ recall, därefter unigrams.
    return max(VARIANTS, key=lambda name: (
        scores[name]["macro_f1"], scores[name]["negative_recall"], name == "tfidf_unigram"
    ))


def run_training(train_path: Path, validation_path: Path, output_dir: Path, model_path: Path, report_path: Path):
    train, validation = load_reviews(train_path), load_reviews(validation_path)
    if set(train["label"]) != {0, 1}:
        raise ValueError("Träningsmaterialet måste innehålla båda klasserna.")
    for column in ("review_id", "duplicate_group_id"):
        if set(train[column]) & set(validation[column]):
            raise ValueError(f"{column} överlappar mellan träning och validering.")
    # Extra kontroll utöver de sparade grupp-ID:na.
    normalize = lambda s: s.str.replace(r"\s+", " ", regex=True).str.strip().str.casefold()
    if set(normalize(train["text"])) & set(normalize(validation["text"])):
        raise ValueError("Normaliserade texter överlappar mellan träning och validering.")

    models, predictions, scores = {}, {}, {}
    baseline = predict_baseline(fit_baseline(train["label"]), validation, "validation")
    baseline_scores = evaluate_predictions(validation, baseline)
    for name, ngrams in VARIANTS.items():
        model = build_pipeline(ngrams)
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            model.fit(train["text"], train["label"])
        models[name] = model
        predictions[name] = predict_model(model, validation, name, "validation")
        scores[name] = evaluate_predictions(validation, predictions[name])
    selected = select_variant(scores)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, prediction in predictions.items():
        prediction.to_csv(output_dir / f"{name}_validation_predictions.csv", index=False)
    predictions[selected].to_csv(output_dir / "model_validation_predictions.csv", index=False)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(models[selected], model_path)

    rows = [f"| majority_baseline | – | {baseline_scores['macro_f1']:.6f} | {baseline_scores['negative_recall']:.6f} |"]
    rows += [f"| {name} | {VARIANTS[name]} | {scores[name]['macro_f1']:.6f} | {scores[name]['negative_recall']:.6f} |" for name in VARIANTS]
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    delta = scores[selected]["macro_f1"] - baseline_scores["macro_f1"]
    report = f'''# Modellval: TF-IDF och logistisk regression

## Metod

En Pipeline anpassar TF-IDF och LogisticRegression enbart på train.csv.
Endast text används som indata. ID och etikett används inte som egenskaper.
Två alternativ jämförs: enskilda ord (1, 1) samt enskilda ord och ordpar (1, 2).
Gemensamma inställningar: C=1.0, max_iter=1000, random_state=42,
stop_words=None. Övriga inställningar är scikit-learns standardvärden.
Scikit-learn-version: {sklearn.__version__}.

## Underlag

- Träningsrader: {len(train)}.
- Valideringsrader: {len(validation)}.
- SHA256 train.csv: `{digest(train_path)}`.
- SHA256 validation.csv: `{digest(validation_path)}`.
- Baseline beräknas på exakt samma träningsetiketter som textmodellerna.
- Alla alternativ utvärderas med samma evaluate_predictions och samma valideringsrader.

## Valideringsresultat

| Modell | ngram_range | Macro-F1 | Negativ recall |
| --- | --- | --- | --- |
{chr(10).join(rows)}

Vald textmodell: **{selected}**.
Urval: högst macro-F1, därefter negativ recall, därefter enskilda ord.
Urvalet använder orundade värden; tabellen visar sex decimaler.
Skillnad i macro-F1 mot baseline: {delta:+.6f}.
Macro-F1 ger de två klassernas F1 lika vikt. Negativ recall visar andelen
negativa recensioner som hittas. Odefinierade mått sätts till 0.

## Sparad modell och begränsningar

Vald pipeline sparas med joblib och återanvänds utan ny fit vid sluttestet.
Ingen omträning på train + validation görs. Testdata har inte lästs.
Validering har använts för modellval och är därför inte ett oberoende sluttest.
Resultaten gäller detta begränsade engelskspråkiga Yelp-underlag och visar
inte hur modellen fungerar på svenska eller på alla restaurangers omdömen.
Fler mått, confusion matrix och felanalys kompletteras i efterföljande uppgifter.
'''
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
    print(f"Vald modell: {selected}\nModell: {model_path}\nRapport: {report_path}")
    return selected, scores


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=ROOT / "data/splits/train.csv")
    parser.add_argument("--validation", type=Path, default=ROOT / "data/splits/validation.csv")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/processed")
    parser.add_argument("--model", type=Path, default=ROOT / "models/tfidf_logistic_regression.joblib")
    parser.add_argument("--report", type=Path, default=ROOT / "reports/model_selection.md")
    args = parser.parse_args()
    run_training(args.train, args.validation, args.output_dir, args.model, args.report)


if __name__ == "__main__":
    main()
