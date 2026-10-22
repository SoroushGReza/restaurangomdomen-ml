"""Slutjämförelse av sparad baseline och sparad textmodell på testdata.

Skriptet anpassar ingen modell och läser inte valideringsresultat in i
testfilen. Modellinställningarna ska inte ändras efter att testresultatet setts.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import sklearn

if __package__:
    from .evaluate import evaluate_predictions, explain_missing_predictions
    from .train_baseline import MODEL_NAME, load_baseline, predict_baseline
    from .train_model import load_reviews, predict_model
else:
    from evaluate import evaluate_predictions, explain_missing_predictions
    from train_baseline import MODEL_NAME, load_baseline, predict_baseline
    from train_model import load_reviews, predict_model

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRAIN_PATH = PROJECT_ROOT / "data" / "splits" / "train.csv"
DEFAULT_VALIDATION_PATH = PROJECT_ROOT / "data" / "splits" / "validation.csv"
DEFAULT_TEST_PATH = PROJECT_ROOT / "data" / "splits" / "test.csv"
DEFAULT_BASELINE_PATH = PROJECT_ROOT / "models" / "baseline.json"
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "tfidf_logistic_regression.joblib"
DEFAULT_METRICS_PATH = PROJECT_ROOT / "reports" / "metrics_test.csv"
DEFAULT_REPORT_PATH = PROJECT_ROOT / "reports" / "results.md"
DEFAULT_FIGURE_DIR = PROJECT_ROOT / "reports" / "figures"

METRIC_FIELDS = (
    ("macro_f1", "Macro-F1"),
    ("negative_recall", "Negativ recall"),
    ("accuracy", "Accuracy"),
)


def validate_split_separation(parts: dict[str, pd.DataFrame]) -> None:
    """Kontrollera etiketter, tomma texter och att delmängderna inte överlappar."""
    if len(parts) < 2:
        raise ValueError("Minst två delmängder krävs för överlappskontrollen.")
    for name, frame in parts.items():
        if frame.empty:
            raise ValueError(f"{name} innehåller inga rader.")
        if not frame["label"].isin([0, 1]).all():
            raise ValueError(f"{name} innehåller ogiltiga etiketter.")
        if frame["text"].str.strip().eq("").any():
            raise ValueError(f"{name} innehåller tomma texter.")
        if frame["review_id"].duplicated().any():
            raise ValueError(f"{name} innehåller duplicerade review_id.")
    names = list(parts)
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            for column in ("review_id", "duplicate_group_id"):
                if set(parts[left][column]) & set(parts[right][column]):
                    raise ValueError(f"{column} överlappar mellan {left} och {right}.")


def model_name_from_pipeline(model) -> str:
    ngram_range = tuple(model.named_steps["tfidf"].ngram_range)
    if ngram_range == (1, 1):
        return "tfidf_unigram"
    if ngram_range == (1, 2):
        return "tfidf_unigram_bigram"
    return f"tfidf_{ngram_range[0]}_{ngram_range[1]}"


def metrics_row(model_name: str, scores: dict, row_count: int) -> dict:
    matrix = scores["confusion_matrix"]
    return {
        "model": model_name,
        "split": "test",
        "n": row_count,
        "accuracy": scores["accuracy"],
        "macro_f1": scores["macro_f1"],
        "negative_recall": scores["negative_recall"],
        "precision_0": scores["precision"][0],
        "recall_0": scores["recall"][0],
        "f1_0": scores["f1"][0],
        "precision_1": scores["precision"][1],
        "recall_1": scores["recall"][1],
        "f1_1": scores["f1"][1],
        "cm_true0_pred0": matrix[0][0],
        "cm_true0_pred1": matrix[0][1],
        "cm_true1_pred0": matrix[1][0],
        "cm_true1_pred1": matrix[1][1],
    }


def save_confusion_figure(matrix: list[list[int]], path: Path, title: str) -> None:
    figure, axis = plt.subplots(figsize=(4.8, 4.2))
    axis.imshow(matrix, cmap="Blues")
    axis.set_xticks([0, 1], ["Negativ (0)", "Positiv (1)"])
    axis.set_yticks([0, 1], ["Negativ (0)", "Positiv (1)"])
    axis.set_xlabel("Förutsagd klass")
    axis.set_ylabel("Verklig klass")
    axis.set_title(title)
    for row_index, row in enumerate(matrix):
        for column_index, value in enumerate(row):
            axis.text(column_index, row_index, str(value), ha="center", va="center")
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=120)
    plt.close(figure)


def save_comparison_figure(scores_by_model: dict[str, dict], path: Path) -> None:
    figure, axis = plt.subplots(figsize=(7, 4.2))
    metric_keys = [key for key, _ in METRIC_FIELDS]
    metric_labels = [label for _, label in METRIC_FIELDS]
    model_names = list(scores_by_model)
    width = 0.35
    positions = list(range(len(metric_keys)))
    for offset, model_name in enumerate(model_names):
        values = [scores_by_model[model_name][key] for key in metric_keys]
        axis.bar([position + offset * width for position in positions], values, width=width, label=model_name)
    axis.set_xticks([position + width / 2 for position in positions], metric_labels)
    axis.set_ylim(0, 1)
    axis.set_ylabel("Värde")
    axis.set_title("Jämförelse på testdata")
    axis.legend()
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=120)
    plt.close(figure)


def _best_model(scores_by_model: dict[str, dict], metric: str) -> str:
    ranking = sorted(scores_by_model, key=lambda name: scores_by_model[name][metric], reverse=True)
    if scores_by_model[ranking[0]][metric] == scores_by_model[ranking[1]][metric]:
        return "lika resultat"
    return ranking[0]


def render_results(
    test_reviews: pd.DataFrame,
    baseline_model,
    text_model,
    text_model_name: str,
    scores_by_model: dict[str, dict],
) -> str:
    classifier = text_model.named_steps["classifier"]
    vectorizer = text_model.named_steps["tfidf"]
    counts = test_reviews["label"].value_counts().to_dict()
    header = "| Mått | " + " | ".join(scores_by_model) + " | Bäst |"
    separator = "| --- | " + " | ".join("---" for _ in scores_by_model) + " | --- |"
    metric_rows = []
    for key, label in METRIC_FIELDS:
        values = " | ".join(f"{scores[key]:.6f}" for scores in scores_by_model.values())
        metric_rows.append(f"| {label} | {values} | {_best_model(scores_by_model, key)} |")
    class_rows = []
    for label in (0, 1):
        for metric_name, title in (("precision", "Precision"), ("recall", "Recall"), ("f1", "F1")):
            values = " | ".join(f"{scores[metric_name][label]:.6f}" for scores in scores_by_model.values())
            class_rows.append(f"| {title} klass {label} | {values} |")
    matrix_lines = []
    for model_name, scores in scores_by_model.items():
        matrix = scores["confusion_matrix"]
        matrix_lines.append(
            f"- {model_name}: verklig 0 förutsades som [0: {matrix[0][0]}, 1: {matrix[0][1]}]; "
            f"verklig 1 förutsades som [0: {matrix[1][0]}, 1: {matrix[1][1]}]. "
            f"{explain_missing_predictions(scores['classes_not_predicted'])}"
        )
    return "\n".join(
        [
            "# Slututvärdering på testdata",
            "",
            "## Metod",
            "",
            "Baseline och textmodellen är de sparade modellerna från uppgift 05 och 06.",
            "Ingen modell anpassas här, och testdata används inte för modellval.",
            "Båda modellerna får exakt samma testrader. Prediktionerna matchas mot facit via `review_id`.",
            "Confusion matrix har klassordningen [0, 1], med verklig klass på raderna och förutsagd klass i kolumnerna.",
            "Valideringsresultaten ligger kvar i `reports/model_selection.md` och blandas inte in i `reports/metrics_test.csv`.",
            "",
            "## Personuppgifter",
            "",
            "Datakvalitetskontrollen identifierade 4 möjliga namnträffar.",
            "Skolans krav gäller behandling av personuppgifter i projektet, inte enbart publicering av exempel.",
            "Gruppen behöver stämma av med läraren hur träffarna ska hanteras innan fortsatt behandling.",
            "Den här utvärderingen gör ingen ny bearbetning av råfilen och citerar inga recensionstexter.",
            "",
            "## Testunderlag",
            "",
            f"- Rader: {len(test_reviews)}.",
            f"- Negativa (0): {int(counts.get(0, 0))}.",
            f"- Positiva (1): {int(counts.get(1, 0))}.",
            "- Träning, validering och test har giltiga etiketter, inga tomma texter och inga gemensamma ID:n eller dubblettgrupper.",
            "",
            "## Modellinställningar",
            "",
            f"- Baseline: alltid klass {baseline_model.majority_class}, beräknad tidigare från träningsmaterialet.",
            f"- Textmodell: `{text_model_name}`.",
            f"- TF-IDF `ngram_range={tuple(vectorizer.ngram_range)}`, `stop_words={vectorizer.stop_words}`.",
            f"- LogisticRegression `C={classifier.C}`, `max_iter={classifier.max_iter}`, `random_state={classifier.random_state}`.",
            f"- Scikit-learn {sklearn.__version__}.",
            "",
            "## Jämförelse",
            "",
            "Högre värde är bättre för alla tre mått. Tabellen redovisar det faktiska testresultatet,",
            "även om textmodellen inte är bättre än baseline på ett mått.",
            "",
            header,
            separator,
            *metric_rows,
            "",
            "| Mått | " + " | ".join(scores_by_model) + " |",
            "| --- | " + " | ".join("---" for _ in scores_by_model) + " |",
            *class_rows,
            "",
            "## Confusion matrix",
            "",
            *matrix_lines,
            "",
            "Figurerna finns i `reports/figures/`.",
            "",
            "## Begränsningar",
            "",
            "Underlaget är ett litet engelskspråkigt Yelp-urval utan neutral klass.",
            "Resultatet visar inte hur modellerna fungerar på svenska eller på alla restaurangers omdömen.",
            "Valideringsdata användes för modellvalet och ingår inte i den här jämförelsen.",
            "Baseline kan få en accuracy nära klassbalansen samtidigt som negativ recall blir 0,",
            "eftersom den alltid väljer samma klass.",
            "",
            "## Regel efter sluttestet",
            "",
            "Modellinställningarna ändras inte efter granskning av testresultatet.",
            "Felanalysen finns i `reports/error_analysis.md` och riskerna i `docs/risks.md`.",
            "Svenska texter och en eventuell neutral klass är fortfarande framtida arbete.",
            "",
        ]
    )


def run_final_evaluation(
    train_path: Path,
    validation_path: Path,
    test_path: Path,
    baseline_path: Path,
    model_path: Path,
    metrics_path: Path,
    report_path: Path,
    figure_dir: Path,
) -> pd.DataFrame:
    model_bytes = Path(model_path).read_bytes()
    baseline_bytes = Path(baseline_path).read_bytes()
    parts = {
        "train": load_reviews(train_path),
        "validation": load_reviews(validation_path),
        "test": load_reviews(test_path),
    }
    validate_split_separation(parts)
    test_reviews = parts["test"].sort_values("review_id", kind="mergesort").reset_index(drop=True)
    baseline_model = load_baseline(baseline_path)
    text_model = joblib.load(model_path)
    text_model_name = model_name_from_pipeline(text_model)
    predictions = {
        MODEL_NAME: predict_baseline(baseline_model, test_reviews, "test"),
        text_model_name: predict_model(text_model, test_reviews, text_model_name, "test"),
    }
    if Path(model_path).read_bytes() != model_bytes or Path(baseline_path).read_bytes() != baseline_bytes:
        raise RuntimeError("Slututvärderingen fick inte ändra de sparade modellfilerna.")
    expected_ids = test_reviews["review_id"].tolist()
    scores_by_model = {}
    rows = []
    for model_name, prediction in predictions.items():
        if prediction["review_id"].tolist() != expected_ids:
            raise ValueError(f"{model_name} utvärderades inte på samma testrader.")
        scores_by_model[model_name] = evaluate_predictions(test_reviews, prediction)
        rows.append(metrics_row(model_name, scores_by_model[model_name], len(test_reviews)))
    metrics = pd.DataFrame(rows)
    if not metrics["split"].eq("test").all():
        raise ValueError("Testfilen får bara innehålla testresultat.")
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(metrics_path, index=False, encoding="utf-8")
    save_confusion_figure(
        scores_by_model[MODEL_NAME]["confusion_matrix"],
        figure_dir / "confusion_baseline.png",
        "Baseline på testdata",
    )
    save_confusion_figure(
        scores_by_model[text_model_name]["confusion_matrix"],
        figure_dir / "confusion_text_model.png",
        f"{text_model_name} på testdata",
    )
    save_comparison_figure(scores_by_model, figure_dir / "metric_comparison.png")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        render_results(test_reviews, baseline_model, text_model, text_model_name, scores_by_model),
        encoding="utf-8",
    )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Utvärdera sparade modeller på testdata utan ny träning.")
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN_PATH)
    parser.add_argument("--validation", type=Path, default=DEFAULT_VALIDATION_PATH)
    parser.add_argument("--test", type=Path, default=DEFAULT_TEST_PATH)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE_PATH)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS_PATH)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--figures", type=Path, default=DEFAULT_FIGURE_DIR)
    args = parser.parse_args()
    metrics = run_final_evaluation(
        args.train,
        args.validation,
        args.test,
        args.baseline,
        args.model,
        args.metrics,
        args.report,
        args.figures,
    )
    print(metrics.to_string(index=False))
    print(f"Rapport: {args.report}")


if __name__ == "__main__":
    main()
