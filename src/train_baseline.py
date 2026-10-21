"""Träna majoritetsbaseline och spara prediktioner för valideringsdata.

Recensionernas text används inte. Majoritetsklassen bestäms endast från
träningsetiketterna och sparas för att kunna återanvändas vid sluttestet.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_NAME = "majority_baseline"
DEFAULT_TRAIN_PATH = PROJECT_ROOT / "data" / "splits" / "train.csv"
DEFAULT_VALIDATION_PATH = PROJECT_ROOT / "data" / "splits" / "validation.csv"
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "baseline.json"
DEFAULT_PREDICTIONS_PATH = (
    PROJECT_ROOT / "data" / "processed" / "baseline_validation_predictions.csv"
)
DEFAULT_REPORT_PATH = PROJECT_ROOT / "reports" / "baseline.md"


@dataclass(frozen=True)
class MajorityBaseline:
    """Majoritetsklass och klassantal från träningsmaterialet."""

    majority_class: int
    class_counts: dict[int, int]


def _validate_labels(labels: pd.Series) -> None:
    if labels.empty:
        raise ValueError("Underlaget får inte vara tomt.")
    if not pd.api.types.is_integer_dtype(labels) or not labels.isin([0, 1]).all():
        raise ValueError("Etiketterna måste vara heltalen 0 eller 1 utan saknade värden.")


def load_split(path: Path | str) -> pd.DataFrame:
    """Läs bara ID, etikett och dubblettgrupp; recensionstexten används inte."""
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(
            f"Hittar inte {source}. Datauppdelningen från src/split_data.py måste finnas."
        )
    columns = ["review_id", "label", "duplicate_group_id"]
    reviews = pd.read_csv(
        source,
        usecols=columns,
        dtype={column: "int64" for column in columns},
        keep_default_na=False,
    )
    _validate_labels(reviews["label"])
    if reviews["review_id"].duplicated().any():
        raise ValueError(f"review_id måste vara unikt i {source}.")
    return reviews


def fit_baseline(train_labels: pd.Series) -> MajorityBaseline:
    """Välj majoriteten i träning. Vid lika antal väljs alltid klass 0."""
    _validate_labels(train_labels)
    counts = {label: int(train_labels.eq(label).sum()) for label in (0, 1)}
    majority_class = 0 if counts[0] >= counts[1] else 1
    return MajorityBaseline(majority_class=majority_class, class_counts=counts)


def predict_baseline(
    model: MajorityBaseline,
    reviews: pd.DataFrame,
    split: str,
) -> pd.DataFrame:
    """Använd den redan valda klassen, även om utvärderingsdatan är annorlunda."""
    if split not in ("validation", "test"):
        raise ValueError("split måste vara 'validation' eller 'test'.")
    return pd.DataFrame(
        {
            "review_id": reviews["review_id"].to_numpy(),
            "y_true": reviews["label"].to_numpy(),
            "y_pred": model.majority_class,
            "model": MODEL_NAME,
            "split": split,
        }
    )


def save_baseline(model: MajorityBaseline, path: Path | str) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": MODEL_NAME,
        "majority_class": model.majority_class,
        "class_counts": model.class_counts,
        "tie_break_class": 0,
    }
    target.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def load_baseline(path: Path | str) -> MajorityBaseline:
    """Ladda den sparade baselinen utan att läsa eller anpassa träningsdata."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload["model"] != MODEL_NAME:
        raise ValueError("Modellfilen innehåller inte en majoritetsbaseline.")
    counts = {label: int(payload["class_counts"][str(label)]) for label in (0, 1)}
    majority_class = int(payload["majority_class"])
    expected = 0 if counts[0] >= counts[1] else 1
    if min(counts.values()) < 0 or sum(counts.values()) == 0 or majority_class != expected:
        raise ValueError("Modellfilens klassantal och majoritetsklass är inte giltiga.")
    return MajorityBaseline(majority_class=majority_class, class_counts=counts)


def _display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path)


def run_baseline(
    train_path: Path = DEFAULT_TRAIN_PATH,
    validation_path: Path = DEFAULT_VALIDATION_PATH,
    model_path: Path = DEFAULT_MODEL_PATH,
    predictions_path: Path = DEFAULT_PREDICTIONS_PATH,
    report_path: Path = DEFAULT_REPORT_PATH,
) -> tuple[MajorityBaseline, pd.DataFrame]:
    train = load_split(train_path)
    validation = load_split(validation_path)
    for column in ("review_id", "duplicate_group_id"):
        if set(train[column]) & set(validation[column]):
            raise ValueError(f"{column} överlappar mellan träning och validering.")

    model = fit_baseline(train["label"])
    predictions = predict_baseline(model, validation, split="validation")
    save_baseline(model, model_path)
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(predictions_path, index=False, encoding="utf-8")

    label_name = "negativ" if model.majority_class == 0 else "positiv"
    report = f"""# Majoritetsbaseline

## Metod

Baseline väljer den vanligaste klassen enbart från träningsmaterialets etiketter.
Vid lika klassantal väljs klass 0. Samma klass förutsägs för varje valideringsrad.
Recensionernas text och innehåll används inte.

Baseline visar vad som kan uppnås utan textanalys och ger en enkel jämförelsenivå
för TF-IDF och logistisk regression. Majoritetsklassens dominans kan samtidigt
ge missvisande accuracy; jämförelsen ska därför även använda macro-F1 och
recall för negativa recensioner.

## Träningsmaterial

Källa: {_display_path(train_path)}

| Klass | Antal |
| --- | --- |
| Negativ (0) | {model.class_counts[0]} |
| Positiv (1) | {model.class_counts[1]} |
| Totalt | {sum(model.class_counts.values())} |

Vald baselineklass: {model.majority_class} ({label_name}).

## Valideringsprediktioner

- Källa: {_display_path(validation_path)}
- Antal prediktioner: {len(predictions)}
- Fil: {_display_path(predictions_path)}
- Kolumner: review_id, y_true, y_pred, model, split.

Utvärderingsmåtten beräknas i projektets gemensamma utvärdering.
Det här steget läser inte testdata och redovisar inga testresultat.

## Återanvändning vid sluttest

Modellen sparas lokalt i {_display_path(model_path)}.
Funktionen load_baseline laddar den sparade klassen. predict_baseline kan
därefter användas med split="test" utan att majoritetsklassen beräknas om.
"""
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
    return model, predictions


def main() -> None:
    parser = argparse.ArgumentParser(description="Träna och dokumentera majoritetsbaseline.")
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN_PATH)
    parser.add_argument("--validation", type=Path, default=DEFAULT_VALIDATION_PATH)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS_PATH)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT_PATH)
    args = parser.parse_args()
    model, predictions = run_baseline(
        args.train, args.validation, args.model, args.predictions, args.report
    )
    print(f"Träningsrader per klass: {model.class_counts}")
    print(f"Vald baselineklass: {model.majority_class}")
    print(f"Valideringsprediktioner: {len(predictions)}")
    print(f"Modell: {args.model}")
    print(f"Prediktioner: {args.predictions}")
    print(f"Rapport: {args.report}")


if __name__ == "__main__":
    main()
