"""Klassificera en ny engelsk restaurangrecension med den sparade pipelinen.

Demonstrationen tränar inte om modellen. Texterna som skrivs in här är
illustrationer och ingår inte i den uppmätta testprestandan.
Modellen är utvecklad för engelska recensioner och har ingen neutral klass.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "tfidf_logistic_regression.joblib"


def predict_review(model, review: str) -> str:
    """Returnera positiv eller negativ. Tom text avvisas."""
    if review.strip() == "":
        raise ValueError(
            "Skriv en recension. Tom eller blank text kan inte klassificeras."
        )
    prediction = int(model.predict([review])[0])
    if prediction == 1:
        return "positiv"
    if prediction == 0:
        return "negativ"
    raise ValueError(f"Modellen returnerade den okända klassen {prediction}.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Klassificera en engelsk restaurangrecension som positiv eller negativ."
    )
    parser.add_argument(
        "review",
        nargs="?",
        help="Recensionstexten. Utan argument frågar programmet efter den.",
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    args = parser.parse_args()
    review = args.review
    if review is None:
        review = input("Skriv en engelsk restaurangrecension: ")
    if not review.strip():
        parser.error("Skriv en recension. Tom eller blank text kan inte klassificeras.")

    if not args.model.is_file():
        parser.error(
            f"Modellfilen saknas: {args.model}. "
            "Kör src/run_pipeline.py med projektets Pythonmiljö för att skapa modellen."
        )

    model = joblib.load(args.model)
    print("Modellen är utvecklad för engelska recensioner och har ingen neutral klass.")
    print(predict_review(model, review))


if __name__ == "__main__":
    main()
