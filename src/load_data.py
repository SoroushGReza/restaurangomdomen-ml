"""Läs in Yelp-delen av UCI Sentiment Labelled Sentences.

Originalfilen är tabbseparerad, saknar rubrikrad och har en recension per rad:
``text<tab>etikett``. Etikett 0 är negativ och etikett 1 är positiv.
"""

from __future__ import annotations

import argparse
import csv
import io
import zipfile
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW_PATH = PROJECT_ROOT / "data" / "raw" / "yelp_labelled.txt"

UCI_DATASET_URL = "https://archive.ics.uci.edu/dataset/331/sentiment+labelled+sentences"
UCI_ZIP_URL = "https://archive.ics.uci.edu/static/public/331/sentiment+labelled+sentences.zip"
YELP_ZIP_MEMBER = "sentiment labelled sentences/yelp_labelled.txt"

EXPECTED_ROW_COUNT = 1000
EXPECTED_CLASS_COUNTS = {0: 500, 1: 500}
COLUMNS = ["review_id", "text", "label"]

# SHA-256 för yelp_labelled.txt hämtad 2026-10-20 från UCI.
YELP_FILE_SHA256 = "c76468b7b5c6e56a0804d728345c5f84aa2142ddb214420f61cc9cfd4c00d2ea"


def load_yelp_reviews(path: Path | str | None = None) -> pd.DataFrame:
    """Läs den tabbseparerade Yelp-filen till en dataframe.

    Första dataraden används som data, inte som kolumnnamn.
    ``review_id`` är radnumret i originalfilen, med 1 för första raden.
    """
    source = DEFAULT_RAW_PATH if path is None else Path(path)
    if not source.is_file():
        raise FileNotFoundError(
            f"Hittar inte Yelp-filen: {source}. "
            "Hämta och packa upp den enligt docs/data_source.md."
        )

    file_lines = source.read_text(encoding="utf-8").splitlines()
    reviews = pd.read_csv(
        source,
        sep="\t",
        header=None,
        names=["text", "label"],
        encoding="utf-8",
        quoting=csv.QUOTE_NONE,
        dtype={"text": "string", "label": "int64"},
        keep_default_na=False,
    )
    if len(reviews) != len(file_lines):
        raise ValueError(
            f"Inläsningen gav {len(reviews)} rader, men filen har {len(file_lines)} rader. "
            "Kontrollera att första dataraden inte tolkas som rubrik och att "
            "citattecken inte slår ihop rader."
        )

    reviews.insert(0, "review_id", range(1, len(reviews) + 1))
    return reviews


def class_counts(reviews: pd.DataFrame) -> dict[int, int]:
    """Räkna recensioner per etikett."""
    counts = reviews["label"].value_counts().to_dict()
    return {int(label): int(count) for label, count in counts.items()}


def validate_yelp_reviews(reviews: pd.DataFrame) -> None:
    """Kontrollera kolumner, 1 000 rader och 500 recensioner per klass.

    Avvikelsen avbryter kontrollen så att den kan utredas innan datan används.
    """
    problems: list[str] = []
    if list(reviews.columns) != COLUMNS:
        problems.append(f"kolumner {list(reviews.columns)} (förväntat {COLUMNS})")
    if len(reviews) != EXPECTED_ROW_COUNT:
        problems.append(f"radantal {len(reviews)} (förväntat {EXPECTED_ROW_COUNT})")

    counts = class_counts(reviews)
    for label, expected in EXPECTED_CLASS_COUNTS.items():
        actual = counts.get(label, 0)
        if actual != expected:
            problems.append(f"etikett {label}: {actual} rader (förväntat {expected})")
    unexpected = sorted(set(counts) - set(EXPECTED_CLASS_COUNTS))
    if unexpected:
        problems.append(f"oväntade etiketter {unexpected}")

    if problems:
        raise ValueError(
            "Yelp-filen avviker från UCI-beskrivningen: " + "; ".join(problems)
        )


def extract_yelp_file(zip_bytes: bytes, destination: Path | str = DEFAULT_RAW_PATH) -> Path:
    """Spara ``yelp_labelled.txt`` oförändrad från UCI-arkivet."""
    target = Path(destination)
    target.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as archive:
        try:
            payload = archive.read(YELP_ZIP_MEMBER)
        except KeyError as error:
            names = ", ".join(archive.namelist())
            raise FileNotFoundError(
                f"UCI-arkivet saknar {YELP_ZIP_MEMBER}. Filer i arkivet: {names}"
            ) from error
    target.write_bytes(payload)
    return target


def download_yelp_file(
    destination: Path | str = DEFAULT_RAW_PATH,
    url: str = UCI_ZIP_URL,
) -> Path:
    """Hämta UCI-arkivet och spara Yelp-filen lokalt utan att ändra innehållet."""
    request = Request(url, headers={"User-Agent": "restaurangomdomen-ml"})
    try:
        with urlopen(request, timeout=60) as response:
            payload = response.read()
    except URLError as error:
        raise ConnectionError(
            "Kunde inte hämta UCI-arkivet. "
            "Manuell hämtning beskrivs i docs/data_source.md."
        ) from error
    return extract_yelp_file(payload, destination)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Läs in Yelp-recensioner från UCI Sentiment Labelled Sentences."
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Hämta UCI-arkivet och spara yelp_labelled.txt oförändrad i data/raw/.",
    )
    parser.add_argument(
        "--path",
        type=Path,
        default=DEFAULT_RAW_PATH,
        help="Sökväg till yelp_labelled.txt.",
    )
    args = parser.parse_args()

    if args.download:
        saved = download_yelp_file(args.path)
        print(f"Sparade oförändrad originalfil: {saved}")

    reviews = load_yelp_reviews(args.path)
    validate_yelp_reviews(reviews)
    counts = class_counts(reviews)
    print(f"Rader: {len(reviews)}")
    print(f"Negativa (0): {counts.get(0, 0)}")
    print(f"Positiva (1): {counts.get(1, 0)}")
    print(f"Kolumner: {', '.join(reviews.columns)}")
    print("Inläsning stämmer med 1 000 rader och 500 per klass.")


if __name__ == "__main__":
    main()
