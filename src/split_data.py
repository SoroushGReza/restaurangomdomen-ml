"""Dela bearbetade Yelp-recensioner i träning, validering och test.

Uppdelningen är gemensam för baseline och den jämförande modellen.
Ingen textvektor eller modell anpassas här.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

import load_data
import prepare_data

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SPLIT_DIR = PROJECT_ROOT / "data" / "splits"
DEFAULT_REPORT_PATH = PROJECT_ROOT / "reports" / "data_summary.md"

RANDOM_STATE = 42
TRAIN_FRACTION = 0.60
VALIDATION_FRACTION = 0.20
TEST_FRACTION = 0.20
SPLIT_NAMES = ("train", "validation", "test")
HELD_OUT_FRACTION = VALIDATION_FRACTION + TEST_FRACTION


@dataclass(frozen=True)
class LengthStats:
    """Textlängd i antal tecken."""

    count: int
    minimum: int
    lower_quartile: float
    median: float
    mean: float
    upper_quartile: float
    maximum: int


@dataclass(frozen=True)
class TextExample:
    """En träningstext som illustrerar längd och klass."""

    kind: str
    label: int
    review_id: int
    length: int
    text: str


@dataclass(frozen=True)
class SplitSummary:
    """Radantal, klassfördelning och träningstexternas längd."""

    input_rows: int
    group_count: int
    max_group_size: int
    mixed_label_groups: int
    row_counts: dict[str, int]
    group_counts: dict[str, int]
    class_counts: dict[str, dict[int, int]]
    train_lengths: LengthStats
    train_lengths_by_label: dict[int, LengthStats]
    examples: tuple[TextExample, ...]


def load_prepared_reviews(path: Path | str | None = None) -> pd.DataFrame:
    """Läs det bearbetade underlaget från Issue 03."""
    source = prepare_data.DEFAULT_OUTPUT_PATH if path is None else Path(path)
    if not source.is_file():
        raise FileNotFoundError(
            f"Hittar inte bearbetad data: {source}. Kör först src/prepare_data.py."
        )
    reviews = pd.read_csv(
        source,
        dtype={
            "review_id": "int64",
            "text": "string",
            "label": "int64",
            "duplicate_group_id": "int64",
        },
        keep_default_na=False,
    )
    missing = set(prepare_data.OUTPUT_COLUMNS) - set(reviews.columns)
    if missing:
        listed = ", ".join(sorted(missing))
        raise ValueError(f"Bearbetad data saknar kolumner: {listed}")
    return reviews.loc[:, list(prepare_data.OUTPUT_COLUMNS)].copy()


def stratification_label(group: pd.DataFrame) -> int:
    """Välj den etikett som styr stratifieringen för en dubblettgrupp.

    Majoritetsetiketten används. Vid lika antal används etiketten på det
    lägsta ``review_id``, så att valet blir detsamma varje körning.
    """
    counts = group["label"].value_counts()
    top_count = int(counts.max())
    winners = sorted(int(label) for label, count in counts.items() if int(count) == top_count)
    if len(winners) == 1:
        return winners[0]
    earliest = group.sort_values("review_id", kind="mergesort").iloc[0]
    return int(earliest["label"])


def _validate_prepared_reviews(reviews: pd.DataFrame) -> None:
    required = set(prepare_data.OUTPUT_COLUMNS)
    missing = required - set(reviews.columns)
    if missing:
        listed = ", ".join(sorted(missing))
        raise ValueError(f"Bearbetad data saknar kolumner: {listed}")
    if reviews.empty:
        raise ValueError("Bearbetad data innehåller inga rader.")
    if reviews["review_id"].duplicated().any():
        raise ValueError("review_id är inte unikt i bearbetad data.")
    if not reviews["label"].isin([0, 1]).all():
        raise ValueError("Bearbetad data innehåller ogiltiga etiketter.")


def _group_table(reviews: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for group_id, group in reviews.groupby("duplicate_group_id", sort=True):
        rows.append(
            {
                "duplicate_group_id": int(group_id),
                "stratify_label": stratification_label(group),
            }
        )
    return pd.DataFrame(rows)


def _subset_for_groups(reviews: pd.DataFrame, group_ids: pd.Series) -> pd.DataFrame:
    selected = reviews[reviews["duplicate_group_id"].isin(set(group_ids))].copy()
    return selected.sort_values(["review_id"], kind="mergesort").reset_index(drop=True)


def _validate_splits(reviews: pd.DataFrame, splits: dict[str, pd.DataFrame]) -> None:
    seen_ids: set[int] = set()
    seen_groups: set[int] = set()
    for name in SPLIT_NAMES:
        frame = splits[name]
        ids = set(int(value) for value in frame["review_id"])
        groups = set(int(value) for value in frame["duplicate_group_id"])
        if seen_ids & ids:
            raise ValueError(f"review_id förekommer i flera delmängder, bland annat i {name}.")
        if seen_groups & groups:
            raise ValueError(f"duplicate_group_id förekommer i flera delmängder, bland annat i {name}.")
        seen_ids.update(ids)
        seen_groups.update(groups)

    expected_ids = set(int(value) for value in reviews["review_id"])
    expected_groups = set(int(value) for value in reviews["duplicate_group_id"])
    if seen_ids != expected_ids or seen_groups != expected_groups:
        raise ValueError("Uppdelningen täcker inte alla rader och dubblettgrupper.")


def split_reviews(reviews: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Dela raderna i cirka 60 % träning, 20 % validering och 20 % test.

    Slumpen styrs av ``random_state=42``. Stratifieringen görs på
    dubblettgrupper, och alla rader i en grupp följer med till samma delmängd.
    """
    _validate_prepared_reviews(reviews)
    groups = _group_table(reviews)
    train_groups, held_out_groups = train_test_split(
        groups,
        test_size=HELD_OUT_FRACTION,
        random_state=RANDOM_STATE,
        stratify=groups["stratify_label"],
    )
    validation_groups, test_groups = train_test_split(
        held_out_groups,
        test_size=TEST_FRACTION / HELD_OUT_FRACTION,
        random_state=RANDOM_STATE,
        stratify=held_out_groups["stratify_label"],
    )
    grouped_ids = {
        "train": train_groups["duplicate_group_id"],
        "validation": validation_groups["duplicate_group_id"],
        "test": test_groups["duplicate_group_id"],
    }
    splits = {
        name: _subset_for_groups(reviews, grouped_ids[name])
        for name in SPLIT_NAMES
    }
    _validate_splits(reviews, splits)
    return splits


def _length_stats(texts: pd.Series) -> LengthStats:
    lengths = texts.str.len()
    quartiles = lengths.quantile([0.25, 0.75])
    return LengthStats(
        count=int(lengths.count()),
        minimum=int(lengths.min()),
        lower_quartile=float(quartiles.loc[0.25]),
        median=float(lengths.median()),
        mean=float(lengths.mean()),
        upper_quartile=float(quartiles.loc[0.75]),
        maximum=int(lengths.max()),
    )


def select_representative_texts(
    train_reviews: pd.DataFrame,
    excluded_ids: set[int] | None = None,
) -> tuple[TextExample, ...]:
    """Välj korta, typiska och långa träningstexter per klass."""
    blocked = excluded_ids or set()
    usable = train_reviews[~train_reviews["review_id"].isin(blocked)].copy()
    usable["length"] = usable["text"].str.len()
    examples: list[TextExample] = []
    for label in (0, 1):
        part = usable[usable["label"].eq(label)].sort_values(
            ["length", "review_id"],
            kind="mergesort",
        )
        if part.empty:
            continue
        median_length = float(part["length"].median())
        median_row = part.assign(distance=(part["length"] - median_length).abs()).sort_values(
            ["distance", "review_id"],
            kind="mergesort",
        ).iloc[0]
        chosen = (
            ("kort", part.iloc[0]),
            ("median", median_row),
            ("lång", part.iloc[-1]),
        )
        seen: set[int] = set()
        for kind, row in chosen:
            review_id = int(row["review_id"])
            if review_id in seen:
                continue
            seen.add(review_id)
            examples.append(
                TextExample(
                    kind=kind,
                    label=label,
                    review_id=review_id,
                    length=int(row["length"]),
                    text=str(row["text"]),
                )
            )
    return tuple(examples)


def summarize_split(reviews: pd.DataFrame, splits: dict[str, pd.DataFrame]) -> SplitSummary:
    """Sammanfatta uppdelningen och träningstexternas längd."""
    group_sizes = reviews.groupby("duplicate_group_id")["label"].agg(["size", "nunique"])
    train = splits["train"]
    excluded = set(prepare_data.person_reference_ids(reviews))
    return SplitSummary(
        input_rows=len(reviews),
        group_count=int(reviews["duplicate_group_id"].nunique()),
        max_group_size=int(group_sizes["size"].max()),
        mixed_label_groups=int((group_sizes["nunique"] > 1).sum()),
        row_counts={name: len(splits[name]) for name in SPLIT_NAMES},
        group_counts={name: int(splits[name]["duplicate_group_id"].nunique()) for name in SPLIT_NAMES},
        class_counts={name: load_data.class_counts(splits[name]) for name in SPLIT_NAMES},
        train_lengths=_length_stats(train["text"]),
        train_lengths_by_label={
            label: _length_stats(train.loc[train["label"].eq(label), "text"])
            for label in (0, 1)
        },
        examples=select_representative_texts(train, excluded),
    )


def _percent(part: int, whole: int) -> str:
    if whole == 0:
        return "0.0 %"
    return f"{100 * part / whole:.1f} %"


def _cell(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def _length_row(name: str, stats: LengthStats) -> str:
    return (
        f"| {name} | {stats.count} | {stats.minimum} | {stats.lower_quartile:g} | "
        f"{stats.median:g} | {stats.mean:.1f} | {stats.upper_quartile:g} | {stats.maximum} |"
    )


def render_data_summary(summary: SplitSummary) -> str:
    """Skapa rapporten med klassfördelning, textlängder och träningsexempel."""
    label_names = {0: "negativ", 1: "positiv"}
    split_names = {"train": "Träning", "validation": "Validering", "test": "Test"}
    count_rows = []
    for name in SPLIT_NAMES:
        counts = summary.class_counts[name]
        total = summary.row_counts[name]
        count_rows.append(
            "| "
            + " | ".join(
                [
                    split_names[name],
                    str(total),
                    _percent(total, summary.input_rows),
                    str(summary.group_counts[name]),
                    str(counts.get(0, 0)),
                    str(counts.get(1, 0)),
                    _percent(counts.get(1, 0), total),
                ]
            )
            + " |"
        )

    if summary.max_group_size == 1:
        group_sentence = (
            "Varje dubblettgrupp innehåller en rad. "
            "Radandelarna följer därför gruppdelningen direkt."
        )
    else:
        group_sentence = (
            f"Största dubblettgruppen innehåller {summary.max_group_size} rader. "
            "Radandelarna kan därför avvika något från 60/20/20, eftersom en grupp inte delas."
        )

    lines = [
        "# Datauppdelning och inledande analys",
        "",
        "Baseline och den jämförande modellen ska använda samma delmängder.",
        "Uppdelningen görs innan någon textvektor eller modell anpassas.",
        "",
        "## Metod",
        "",
        "Underlaget är `data/processed/reviews.csv` från datakvalitetskontrollen.",
        "En dubblettgrupp identifieras med `duplicate_group_id`.",
        "Alla recensioner i samma grupp placeras i samma delmängd.",
        "",
        "- Träning: 60 % av dubblettgrupperna.",
        "- Validering: 20 % av dubblettgrupperna.",
        "- Test: 20 % av dubblettgrupperna.",
        f"- `random_state={RANDOM_STATE}`.",
        "- Stratifiering görs på gruppens etikett, så att klassfördelningen behålls så långt det går.",
        "- Om en grupp har flera etiketter används majoriteten. Vid lika antal används etiketten på det lägsta `review_id`.",
        "",
        group_sentence,
        (
            f"Antal grupper med mer än en etikett: {summary.mixed_label_groups}."
        ),
        "",
        "## Klassfördelning",
        "",
        "| Delmängd | Rader | Andel | Grupper | Negativa (0) | Positiva (1) | Andel positiva |",
        "| --- | --- | --- | --- | --- | --- | --- |",
        *count_rows,
        "",
        f"Totalt antal rader: {summary.input_rows}.",
        f"Totalt antal dubblettgrupper: {summary.group_count}.",
        "",
        "Inget `review_id` och ingen `duplicate_group_id` förekommer i mer än en delmängd.",
        "",
        "## Textlängd i träningsdata",
        "",
        "Längden räknas i tecken i originaltexten. Ingen tokenisering eller annan",
        "modellförbehandling ingår. Validerings- och testtexter citeras inte,",
        "eftersom de inte ska styra modellutvecklingen.",
        "",
        "| Urval | Rader | Min | Q1 | Median | Medel | Q3 | Max |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
        _length_row("Träning, alla", summary.train_lengths),
        _length_row("Träning, negativa", summary.train_lengths_by_label[0]),
        _length_row("Träning, positiva", summary.train_lengths_by_label[1]),
        "",
        "Träningstexterna är korta recensionsmeningar. "
        f"Medianlängden är {summary.train_lengths.median:g} tecken och "
        f"hälften av texterna ligger mellan {summary.train_lengths.lower_quartile:g} och "
        f"{summary.train_lengths.upper_quartile:g} tecken.",
        "",
        "## Representativa träningstexter",
        "",
        "Exemplen kommer bara från träningsdata: en kort, en typisk och en lång text per klass.",
        "Recensioner som datakvalitetskontrollen markerade som möjliga personnamn skrivs inte ut.",
        "Skolans krav gäller behandling av personuppgifter i projektet, inte enbart publicering av exempel.",
        "Gruppen behöver stämma av med läraren hur träffarna ska hanteras innan fortsatt behandling.",
        "",
        "| Typ | Klass | review_id | Tecken | Text |",
        "| --- | --- | --- | --- | --- |",
    ]
    for example in summary.examples:
        lines.append(
            "| "
            + " | ".join(
                [
                    example.kind,
                    label_names[example.label],
                    str(example.review_id),
                    str(example.length),
                    _cell(example.text),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Användning",
            "",
            "- Träningsdata används för att anpassa textomvandling och modeller.",
            "- Valideringsdata används för modellval.",
            "- Testdata används först när modellvalet är låst.",
            "- Testresultat får inte styra modellutvecklingen.",
            "",
            "Filerna sparas lokalt som `data/splits/train.csv`, `data/splits/validation.csv`",
            "och `data/splits/test.csv`. Mappen `data/splits/` är gitignorerad, förutom `.gitkeep`.",
            "",
            "Kör från projektroten:",
            "",
            "```powershell",
            ".\\.venv\\Scripts\\python.exe src/split_data.py",
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def save_splits(splits: dict[str, pd.DataFrame], directory: Path | str = DEFAULT_SPLIT_DIR) -> dict[str, Path]:
    """Spara de tre delmängderna som CSV."""
    target_dir = Path(directory)
    target_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name in SPLIT_NAMES:
        path = target_dir / f"{name}.csv"
        splits[name].to_csv(path, index=False, encoding="utf-8")
        paths[name] = path
    return paths


def write_data_summary(summary: SplitSummary, path: Path | str = DEFAULT_REPORT_PATH) -> Path:
    """Skriv den inledande analysen."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(render_data_summary(summary), encoding="utf-8")
    return target


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Dela bearbetade Yelp-recensioner i träning, validering och test."
    )
    parser.add_argument("--input", type=Path, default=prepare_data.DEFAULT_OUTPUT_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_SPLIT_DIR)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT_PATH)
    args = parser.parse_args()

    reviews = load_prepared_reviews(args.input)
    splits = split_reviews(reviews)
    saved = save_splits(splits, args.output_dir)
    summary = summarize_split(reviews, splits)
    report_path = write_data_summary(summary, args.report)

    print(f"Rader: {summary.input_rows}")
    for name in SPLIT_NAMES:
        counts = summary.class_counts[name]
        print(
            f"{name}: {summary.row_counts[name]} rader, "
            f"negativa {counts.get(0, 0)}, positiva {counts.get(1, 0)}"
        )
    for name in SPLIT_NAMES:
        print(f"Sparade {name}: {saved[name]}")
    print(f"Rapport: {report_path}")


if __name__ == "__main__":
    main()
