"""Kontrollera datakvalitet och skapa det bearbetade Yelp-underlaget.

Originaltexten behålls. Normaliseringen i den här modulen används bara för att
jämföra dubbletter och är skild från den textbehandling modellen ska använda.
"""

from __future__ import annotations

import argparse
import numbers
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

import load_data

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "reviews.csv"
DEFAULT_REPORT_PATH = PROJECT_ROOT / "reports" / "data_quality.md"
OUTPUT_COLUMNS = ["review_id", "text", "label", "duplicate_group_id"]

_EMAIL_PATTERN = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")
_PHONE_PATTERN = re.compile(
    r"(?<!\d)(?:\+\d{1,3}[\s.-]?)?(?:\(?\d{2,4}\)?[\s.-])\d{2,4}[\s.-]\d{2,4}(?!\d)"
)
_URL_PATTERN = re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)
_LONG_NUMBER_PATTERN = re.compile(r"\d{6,}")
_PERSON_PATTERNS = (
    re.compile(
        r"\b(?:server|waitress|waiter|manager|hostess|chef|bartender)\s+"
        r"[A-Z][a-z]+(?:\s+[A-Z]\.)?"
    ),
    re.compile(r"\b[A-Z][a-z]+\s+our\s+(?:server|waitress|waiter|manager)\b"),
    re.compile(r"\bthanks to\s+[A-Z][a-z]+(?:\s+[A-Z]\.)?"),
    re.compile(r"\b[A-Z][a-z]+\s+went above and beyond\b"),
)


@dataclass(frozen=True)
class RemovedRow:
    """En rad som inte följer med till det bearbetade underlaget."""

    review_id: int
    reason: str
    label: str
    detail: str


@dataclass(frozen=True)
class NearDuplicateGroup:
    """Rader som bara skiljer sig i skiftläge eller blanksteg."""

    duplicate_group_id: int
    review_ids: tuple[int, ...]
    labels: tuple[int, ...]
    texts: tuple[str, ...]


@dataclass(frozen=True)
class PersonalDataSummary:
    """Antal träffar. Själva träffarna skrivs inte ut."""

    email_count: int
    phone_count: int
    url_count: int
    long_number_count: int
    person_reference_count: int


@dataclass(frozen=True)
class PreparationResult:
    """Sammanfattning av grundkontroller, borttagna rader och kvarvarande data."""

    input_rows: int
    removals: tuple[RemovedRow, ...]
    near_duplicate_groups: tuple[NearDuplicateGroup, ...]
    personal_data: PersonalDataSummary
    output_rows: int
    class_counts: dict[int, int]
    min_text_length: int | None
    median_text_length: float | None
    max_text_length: int | None


def normalize_for_duplicate_check(text: str) -> str:
    """Gör text jämförbar utan att ändra originalet.

    Jämförelsen tar bort skillnader i skiftläge och i antal blanksteg.
    """
    return " ".join(text.split()).casefold()


def _is_valid_label(value: object) -> bool:
    if isinstance(value, bool) or pd.isna(value):
        return False
    if isinstance(value, numbers.Integral):
        return int(value) in (0, 1)
    return False


def _format_label(value: object) -> str:
    if pd.isna(value):
        return "-"
    if _is_valid_label(value):
        return str(int(value))
    return str(value)


def _quality_removal(review_id: object, text: object, label: object) -> RemovedRow | None:
    missing = []
    if pd.isna(review_id):
        missing.append("review_id")
    if pd.isna(text):
        missing.append("text")
    if pd.isna(label):
        missing.append("label")
    reported_id = -1 if pd.isna(review_id) else int(review_id)
    if missing:
        return RemovedRow(
            review_id=reported_id,
            reason="saknat värde",
            label=_format_label(label),
            detail="saknas: " + ", ".join(missing),
        )

    reasons = []
    details = []
    if str(text).strip() == "":
        reasons.append("tom text")
        details.append("texten är tom eller består bara av blanksteg")
    if not _is_valid_label(label):
        reasons.append("ogiltig etikett")
        details.append(f"värde: {label}")
    if not reasons:
        return None
    return RemovedRow(
        review_id=reported_id,
        reason="; ".join(reasons),
        label=_format_label(label),
        detail="; ".join(details),
    )


def _pattern_count(reviews: pd.DataFrame, pattern: re.Pattern[str]) -> int:
    count = 0
    for text in reviews["text"]:
        if pd.isna(text):
            continue
        if pattern.search(str(text)):
            count += 1
    return count


def person_reference_ids(reviews: pd.DataFrame) -> list[int]:
    """Returnera review_id för möjliga personnamn, utan att returnera texten."""
    found: set[int] = set()
    for row in reviews.itertuples(index=False):
        if pd.isna(row.text) or pd.isna(row.review_id):
            continue
        if any(pattern.search(str(row.text)) for pattern in _PERSON_PATTERNS):
            found.add(int(row.review_id))
    return sorted(found)


def summarize_personal_data(reviews: pd.DataFrame) -> PersonalDataSummary:
    """Räkna kontaktuppgifter och möjliga personnamn utan att spara texten."""
    return PersonalDataSummary(
        email_count=_pattern_count(reviews, _EMAIL_PATTERN),
        phone_count=_pattern_count(reviews, _PHONE_PATTERN),
        url_count=_pattern_count(reviews, _URL_PATTERN),
        long_number_count=_pattern_count(reviews, _LONG_NUMBER_PATTERN),
        person_reference_count=len(person_reference_ids(reviews)),
    )


def _remove_exact_duplicates(
    group: pd.DataFrame,
) -> tuple[list[dict[str, object]], list[RemovedRow]]:
    """Behåll första förekomsten av varje exakt text i en normaliserad grupp."""
    occurrences_by_text: dict[str, list[tuple[int, int]]] = {}
    for row in group.itertuples(index=False):
        occurrences_by_text.setdefault(str(row.text), []).append(
            (int(row.review_id), int(row.label))
        )

    kept: list[dict[str, object]] = []
    removals: list[RemovedRow] = []
    for text, occurrences in occurrences_by_text.items():
        labels = {label for _, label in occurrences}
        if len(labels) > 1:
            listed_ids = ", ".join(str(review_id) for review_id, _ in occurrences)
            listed_labels = ", ".join(str(label) for label in sorted(labels))
            for review_id, label in occurrences:
                removals.append(
                    RemovedRow(
                        review_id=review_id,
                        reason="motstridiga etiketter",
                        label=str(label),
                        detail=(
                            "identisk text hos review_id "
                            f"{listed_ids}, etiketter {listed_labels}: {text}"
                        ),
                    )
                )
            continue

        first_id, first_label = occurrences[0]
        kept.append({"review_id": first_id, "text": text, "label": first_label})
        for review_id, label in occurrences[1:]:
            removals.append(
                RemovedRow(
                    review_id=review_id,
                    reason="exakt dubblett med samma etikett",
                    label=str(label),
                    detail=f"samma text som review_id {first_id}: {text}",
                )
            )
    return kept, removals


def prepare_reviews(reviews: pd.DataFrame) -> tuple[pd.DataFrame, PreparationResult]:
    """Kontrollera rader, hantera dubbletter och lägg till grupp-ID.

    Nära dubbletter, texter som blir lika efter normalisering men inte är
    exakt lika, behålls med sin originaltext och delar ``duplicate_group_id``.
    """
    required = {"review_id", "text", "label"}
    missing_columns = required - set(reviews.columns)
    if missing_columns:
        listed = ", ".join(sorted(missing_columns))
        raise ValueError(f"Inläsningen saknar kolumner: {listed}")

    source = reviews.loc[:, ["review_id", "text", "label"]].copy()
    personal_data = summarize_personal_data(source)
    quality_removals: list[RemovedRow] = []
    valid_indexes: list[int] = []
    for index, row in source.iterrows():
        removal = _quality_removal(row["review_id"], row["text"], row["label"])
        if removal is None:
            valid_indexes.append(index)
        else:
            quality_removals.append(removal)

    valid = source.loc[valid_indexes].sort_values("review_id", kind="mergesort")
    valid = valid.assign(
        _duplicate_key=valid["text"].map(lambda text: normalize_for_duplicate_check(str(text)))
    )

    kept_rows: list[dict[str, object]] = []
    duplicate_removals: list[RemovedRow] = []
    near_groups: list[NearDuplicateGroup] = []
    for _, group in valid.groupby("_duplicate_key", sort=False):
        ordered = group.sort_values("review_id", kind="mergesort")
        kept_for_group, removals = _remove_exact_duplicates(ordered)
        duplicate_removals.extend(removals)
        if not kept_for_group:
            continue
        group_id = min(int(row["review_id"]) for row in kept_for_group)
        for row in kept_for_group:
            row["duplicate_group_id"] = group_id
            kept_rows.append(row)
        if len(kept_for_group) > 1:
            kept_for_group.sort(key=lambda row: int(row["review_id"]))
            near_groups.append(
                NearDuplicateGroup(
                    duplicate_group_id=group_id,
                    review_ids=tuple(int(row["review_id"]) for row in kept_for_group),
                    labels=tuple(int(row["label"]) for row in kept_for_group),
                    texts=tuple(str(row["text"]) for row in kept_for_group),
                )
            )

    prepared = pd.DataFrame(kept_rows, columns=OUTPUT_COLUMNS)
    if prepared.empty:
        prepared = pd.DataFrame(columns=OUTPUT_COLUMNS)
    prepared["review_id"] = prepared["review_id"].astype("int64")
    prepared["text"] = prepared["text"].astype("string")
    prepared["label"] = prepared["label"].astype("int64")
    prepared["duplicate_group_id"] = prepared["duplicate_group_id"].astype("int64")
    prepared = prepared.sort_values("review_id", kind="mergesort").reset_index(drop=True)

    lengths = prepared["text"].str.len()
    removals = tuple(sorted([*quality_removals, *duplicate_removals], key=lambda row: row.review_id))
    result = PreparationResult(
        input_rows=len(source),
        removals=removals,
        near_duplicate_groups=tuple(sorted(near_groups, key=lambda group: group.duplicate_group_id)),
        personal_data=personal_data,
        output_rows=len(prepared),
        class_counts=load_data.class_counts(prepared) if len(prepared) else {},
        min_text_length=None if lengths.empty else int(lengths.min()),
        median_text_length=None if lengths.empty else float(lengths.median()),
        max_text_length=None if lengths.empty else int(lengths.max()),
    )
    return prepared, result


def _cell(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def _markdown_table(headers: list[str], rows: list[list[object]]) -> str:
    if not rows:
        return ""
    head = "| " + " | ".join(headers) + " |"
    separator = "| " + " | ".join("---" for _ in headers) + " |"
    body = ["| " + " | ".join(_cell(value) for value in row) + " |" for row in rows]
    return "\n".join([head, separator, *body])


def _removal_section(title: str, empty_text: str, rows: list[RemovedRow]) -> list[str]:
    lines = [f"### {title}", ""]
    if not rows:
        lines.extend([empty_text, ""])
        return lines
    table = _markdown_table(
        ["review_id", "etikett", "orsak", "förklaring"],
        [[row.review_id, row.label, row.reason, row.detail] for row in rows],
    )
    lines.extend([table, ""])
    return lines


def render_load_error_report(message: str) -> str:
    """Beskriv ett fel som stoppade inläsningen innan någon bearbetning gjordes."""
    return "\n".join(
        [
            "# Datakvalitet",
            "",
            "Inläsningen av Yelp-filen misslyckades. Ingen bearbetad fil skrevs.",
            "",
            f"Fel: {message}",
            "",
        ]
    )


def render_quality_report(result: PreparationResult) -> str:
    """Skapa datakvalitetsrapporten utan att återge hittade personuppgifter."""
    personal = result.personal_data
    quality_rows = [
        row
        for row in result.removals
        if row.reason.startswith(("saknat värde", "tom text", "ogiltig etikett"))
    ]
    duplicate_rows = [row for row in result.removals if row.reason == "exakt dubblett med samma etikett"]
    conflict_rows = [row for row in result.removals if row.reason == "motstridiga etiketter"]
    mixed_groups = [group for group in result.near_duplicate_groups if len(set(group.labels)) > 1]
    class_lines = [
        f"| Negativa (0) | {result.class_counts.get(0, 0)} |",
        f"| Positiva (1) | {result.class_counts.get(1, 0)} |",
    ]

    lines = [
        "# Datakvalitet",
        "",
        "Bearbetningen utgår från Yelp-filen som `src/load_data.py` läser in.",
        "Originaltexten behålls i kolumnen `text`. Normaliseringen nedan används",
        "bara för att upptäcka dubbletter och ingår inte i modellens textbehandling.",
        "",
        "## Inläsning och grundkontroller",
        "",
        "Filen läses med `load_yelp_reviews`. Den kontrollen fångar en saknad fil,",
        "en första rad som felaktigt blivit rubrik och ett radantal som inte stämmer",
        "med originalfilen. Därefter kontrolleras varje rad.",
        "",
        "| Kontroll | Antal rader |",
        "| --- | --- |",
        f"| Inlästa rader | {result.input_rows} |",
        f"| Saknade värden | {sum(row.reason == 'saknat värde' for row in quality_rows)} |",
        f"| Tomma texter | {sum(row.reason.startswith('tom text') for row in quality_rows)} |",
        f"| Ogiltiga etiketter | {sum('ogiltig etikett' in row.reason for row in quality_rows)} |",
        "",
    ]
    lines.extend(
        _removal_section(
            "Borttaget i grundkontrollen",
            "Inga rader togs bort för tom text, saknat värde eller ogiltig etikett.",
            quality_rows,
        )
    )
    lines.extend(
        [
            "## Dubbletter",
            "",
            "Jämförelsenormaliseringen gör tre saker med en kopia av texten:",
            "",
            "- blanksteg i början och slutet tas bort",
            "- upprepade blanksteg blir ett enda blanksteg",
            "- skiftläge jämnas ut med `casefold`",
            "",
            "Regler:",
            "",
            "- Exakt samma text och samma etikett behålls en gång. Första `review_id` behålls.",
            "- Exakt samma text med motstridiga etiketter tas bort helt och räknas nedan.",
            "- Texter som blir lika först efter normaliseringen är nära dubbletter.",
            "  Originaltexterna behålls och raderna får ett gemensamt `duplicate_group_id`.",
            "",
        ]
    )
    lines.extend(
        _removal_section(
            "Exakta dubbletter med samma etikett",
            "Inga exakta dubbletter med samma etikett togs bort.",
            duplicate_rows,
        )
    )
    lines.extend(
        _removal_section(
            "Identiska texter med motstridiga etiketter",
            "Inga identiska texter med motstridiga etiketter hittades.",
            conflict_rows,
        )
    )

    lines.extend(["### Nära dubbletter", ""])
    if not result.near_duplicate_groups:
        lines.extend(
            [
                "Inga textvarianter som bara skiljer sig i skiftläge eller blanksteg hittades.",
                "Varje kvarvarande rad har därför ett eget `duplicate_group_id`, satt till radens `review_id`.",
                "",
            ]
        )
    else:
        lines.extend(
            [
                "Följande rader är nära dubbletter. Originaltexten är oförändrad.",
                "`duplicate_group_id` är det lägsta `review_id` i gruppen, så att gruppen",
                "kan hållas samlad vid den senare datauppdelningen.",
                "",
            ]
        )
        for group in result.near_duplicate_groups:
            lines.append(f"- Grupp `{group.duplicate_group_id}`: review_id {', '.join(map(str, group.review_ids))}.")
            for review_id, label, text in zip(group.review_ids, group.labels, group.texts, strict=True):
                lines.append(f"  - `{review_id}`, etikett {label}: {_cell(text)}")
        lines.append("")
        if mixed_groups:
            lines.extend(
                [
                    "Vissa nära dubbletter har olika etiketter. De är inte identiska texter",
                    "och behålls därför, men etikettkonflikten är ett kvarstående kvalitetsproblem.",
                    "Grupperna ska hållas samlade vid datauppdelningen.",
                    "",
                ]
            )

    lines.extend(
        [
            "## Personuppgifter",
            "",
            "Recensionerna kontrollerades efter e-postadresser, telefonnummer,",
            "webbadresser och sifferföljder med minst sex siffror. Möjliga personnamn",
            "markerades när ett förnamn står direkt vid en yrkesroll som server,",
            "waitress, waiter eller manager, eller i formuleringarna `thanks to`",
            "och `went above and beyond`.",
            "",
            "| Kontroll | Träffar |",
            "| --- | --- |",
            f"| E-post | {personal.email_count} |",
            f"| Telefonnummer | {personal.phone_count} |",
            f"| Webbadress | {personal.url_count} |",
            f"| Lång sifferföljd | {personal.long_number_count} |",
            f"| Möjligt personnamn | {personal.person_reference_count} |",
            "",
            "Träffarnas text publiceras inte. Möjliga personnamn ska tas upp med",
            "läraren innan meningarna används som exempel i redovisning eller presentation,",
            "enligt skolans hantering av personuppgifter. Raderna är kvar i underlaget",
            "eftersom kontrollen inte kräver att de tas bort.",
            "",
            "## Kvarstående kvalitetsproblem",
            "",
        ]
    )
    if result.output_rows == 0:
        lines.append("Inga rader återstår efter bearbetningen.")
    else:
        lines.append(
            "Inga tomma texter, saknade värden eller ogiltiga etiketter återstår. "
            f"Klassfördelningen är {result.class_counts.get(0, 0)} negativa och "
            f"{result.class_counts.get(1, 0)} positiva. "
            f"Textlängden är minst {result.min_text_length}, median "
            f"{result.median_text_length:g} och högst {result.max_text_length} tecken. "
            "Stavning och skiljetecken från källan är kvar, eftersom originaltexten bevaras."
        )
    if not mixed_groups and not conflict_rows and personal.person_reference_count == 0:
        lines.append("Inga ytterligare kvalitetsproblem noterades.")
    elif personal.person_reference_count:
        lines.extend(
            [
                "",
                "Möjliga personnamn är det kvarstående förhållande som behöver ett beslut",
                "tillsammans med läraren innan exempel publiceras.",
            ]
        )
    lines.extend(
        [
            "",
            "## Sparat resultat",
            "",
            "| Uppgift | Värde |",
            "| --- | --- |",
            "| Fil | `data/processed/reviews.csv` |",
            f"| Rader | {result.output_rows} |",
            "| Kolumner | `review_id`, `text`, `label`, `duplicate_group_id` |",
            *class_lines,
            "",
            "Filen sparas lokalt. Innehållet i `data/processed/` är gitignorerat,",
            "förutom `.gitkeep`.",
            "",
            "Kör från projektroten:",
            "",
            "```powershell",
            ".\\.venv\\Scripts\\python.exe src/prepare_data.py",
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def save_prepared_reviews(reviews: pd.DataFrame, path: Path | str = DEFAULT_OUTPUT_PATH) -> Path:
    """Spara det bearbetade underlaget som CSV."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    reviews.to_csv(target, index=False, encoding="utf-8")
    return target


def write_quality_report(
    result: PreparationResult,
    path: Path | str = DEFAULT_REPORT_PATH,
) -> Path:
    """Skriv datakvalitetsrapporten."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(render_quality_report(result), encoding="utf-8")
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description="Kontrollera Yelp-datan och spara det bearbetade underlaget.")
    parser.add_argument("--input", type=Path, default=load_data.DEFAULT_RAW_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT_PATH)
    args = parser.parse_args()

    try:
        reviews = load_data.load_yelp_reviews(args.input)
    except (OSError, UnicodeError, ValueError, pd.errors.ParserError) as error:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(render_load_error_report(str(error)), encoding="utf-8")
        print(f"Inläsningsfel: {error}", file=sys.stderr)
        raise SystemExit(1) from error

    prepared, result = prepare_reviews(reviews)
    saved = save_prepared_reviews(prepared, args.output)
    report_path = write_quality_report(result, args.report)
    print(f"Inlästa rader: {result.input_rows}")
    print(f"Borttagna rader: {len(result.removals)}")
    print(f"Kvarvarande rader: {result.output_rows}")
    print(f"Negativa (0): {result.class_counts.get(0, 0)}")
    print(f"Positiva (1): {result.class_counts.get(1, 0)}")
    print(f"Nära dubblettgrupper: {len(result.near_duplicate_groups)}")
    print(f"Sparade data: {saved}")
    print(f"Rapport: {report_path}")


if __name__ == "__main__":
    main()
