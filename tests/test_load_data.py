import hashlib
import io
import zipfile
from pathlib import Path

import pandas as pd
import pytest

import load_data


def _write_reviews(path: Path, rows: list[tuple[str, int]]) -> None:
    lines = [f"{text}\t{label}" for text, label in rows]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_first_row_is_data_and_ids_follow_line_numbers(tmp_path: Path) -> None:
    source = tmp_path / "yelp_labelled.txt"
    _write_reviews(
        source,
        [
            ("text", 0),
            ('As for the "mains," also uninspired.', 0),
            ("NA", 1),
        ],
    )

    reviews = load_data.load_yelp_reviews(source)

    assert list(reviews.columns) == ["review_id", "text", "label"]
    assert reviews["review_id"].tolist() == [1, 2, 3]
    assert reviews.loc[0, "text"] == "text"
    assert reviews.loc[1, "text"] == 'As for the "mains," also uninspired.'
    assert reviews.loc[2, "text"] == "NA"
    assert reviews["label"].tolist() == [0, 0, 1]
    assert reviews["review_id"].dtype == "int64"
    assert reviews["label"].dtype == "int64"
    assert pd.api.types.is_string_dtype(reviews["text"])


def test_missing_file_raises_file_not_found(tmp_path: Path) -> None:
    missing = tmp_path / "saknas.txt"

    with pytest.raises(FileNotFoundError, match="docs/data_source.md"):
        load_data.load_yelp_reviews(missing)


def test_validate_accepts_expected_distribution() -> None:
    reviews = pd.DataFrame(
        {
            "review_id": range(1, 1001),
            "text": pd.Series(["ok"] * 1000, dtype="string"),
            "label": [0] * 500 + [1] * 500,
        }
    )

    load_data.validate_yelp_reviews(reviews)


def test_validate_reports_row_and_class_deviations() -> None:
    reviews = pd.DataFrame(
        {
            "review_id": [1, 2, 3],
            "text": pd.Series(["a", "b", "c"], dtype="string"),
            "label": [1, 1, 1],
        }
    )

    with pytest.raises(ValueError, match="radantal 3") as error:
        load_data.validate_yelp_reviews(reviews)

    message = str(error.value)
    assert "etikett 0: 0 rader" in message
    assert "etikett 1: 3 rader" in message


def test_extract_yelp_file_keeps_original_bytes(tmp_path: Path) -> None:
    payload = 'The crêpe was delicate and thin and moist.\t1\n'.encode("utf-8")
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr(load_data.YELP_ZIP_MEMBER, payload)
        archive.writestr("sentiment labelled sentences/amazon_cells_labelled.txt", b"other\t1\n")

    destination = tmp_path / "yelp_labelled.txt"
    saved = load_data.extract_yelp_file(buffer.getvalue(), destination)

    assert saved.read_bytes() == payload
    assert not (tmp_path / "amazon_cells_labelled.txt").exists()


def test_extract_yelp_file_reports_missing_member() -> None:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("sentiment labelled sentences/readme.txt", b"info")

    with pytest.raises(FileNotFoundError, match="saknar"):
        load_data.extract_yelp_file(buffer.getvalue(), Path("unused.txt"))


def test_official_yelp_file_matches_uci_description() -> None:
    source = load_data.DEFAULT_RAW_PATH
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    reviews = load_data.load_yelp_reviews(source)

    load_data.validate_yelp_reviews(reviews)
    assert digest == load_data.YELP_FILE_SHA256
    assert reviews.loc[0, "text"] == "Wow... Loved this place."
    assert int(reviews.loc[0, "label"]) == 1
    assert int(reviews.loc[0, "review_id"]) == 1

    quoted = reviews.loc[reviews["review_id"].eq(229)].iloc[0]
    assert quoted["text"] == 'As for the "mains," also uninspired.'
    assert int(quoted["label"]) == 0

    accented = reviews.loc[reviews["review_id"].eq(824)].iloc[0]
    assert accented["text"] == "The crêpe was delicate and thin and moist."

    rebuilt = "".join(
        f"{text}\t{int(label)}\n" for text, label in zip(reviews["text"], reviews["label"], strict=True)
    )
    assert rebuilt.encode("utf-8") == source.read_bytes()
