import pandas as pd
import pytest

import load_data
import prepare_data
import split_data


def _reviews(rows: list[tuple[int, str, int, int]]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "review_id": [review_id for review_id, _, _, _ in rows],
            "text": pd.Series([text for _, text, _, _ in rows], dtype="string"),
            "label": [label for _, _, label, _ in rows],
            "duplicate_group_id": [group_id for _, _, _, group_id in rows],
        }
    )


def _population(per_class: int = 40) -> pd.DataFrame:
    rows = []
    review_id = 1
    for label in (0, 1):
        for _ in range(per_class):
            rows.append((review_id, f"Review {review_id} is label {label}.", label, review_id))
            review_id += 1
    rows.extend(
        [
            (9001, "Great  food.", 1, 9001),
            (9002, "great food.", 1, 9001),
            (9003, "GREAT FOOD.", 1, 9001),
        ]
    )
    return _reviews(rows)


def test_duplicate_group_stays_together() -> None:
    splits = split_data.split_reviews(_population())
    locations = {
        name
        for name, frame in splits.items()
        if set(frame["review_id"]).issuperset({9001, 9002, 9003})
    }

    assert locations == {next(iter(locations))}
    assert sum(9001 in set(frame["review_id"]) for frame in splits.values()) == 1


def test_split_is_reproducible_when_rows_are_shuffled() -> None:
    reviews = _population()
    shuffled = reviews.sample(frac=1, random_state=7).reset_index(drop=True)

    first = split_data.split_reviews(reviews)
    second = split_data.split_reviews(shuffled)

    for name in split_data.SPLIT_NAMES:
        assert first[name]["review_id"].tolist() == second[name]["review_id"].tolist()


def test_stratification_label_uses_majority_and_then_earliest_id() -> None:
    majority = _reviews(
        [
            (1, "A", 1, 10),
            (2, "B", 1, 10),
            (3, "C", 0, 10),
        ]
    )
    tie = _reviews(
        [
            (8, "A", 0, 4),
            (6, "B", 1, 4),
        ]
    )

    assert split_data.stratification_label(majority) == 1
    assert split_data.stratification_label(tie) == 1


def test_representative_texts_skip_person_references() -> None:
    train = _reviews(
        [
            (1, "Alex our server was kind.", 1, 1),
            (2, "A decent meal today.", 1, 2),
            (3, "This was a really long and positive description of the meal.", 1, 3),
            (4, "Bad.", 0, 4),
            (5, "The meal was slow and cold.", 0, 5),
        ]
    )

    examples = split_data.select_representative_texts(train, excluded_ids={1})

    assert all("Alex" not in example.text for example in examples)
    assert any(example.text == "A decent meal today." for example in examples)


def test_report_does_not_quote_held_out_text() -> None:
    reviews = _population()
    reviews.loc[reviews["review_id"].eq(1), "text"] = "Alex our server was kind."
    splits = split_data.split_reviews(reviews)
    summary = split_data.summarize_split(reviews, splits)
    report = split_data.render_data_summary(summary)

    assert "random_state=42" in report
    assert "data/splits/validation.csv" in report
    for name in ("validation", "test"):
        for text in splits[name]["text"]:
            assert text not in report
    assert "Alex our server was kind." not in report


def test_save_splits_writes_three_files(tmp_path) -> None:
    splits = split_data.split_reviews(_population())

    paths = split_data.save_splits(splits, tmp_path)

    assert set(paths) == {"train", "validation", "test"}
    for path in paths.values():
        assert path.exists()
        assert path.parent == tmp_path


def test_missing_prepared_file_raises(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="prepare_data.py"):
        split_data.load_prepared_reviews(tmp_path / "saknas.csv")


def test_official_yelp_split_is_separated_and_stratified() -> None:
    prepared, _ = prepare_data.prepare_reviews(load_data.load_yelp_reviews())
    shuffled = prepared.sample(frac=1, random_state=1).reset_index(drop=True)
    splits = split_data.split_reviews(prepared)
    repeated = split_data.split_reviews(shuffled)

    total = len(prepared)
    positive_rate = prepared["label"].mean()
    seen_ids: set[int] = set()
    seen_groups: set[int] = set()
    for name, expected_fraction in (
        ("train", 0.60),
        ("validation", 0.20),
        ("test", 0.20),
    ):
        frame = splits[name]
        assert frame["review_id"].tolist() == repeated[name]["review_id"].tolist()
        assert abs(len(frame) / total - expected_fraction) < 0.02
        assert abs(frame["label"].mean() - positive_rate) < 0.02
        ids = set(frame["review_id"])
        groups = set(frame["duplicate_group_id"])
        assert seen_ids.isdisjoint(ids)
        assert seen_groups.isdisjoint(groups)
        seen_ids.update(ids)
        seen_groups.update(groups)

    assert seen_ids == set(prepared["review_id"])
    assert seen_groups == set(prepared["duplicate_group_id"])
    assert list(splits["train"].columns) == ["review_id", "text", "label", "duplicate_group_id"]

    summary = split_data.summarize_split(prepared, splits)
    report = split_data.render_data_summary(summary)
    sensitive_texts = prepared.loc[
        prepared["review_id"].isin(prepare_data.person_reference_ids(prepared)),
        "text",
    ]
    for text in sensitive_texts:
        assert text not in report
    for name in ("validation", "test"):
        longest = splits[name].loc[splits[name]["text"].str.len().idxmax(), "text"]
        assert longest not in report
