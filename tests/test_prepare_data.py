import hashlib
from pathlib import Path

import pandas as pd
import pytest

import load_data
import prepare_data


def _reviews(rows: list[tuple[int, str, int]]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "review_id": [review_id for review_id, _, _ in rows],
            "text": pd.Series([text for _, text, _ in rows], dtype="string"),
            "label": [label for _, _, label in rows],
        }
    )


def test_exact_duplicate_with_same_label_is_kept_once() -> None:
    prepared, result = prepare_data.prepare_reviews(
        _reviews(
            [
                (4, "I love this place.", 1),
                (9, "I love this place.", 1),
            ]
        )
    )

    assert prepared["review_id"].tolist() == [4]
    assert prepared.loc[0, "text"] == "I love this place."
    assert int(prepared.loc[0, "label"]) == 1
    assert int(prepared.loc[0, "duplicate_group_id"]) == 4
    assert [row.review_id for row in result.removals] == [9]
    assert result.removals[0].reason == "exakt dubblett med samma etikett"


def test_identical_texts_with_conflicting_labels_are_removed() -> None:
    prepared, result = prepare_data.prepare_reviews(
        _reviews(
            [
                (2, "The food was terrible.", 0),
                (6, "The food was terrible.", 1),
            ]
        )
    )

    assert prepared.empty
    assert [row.reason for row in result.removals] == [
        "motstridiga etiketter",
        "motstridiga etiketter",
    ]


def test_near_duplicates_keep_original_text_and_share_group_id() -> None:
    prepared, result = prepare_data.prepare_reviews(
        _reviews(
            [
                (3, "Great  food.", 1),
                (8, "great food.", 1),
            ]
        )
    )

    assert prepared["text"].tolist() == ["Great  food.", "great food."]
    assert prepared["duplicate_group_id"].tolist() == [3, 3]
    assert result.removals == ()
    assert result.near_duplicate_groups[0].review_ids == (3, 8)
    assert "_duplicate_key" not in prepared.columns


def test_near_duplicates_with_different_labels_stay_and_are_reported() -> None:
    _, result = prepare_data.prepare_reviews(
        _reviews(
            [
                (1, "Good.", 1),
                (2, "good.", 0),
            ]
        )
    )

    report = prepare_data.render_quality_report(result)
    assert result.near_duplicate_groups[0].labels == (1, 0)
    assert "etikettkonflikten är ett kvarstående kvalitetsproblem" in report
    assert "Grupp `1`: review_id 1, 2, etiketter 1, 0." in report
    assert "Good." not in report
    assert "good." not in report


def test_empty_missing_and_invalid_rows_are_removed() -> None:
    reviews = pd.DataFrame(
        {
            "review_id": [1, 2, 3, 4],
            "text": pd.Series(["Bra.", "   ", pd.NA, "Ogiltig"], dtype="string"),
            "label": [1, 0, 1, 5],
        }
    )

    prepared, result = prepare_data.prepare_reviews(reviews)

    assert prepared["review_id"].tolist() == [1]
    assert [row.review_id for row in result.removals] == [2, 3, 4]
    assert [row.reason for row in result.removals] == [
        "tom text",
        "saknat värde",
        "ogiltig etikett",
    ]


def test_missing_column_is_reported_as_load_problem() -> None:
    reviews = pd.DataFrame({"review_id": [1], "text": ["Hej"]})

    with pytest.raises(ValueError, match="saknar kolumner"):
        prepare_data.prepare_reviews(reviews)


def test_report_counts_personal_data_without_quoting_it() -> None:
    prepared, result = prepare_data.prepare_reviews(
        _reviews(
            [
                (1, "Alex our server was kind.", 1),
                (2, "Mail me at student@example.com please.", 0),
                (3, "Call 070-123 45 67.", 0),
            ]
        )
    )
    report = prepare_data.render_quality_report(result)

    assert len(prepared) == 3
    assert result.personal_data.person_reference_count == 1
    assert result.personal_data.email_count == 1
    assert result.personal_data.phone_count == 1
    assert "Alex" not in report
    assert "student@example.com" not in report
    assert "070-123" not in report
    assert "Möjligt personnamn | 1" in report


def test_load_error_report_does_not_write_prepared_data() -> None:
    report = prepare_data.render_load_error_report("Hittar inte Yelp-filen.")

    assert "misslyckades" in report
    assert "Hittar inte Yelp-filen." in report


def test_prepared_csv_roundtrip_preserves_text(tmp_path: Path) -> None:
    prepared, _ = prepare_data.prepare_reviews(
        _reviews([(1, 'As for the "mains," also uninspired.', 0), (2, "NA", 1)])
    )
    path = tmp_path / "reviews.csv"

    prepare_data.save_prepared_reviews(prepared, path)
    loaded = pd.read_csv(
        path,
        dtype={"text": "string", "review_id": "int64", "label": "int64", "duplicate_group_id": "int64"},
        keep_default_na=False,
    )

    pd.testing.assert_frame_equal(loaded, prepared)


def test_official_yelp_preparation_removes_only_exact_duplicates() -> None:
    source = load_data.DEFAULT_RAW_PATH
    digest_before = hashlib.sha256(source.read_bytes()).hexdigest()
    original = load_data.load_yelp_reviews(source)

    prepared, result = prepare_data.prepare_reviews(original)

    assert hashlib.sha256(source.read_bytes()).hexdigest() == digest_before
    assert result.input_rows == 1000
    assert result.output_rows == 996
    assert result.class_counts == {0: 497, 1: 499}
    assert [row.review_id for row in result.removals] == [815, 817, 844, 847]
    assert {row.reason for row in result.removals} == {"exakt dubblett med samma etikett"}
    assert result.near_duplicate_groups == ()
    assert result.personal_data.email_count == 0
    assert result.personal_data.phone_count == 0
    assert result.personal_data.url_count == 0
    assert result.personal_data.long_number_count == 0
    assert result.personal_data.person_reference_count == 4
    assert prepared["duplicate_group_id"].tolist() == prepared["review_id"].tolist()

    original_by_id = original.set_index("review_id")
    for row in prepared.itertuples(index=False):
        assert row.text == original_by_id.loc[row.review_id, "text"]
        assert int(row.label) == int(original_by_id.loc[row.review_id, "label"])

    report = prepare_data.render_quality_report(result)
    assert "I love this place." not in report
    assert "The food was terrible." not in report
    assert "I won't be back." not in report
    assert "I would not recommend this place." not in report
    assert "| 815 | 1 | exakt dubblett med samma etikett | samma text som review_id 335 |" in report
    assert "Möjligt personnamn | 4" in report
    assert "4 möjliga namnträffar har identifierats." in report
    assert "behandling av personuppgifter" in report
    assert "innan fortsatt behandling" in report
    assert "godkännande har lämnats" not in report


def test_report_omits_personal_data_in_duplicate_sections() -> None:
    prepared, result = prepare_data.prepare_reviews(
        _reviews(
            [
                (10, "Alex our server was kind.", 1),
                (11, "Alex our server was kind.", 1),
                (20, "Mail me at student@example.com please.", 0),
                (21, "Mail me at student@example.com please.", 1),
                (30, "Thanks to Sam T. for a great meal.", 1),
                (31, "thanks  to sam t. for a great meal.", 1),
                (40, "Call 070-123 45 67 tonight.", 0),
                (41, "call  070-123 45 67 tonight.", 1),
            ]
        )
    )
    report = prepare_data.render_quality_report(result)
    kept = prepared.set_index("review_id")

    assert kept.loc[10, "text"] == "Alex our server was kind."
    assert kept.loc[30, "text"] == "Thanks to Sam T. for a great meal."
    assert int(kept.loc[30, "duplicate_group_id"]) == int(kept.loc[31, "duplicate_group_id"])
    assert 11 not in kept.index
    assert 20 not in kept.index
    assert 21 not in kept.index

    for sensitive in (
        "Alex our server was kind.",
        "Alex",
        "student@example.com",
        "Sam T.",
        "sam t.",
        "070-123 45 67",
    ):
        assert sensitive not in report

    assert "| 11 | 1 | exakt dubblett med samma etikett | samma text som review_id 10 |" in report
    assert "| 20 | 0 | motstridiga etiketter | identisk text hos review_id 20, 21, etiketter 0, 1 |" in report
    assert "| 21 | 1 | motstridiga etiketter | identisk text hos review_id 20, 21, etiketter 0, 1 |" in report
    assert "Grupp `30`: review_id 30, 31, etiketter 1, 1." in report
    assert "Grupp `40`: review_id 40, 41, etiketter 0, 1." in report
