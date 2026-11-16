import pandas as pd
import pytest

from src.eda import (
    compute_class_distribution,
    compute_code_mixing_index,
    compute_language_ratios,
    compute_length_stats,
    compute_vocabulary_stats,
    compute_zipf_counts,
    require_language_tags,
)


def test_class_distribution_counts_labels():
    frame = pd.DataFrame({"label": [0, 0, 1, 2]})

    result = compute_class_distribution(frame)

    assert result.to_dict("records") == [
        {"label": 0, "count": 2},
        {"label": 1, "count": 1},
        {"label": 2, "count": 1},
    ]


def test_length_stats_reports_token_and_character_lengths():
    frame = pd.DataFrame({"text_clean": ["a bb", "ccc"]})

    result = compute_length_stats(frame, lambda text: text.split())

    assert result["token_count"].tolist() == [2, 1]
    assert result["character_count"].tolist() == [4, 3]


def test_vocabulary_stats_reports_oov_rate():
    result = compute_vocabulary_stats(["a b"], ["b c"])

    assert result == {"train_vocab_size": 2, "test_vocab_size": 2, "oov_rate": 0.5}


def test_zipf_counts_are_sorted_by_frequency():
    result = compute_zipf_counts(["a b a"])

    assert result.iloc[0].to_dict() == {"token": "a", "count": 2, "rank": 1}


def test_cmi_and_language_ratios():
    assert compute_code_mixing_index(["Hin", "Eng", "Hin", "O"]) == 0.5
    result = compute_language_ratios([["Hin", "Eng"], ["Hin"]])
    assert result.to_dict("records") == [
        {"language": "Eng", "count": 1, "ratio": 1 / 3},
        {"language": "Hin", "count": 2, "ratio": 2 / 3},
    ]


def test_language_tags_are_required():
    with pytest.raises(ValueError, match="language tags"):
        require_language_tags(pd.DataFrame({"text_clean": ["hello"]}))
