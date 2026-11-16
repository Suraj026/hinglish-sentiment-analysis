import pandas as pd

from src.error_analysis import (
    assign_cmi_bucket,
    assign_length_bucket,
    collect_misclassifications,
    summarize_error_categories,
    tag_error_category,
)


def test_bucket_assignment_is_deterministic():
    assert assign_length_bucket(pd.Series([1, 5, 10])).tolist() == [
        "short",
        "medium",
        "long",
    ]
    assert assign_cmi_bucket(pd.Series([0.1, 0.5, 0.9])).tolist() == [
        "low",
        "medium",
        "high",
    ]


def test_misclassifications_join_on_uid_and_are_limited():
    predictions = pd.DataFrame(
        {"uid": ["a", "b"], "y_true": [0, 1], "y_pred": [1, 1]}
    )
    text = pd.DataFrame(
        {"uid": ["a", "b"], "text_clean": ["not good", "fine"], "label": [0, 1]}
    )

    errors = collect_misclassifications(predictions, text, limit=1)

    assert len(errors) == 1
    assert errors.iloc[0]["uid"] == "a"


def test_error_category_and_summary():
    row = pd.Series({"text_clean": "not good 😂"})
    assert tag_error_category(row) in {"negation", "emoji-only signal"}
    summary = summarize_error_categories(
        pd.DataFrame({"category": ["negation", "negation", "spelling variation"]})
    )
    assert summary.set_index("category").loc["negation", "count"] == 2
