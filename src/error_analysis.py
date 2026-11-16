from __future__ import annotations

import re

import pandas as pd


def assign_length_bucket(lengths: pd.Series) -> pd.Series:
    return pd.cut(
        lengths,
        bins=[-float("inf"), 4, 8, float("inf")],
        labels=["short", "medium", "long"],
    ).astype(str)


def assign_cmi_bucket(cmi: pd.Series) -> pd.Series:
    return pd.cut(
        cmi,
        bins=[-float("inf"), 0.33, 0.66, float("inf")],
        labels=["low", "medium", "high"],
    ).astype(str)


def collect_misclassifications(
    predictions: pd.DataFrame, text_frame: pd.DataFrame, limit: int = 30
) -> pd.DataFrame:
    errors = predictions[predictions["y_true"] != predictions["y_pred"]]
    joined = errors.merge(text_frame, on="uid", how="left", validate="one_to_one")
    return joined.head(limit).copy()


def tag_error_category(row: pd.Series) -> str:
    text = str(row.get("text_clean", "")).lower()
    if re.search(r"\b(no|not|nahi|nahin|never)\b", text):
        return "negation"
    if any(char in text for char in "😀😂😊😢😡❤️👍👎"):
        return "emoji-only signal"
    if re.search(r"(.)\1{2,}", text):
        return "spelling variation"
    return "neutral/negative confusion"


def summarize_error_categories(errors: pd.DataFrame) -> pd.DataFrame:
    return (
        errors["category"]
        .value_counts()
        .rename_axis("category")
        .reset_index(name="count")
    )
