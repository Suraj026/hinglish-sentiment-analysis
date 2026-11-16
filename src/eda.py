from __future__ import annotations

from collections import Counter
from typing import Callable, Iterable

import pandas as pd


def compute_class_distribution(frame: pd.DataFrame) -> pd.DataFrame:
    return (
        frame["label"]
        .value_counts()
        .sort_index()
        .rename_axis("label")
        .reset_index(name="count")
    )


def compute_length_stats(
    frame: pd.DataFrame, tokens: Callable[[str], list[str]]
) -> pd.DataFrame:
    result = frame.copy()
    result["token_count"] = result["text_clean"].map(
        lambda text: len(tokens(str(text)))
    )
    result["character_count"] = result["text_clean"].map(
        lambda text: len(str(text))
    )
    return result


def _token_set(texts: Iterable[str]) -> set[str]:
    return {
        token
        for text in texts
        for token in str(text).split()
    }


def compute_vocabulary_stats(
    train_texts: Iterable[str], test_texts: Iterable[str]
) -> dict[str, float | int]:
    train_vocab = _token_set(train_texts)
    test_vocab = _token_set(test_texts)
    return {
        "train_vocab_size": len(train_vocab),
        "test_vocab_size": len(test_vocab),
        "oov_rate": (
            len(test_vocab - train_vocab) / len(test_vocab)
            if test_vocab
            else 0.0
        ),
    }


def compute_zipf_counts(texts: Iterable[str]) -> pd.DataFrame:
    counts = Counter(token for text in texts for token in str(text).split())
    rows = [
        {"token": token, "count": count, "rank": rank}
        for rank, (token, count) in enumerate(
            sorted(counts.items(), key=lambda item: (-item[1], item[0])),
            start=1,
        )
    ]
    return pd.DataFrame(rows, columns=["token", "count", "rank"])


def compute_code_mixing_index(language_tags: Iterable[str]) -> float:
    tags = [str(tag) for tag in language_tags]
    if not tags:
        return 0.0
    non_other = [tag for tag in tags if tag.lower() not in {"o", "other"}]
    if not non_other:
        return 0.0
    languages = {tag.lower() for tag in non_other}
    return 0.0 if len(languages) <= 1 else 1.0 - 1.0 / len(languages)


def compute_language_ratios(
    tagged_tokens: Iterable[Iterable[str]],
) -> pd.DataFrame:
    counts = Counter(tag for row in tagged_tokens for tag in row)
    total = sum(counts.values())
    rows = [
        {"language": language, "count": count, "ratio": count / total}
        for language, count in sorted(counts.items())
    ]
    return pd.DataFrame(rows, columns=["language", "count", "ratio"])


def require_language_tags(frame: pd.DataFrame) -> None:
    if "language_tags" not in frame.columns:
        raise ValueError("language tags are required for language-ratio analysis")
