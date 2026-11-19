from pathlib import Path

import pandas as pd

from src.preprocess import PreprocessConfig, Preprocessor, process_split


def test_normalized_text_removes_urls_mentions_symbols_and_emoji():
    preprocessor = Preprocessor(PreprocessConfig()).fit(["bahut acchaaa"])

    result = preprocessor.transform_text("Soooo acchaaa! https://x.test @user 😊")

    assert result == "soo acchaa"


def test_normalization_removes_hashtag_marker_and_missing_tokens():
    processor = Preprocessor(PreprocessConfig()).fit(
        ["achha achha accha"]
    )

    assert processor.transform_text(
        "Soooo good!!! 😊 #Movie Accha nan"
    ) == "soo good movie achha"


def test_raw_mode_preserves_case_and_repeated_spelling():
    preprocessor = Preprocessor(
        PreprocessConfig(normalize=False, lowercase=False)
    ).fit(["Soooo"])

    assert preprocessor.transform_text("Soooo") == "Soooo"


def test_empty_text_returns_empty_tokens():
    preprocessor = Preprocessor(PreprocessConfig()).fit(["hello"])

    assert preprocessor.transform([""]) == [[]]


def test_preprocessor_persistence_round_trip(tmp_path: Path):
    preprocessor = Preprocessor(PreprocessConfig()).fit(["nahi nahin"])
    path = tmp_path / "preprocessor.json"
    preprocessor.save(path)

    restored = Preprocessor.load(path)

    assert restored.transform_text("nahi") == preprocessor.transform_text("nahi")


def test_spelling_variants_are_learned_from_training_counts():
    preprocessor = Preprocessor(PreprocessConfig()).fit(
        ["achha achha achha accha"]
    )

    assert preprocessor.transform_text("accha") == "achha"


def test_process_split_preserves_uid_and_labels(tmp_path: Path):
    source = tmp_path / "source.csv"
    target = tmp_path / "target.csv"
    pd.DataFrame(
        {"uid": ["uid-1"], "text_clean": ["Soooo good"], "label": [2]}
    ).to_csv(source, index=False)

    process_split(
        source,
        target,
        Preprocessor(PreprocessConfig()).fit(["Soooo good"]),
    )

    result = pd.read_csv(target)
    assert list(result.columns) == ["uid", "text_clean", "label"]
    assert result.loc[0, "uid"] == "uid-1"
    assert result.loc[0, "label"] == 2
