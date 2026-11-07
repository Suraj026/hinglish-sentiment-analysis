from pathlib import Path

import pandas as pd

from src.classical import train_classical_model
from src.contracts import LABEL_TO_ID, validate_split_frame
from src.handoff import build_manifest, write_manifest
from src.preprocess import PreprocessConfig, Preprocessor


def test_reproducibility_contracts_and_fixture_pipeline(tmp_path: Path):
    frame = pd.DataFrame(
        {
            "uid": ["uid-1", "uid-2", "uid-3"],
            "text_clean": ["Soooo good", "bad result", "okay result"],
            "label": [2, 0, 1],
        }
    )
    validate_split_frame(frame, "train")
    processor = Preprocessor(PreprocessConfig()).fit(frame.text_clean)
    model, vectorizer = train_classical_model(
        [processor.transform_text(text) for text in frame.text_clean],
        frame.label,
        "logistic_regression",
        "word",
    )

    assert LABEL_TO_ID["positive"] == 2
    assert model is not None
    assert vectorizer is not None


def test_handoff_manifest_contains_schema_and_checksums(tmp_path: Path):
    split = tmp_path / "train.csv"
    pd.DataFrame(
        {"uid": ["uid-1"], "text_clean": ["hello"], "label": [2]}
    ).to_csv(split, index=False)
    manifest = build_manifest({"train": split}, "H1", seed=42)
    output = tmp_path / "H1_MANIFEST.json"
    write_manifest(manifest, output)

    assert manifest["handoff"] == "H1"
    assert manifest["seed"] == 42
    assert manifest["files"]["train"]["sha256"]
    assert manifest["files"]["train"]["row_count"] == 1
    assert manifest["files"]["train"]["columns"] == ["uid", "text_clean", "label"]
    assert manifest["files"]["train"]["label_counts"] == {"2": 1}
    assert output.exists()
