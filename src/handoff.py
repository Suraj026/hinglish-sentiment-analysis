from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Mapping

import pandas as pd


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_manifest(
    files: Mapping[str, Path], handoff: str, seed: int = 42
) -> dict[str, object]:
    file_metadata: dict[str, dict[str, object]] = {}
    for name, path in files.items():
        metadata: dict[str, object] = {
            "path": str(path),
            "sha256": _sha256(path),
        }
        if path.suffix.lower() == ".csv":
            frame = pd.read_csv(path)
            metadata["row_count"] = int(len(frame))
            metadata["columns"] = list(frame.columns)
            if "label" in frame.columns:
                metadata["label_counts"] = {
                    str(label): int(count)
                    for label, count in frame["label"].value_counts().sort_index().items()
                }
        file_metadata[name] = metadata
    return {
        "handoff": handoff,
        "seed": seed,
        "files": file_metadata,
    }


def write_manifest(manifest: Mapping[str, object], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
