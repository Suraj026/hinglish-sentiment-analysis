from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable

import pandas as pd
from nltk.tokenize import TweetTokenizer

from .contracts import validate_split_frame


@dataclass
class PreprocessConfig:
    normalize: bool = True
    lowercase: bool = True
    hashtag_mode: str = "keep"


@dataclass
class Preprocessor:
    config: PreprocessConfig
    spelling_map: dict[str, str] = field(default_factory=dict)
    _tokenizer: TweetTokenizer = field(
        default_factory=lambda: TweetTokenizer(
            preserve_case=True, reduce_len=False, strip_handles=False
        ),
        init=False,
        repr=False,
    )

    def fit(self, train_texts: Iterable[str]) -> "Preprocessor":
        counts: dict[str, int] = {}
        for text in train_texts:
            for token in self._tokenizer.tokenize(str(text or "")):
                lowered = token.lower()
                counts[lowered] = counts.get(lowered, 0) + 1
        self.spelling_map = {
            "nahin": "nahi",
            "nahee": "nahi",
            "nahein": "nahi",
        }
        return self

    def transform(self, texts: Iterable[str]) -> list[list[str]]:
        return [self._transform_tokens(str(text or "")) for text in texts]

    def transform_text(self, text: str) -> str:
        return " ".join(self._transform_tokens(text))

    def _transform_tokens(self, text: str) -> list[str]:
        if not text:
            return []
        tokens = self._tokenizer.tokenize(text)
        output: list[str] = []
        for token in tokens:
            if not self.config.normalize:
                output.append(token)
                continue
            if re.fullmatch(r"https?://\S+|www\.\S+", token, flags=re.I):
                continue
            if token.startswith("@") and len(token) > 1:
                continue
            if token.lower() in {"nan", "none", "null"}:
                continue
            if token.startswith("#"):
                if self.config.hashtag_mode == "drop":
                    continue
                token = token[1:]
            if not token or not any(character.isalnum() for character in token):
                continue
            token = "".join(character for character in token if character.isalnum())
            if not token:
                continue
            if self.config.lowercase:
                token = token.lower()
            token = re.sub(r"(.)\1{2,}", r"\1\1", token)
            token = self.spelling_map.get(token.lower(), token)
            output.append(token)
        return output

    def save(self, path: Path) -> None:
        path.write_text(
            json.dumps(
                {
                    "config": asdict(self.config),
                    "spelling_map": self.spelling_map,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path) -> "Preprocessor":
        payload = json.loads(path.read_text(encoding="utf-8"))
        preprocessor = cls(PreprocessConfig(**payload["config"]))
        preprocessor.spelling_map = payload["spelling_map"]
        return preprocessor


def process_split(
    input_path: Path, output_path: Path, preprocessor: Preprocessor
) -> None:
    frame = pd.read_csv(input_path)
    validate_split_frame(frame, input_path.name)
    frame["text_clean"] = [
        preprocessor.transform_text(text) for text in frame["text_clean"]
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)
