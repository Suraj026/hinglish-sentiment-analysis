from __future__ import annotations

import json
import re
import html
import unicodedata
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
        spelling_map: dict[str, str] = {}
        eligible = [
            (token, count)
            for token, count in counts.items()
            if len(token) >= 3 and token.isalpha()
        ]
        candidate_index: dict[tuple[int, str], list[str]] = {}
        for token, _ in eligible:
            for key in ((len(token), token[0]), (len(token), token[-1])):
                candidate_index.setdefault(key, []).append(token)
        for variant, variant_count in eligible:
            candidate_names: set[str] = set()
            length = len(variant)
            candidate_names.update(candidate_index.get((length, variant[0]), []))
            candidate_names.update(candidate_index.get((length, variant[-1]), []))
            candidates = []
            for candidate in candidate_names:
                candidate_count = counts[candidate]
                if candidate != variant and (
                    len(candidate) == len(variant)
                    and self._near_spelling_distance(candidate, variant) == 1
                ):
                    candidates.append((candidate, candidate_count))
            if not candidates:
                continue
            candidate, candidate_count = min(
                candidates,
                key=lambda item: (
                    -item[1],
                    len(item[0]),
                    item[0],
                ),
            )
            if candidate_count >= variant_count * 2:
                spelling_map[variant] = candidate
        self.spelling_map = spelling_map
        return self

    @staticmethod
    def _near_spelling_distance(left: str, right: str) -> int:
        if len(left) == len(right):
            mismatches = [
                index
                for index, (left_char, right_char) in enumerate(zip(left, right))
                if left_char != right_char
            ]
            if len(mismatches) == 1:
                return 1
            if len(mismatches) == 2:
                first, second = mismatches
                if (
                    second == first + 1
                    and left[first] == right[second]
                    and left[second] == right[first]
                ):
                    return 1
        if abs(len(left) - len(right)) == 1:
            shorter = left if len(left) < len(right) else right
            longer = right if len(left) < len(right) else left
            first_difference = next(
                (
                    index
                    for index, (short_char, long_char)
                    in enumerate(zip(shorter, longer))
                    if short_char != long_char
                ),
                len(shorter),
            )
            if shorter == longer[:first_difference] + longer[first_difference + 1 :]:
                return 1
        return 2

    def transform(self, texts: Iterable[str]) -> list[list[str]]:
        return [self._transform_tokens(str(text or "")) for text in texts]

    def transform_text(self, text: str) -> str:
        return " ".join(self._transform_tokens(text))

    def _transform_tokens(self, text: str) -> list[str]:
        if not text:
            return []
        text = unicodedata.normalize("NFKC", html.unescape(str(text)))
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
