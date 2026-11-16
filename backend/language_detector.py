"""Simple Hinglish / language-mix detector using script analysis."""

from __future__ import annotations

import re
import unicodedata


# Unicode ranges
_DEVANAGARI_RANGE = re.compile(r"[\u0900-\u097F]")
_LATIN_WORD = re.compile(r"\b[A-Za-z]+\b")
_WORD_SPLIT = re.compile(r"\s+")


def detect_language_mix(text: str) -> dict:
    """Analyse *text* for Devanagari vs Latin script content.

    Returns:
        {
          "hindi_ratio":  float in [0, 1],   # fraction of chars that are Devanagari
          "english_ratio": float in [0, 1],  # fraction of words that are pure-Latin
          "is_hinglish": bool,               # True when both scripts are present
        }
    """
    if not text or not text.strip():
        return {"hindi_ratio": 0.0, "english_ratio": 0.0, "is_hinglish": False}

    # --- character-level Devanagari ratio ---
    total_alpha = sum(1 for ch in text if ch.isalpha())
    devanagari_chars = len(_DEVANAGARI_RANGE.findall(text))
    hindi_ratio = devanagari_chars / total_alpha if total_alpha else 0.0

    # --- word-level Latin ratio ---
    words = [w for w in _WORD_SPLIT.split(text.strip()) if w]
    if words:
        latin_words = sum(1 for w in words if _LATIN_WORD.fullmatch(w))
        english_ratio = latin_words / len(words)
    else:
        english_ratio = 0.0

    # Hinglish = meaningful presence of both scripts
    is_hinglish = hindi_ratio > 0.05 and english_ratio > 0.05

    return {
        "hindi_ratio": round(hindi_ratio, 4),
        "english_ratio": round(english_ratio, 4),
        "is_hinglish": is_hinglish,
    }
