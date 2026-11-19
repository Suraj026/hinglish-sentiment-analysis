"""Pydantic schemas for the HinglishPulse API."""

from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, Field


class AnalyzeRequest(BaseModel):
    text: str
    model_name: str = "bilstm_conv"


class BatchAnalyzeRequest(BaseModel):
    texts: list[str]
    model_name: str = "bilstm_conv"


class LanguageMix(BaseModel):
    hindi_ratio: float
    english_ratio: float
    is_hinglish: bool


class SentimentScores(BaseModel):
    negative: float
    neutral: float
    positive: float


class SentimentResult(BaseModel):
    text: str
    model_name: str
    sentiment: str  # "negative" | "neutral" | "positive"
    confidence: float
    scores: SentimentScores
    processing_time_ms: float
    language_mix: LanguageMix
    tokens: list[str]


class BatchResult(BaseModel):
    results: list[SentimentResult]
    model_name: str
    total_time_ms: float
    count: int


class ModelInfo(BaseModel):
    name: str
    display_name: str
    description: str
    accuracy: float
    macro_f1: float
    weighted_f1: float
    inference_count: int
    avg_latency_ms: float


class BenchmarkData(BaseModel):
    models: list[dict]


class StatsResponse(BaseModel):
    total_analyzed: int
    avg_latency_ms: float
    model_distribution: dict[str, int]
    sentiment_distribution: dict[str, int]
    recent_requests: list[dict]
