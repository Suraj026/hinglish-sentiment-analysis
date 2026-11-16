"""FastAPI application for HinglishPulse sentiment analysis."""

from __future__ import annotations

import csv
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from backend.model_loader import _MODEL_CONFIGS, registry
from backend.schemas import (
    AnalyzeRequest,
    BatchAnalyzeRequest,
    BatchResult,
    BenchmarkData,
    ModelInfo,
    SentimentResult,
    StatsResponse,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parent.parent
_RESULTS_CSV = _REPO_ROOT / "results" / "results.csv"

# ---------------------------------------------------------------------------
# Lifespan — load models at startup
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    logger.info("HinglishPulse backend starting — loading models …")
    registry.initialize()
    logger.info(
        "Models ready: %d loaded", registry.model_count
    )
    yield
    logger.info("HinglishPulse backend shutting down.")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="HinglishPulse API",
    description="Hinglish sentiment analysis — LSTM / BiLSTM / CNN / BiLSTM+Conv",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Static benchmark data loaded from results/results.csv
# ---------------------------------------------------------------------------


def _load_benchmark_data() -> list[dict]:
    """Parse results/results.csv into a list of model performance dicts."""
    if not _RESULTS_CSV.exists():
        logger.warning("results.csv not found at %s", _RESULTS_CSV)
        return []

    rows: list[dict] = []
    seen: set[str] = set()
    with _RESULTS_CSV.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            model_name = row["model"]
            if model_name in seen:
                continue
            seen.add(model_name)
            rows.append({
                "model": model_name,
                "embedding": row.get("embedding", "random"),
                "accuracy": round(float(row["accuracy"]), 4),
                "macro_f1": round(float(row["macro_f1"]), 4),
                "weighted_f1": round(float(row["weighted_f1"]), 4),
                "seed": int(row.get("seed", 42)),
            })
    return rows


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/api/health", summary="Health check")
def health() -> dict:
    return {
        "status": "ok",
        "models_loaded": registry.models_loaded,
        "model_count": registry.model_count,
    }


@app.get("/api/models", response_model=list[ModelInfo], summary="List available models")
def list_models() -> list[ModelInfo]:
    return [ModelInfo(**m) for m in registry.get_model_info()]


@app.post("/api/analyze", response_model=SentimentResult, summary="Analyze single text")
def analyze(req: AnalyzeRequest) -> SentimentResult:
    text = req.text.strip()
    if not text:
        raise HTTPException(status_code=422, detail="Text cannot be empty")
    try:
        return registry.predict(text=text, model_name=req.model_name)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Inference failed for model %s", req.model_name)
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}") from exc


@app.post("/api/analyze/batch", response_model=BatchResult, summary="Analyze a batch of texts")
def analyze_batch(req: BatchAnalyzeRequest) -> BatchResult:
    if not req.texts:
        raise HTTPException(status_code=422, detail="texts list cannot be empty")
    import time
    t0 = time.perf_counter()
    results: list[SentimentResult] = []
    for text in req.texts:
        clean = text.strip()
        if not clean:
            continue
        try:
            res = registry.predict(text=clean, model_name=req.model_name)
            results.append(res)
        except Exception as exc:
            logger.warning("Batch item failed: %s (%s)", clean[:40], exc)

    total_time_ms = round((time.perf_counter() - t0) * 1000.0, 2)
    return BatchResult(
        results=results,
        model_name=req.model_name,
        total_time_ms=total_time_ms,
        count=len(results),
    )


@app.get("/api/stats", response_model=StatsResponse, summary="Aggregate usage stats")
def get_stats() -> StatsResponse:
    stats = registry.get_stats()
    model_dist = [
        {"model": name, "count": count, "percentage": round(count / stats["total_analyzed"] * 100, 1) if stats["total_analyzed"] else 0.0}
        for name, count in stats["model_distribution"].items()
    ]
    total_sent = sum(stats["sentiment_distribution"].values()) or 1
    sent_dist = {
        k: round(v / total_sent, 4) for k, v in stats["sentiment_distribution"].items()
    }
    return StatsResponse(
        total_analyzed=stats["total_analyzed"],
        avg_latency_ms=stats["avg_latency_ms"],
        model_distribution=model_dist,
        sentiment_distribution=sent_dist,
        recent_requests=stats["recent_requests"],
    )


@app.get("/api/benchmark", summary="Model benchmark data")
def get_benchmark() -> BenchmarkData:
    models_info = [ModelInfo(**m) for m in registry.get_model_info()]
    csv_rows = _load_benchmark_data()
    return BenchmarkData(models=models_info, benchmark_runs=csv_rows)
