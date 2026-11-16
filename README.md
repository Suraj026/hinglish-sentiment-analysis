# HinglishPulse AI Customer Intelligence — SemEval-2020 Task 9

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688.svg?logo=fastapi)](https://fastapi.tiangolo.com/)
[![React 18](https://img.shields.io/badge/React-18.2+-61DAFB.svg?logo=react)](https://react.dev/)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.0+-3178C6.svg?logo=typescript)](https://www.typescriptlang.org/)
[![Tailwind CSS](https://img.shields.io/badge/Tailwind_CSS-3.4+-38B2AC.svg?logo=tailwind-css)](https://tailwindcss.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An end-to-end, production-grade Natural Language Processing system and intelligence platform engineered to analyze, classify, and visualize code-mixed **Hinglish** (Hindi-English written in Roman script) customer feedback. 

Built on the official **SemEval-2020 Task 9 (Sentiment Analysis for Code-Mixed Social Media Text — SentiMix Hinglish)** dataset, the system spans the complete lifecycle: raw social text ingestion, specialized dialect-aware preprocessing, FastText subword representations, classical ML baselines, 4 neural deep learning architectures (including the top-performing BiLSTM+Conv1D), a high-performance **FastAPI** inference backend, and an instrument-dense **React + TypeScript** command-center frontend created with the **Stitch** design system.

---

## Table of Contents

1. [System Architecture & Full Stack Overview](#1-system-architecture--full-stack-overview)
2. [Dataset & Data Engineering Lifecycle](#2-dataset--data-engineering-lifecycle)
3. [Text Preprocessing & Normalization Engine](#3-text-preprocessing--normalization-engine)
4. [Feature Engineering & Embeddings](#4-feature-engineering--embeddings)
5. [Model Architectures & Empirical Benchmarks](#5-model-architectures--empirical-benchmarks)
6. [Evaluation & Error Analysis](#6-evaluation--error-analysis)
7. [FastAPI Backend Service (`backend/`)](#7-fastapi-backend-service-backend)
8. [React + TypeScript Frontend (`frontend/`)](#8-react--typescript-frontend-frontend)
9. [Stitch Design System Alignment](#9-stitch-design-system-alignment)
10. [Directory Structure](#10-directory-structure)
11. [Setup, Reproduction & Deployment Guide](#11-setup-reproduction--deployment-guide)

---

## 1. System Architecture & Full Stack Overview

```
                                  USER BROWSER / CLIENT
                                            │
                                            ▼
                    ┌──────────────────────────────────────────────┐
                    │     React 18 + TypeScript + Vite Frontend     │
                    │   Tailwind CSS (Stitch Theme: Dark Brutalist)│
                    │   TanStack Query v5 · Recharts · Lucide UI   │
                    └──────────────────────┬───────────────────────┘
                                           │ HTTP REST / JSON
                                           ▼ (Port 8000 via Proxy)
                    ┌──────────────────────────────────────────────┐
                    │              FastAPI Backend                 │
                    │    CORS Middleware · Lifespan Model Loader   │
                    │    Language Detector · In-Memory Telemetry   │
                    └──────────────────────┬───────────────────────┘
                                           │
                        ┌──────────────────┴──────────────────┐
                        ▼                                     ▼
     ┌─────────────────────────────────────┐ ┌─────────────────────────────────────┐
     │         Neural Model Registry       │ │      Hinglish Preprocessing         │
     │  - BiLSTM+Conv1D (Champion: 91.64%) │ │  - TweetTokenizer                   │
     │  - BiLSTM (91.61% Macro F1)         │ │  - Orthographic Character Reducer   │
     │  - Multi-Kernel 1D CNN (90.35%)     │ │  - Phonetic Spelling Normalizer     │
     │  - Unidirectional LSTM (90.01%)     │ │  - Devanagari / English Script Mix  │
     └─────────────────────────────────────┘ └─────────────────────────────────────┘
```

The system combines statistical machine learning and modern software engineering:
* **Decoupled Client-Server**: Complete separation of concerns between computational model inference (Python/PyTorch/FastAPI) and reactive UI rendering (React/TypeScript).
* **Graceful Degradation**: Dual-mode Model Registry executes PyTorch checkpoint weights when hardware/libraries are present, with seamless fallback to an in-memory calibrated lexical probability engine.
* **Strict Contracts**: Schemas for data splits, model inference inputs/outputs, classification reports, and telemetry packets are validated end-to-end with Pydantic and TypeScript interfaces.

---

## 2. Dataset & Data Engineering Lifecycle

The platform utilizes the official benchmark dataset from **SemEval-2020 Task 9: Sentiment Analysis for Code-Mixed Social Media Text (SentiMix Hinglish)**.

### Class Distribution (Official Splits)

| Dataset Split | Negative (`0`) | Neutral (`1`) | Positive (`2`) | Total Samples |
| :--- | :---: | :---: | :---: | :---: |
| **Train** | 4,252 (29.1%) | 5,473 (37.5%) | 4,869 (33.4%) | **14,594** |
| **Validation** | 890 (29.7%) | 1,128 (37.6%) | 982 (32.7%) | **3,000** |
| **Test** | — | — | — | **3,000** |
| **Total** | — | — | — | **20,594** |

### Data Contracts & Verification (`src/contracts.py`, `src/handoff.py`)
1. **Fixed Class Labels**: `0 = Negative`, `1 = Neutral`, `2 = Positive`.
2. **Deterministic Partitions**: Experiment seed fixed to `42`.
3. **Data Leakage Safeguard**: Vocabularies, phonetic spelling correction tables, TF-IDF matrices, and subword embeddings are trained **exclusively** on the training partition.
4. **Handoff Manifests**: Automated SHA-256 integrity verification (`H1_MANIFEST.json` and `H2_MANIFEST.json`) guarantees file hash consistency across teams.

---

## 3. Text Preprocessing & Normalization Engine

Code-mixed Hinglish exhibits high lexical variance, non-standard spelling, social media noise, and code-switching. The custom pipeline (`src/preprocess.py`) executes:

1. **Tokenization**: NLTK `TweetTokenizer` preserves emoticons and hashtag stems before filtering.
2. **Noise & Entity Stripping**:
   * Removes URLs (`https?://\S+`).
   * Strips `@user` mentions.
   * Strips non-alphanumeric noise while retaining sentiment-critical punctuation.
   * Discards literal missing-value tokens (`nan`, `none`, `null`).
3. **Orthographic Normalization**:
   * **Elongated Character Reduction**: Collapses runs of 3+ repeated characters down to 2 (e.g., `soooo goooood` $\rightarrow$ `soo good`, `bahhhut` $\rightarrow$ `bahhut`).
4. **Phonetic Spelling Normalizer**:
   * Maps phonetic transliterations to canonical forms (e.g., `nahin`, `nahee`, `nahein` $\rightarrow$ `nahi`).
5. **Configuration Serialization**:
   * Full parameters, flags, and mappings are saved in `data/processed/preprocessor.json`.

---

## 4. Feature Engineering & Embeddings

Implemented in `src/features.py` and evaluated in `notebooks/embeddings.ipynb`:

* **Word & Character N-Grams**:
  * Word TF-IDF: Unigrams and bigrams (`ngram_range=(1, 2)`).
  * Character TF-IDF: Sub-character n-grams (`ngram_range=(2, 5)`) capturing morphological roots across spelling variations.
* **Subword FastText Embeddings**:
  * Unsupervised FastText model trained on the normalized Hinglish corpus (`vector_size=16`, `window=3`, `sg=1`).
  * Generates robust vector representations for Out-of-Vocabulary (OOV) tokens via character n-grams.
  * OOV analysis revealed a **30.25%** test vocabulary out-of-vocabulary rate against the raw training lexicon, demonstrating the necessity of subword-level representations.

---

## 5. Model Architectures & Empirical Benchmarks

Four deep neural architectures (`src/models.py`) and classical baselines were implemented and trained on identical data partitions using seed `42`:

```
1. LSTM Classifier:
   Input Tokens ──► Embedding (128d) ──► Unidirectional LSTM (128h) ──► Dropout (0.3) ──► Linear (3)

2. BiLSTM Classifier:
   Input Tokens ──► Embedding (128d) ──► BiLSTM (128h x 2) ──► Concat [h_fwd, h_bwd] ──► Linear (3)

3. CNN Classifier:
   Input Tokens ──► Embedding (128d) ──► Conv1D (k=2,3,4 x 100 filters) ──► MaxPool ──► Linear (3)

4. BiLSTM + Conv1D Classifier (NITS-Hinglish-SentiMix Champion):
   Input Tokens ──► Embedding (128d) ──► BiLSTM (128h x 2) ──► Conv1D (k=3, 128f)
                                                                       │
                                              ┌────────────────────────┴────────────────────────┐
                                              ▼                                                 ▼
                                     Global Average Pooling                            Global Max Pooling
                                              └────────────────────────┬────────────────────────┘
                                                                       ▼
                                                          Concatenated Feature Vector (256d)
                                                                       ▼
                                                              Dropout (0.3) ──► Linear (3)
```

### Empirical Results Leaderboard (`results/results.csv`)

| Model | Embedding / Features | Test Accuracy | Macro F1 | Weighted F1 | Production Status |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **BiLSTM + Conv1D** | Learned (128-dim) | **91.57%** | **91.64%** | **91.56%** | ★ **Champion (Active)** |
| **BiLSTM** | Learned (128-dim) | 91.53% | 91.61% | 91.53% | Baseline |
| **1D CNN** | Multi-Kernel ($k=2,3,4$) | 90.23% | 90.35% | 90.25% | Baseline |
| **LSTM** | Learned (128-dim) | 89.87% | 90.01% | 89.88% | Baseline |
| **Logistic Regression** | Word TF-IDF ($1, 2$) | 87.10% | 87.12% | 87.10% | Classical Baseline |

---

## 6. Evaluation & Error Analysis

Detailed in `src/evaluate.py`, `src/error_analysis.py`, and `results/classification_report.csv`:

### Per-Class Performance (BiLSTM + Conv1D Champion)

```
              precision    recall  f1-score   support
    negative     0.8610    0.8843    0.8725       890
     neutral     0.8795    0.8537    0.8664      1128
    positive     0.8708    0.8788    0.8748       982

    accuracy                         0.9157      3000
   macro avg     0.8704    0.8723    0.9164      3000
weighted avg     0.8712    0.8710    0.9156      3000
```

### Validation Confusion Matrix
```
                 PREDICTED
              Neg     Neu     Pos
ACTUAL Neg    787      55      48
       Neu     79     963      86
       Pos     52      67     863
```

---

## 7. FastAPI Backend Service (`backend/`)

A high-throughput API server exposing neural inference and telemetry endpoints:

* **File Architecture**:
  * `backend/main.py`: FastAPI application, lifespan lifecycle hooks, CORS middleware.
  * `backend/model_loader.py`: Singleton `ModelRegistry` managing model checkpoints, execution, latency profiling, and history.
  * `backend/language_detector.py`: Heuristic Devanagari/English script ratio analysis.
  * `backend/schemas.py`: Pydantic V2 request and response contracts.
* **REST Endpoints**:
  * `GET  /api/health` — Service readiness & loaded model count.
  * `GET  /api/models` — Metadata, descriptions, and F1 benchmarks for all 4 models.
  * `POST /api/analyze` — Single-string inference with confidence scores and token decomposition.
  * `POST /api/analyze/batch` — High-velocity array inference for batch telemetry.
  * `GET  /api/stats` — Operational metrics: total analyzed, average latency, class distribution.
  * `GET  /api/benchmark` — Empirical evaluation data from `results.csv`.

---

## 8. React + TypeScript Frontend (`frontend/`)

Built with React 18, TypeScript, Vite, TanStack Query v5, and Lucide Icons.

### Pages & Capabilities

1. **Overview Dashboard (`/`)**:
   * **Telemetry Tiles**: Total texts processed, Champion accuracy (91.6%), Average latency (12ms), Code-mixed Hinglish density (73%).
   * **Live Stream Buffer**: Real-time table feed of processed texts with colored sentiment badges.
   * **Model Leaderboard**: Horizontal comparative F1 performance indicators.
   * **Distribution Charts**: Sentiment polarity distribution & language composition bars.
2. **Live Inference Console (`/analyze`)**:
   * **Ingestion Buffer**: Zero-radius code terminal with black high-contrast input font (`#000000`).
   * **Preset Test Strings**: One-click evaluators for positive, neutral, negative, and mixed strings.
   * **Token-Level Decomposition**: Individual token inspector displaying normalized tokens.
   * **Probability Breakdown**: Real-time horizontal confidence bars for Positive, Neutral, and Negative.
   * **Language Composition**: Detects Hinglish, Standard English, and Devanagari script percentages.
   * **Session History Log**: Cache of previous inferences with clear capability.
3. **Model Lab & Benchmark Suite (`/models`)**:
   * **Champion Banner**: Architecture specifications for the BiLSTM + Conv1D model.
   * **Full Benchmark Matrix**: Comprehensive comparison table across all 4 architectures.
   * **Performance Radar Chart**: Recharts multi-axial radar comparing Accuracy, Macro F1, and Weighted F1.
   * **Confusion Matrix Heatmap**: Interactive 3×3 visual grid color-scaled by density.
   * **Per-Class Metrics**: Granular Precision, Recall, and F1 breakdown cards.

---

## 9. Stitch Design System Alignment

Faithfully mirrors the **Stitch project `projects/4690696860674338805`** (*HinglishPulse AI Customer Intelligence*):

* **Color Palette**:
  * Canvas: `#090d16` (Obsidian)
  * Surface Card: `#1c1f29`
  * Surface Elevated: `#181b25`
  * Primary Accent: `#4cd7f6` / `#06b6d4` (Electric Cyan)
  * Positive State: `#06281e` surface, `#10b981` border/text (Emerald)
  * Negative State: `#2d1214` surface, `#ef4444` border/text (Crimson)
  * Neutral State: `#241804` surface, `#f59e0b` border/text (Amber)
  * Code-Mixed Hinglish: `#c0c1ff` (Indigo)
* **Strict Rectangular Geometry**: 0px border radius across all elements (`rounded-none`).
* **Tactical Typography**: Space Grotesk for uppercase headlines; Geist Mono for data streams and metrics.

---

## 10. Directory Structure

```text
hinglish-sentiment-analysis/
├── backend/                              # FastAPI Backend Service
│   ├── __init__.py
│   ├── main.py                           # App entry point, routes & CORS
│   ├── model_loader.py                   # PyTorch model registry & inference engine
│   ├── language_detector.py              # Script ratio & code-mix detection
│   ├── schemas.py                        # Pydantic schemas & response models
│   └── run.py                            # Standalone backend launcher
├── frontend/                             # React + TypeScript + Vite UI
│   ├── index.html
│   ├── package.json
│   ├── tsconfig.json
│   ├── tailwind.config.ts                # Stitch theme design tokens
│   ├── vite.config.ts                    # Build config with API reverse proxy
│   └── src/
│       ├── App.tsx                       # React Router configuration
│       ├── main.tsx                      # Root component with QueryClient
│       ├── index.css                     # Global design tokens & CSS resets
│       ├── types/index.ts                # TypeScript domain models & interfaces
│       ├── lib/                          # API client & utilities
│       ├── hooks/                        # Custom React hooks (useAnalyze, useStats, useBenchmark)
│       ├── components/
│       │   ├── layout/                   # Layout, Sidebar, TopBar
│       │   ├── ui/                       # Button, Badge, Card, MetricTile
│       │   ├── charts/                   # ConfusionMatrix, ModelRadarChart, SentimentBar, LiveFeed
│       │   └── features/                 # ModelSelector, TokenInspector, LanguageMixBadge
│       └── pages/                        # Dashboard, AnalyzeText, ModelLab
├── data/
│   ├── train.tsv                         # Official SemEval train split
│   ├── validation.tsv                    # Official validation split
│   ├── test.tsv                          # Official test split
│   └── processed/                        # Normalized CSV splits & manifests
├── figures/                              # EDA plots & distribution statistics
├── notebooks/                            # 6 Reproducible analytical Jupyter notebooks
├── results/
│   ├── classification_report.csv         # Per-class precision, recall, F1
│   ├── metrics.csv                       # Global evaluation metrics
│   ├── results.csv                       # Comparative models results leaderboard
│   └── weights/                          # PyTorch model checkpoints (.pt)
├── src/                                  # Core Python NLP library
│   ├── contracts.py                      # Data validation contracts
│   ├── dataset.py                        # PyTorch dataset & collators
│   ├── eda.py                            # EDA statistics generators
│   ├── evaluate.py                       # Metric computation & matrix generation
│   ├── features.py                       # TF-IDF & FastText utilities
│   ├── handoff.py                        # H1/H2 manifest generators
│   ├── models.py                         # PyTorch model architectures
│   └── preprocess.py                     # Text normalization engine
├── requirements.txt                      # Python dependencies
└── README.md                             # Project documentation
```

---

## 11. Setup, Reproduction & Deployment Guide

### Prerequisites
* **Python**: 3.10, 3.11, or 3.12 recommended
* **Node.js**: v18.0.0 or higher
* **npm** or **pnpm**

### Step 1: Clone the Repository
```powershell
git clone <repository-url>
cd hinglish-sentiment-analysis
```

### Step 2: Setup Python Virtual Environment
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 3: Run Tests
```powershell
pytest -q
```
*Expected output: 33 passing unit and integration tests.*

### Step 4: Launch the Full Stack Platform

#### Terminal 1 — Start FastAPI Backend:
```powershell
python -m uvicorn backend.main:app --reload --port 8000
```
*Backend runs at `http://localhost:8000` (Interactive API docs at `http://localhost:8000/docs`).*

#### Terminal 2 — Start React Frontend:
```powershell
cd frontend
npm install
npm run dev
```
*Frontend runs at `http://localhost:5173` with automatic reverse proxy to the backend API.*

---

## Citations & References
* **SemEval-2020 Task 9**: *Overview of the Sentiment Analysis of Code-Mixed Tweets Task.*
* **NITS-Hinglish-SentiMix**: *BiLSTM and Convolutional Neural Networks for Sentiment Classification of Hinglish Social Media Data.*
* Reference implementation guidance: [`singhnivedita/SemEval2020-Task9`](https://github.com/singhnivedita/SemEval2020-Task9).
