# Project Summary: Hinglish Sentiment Analysis (SemEval-2020 Task 9)

**Branch:** `analysis`  
**Role:** Person B (Data engineering, NLP exploratory analysis, classical baselines, evaluation framework, error analysis, handoff protocols, and demo integration)  
**Partner Role:** Person A (Neural models, training, model checkpoints, neural predictions, and attention exports on `models` branch)

---

## 1. Executive Summary

This repository implements the end-to-end data processing, exploratory data analysis (EDA), classical machine learning baselines, evaluation harness, error analysis pipeline, Gradio web demonstration, and reproducible handoff infrastructure for **SemEval-2020 Task 9 (Sentiment Analysis for Code-Mixed Social Media Text — Hinglish)**.

All shared data contracts, feature pipelines, reproducible test suites (33 unit/integration tests), and validation results have been completed and verified.

---

## 2. Shared Project Contracts & Standards

To ensure zero-friction integration between Person B (analysis/data) and Person A (deep learning/models), strict shared schemas and contracts are defined and enforced in [`src/contracts.py`](file:///home/cse/g13_nlp/hinglish-sentiment-analysis/src/contracts.py):

*   **Fixed Label Space:**
    *   `0`: Negative
    *   `1`: Neutral
    *   `2`: Positive
*   **Split Dataframe Schema (`SPLIT_COLUMNS`):**
    *   Columns: `uid`, `text_clean`, `label`
    *   Constraint: Unique `uid`, non-null clean text, valid labels `[0, 1, 2]`.
*   **Prediction Schema (`PREDICTION_COLUMNS`):**
    *   Columns: `uid`, `y_true`, `y_pred`, `p_neg`, `p_neu`, `p_pos`
    *   Constraint: Probabilities sum to 1.0, labels match valid categories.
*   **Evaluation Metric Table Schema (`RESULT_COLUMNS`):**
    *   Columns: `model`, `embedding`, `normalized`, `accuracy`, `macro_f1`, `weighted_f1`, `seed`
*   **Experimental Rigor:**
    *   Random seed: `42` across all models and splits.
    *   Fitting restriction: Vocabularies, spelling maps, TF-IDF vectorizers, and FastText embeddings are fit **strictly on the training split** to prevent data leakage.

---

## 3. Dataset & Data Handoffs

### 3.1 Dataset Breakdown (Official SemEval-2020 Task 9 Splits)
*   **Raw Data:** Stored in `data/` (`train.tsv`, `validation.tsv`, `test.tsv`).
*   **Processed Data:** Stored in `data/processed/` (`train.csv`, `val.csv`, `test.csv`).

| Split | Total Samples | Negative (0) | Neutral (1) | Positive (2) |
| :--- | :---: | :---: | :---: | :---: |
| **Train** | 14,594 | 4,252 (29.1%) | 5,473 (37.5%) | 4,869 (33.4%) |
| **Validation** | 3,000 | 890 (29.7%) | 1,128 (37.6%) | 982 (32.7%) |
| **Test** | 3,000 | — | — | — |
| **Total** | **20,594** | — | — | — |

### 3.2 Handoff Manifests (`src/handoff.py`)
*   **H1 Quick Data Manifest (`data/processed/H1_MANIFEST.json`):**
    *   Tracks file paths, row counts, exact column sets, label distributions, and SHA-256 hashes (`train.csv`: `dd3d3a...`, `val.csv`: `25d327...`, `test.csv`: `1058ce...`).
*   **H2 Final Preprocessing & Embeddings:**
    *   Preserves exact serialized configuration in `data/processed/preprocessor.json`.
    *   Recommends sequence length based on 95th-percentile length statistics.

---

## 4. Preprocessing Pipeline (`src/preprocess.py`)

A specialized, reproducible text normalization engine designed specifically for code-mixed Romanized Hindi and English (Hinglish):

1.  **Tokenization:** NLTK `TweetTokenizer` preserving case flags and emoticons before filtering.
2.  **Entity & Artifact Cleaning:**
    *   Removes URLs (`http://...`, `https://...`).
    *   Removes user handles/mentions (`@user`).
    *   Strips redundant punctuation, emoji symbols, and hashtag `#` signs while preserving token stems.
    *   Cleans corrupted missing-value tokens (`nan`).
3.  **Orthographic Normalization:**
    *   Character collapse: Truncates repeated characters to at most 2 (e.g., `acchaaa` $\rightarrow$ `acchaa`, `soooo` $\rightarrow$ `soo`).
    *   Lowercasing: Configurable lowercase normalization.
4.  **Hinglish Spelling Normalizer:**
    *   Standardizes frequent dialectal phonetic variations learned from train data (e.g., `nahin`, `nahee`, `nahein` $\rightarrow$ `nahi`).
5.  **Artifact Persistence:**
    *   Saves and reloads state via JSON serialization (`data/processed/preprocessor.json`).

---

## 5. Exploratory Data Analysis & Linguistic Figures

Implemented in [`src/eda.py`](file:///home/cse/g13_nlp/hinglish-sentiment-analysis/src/eda.py) and executed via [`notebooks/eda.ipynb`](file:///home/cse/g13_nlp/hinglish-sentiment-analysis/notebooks/eda.ipynb). All figures and statistical tables are stored in `figures/`:

*   **Class Distribution:** Verified mild class imbalance with Neutral (37.5%) being the majority class, followed by Positive (33.4%) and Negative (29.1%) (`figures/class_distribution.png`, `figures/class_distribution.csv`).
*   **Sequence & Character Lengths:**
    *   Computed token and character distributions across all tweets (`figures/train_length_stats.csv`, `figures/token_length_histogram.png`).
    *   Identified 95th percentile token length for sequence truncation in downstream neural models.
*   **Vocabulary & Out-of-Vocabulary (OOV) Rate:**
    *   Training Vocabulary: 30,978 unique tokens.
    *   Test Vocabulary: 10,914 unique tokens.
    *   OOV Rate: **30.25%** (`figures/vocabulary_stats.csv`), confirming high lexical variance and strong motivation for subword / character-level modeling.
*   **Zipf Frequency Distribution:** Generated token rank-frequency distributions (`figures/zipf_counts.csv`), revealing high frequency of Hinglish function words (`hai`, `ki`, `ko`, `nahi`, `ka`, `ke`, `se`).

---

## 6. Feature Engineering & Subword Embeddings (`src/features.py`)

1.  **TF-IDF Vectorizers:**
    *   **Word N-grams:** Unigrams and bigrams (`ngram_range=(1, 2)`).
    *   **Character N-grams:** 2-gram to 5-gram subwords (`ngram_range=(2, 5)`) to capture spelling drift, phonetic Romanization, and morphology.
2.  **Domain-Adapted FastText Embeddings:**
    *   Trained on raw training tweets using subword skip-gram (`sg=1`, dimension 16, window 3, 20 epochs).
    *   Exported vocabulary mappings and embedding matrix builder with special tokens (`<pad>`, `<unk>`).
    *   Nearest neighbor exploration to validate semantic coherence under phonetic variations.

---

## 7. Classical Baseline Models (`src/classical.py`)

Classical sentiment classification pipelines trained on the training split and evaluated on the official validation split:

1.  **Algorithms:**
    *   **Logistic Regression:** Multinomial with SAGA solver (`max_iter=1000`, `seed=42`).
    *   **Linear SVM:** LinearSVC with probability calibration using temperature/softmax adapter on decision function margins.
2.  **Feature Sets:** Word TF-IDF and Character TF-IDF.
3.  **Top N-gram Interpretability:** Helper `extract_top_ngrams()` extracts the most predictive linguistic signals per sentiment class.

---

## 8. Evaluation Framework & Results

Implemented in [`src/evaluate.py`](file:///home/cse/g13_nlp/hinglish-sentiment-analysis/src/evaluate.py) and tracked in `results/`:

### 8.1 Validation Benchmark Results (`results/results.csv`, `results/metrics.csv`)
*   **Model:** Logistic Regression (Word TF-IDF, normalized)
*   **Accuracy:** **87.10%** (0.8710)
*   **Macro F1-Score:** **87.12%** (0.8712)
*   **Weighted F1-Score:** **87.10%** (0.8710)

### 8.2 Detailed Classification Report (`results/classification_report.csv`)

| Class | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| **Negative (0)** | 0.8611 | 0.8843 | 0.8725 | 890 |
| **Neutral (1)** | 0.8795 | 0.8537 | 0.8664 | 1,128 |
| **Positive (2)** | 0.8708 | 0.8788 | 0.8748 | 982 |
| **Macro Average** | **0.8704** | **0.8723** | **0.8712** | 3,000 |
| **Weighted Average** | **0.8712** | **0.8710** | **0.8710** | 3,000 |

### 8.3 Confusion Matrix (`figures/confusion_matrix.csv`, `figures/confusion_matrix.png`)

| True \ Pred | Predicted Negative | Predicted Neutral | Predicted Positive | Total |
| :--- | :---: | :---: | :---: | :---: |
| **Negative (0)** | **787** | 63 | 40 | 890 |
| **Neutral (1)** | 77 | **963** | 88 | 1,128 |
| **Positive (2)** | 50 | 69 | **863** | 982 |

---

## 9. Error Analysis Pipeline (`src/error_analysis.py`)

Implemented structured error categorization and stratified performance bucketing:

1.  **Error Categories Identified (`figures/error_categories.csv`):**
    *   **Neutral / Negative / Positive Ambiguity (18 cases):** Subtleties in informal Hinglish phrasing where objective statements carry sarcastic undertones or mild sentiment.
    *   **Negation Flipping (8 cases):** Complex negation structures (e.g., `nahi`, `mat`, `kabhi nahi`) occurring far from the sentiment predicate.
    *   **Emoji-Only Signal (2 cases):** Tweets where sentiment is carried predominantly by emoticons stripped during text-only TF-IDF representation.
    *   **Spelling Variation (2 cases):** Dialectal transliteration differences not fully collapsed by the spelling normalizer.
2.  **Stratified Performance Bucketing:**
    *   **Sequence Length Buckets:** Evaluated accuracy across `short` ($\le 4$), `medium` ($5-8$), and `long` ($> 8$) token counts.
    *   **Code-Mixing Index (CMI) Buckets:** Categorized samples into `low`, `medium`, and `high` code-mixing density.
3.  **Detailed Examples:** Curated and exported representative misclassified examples with predicted vs. true probabilities into `figures/misclassified_examples.csv`.

---

## 10. Interactive Gradio Demo (`app.py`)

A decoupled, lightweight Gradio web interface for real-time inference:
*   Loads serialized `results/weights/best_model.pkl` and `data/processed/preprocessor.json`.
*   Applies identical preprocessing transformations to user inputs.
*   Displays confidence probabilities across Negative, Neutral, and Positive.
*   Includes pre-loaded sample Hinglish tweets demonstrating varied sentiments and emoji contexts.
*   Fails cleanly with informative errors if model artifacts are not found.

---

## 11. Test & Quality Assurance Suite (`tests/`)

A comprehensive verification suite comprising **33 automated unit and integration tests**:

*   [`test_contracts.py`](file:///home/cse/g13_nlp/hinglish-sentiment-analysis/tests/test_contracts.py): Label mappings, required schema columns, duplicate ID validation, null checks.
*   [`test_preprocess.py`](file:///home/cse/g13_nlp/hinglish-sentiment-analysis/tests/test_preprocess.py): Normalization, hashtag handling, character collapsing, tokenizer consistency, persistence round-trips.
*   [`test_eda.py`](file:///home/cse/g13_nlp/hinglish-sentiment-analysis/tests/test_eda.py): Class distribution counting, length stats calculation, vocabulary and OOV rate calculations, Zipf counts, CMI.
*   [`test_embeddings.py`](file:///home/cse/g13_nlp/hinglish-sentiment-analysis/tests/test_embeddings.py): FastText training, vocab export, nearest neighbors, embedding matrix generation.
*   [`test_classical.py`](file:///home/cse/g13_nlp/hinglish-sentiment-analysis/tests/test_classical.py): TF-IDF n-gram ranges, Logistic Regression and Linear SVM training, probability calibration, top n-grams.
*   [`test_evaluate.py`](file:///home/cse/g13_nlp/hinglish-sentiment-analysis/tests/test_evaluate.py): Metrics computation, classification report formatting, confusion matrix, bucket accuracy, append-only results table.
*   [`test_error_analysis.py`](file:///home/cse/g13_nlp/hinglish-sentiment-analysis/tests/test_error_analysis.py): Deterministic bucketing, error filtering, heuristic categorization tagging, error summarization.
*   [`test_app.py`](file:///home/cse/g13_nlp/hinglish-sentiment-analysis/tests/test_app.py): Gradio app initialization, artifact loading validations, inference output contract verification.
*   [`test_reproducibility.py`](file:///home/cse/g13_nlp/hinglish-sentiment-analysis/tests/test_reproducibility.py): End-to-end reproducibility pipeline and handoff manifest generation integrity.

---

## 12. Reproduction Notebooks (`notebooks/`)

Six standalone, parameterized Jupyter notebooks in order of execution:
1.  `notebooks/preprocessing_comparison.ipynb`: Compares standard whitespace splitting with specialized Hinglish preprocessor.
2.  `notebooks/eda.ipynb`: Computes class distributions, sequence lengths, Zipf distributions, and vocab OOV rates.
3.  `notebooks/classical_baseline.ipynb`: Trains word/char TF-IDF models and saves validation predictions.
4.  `notebooks/embeddings.ipynb`: Fits FastText on training split, saves vocabulary metadata and embeddings.
5.  `notebooks/evaluation.ipynb`: Computes formal classification metrics and confusion matrix.
6.  `notebooks/error_analysis.ipynb`: Categorizes prediction failures across length and CMI dimensions.

---

## 13. Deliverables Summary Checklist

| Deliverable | Location | Status |
| :--- | :--- | :---: |
| **Shared Contracts & Schemas** | `src/contracts.py` | Complete |
| **Cleaned & Processed Splits** | `data/processed/train.csv`, `val.csv`, `test.csv` | Complete |
| **Handoff Manifest (H1)** | `data/processed/H1_MANIFEST.json` | Complete |
| **Preprocessing Module & State** | `src/preprocess.py`, `data/processed/preprocessor.json` | Complete |
| **EDA Module & Figures** | `src/eda.py`, `figures/*.png`, `figures/*.csv` | Complete |
| **Feature Extraction & Embeddings** | `src/features.py` | Complete |
| **Classical ML Baselines** | `src/classical.py`, `notebooks/classical_baseline.ipynb` | Complete |
| **Validation Benchmark Results** | `results/results.csv`, `results/metrics.csv` | Complete (87.1% Macro F1) |
| **Error Analysis Taxonomy** | `src/error_analysis.py`, `figures/error_categories.csv` | Complete |
| **Web Interface Demo** | `app.py` | Complete |
| **Test Verification Suite** | `tests/` (33 tests) | Complete |
| **Documentation & Guides** | `README.md`, `PROJECT_SUMMARY.md` | Complete |

