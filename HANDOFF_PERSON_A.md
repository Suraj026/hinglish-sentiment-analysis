# Phase 1 Handoff: Person B → Person A (Neural Modeling)

**Branch:** `analysis`  
**Target Branch:** `models`  
**Execution Environment:** `roboflow` conda environment (`/home/cse/miniforge3/envs/roboflow/bin/python`)  
**Handoff Package Version:** `H2`  
**Manifest:** [`data/processed/H2_MANIFEST.json`](file:///home/cse/g13_nlp/hinglish-sentiment-analysis/data/processed/H2_MANIFEST.json)

---

## 1. Overview & Handoff Objectives

This document completes **Phase 1** of the next-phase roadmap: handing over data, schemas, preprocessor configuration, embeddings, sequence length recommendations, and validation tooling from Person B to Person A.

All data splits and feature representations have been fit **strictly on the training split** to guarantee zero data leakage.

---

## 2. Canonical Artifact Paths

| Artifact | Canonical Path | Description | SHA-256 Checksum |
| :--- | :--- | :--- | :--- |
| **Train Split** | `data/processed/train.csv` | 14,594 rows, normalized text, labels | `08784cb27f129d206d40b67e5251ee2897b7777b059b7082871f5d2924cd13f1` |
| **Validation Split** | `data/processed/val.csv` | 3,000 rows, normalized text, labels | `d6e059e457ce5476e993422647798767d83ebabd3152c94c46ecf7f2ef8cf65e` |
| **Test Split** | `data/processed/test.csv` | 3,000 rows, normalized text | `e56de949389366b49c8c873a238a8bbadf3db6920afa526da42553f1fbfb2478` |
| **Preprocessor State** | `data/processed/preprocessor.json` | Serialized config & Hinglish spelling map | `6da732aade705755f84a2d875ae6dabcfa0888929423dafe90cdc2cc59e7b90d` |
| **H2 Manifest** | `data/processed/H2_MANIFEST.json` | Verified artifact manifest with SHA-256 | — |
| **FastText Model** | `embeddings/fasttext.model` | Train-only subword skipgram model (dim=16) | `a6adeaa4762fad45447c30201d4125ff7296543440ee4b0911a617cce6ecd2a6` |
| **Vocabulary Map** | `embeddings/vocab.json` | Token-to-index mapping (28,947 tokens) | `ee983a3c640a7e9efeb1b1602547575c9ab85a457942530ba3faa95b3d04721d` |

---

## 3. Shared Contracts & Schemas ([`src/contracts.py`](file:///home/cse/g13_nlp/hinglish-sentiment-analysis/src/contracts.py))

### 3.1 Fixed Label Mapping
```python
LABEL_TO_ID = {
    "negative": 0,
    "neutral": 1,
    "positive": 2,
}
```
*Always output integer labels in `{0, 1, 2}` matching this order.*

### 3.2 Processed Split Dataframes
Columns: `uid,text_clean,label` (Test has `uid,text_clean`)
*   `uid`: Integer/string tweet identifier (preserved from SemEval).
*   `text_clean`: Preprocessed Hinglish string.
*   `label`: Integer target in `[0, 1, 2]`.

> [!TIP]
> When loading CSVs in pandas, use `pd.read_csv(path, keep_default_na=False)` so that tweets containing only stripped tokens remain empty strings `""` rather than `NaN`.

Example row:
```csv
uid,text_clean,label
3,pakistan ka ghra tauq pakistan israel ko tasleem nahein kerta isko palestin kehta - occupi palestin nan,0
```

### 3.3 Prediction File Schema
Path: `results/predictions/<model_name>.csv`  
Columns: `uid,y_true,y_pred,p_neg,p_neu,p_pos`

Requirements:
1. Probabilities `p_neg, p_neu, p_pos` must be non-negative floats summing to $1.0 \pm 10^{-4}$.
2. `y_pred` must equal `argmax([p_neg, p_neu, p_pos])`.
3. Validation and testing frames must match row counts exactly.

Example row:
```csv
uid,y_true,y_pred,p_neg,p_neu,p_pos
27137,1,0,0.55753,0.35871,0.08375
```

---

## 4. Sequence Length & Text Recommendations

*   **Initial Text Input:** Use **normalized text** (`text_clean` from `train.csv` and `val.csv`) for the baseline neural models (LSTM, BiLSTM, CNN, BiLSTM+Conv). The raw vs. normalized comparison will be run as an ablation in Phase 4.
*   **Recommended Maximum Sequence Length ($T_{\text{max}}$):**
    *   **95th percentile:** **22 tokens** (Recommended for efficient LSTM/CNN training).
    *   **99th percentile:** **25 tokens**.
    *   **Max observed length:** **104 tokens**.
    *   *Padding convention:* Pad or truncate sequences to **22** tokens with post-padding/pre-padding using token index `0` for `<pad>`.

---

## 5. Embeddings & Vocabulary Specification

*   **Vocabulary:** 28,947 unique tokens extracted from training tweets.
*   **Special Tokens:**
    *   Index `0`: `<pad>` (zero vector).
    *   Index `1`: `<unk>` (random or mean vector).
    *   Indices $2 \dots 28948$: Vocabulary tokens from `embeddings/vocab.json`.
*   **FastText Model:** Subword skipgram model trained with `vector_size=16`, `window=3`, `epochs=20`.
*   **Embedding Matrix Builder:** Person A can directly invoke `src.features.build_embedding_matrix(fasttext_path, token_to_id, embedding_dim=16)`.

---

## 6. Smoke-Test Verification & Results

A complete smoke-test script has been implemented in [`scripts/smoke_test_handoff.py`](file:///home/cse/g13_nlp/hinglish-sentiment-analysis/scripts/smoke_test_handoff.py) and added to the automated test suite in [`tests/test_handoff.py`](file:///home/cse/g13_nlp/hinglish-sentiment-analysis/tests/test_handoff.py).

### Smoke-Test Run Output:
```json
{
  "status": "PASSED",
  "train_rows": 14594,
  "val_rows": 3000,
  "train_label_distribution": {
    "0": 4252,
    "1": 5473,
    "2": 4869
  },
  "val_label_distribution": {
    "0": 890,
    "1": 1128,
    "2": 982
  },
  "label_mapping": {
    "negative": 0,
    "neutral": 1,
    "positive": 2
  },
  "vocab_size": 28947,
  "embedding_dim": 16,
  "embedding_matrix_shape": [28949, 16],
  "recommended_p95_seq_length": 22,
  "smoke_test_prediction_file": "results/predictions/smoke_test.csv",
  "smoke_test_metrics": {
    "accuracy": 0.3447,
    "macro_f1": 0.3441,
    "weighted_f1": 0.3451
  }
}
```

---

## 7. Phase 1 Exit Criteria Checklist

- [x] **Data Loading:** Train (14,594) and validation (3,000) splits load cleanly without manual intervention.
- [x] **Contracts & Labels:** Both contributors agree on label IDs (`negative=0`, `neutral=1`, `positive=2`) and schema conventions.
- [x] **Sequence Length:** Empirically verified 95th percentile length = 22 tokens.
- [x] **FastText & Vocabulary:** Verified 16-dim FastText model and 28,947-word vocabulary in `embeddings/`.
- [x] **Smoke-Test Prediction:** Valid prediction file generated at `results/predictions/smoke_test.csv` and successfully evaluated by Person B's metric evaluator.
- [x] **Test Suite:** 34 / 34 pytest tests pass in `roboflow` environment.

