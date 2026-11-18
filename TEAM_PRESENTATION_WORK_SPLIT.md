# Hinglish Sentiment Analysis: Team Work Split and Presentation Guide

**Repository:** `Suraj026/hinglish-sentiment-analysis`

This document identifies the actual files implemented in the repository, explains what each person should present, and states the final measured outputs. It is designed as a handoff for teammates and a rehearsal guide.

## 1. Project summary for the opening

The project analyzes code-mixed Hinglish tweets from SemEval-2020 Task 9 and predicts one of three sentiments:

- `0`: Negative
- `1`: Neutral
- `2`: Positive

The system covers:

1. dataset cleaning and Hinglish-specific preprocessing;
2. exploratory and linguistic analysis;
3. TF-IDF classical baselines and neural baselines;
4. FastText embeddings and recurrent/CNN architectures;
5. validation metrics, confusion matrices, and error analysis;
6. a FastAPI and React demo for live inference.

The official split used in the repository contains:

| Split | Rows | Negative | Neutral | Positive |
|---|---:|---:|---:|---:|
| Train | 14,594 | 4,252 | 5,473 | 4,869 |
| Validation | 3,000 | 890 | 1,128 | 982 |
| Test | 3,000 | No labels | - | - |

The configured random seed is `42`.

---

## 2. Person 1: Data, NLP Analysis, and Demo

### File ownership

Primary files:

- `src/preprocess.py`
- `src/eda.py`
- `src/dataset.py`
- `src/handoff.py`
- `src/contracts.py`
- `data/processed/`
- `notebooks/eda.ipynb`
- `notebooks/preprocessing_comparison.ipynb`
- `backend/` and `frontend/` for the current full-stack demo
- `PROJECT_SUMMARY.md` and `README.md`

For the original Gradio design, `app.py` is referenced in project documentation, but the repository currently uses FastAPI plus React instead.

### Work completed

**Quick and final data preparation**

- Loads the official train, validation, and test splits.
- Validates unique IDs, text columns, and the fixed three-label mapping.
- Publishes the H1 and H2 handoff manifests.
- Fits preprocessing resources only on the training data.

**Hinglish preprocessing**

- Uses NLTK `TweetTokenizer`.
- Lowercases text when normalization is enabled.
- Removes URLs and `@mentions`.
- Removes missing-value tokens such as `nan`, `none`, and `null`.
- Keeps hashtag stems by converting `#tag` to `tag`.
- Collapses runs of three or more repeated characters to at most two characters.
- Maps `nahin`, `nahee`, and `nahein` to `nahi`.
- Saves the configuration and spelling map in `data/processed/preprocessor.json`.

**EDA**

- Computes class distribution.
- Computes character and token lengths.
- Calculates train/test vocabulary size and OOV rate.
- Generates Zipf frequency counts.
- Provides language-tag ratios and code-mixing calculations.

**Demo integration**

- The FastAPI backend exposes health, model, analysis, batch, benchmark, and telemetry endpoints.
- The React frontend provides a live inference console and model dashboard.
- The demo returns label probabilities and language mix details.
- The demo includes a fallback lexicon mode if PyTorch checkpoint loading is unavailable.

### Code explanation

`Preprocessor` in `src/preprocess.py` is the central component. `fit()` learns corpus resources and `transform_text()` applies them consistently. The saved JSON makes the preprocessing reproducible and allows Person 3 to use exactly the same cleaning rules.

`HinglishDataset` and `create_dataloader` in `src/dataset.py` convert cleaned text into token IDs, pad sequences, and create PyTorch batches.

`H1_MANIFEST.json` and `H2_MANIFEST.json` record file hashes and row counts so the team can verify that every handoff is complete and unchanged.

### Results obtained

- Train vocabulary and test-vocabulary resources were exported.
- H1: 14,594 train rows and 3,000 validation rows.
- H2: 28,947 FastText vocabulary entries.
- Recommended sequence length: 95th percentile = 22 tokens.
- H2 also records p90 = 21, p95 = 22, p99 = 25, and maximum = 104 tokens.
- EDA figures exist in `figures/class_distribution.png` and `figures/token_length_histogram.png`.
- Preprocessor state is stored in `data/processed/preprocessor.json`.

### Suggested presentation content

- Raw tweet → preprocessing steps → clean tweet.
- Class distribution chart.
- Vocabulary and OOV motivation.
- Zipf plot and code-mixing findings.
- Demo walkthrough with positive, negative, neutral, and Hinglish examples.

---

## 3. Person 2: Classical Baseline, Embeddings, and CNN

### File ownership

Primary files:

- `src/features.py`
- `src/classical.py`
- `src/models.py`, specifically `CNNClassifier`
- `embeddings/fasttext.model`
- `embeddings/vocab.json`
- `notebooks/classical_baseline.ipynb`
- `notebooks/embeddings.ipynb`

### Work completed

**TF-IDF baselines**

- Word TF-IDF uses word unigrams and bigrams.
- Character TF-IDF uses character n-grams from 2 to 5.
- Logistic Regression uses the SAGA solver.
- Linear SVM uses `LinearSVC`.
- SVM probabilities are calculated from calibrated softmax-normalized decision scores.

**FastText embeddings**

- Trains FastText with:
  - skip-gram (`sg=1`);
  - 16-dimensional vectors;
  - window size 3;
  - minimum count 1;
  - 20 epochs;
  - seed 42.
- Exports `embeddings/fasttext.model` and `embeddings/vocab.json`.
- Uses subword vectors for spelling-variant robustness and OOV words.

**CNN**

- Embedding layer converts token IDs to vectors.
- Parallel Conv1D kernels of sizes 2, 3, and 4 detect local n-gram patterns.
- ReLU activates convolution output.
- Global max pooling retains the strongest feature per kernel.
- Concatenated pooled features enter a dropout and softmax-ready linear classifier.

### Code explanation

`build_word_tfidf()` and `build_char_tfidf()` create the two feature representations. `train_classical_model()` fits vectorizers and classifiers strictly on training text.

`extract_top_ngrams()` ranks coefficients for a linear model and reports the strongest n-grams for each class.

`CNNClassifier.forward()` transposes the embedding tensor to Conv1D format, applies each kernel, takes global max pooling, concatenates features, and predicts across the three sentiment labels.

### Results obtained

The strongest recorded classical result is Logistic Regression with word TF-IDF:

| Metric | Result |
|---|---:|
| Accuracy | 87.17% |
| Macro F1 | 87.19% |
| Weighted F1 | 87.16% |

A second Logistic Regression run is recorded at:

- Accuracy: 87.10%
- Macro F1: 87.12%
- Weighted F1: 87.10%

The repository leaderboard also records:

| Model | Embedding/features | Accuracy | Macro F1 |
|---|---|---:|---:|
| 1D CNN | random | 90.23% | 90.35% |
| Logistic Regression | word TF-IDF | 87.10% | 87.12% |

FastText exports and nearest-neighbor support are implemented, but the complete character-TF-IDF versus word-TF-IDF and random-versus-FastText neural ablation matrix is not fully represented in `results/results.csv`.

### Suggested presentation content

- Compare bag-of-n-grams to learned embeddings.
- Explain why character TF-IDF helps spelling variation.
- Explain FastText subword handling for OOV and Hinglish variants.
- Show top positive, neutral, and negative n-grams.
- Explain why parallel convolution kernels capture different phrase lengths.

---

## 4. Person 3: Recurrent Models

### File ownership

Primary files:

- `src/models.py`
- `src/train.py`
- `src/dataset.py`
- `embeddings/vocab.json`
- `embeddings/fasttext.model`
- `results/weights/`
- `results/predictions/`
- `results/attention/`

### Work completed

**Data pipeline**

- Loads the serialized preprocessor.
- Builds or loads vocabulary mappings.
- Pads sequences to the configured maximum length.
- Creates shuffled training and deterministic validation DataLoaders.
- Uses a fixed label mapping.

**Architectures**

The repository implements:

1. unidirectional LSTM;
2. bidirectional LSTM;
3. multi-kernel 1D CNN;
4. BiLSTM followed by Conv1D, average pooling, and max pooling.

**Training**

- Seed: `42`.
- Adam optimizer.
- Weight decay.
- Cross-entropy loss.
- Gradient clipping with maximum norm `5.0`.
- Early stopping based on validation macro F1.
- Restores the best validation checkpoint.
- Exports validation predictions and appends benchmark rows.
- Performs a small-batch overfit sanity test.

### Code explanation

`LSTMClassifier` reads the final forward hidden state and sends it to a three-class linear classifier.

`BiLSTMClassifier` concatenates the final forward and backward hidden states, allowing the model to use context from both directions.

`BiLSTMConvClassifier`:

1. creates bidirectional sequence representations;
2. applies a kernel-size-3 Conv1D;
3. applies global average pooling;
4. applies global max pooling;
5. concatenates both feature vectors;
6. performs dropout and classification.

`train_baseline_model()` handles vocabulary loading, batching, training, validation, early stopping, checkpoint saving, and prediction export.

### Results obtained

The recorded validation leaderboard is:

| Model | Embedding | Accuracy | Macro F1 | Weighted F1 |
|---|---|---:|---:|---:|
| LSTM | random | 89.87% | 90.01% | 89.88% |
| BiLSTM | random | 91.53% | 91.61% | 91.53% |
| CNN | random | 90.23% | 90.35% | 90.25% |
| BiLSTM + Conv1D | random | 91.57% | 91.64% | 91.56% |

**Best implemented neural model:** BiLSTM + Conv1D

It improves over Logistic Regression by approximately:

- 4.40 percentage points in accuracy;
- 4.52 percentage points in macro F1.

### Presentation gaps to disclose

The repository does **not** currently implement:

- separate GRU and BiGRU classes;
- additive attention or a BiLSTM/BiGRU attention variant;
- attention-weight JSON export;
- actual FastText-versus-random and normalized-versus-raw matrices;
- saved neural weight files under `results/weights/` in the current checkout;
- saved neural prediction files under `results/predictions/`.

The split's requested recurrent-model extras must be described as future extensions or added before the presentation if they are claimed.

### Suggested presentation content

- Show the LSTM and BiLSTM architecture diagrams.
- Explain forward/backward context in a Hinglish sentence.
- Explain CNN feature extraction after the BiLSTM.
- Present the benchmark table.
- Explain early stopping and seed `42`.
- Explain that the champion combines sequence context and n-gram-style local features.

---

## 5. Person 4: Evaluation, Error Analysis, and Transformer Comparison

### File ownership

Primary files:

- `src/evaluate.py`
- `src/error_analysis.py`
- `notebooks/evaluation.ipynb`
- `notebooks/error_analysis.ipynb`
- `results/results.csv`
- `results/metrics.csv`
- `results/classification_report.csv`
- `figures/confusion_matrix.png`
- `figures/`

Transformer comparison needs a separate script not currently present.

### Work completed

**Evaluation**

- Computes accuracy, macro F1, and weighted F1.
- Builds a per-class classification report.
- Builds a three-class confusion matrix.
- Supports accuracy calculations by CMI bucket and tweet-length bucket.
- Validates the shared prediction schema.
- Provides append-only benchmark-row support.

**Error analysis**

- Joins misclassified validation rows with original text.
- Limits analysis to representative misclassifications.
- Tags:
  - negation;
  - emoji-only signal;
  - spelling variation;
  - neutral/negative confusion.
- Summarizes category counts.
- Creates short, medium, and long length buckets.
- Creates low, medium, and high CMI buckets.

**Visual outputs**

- Class-distribution figure.
- Token-length histogram.
- Confusion-matrix figure.
- Benchmark results CSV.
- Per-class metrics CSV.

### Code explanation

`compute_metrics()` validates `uid`, true label, predicted label, and class probabilities before calculating metrics.

`build_classification_report()` uses scikit-learn with labels `[0, 1, 2]` and names `negative`, `neutral`, and `positive`.

`compute_bucket_accuracy()` groups predictions by length or CMI bucket and calculates accuracy for each group.

`collect_misclassifications()` merges incorrect predictions with text by UID. `tag_error_category()` applies heuristic rules for reporting representative failure types.

### Results obtained

The available classification report is associated with the recorded Logistic Regression benchmark:

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| Negative | 86.11% | 88.43% | 87.25% | 890 |
| Neutral | 87.95% | 85.37% | 86.64% | 1,128 |
| Positive | 87.08% | 87.88% | 87.48% | 982 |
| Macro average | 87.04% | 87.23% | 87.12% | 3,000 |

The confusion matrix for the available prediction set:

| Actual \ Predicted | Negative | Neutral | Positive |
|---|---:|---:|---:|
| Negative | 787 | 63 | 40 |
| Neutral | 77 | 963 | 88 |
| Positive | 50 | 69 | 863 |

Overall recorded accuracy: 87.10%.

Error categories are supported by code, but no final categorized CSV or selected 20-30-example audit is present as a repository artifact in the current checkout.

### Presentation gaps to disclose

The following requested deliverables are not currently implemented or exported:

- MuRIL fine-tuning script;
- IndicBERT fine-tuning script;
- recurrent-versus-transformer benchmark row;
- attention heatmap weights;
- final CMI-bucket accuracy table;
- final tweet-length-bucket accuracy table;
- an exported list of 20-30 manually tagged errors.

### Suggested presentation content

- Show macro F1 as the primary benchmark metric.
- Present the confusion matrix.
- Explain the negative/neutral boundary.
- Show bucket and error-analysis charts if generated.
- State clearly whether the transformer comparison is a planned extension or requires a new result.

---

## 6. Shared files and contracts

These files support everyone and should not be edited by one person without team approval:

- `src/contracts.py`: fixed labels and prediction/result schemas.
- `data/processed/H1_MANIFEST.json`: quick handoff verification.
- `data/processed/H2_MANIFEST.json`: preprocessing and embedding verification.
- `data/processed/preprocessor.json`: saved preprocessing state.
- `embeddings/vocab.json`: shared neural vocabulary.
- `embeddings/fasttext.model`: shared embedding source.
- `results/results.csv`: shared benchmark table.
- `results/metrics.csv`: shared metric summary.
- `tests/`: 33-test repository verification suite.

The shared prediction contract is:

```text
uid, y_true, y_pred, p_neg, p_neu, p_pos
```

The shared result contract is:

```text
model, embedding, normalized, accuracy, macro_f1, weighted_f1, seed
```

---

## 7. Recommended presentation order

### Slide 1: Problem and dataset

Presented by Person 1:

- Hinglish code-mixed tweets.
- Three sentiment labels.
- 14,594 train and 3,000 validation examples.
- Mild class imbalance.

### Slide 2: Preprocessing

Presented by Person 1:

- Before/after normalization examples.
- URL and mention removal.
- Repeated characters and spelling variants.
- H1/H2 handoff and reproducibility.

### Slide 3: EDA and linguistic findings

Presented by Person 1:

- Class distribution.
- Tweet-length distribution.
- Vocabulary, OOV, Zipf, CMI, and language ratios.

### Slide 4: Classical baselines

Presented by Person 2:

- Word and character TF-IDF.
- Logistic Regression and Linear SVM.
- Top n-grams.
- 87.12% to 87.19% macro-F1 range.

### Slide 5: Embeddings

Presented by Person 2:

- FastText subword vectors.
- 28,947 vocabulary entries.
- 16 dimensions.
- Nearest neighbors and spelling variation.

### Slide 6: Neural architectures

Presented by Person 3:

- LSTM.
- BiLSTM.
- CNN.
- BiLSTM+Conv.
- Training and early stopping.

### Slide 7: Neural benchmark table

Presented by Person 3:

- BiLSTM+Conv champion at 91.64% macro F1.
- BiLSTM close at 91.61% macro F1.
- CNN at 90.35% macro F1.
- LSTM at 90.01% macro F1.

### Slide 8: Evaluation

Presented by Person 4:

- Confusion matrix.
- Precision, recall, and F1 by class.
- Macro-F1 as the primary comparison metric.

### Slide 9: Error analysis

Presented by Person 4:

- Negation.
- Spelling variation.
- Emoji-only signals.
- Sarcasm or ambiguous neutral/negative cases.
- Four representative examples.

### Slide 10: Demo

Presented by Person 1:

- React/FastAPI interface.
- Text input.
- Selected model.
- Predicted label and probabilities.
- Language-mix details.

### Slide 11: Limitations and future work

Presented jointly:

- Add GRU/BiGRU and additive attention.
- Export attention weights and heatmaps.
- Complete normalized/raw and random/FastText ablations.
- Fine-tune MuRIL or IndicBERT.
- Perform manual 20-30-example error tagging.
- Ensure real checkpoint-backed demo inference is clearly distinguished from fallback inference.

---

## 8. Key sentences to rehearse

**Person 1:**

> We created a reproducible preprocessing and handoff pipeline. It uses TweetTokenizer, Hinglish spelling normalization, URL and mention removal, character collapsing, and serialized preprocessing state.

**Person 2:**

> Character TF-IDF captures spelling variation, while word TF-IDF captures semantic phrases. FastText extends this with subword vectors, and the CNN learns local Hinglish patterns through multiple convolution kernel sizes.

**Person 3:**

> The neural experiments use the same train and validation split and seed 42. BiLSTM reads context in both directions, while BiLSTM+Conv combines bidirectional sequence context with local convolution features and pooling.

**Person 4:**

> We evaluate with macro F1 because all three sentiment classes matter. The available per-class metrics show that the strongest confusions occur between neutral and sentiment-bearing classes, making neutral classification and negation important error categories.

---

## 9. Final honesty check before presenting

Do not claim the following unless the missing files are produced:

- attention heatmaps;
- exported GRU or attention weights;
- complete recurrent embedding ablation;
- a transformer-versus-recurrent table;
- FastText CNN or recurrent checkpoints;
- real neural checkpoint inference in the demo when only fallback lexicon inference is running.
