# Hinglish Sentiment Analysis — Next-Phase Execution Roadmap

**Project:** SemEval-2020 Task 9 — Hinglish sentiment classification  
**Current branch:** `analysis`  
**Current ownership:** Person B — data engineering, NLP analysis, classical baselines, evaluation, error analysis, and demo integration  
**Partner ownership:** Person A — neural architectures, training, model checkpoints, neural predictions, and attention exports  
**Primary goal:** Move from the completed/reported Person B implementation to a reproducible comparison of classical and neural models, then integrate the best model into the final demo and presentation.

> **Status convention**
> - **Reported complete** means the supplied project summary says the work is implemented. Re-run the relevant tests or commands before treating it as verified in the current checkout.
> - **Next** means this is part of the remaining project work.
> - **Optional** means it should only be attempted after the required comparison and demo work are complete.

---

## 1. Where the project stands now

The current implementation summary describes a substantial Person B / `analysis`-branch implementation. The next phase should therefore focus on verification, the handoff to Person A, neural-model experiments, and final integration—not on rebuilding the existing analysis pipeline from scratch.

### 1.1 Work reported as implemented

| Area | Reported implementation | What to do now |
|---|---|---|
| Shared contracts | `src/contracts.py`; fixed labels and CSV schemas | Run the contract tests and confirm Person A uses the same definitions |
| Processed data | `data/processed/train.csv`, `val.csv`, `test.csv`; H1 manifest | Verify row counts, columns, labels, unique IDs, hashes, and exact file paths |
| Preprocessing | `src/preprocess.py`; serialized state in `data/processed/preprocessor.json` | Verify save/load consistency and create the final raw/normalized variants needed for ablation |
| EDA | `src/eda.py`, EDA notebook, figures and CSV summaries | Regenerate the figures from the current data and retain the final outputs |
| Features and embeddings | Word/character TF-IDF support; FastText utilities and vocabulary export | Verify the actual saved FastText model, vocabulary, dimension, and train-only fitting |
| Classical models | Logistic Regression and Linear SVM pipeline support | Confirm which configurations have completed runs and saved prediction files |
| Evaluation | Metrics, report, confusion matrix, and append-only result tracking | Verify every result can be regenerated from its prediction CSV |
| Error analysis | Error categorization and length/CMI bucketing utilities | Review the selected examples manually and regenerate the final analysis for the winning model |
| Gradio | `app.py` and a serialized `best_model.pkl` are reported | Identify whether the current artifact is classical or neural before integrating the final model |
| Tests | 33 unit/integration tests are reported | Run the full test suite in the current environment and record the result |

### 1.2 Current benchmark to preserve

The supplied summary reports the following **validation** result for normalized Word TF-IDF + Logistic Regression:

| Metric | Reported score |
|---|---:|
| Accuracy | 0.8710 |
| Macro F1 | 0.8712 |
| Weighted F1 | 0.8710 |

Reported per-class F1:

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| Negative | 0.8611 | 0.8843 | 0.8725 | 890 |
| Neutral | 0.8795 | 0.8537 | 0.8664 | 1,128 |
| Positive | 0.8708 | 0.8788 | 0.8748 | 982 |

Keep this result as the current classical reference point. Do not overwrite it when new experiments are run. The summary reports a confusion matrix and error-category counts as well; regenerate those from the saved predictions before putting them in the final report.

### 1.3 Important points to confirm before training

1. **Test labels:** The supplied split summary lists 3,000 test examples but does not show test-label counts. Check the actual file and official dataset instructions. Until labeled test targets are confirmed, use the validation split for model selection and do not report test metrics.
2. **Current Gradio artifact:** The summary says `app.py` loads `results/weights/best_model.pkl`. Confirm whether this is a scikit-learn model or a neural checkpoint. A PyTorch/Keras model cannot be treated as a scikit-learn pickle without a matching loader and inference path.
3. **FastText settings:** The summary reports a 16-dimensional FastText model trained on raw training tweets. Confirm the saved model and embedding matrix have the same dimension and vocabulary mapping that Person A will use.
4. **Actual experiment coverage:** The project summary reports model support, but the specific headline result supplied is for Logistic Regression with Word TF-IDF. Distinguish implemented code from completed, saved, reproducible runs in the final results table.
5. **Language tags / CMI:** Confirm the downloaded dataset actually contains usable token-language tags before describing language ratios or tag-derived CMI as measured facts. If not, document the alternative method and its limitations.

---

## 2. Non-negotiable experiment rules

These rules apply to every experiment, regardless of who runs it.

- Use the official train, validation, and test files without changing their membership.
- Fit vocabularies, spelling maps, TF-IDF vectorizers, and FastText only on the **training split**. Use the fitted artifacts to transform validation and test data.
- Keep the label mapping fixed everywhere: `0 = negative`, `1 = neutral`, `2 = positive`.
- Keep the common random seed at `42` for the first comparison. Record any additional seeds explicitly.
- Use validation macro-F1 for model selection and early stopping. Do not tune on the test split.
- Save a metrics row, validation predictions, configuration, and model artifact for every completed run.
- Compare models using the same split and a documented training/tuning budget. State any unavoidable differences.
- Never silently change a shared CSV schema or preprocessing behavior. Discuss the change, update the contract, and update tests first.
- Report scores honestly. A result lower than the reference paper or another model is still a valid result when the experiment is reproducible.

---

## 3. Shared file contracts for Person A and Person B

Agree on these formats before Person A starts the full model run matrix. The current implementation summary says these contracts already exist in `src/contracts.py`; use that module as the source of truth rather than creating a second, slightly different schema.

### 3.1 Processed split files

**Files:**

- `data/processed/train.csv`
- `data/processed/val.csv`
- `data/processed/test.csv`

**Required columns:**

| Column | Meaning |
|---|---|
| `uid` | Stable unique ID for a tweet |
| `text_clean` | Cleaned text supplied to the model pipeline |
| `label` | Integer class label: 0, 1, or 2, where available |

Requirements:

- `uid` must be unique within each split.
- `text_clean` must not be null or an accidental string such as `nan`.
- Labels must be in `[0, 1, 2]` where labels are available.
- Keep the original tweet text in a separate source column/file if the current data pipeline needs it for error inspection; do not replace the agreed model-input column without coordination.

### 3.2 Prediction files

**Recommended path:** `results/predictions/<model>_<configuration>.csv`

**Required columns:**

| Column | Meaning |
|---|---|
| `uid` | ID from the corresponding split; use this to join predictions |
| `y_true` | Gold label |
| `y_pred` | Predicted label |
| `p_neg` | Probability for negative |
| `p_neu` | Probability for neutral |
| `p_pos` | Probability for positive |

Requirements:

- Probabilities must be finite, between 0 and 1, and sum to approximately 1 for each row.
- `y_true` and `y_pred` must use the shared label mapping.
- Each expected split row must appear exactly once; do not rely on row order when joining results.
- Save validation predictions for model comparison. Save test predictions separately, and only calculate test metrics if gold test labels are genuinely available.

### 3.3 Results table

**File:** `results/results.csv`

The existing required columns are:

`model, embedding, normalized, accuracy, macro_f1, weighted_f1, seed`

Add one row per completed run; do not overwrite another person's rows. Keep the existing columns stable. Any additional fields—such as learning rate, hidden size, dropout, max length, or run ID—should be added only after both contributors agree and the contract tests are updated.

### 3.4 Neural checkpoint contents

For each saved neural model, Person A should preserve enough metadata to reproduce inference:

- model state/checkpoint;
- architecture name and model hyperparameters;
- vocabulary/token-to-index mapping or a stable reference to it;
- embedding type and dimension;
- FastText model / embedding-matrix version when applicable;
- maximum sequence length and padding/truncation convention;
- label mapping;
- preprocessing configuration/version;
- random seed and selected epoch.

A checkpoint without its vocabulary, sequence length, and preprocessing configuration is not a complete demo artifact.

---

## 4. Phase-wise execution plan

## Phase 0 — Freeze and verify the current `analysis` branch

**Owner:** Person B  
**Priority:** Required before the model handoff  
**Outcome:** A clean, reproducible baseline and a known-good handoff package.

### Tasks

1. Pull the latest `analysis` branch and inspect `git status`.
2. Run the complete test suite (the summary reports 33 tests) and save the console output or CI result.
3. Regenerate the processed-data manifest and compare row counts, columns, labels, and hashes with the committed manifest.
4. Run preprocessing save/load checks: processing the same text before and after loading `preprocessor.json` should produce the same output.
5. Regenerate the current Logistic Regression validation prediction file and calculate accuracy, macro-F1, weighted-F1, and the classification report.
6. Confirm the saved prediction file aligns to the validation split by `uid` and that probability columns are valid.
7. Record which artifacts actually exist. Do not mark an experiment complete merely because its training code exists.
8. Commit any corrections and tag or otherwise record the known-good baseline commit.

### Suggested checks

Use the repository's actual test command and entry points. If the project uses pytest, a typical command is:

```bash
pytest -q
```

Do not assume a command or script name that is not present in the repository; check `README.md` and the current CLI/notebooks first.

### Exit criteria

- [ ] Full test suite passes, or failures are documented and assigned.
- [ ] Processed split schemas and label mapping are verified.
- [ ] Baseline validation metrics are reproducible from saved predictions.
- [ ] The existing results row is preserved.
- [ ] The current Gradio model artifact type is identified.
- [ ] No uncommitted changes remain before handoff.

---

## Phase 1 — Complete the Person B → Person A handoff

**Owners:** Person B prepares; Person A validates  
**Priority:** Required  
**Outcome:** Person A can train without reimplementing or guessing the data pipeline.

### Person B tasks

1. Send Person A the exact processed data paths and the `src/contracts.py` definitions.
2. Share the serialized preprocessing configuration: `data/processed/preprocessor.json`.
3. Share the vocabulary and embedding artifacts that are actually present, including `embeddings/fasttext.model` and `embeddings/vocab.json` if those are the current paths.
4. State the current sequence-length recommendation derived from the training-length distribution. The work split proposes the 95th percentile; record the actual computed value rather than hardcoding an assumed number.
5. Explain whether Person A should initially use raw or normalized text. For a first reproducible baseline, select one version and record it. The normalization comparison will be run later as an ablation.
6. Share one small example of each CSV contract and document the label order.

### Person A tasks

1. Pull the `analysis` branch or the agreed handoff commit.
2. Load the processed train and validation files using the agreed columns.
3. Validate labels, IDs, and text values through `src/contracts.py`.
4. Load the exact vocabulary/embedding artifacts supplied by Person B; do not independently build a different vocabulary for the same comparison.
5. Return a small smoke-test output confirming the number of examples loaded, label mapping, vocabulary size, embedding dimension, and maximum sequence length.

### Exit criteria

- [x] Person A can load train/validation data without manual file edits.
- [x] Both contributors agree on the same label mapping and sequence conventions.
- [x] The FastText dimension and vocabulary mapping are confirmed.
- [x] One smoke-test prediction file follows the agreed prediction schema.

---

## Phase 2 — Build and run the neural baseline models

**Owner:** Person A  
**Support:** Person B checks data compatibility and evaluates prediction files  
**Priority:** Required  
**Outcome:** Reproducible neural baselines that can be compared with the classical Logistic Regression reference.

### Model order

Implement and validate models in this order so that errors are found early:

1. **LSTM** — embedding → LSTM → classification layer.
2. **BiLSTM** — embedding → bidirectional LSTM → classification layer.
3. **CNN** — embedding → one or more 1D convolution blocks → pooling → classification layer.
4. **LSTM+Conv / BiLSTM+Conv** — recurrent sequence output → 1D convolution (the project guide describes a kernel size of 3 for the BiLSTM-convolution design) → global average and max pooling → concatenation → classification layer.

Follow the architecture descriptions in the NITS-Hinglish-SentiMix paper and cite that paper as the design reference. Write the implementation yourselves; do not copy a repository implementation line by line.

### Training procedure

1. Set seed `42` and record the framework/device used.
2. Build the sequence inputs from the shared vocabulary and agreed preprocessing output.
3. Use a fixed maximum sequence length based on the measured training distribution.
4. Start with random-initialized embeddings for the first baseline pass.
5. Use the same optimizer/loss/early-stopping policy across comparable models. The guide proposes Adam, cross-entropy, and early stopping on validation macro-F1.
6. Perform a small-batch overfit test before full training. The model should be capable of fitting a tiny batch; otherwise debug the pipeline before consuming GPU time.
7. Train each baseline and save the best validation checkpoint.
8. Export validation predictions in the shared format and append a results row.
9. Record training time, best epoch, and configuration in a run log, even if these details are not part of the minimum results schema.

### Person B tasks during this phase

- Run the evaluation pipeline on each valid prediction CSV.
- Verify row counts and UID alignment.
- Check probability validity and label mapping.
- Add results without changing Person A's model code.
- Report malformed files quickly so Person A can correct the exporter once rather than debugging downstream charts.

### Exit criteria

- [ ] At least LSTM and BiLSTM train end-to-end and produce valid validation predictions.
- [ ] The remaining planned baseline architectures are complete, or any dropped model is documented with a reason.
- [ ] Every completed run has a checkpoint, configuration, predictions, and results row.
- [ ] Baseline metrics can be regenerated by Person B's evaluation code.

### Time-constrained fallback

If the baseline phase falls behind, preserve LSTM and BiLSTM first. Drop or defer LSTM+Conv or CNN rather than leaving all model results incomplete.

---

## Phase 3 — Add neural extensions and embedding comparison

**Owner:** Person A  
**Support:** Person B supplies/validates embeddings and tracks results  
**Priority:** Required extensions; prioritize in the order below  
**Outcome:** Measure whether gated recurrent units, attention, and FastText embeddings improve performance.

### Architectures to add

1. **GRU**
2. **BiGRU**
3. **BiLSTM with additive attention**

For additive attention, the guide proposes a score over each time step followed by a softmax over time. Return the attention weights as well as the class probabilities so that selected examples can be visualized later.

### Embedding comparison

Compare:

- random initialization;
- corpus-trained FastText embeddings.

Use the same vocabulary mapping and compatible embedding dimensions. Confirm that padding/unknown-token rows are handled consistently. FastText should be trained only on the training tweets, as specified by the project protocol.

### Recommended experiment sequence

To avoid an unnecessarily large experiment grid:

1. Run the neural baseline models with random embeddings.
2. Run GRU, BiGRU, and BiLSTM-attention with random embeddings.
3. Run FastText embeddings for the strongest one or two baseline architectures and the strongest extension.
4. If compute allows, complete the full model × embedding comparison.
5. Keep hyperparameter tuning light and equally budgeted across the models being compared. The guides suggest exploring hidden size, dropout, and learning rate rather than performing an unlimited search.

### Attention exports

For three selected validation tweets, Person A should export:

- tweet ID;
- token sequence used by the model;
- attention weight for each token;
- predicted class and gold class.

A JSON file per example under `results/attention/` is suitable. Ensure the tokens and weights have the same length and correspond to the exact preprocessed input.

### Exit criteria

- [ ] GRU and BiGRU produce valid validation predictions.
- [ ] BiLSTM-attention returns usable token-level weights.
- [ ] Random vs. FastText is compared on matched configurations.
- [ ] The experiment table clearly distinguishes completed runs from planned runs.
- [ ] Attention exports are available for selected examples.

---

## Phase 4 — Run the ablations and consolidate the comparison

**Owners:** Person B coordinates; Person A supplies neural predictions  
**Priority:** Required  
**Outcome:** A single, defensible results table and ablation figures.

### Required ablations

1. **Normalization:** run the best recurrent model with normalized and raw/non-normalized input, keeping all other settings fixed.
2. **Embedding:** compare random initialization against FastText on matched model/data settings.
3. **Classical features:** compare word n-gram TF-IDF against character n-gram TF-IDF. Include Logistic Regression and Linear SVM only where completed runs are available.

### Optional ablation

- Word2Vec can be included only if it has already been trained successfully and the comparison can be completed without delaying the required work.
- MuRIL or IndicBERT fine-tuning is optional. Attempt it only after the required baselines, ablations, and demo path are secure. The project guides explicitly allow skipping it when time or GPU access is limited.

### Fair-comparison checklist

For every pair of rows used to claim an improvement, confirm:

- same validation split;
- same label mapping;
- same preprocessing except for the ablation variable;
- same seed for the primary comparison;
- same or clearly documented tuning budget;
- no test-set tuning;
- metrics generated by the same evaluation implementation.

### Outputs

- `results/results.csv` — all completed runs;
- `figures/model_comparison_macro_f1.png` — model comparison;
- `figures/embedding_ablation.png` — random vs. FastText;
- `figures/normalization_ablation.png` — normalization on vs. off;
- `figures/tfidf_feature_ablation.png` — word vs. character features, if both are available.

Use the actual filenames already established in the repository where possible; avoid maintaining duplicate charts under competing names.

### Exit criteria

- [ ] Classical and neural results appear in one table.
- [ ] Each plotted value can be traced to a row in `results.csv` and an underlying prediction file.
- [ ] Ablation comparisons change one main factor at a time.
- [ ] Missing experiments are clearly labelled as not run, not silently omitted from claims.

---

## Phase 5 — Final evaluation and error analysis

**Owner:** Person B  
**Input:** Person A's final validation predictions and attention exports  
**Priority:** Required  
**Outcome:** Explain what the best model gets wrong, not just its headline score.

### Evaluation for the selected models

1. Generate accuracy, macro-F1, weighted-F1, and per-class precision/recall/F1.
2. Generate the confusion matrix.
3. Compare the best classical model with the best neural model.
4. Calculate performance by tweet-length bucket using the project's documented boundaries.
5. Calculate performance by low/medium/high Code-Mixing Index bucket, provided the CMI method is valid for the available data.
6. For the selected attention model, plot token-level attention weights for two or three representative tweets.

### Error review

1. Extract approximately 20–30 misclassified validation examples from the best model, if that many are available.
2. Review the tweet text and model probabilities manually.
3. Assign defensible categories, such as:
   - sarcasm or implicit sentiment;
   - negation scope;
   - spelling/transliteration variation;
   - neutral vs. negative/positive ambiguity;
   - emoji or emoticon signal;
   - insufficient context or genuinely ambiguous label.
4. Count the reviewed examples by category.
5. Select three or four clear examples for the presentation, with a short explanation of why the prediction failed.

The current summary reports examples in ambiguity, negation, emoji-only, and spelling-variation categories. Treat those as preliminary until the examples are manually checked and the analysis is rerun for the final selected model.

### Exit criteria

- [ ] The selected model's confusion matrix and per-class report are saved.
- [ ] Length/CMI bucket analysis is reproducible and its methodology is described.
- [ ] Misclassified examples have been manually inspected.
- [ ] Three or four presentation examples have a concise, evidence-based explanation.
- [ ] Attention visualizations align with the model's actual token sequence.

---

## Phase 6 — Select the final model and integrate the Gradio demo

**Owners:** Person B leads integration; Person A supplies the final neural artifact and inference interface  
**Priority:** Required  
**Outcome:** A stable demo that uses the exact model and preprocessing pipeline reported in the results.

### Model-selection decision

Choose the demo model using validation macro-F1, per-class behavior, reliability, and practical inference requirements. Do not choose a model solely because it has the highest accuracy. Record why it was selected and preserve the classical baseline as a fallback.

### Before changing `app.py`

The current implementation summary says the app loads `results/weights/best_model.pkl`. First identify whether that artifact is a scikit-learn pipeline or another model type. Then define a clear inference interface for the final selection.

The app must not assume that every model supports the same `.predict_proba()` API. A scikit-learn model and a PyTorch/Keras checkpoint need different loading and inference code.

### Demo requirements

- input box for a Hinglish tweet;
- predicted class: Negative, Neutral, or Positive;
- probability/confidence for all three classes;
- four preloaded example tweets covering different sentiment classes and text styles;
- the same preprocessing configuration used during training;
- clear error messages when model or preprocessing artifacts are missing;
- optional attention highlight only if the final selected model supports it correctly.

### Testing

1. Test at least 20 new, manually collected or otherwise held-out tweets that were not used to tune the model.
2. Include spelling variants, mixed Hindi/English tokens, punctuation, emojis, short texts, and longer texts.
3. Check that app preprocessing matches the training preprocessing.
4. Check that the three probabilities are finite and sum to approximately 1.
5. Test the app from a fresh process/environment using only documented setup steps.
6. Record a one-minute backup screen capture for the presentation.

### Exit criteria

- [ ] App loads the chosen model and matching metadata from saved artifacts.
- [ ] Training and inference preprocessing are consistent.
- [ ] All 20 test inputs return valid labels and probabilities, or failures are documented.
- [ ] Four examples are available in the UI.
- [ ] A backup demo recording exists.

---

## Phase 7 — Reproducibility, final runs, and repository cleanup

**Owners:** Person A and Person B  
**Priority:** Required for the final chosen model; additional seeds are strongly recommended when time permits  
**Outcome:** Results another person can reproduce from the README.

### Tasks

1. Re-run the selected classical and neural configurations from clean commands.
2. If time permits, run the best configuration with two or three seeds and report mean and standard deviation. Do not mix multi-seed averages with single-seed rows without labelling them.
3. Freeze the final results table and plots after the final run.
4. Remove temporary files and stale checkpoints only after confirming they are not needed for reproduction.
5. Ensure raw data is not committed if the dataset's license or distribution terms prohibit it.
6. Update `README.md` with environment setup, data placement, preprocessing, training commands, evaluation commands, demo launch, and citations.
7. Make sure no absolute local paths (for example, a developer's home directory) are required to run the project.
8. Run the full test suite again after integration.

### Exit criteria

- [ ] Final results can be regenerated using documented commands.
- [ ] Every reported score has a known split, model, embedding setting, normalization setting, and seed.
- [ ] README instructions work in a fresh environment.
- [ ] Full tests pass after the final merge.
- [ ] No restricted raw dataset files are accidentally committed.

---

## Phase 8 — Final report and presentation

**Owners:** Both  
**Priority:** Required  
**Outcome:** A clear story from the linguistic challenge to the measured results and working demo.

### Suggested presentation outline

1. **Problem statement:** why Hinglish sentiment classification is challenging; include a few representative examples.
2. **Dataset and EDA:** split sizes, class distribution, tweet lengths, vocabulary/OOV observations, and language/CMI analysis where supported.
3. **Preprocessing:** show before/after examples and explain normalization choices.
4. **Classical baseline and embeddings:** TF-IDF, Logistic Regression/Linear SVM, FastText, and a useful nearest-neighbor or spelling-variation illustration.
5. **Neural architectures:** explain the paper-based baselines and the GRU/BiGRU/attention extensions.
6. **Training protocol and results:** show the comparison table and state the fairness rules.
7. **Ablations:** normalization and embedding comparisons.
8. **Error analysis:** confusion matrix, failure categories, and three or four annotated errors.
9. **Live demo:** demonstrate a few inputs and show probabilities.
10. **Conclusion:** what the paper provided, what the team implemented/extended, limitations, and next steps.

The longer implementation guide proposes approximately 12 slides; the shorter project plan proposes approximately 10. Use the 10-section outline above as the content structure and split sections into extra slides only where the material is readable.

### Prepare answers to these questions

- Why is Hinglish/code-mixed text difficult for standard sentiment models?
- Why compare word-level and character-level TF-IDF?
- What does FastText offer for spelling/transliteration variation?
- Why compare LSTM, BiLSTM, GRU, BiGRU, and attention?
- Why use macro-F1 in addition to accuracy?
- Why did normalization help or hurt?
- Which errors remain difficult, and what evidence supports that explanation?
- How does the implementation differ from the referenced paper?
- What are the limitations of the dataset, splits, and evaluation?

### Exit criteria

- [ ] Every number on the slides matches the frozen results table.
- [ ] Every chart is generated from saved project outputs.
- [ ] Paper and dataset references are cited.
- [ ] Both contributors know their slide ownership and have rehearsed twice.
- [ ] Live demo and backup recording are ready.

---

## 5. Responsibility matrix

| Workstream | Person A — Models | Person B — Data / Analysis / Delivery |
|---|---|---|
| Shared data contracts | Consume and validate | Maintain and document |
| Preprocessing | Use the agreed pipeline | Own implementation and artifacts |
| Vocabulary / FastText | Load agreed vocabulary and embeddings | Train/export and document |
| EDA / CMI / Zipf / OOV | Provide input requirements | Own and validate |
| Classical TF-IDF models | — | Own |
| LSTM / BiLSTM / LSTM+Conv / CNN | Own | Evaluate outputs |
| GRU / BiGRU / BiLSTM-attention | Own | Evaluate outputs and plot attention |
| Neural training and checkpoints | Own | Check artifact/prediction contracts |
| Results table and ablations | Supply complete run outputs | Consolidate and plot |
| Confusion matrix / error analysis | Supply predictions | Own |
| Gradio model interface | Supply loader/inference contract | Integrate and test app |
| README and presentation | Model/training sections | Data/analysis/demo sections |
| Final integration and rehearsal | Both | Both |

### Folder ownership

Continue the no-overlapping-edits approach from the project plan:

- **Person B:** `data/processed/`, `embeddings/`, `notebooks/`, `src/preprocess.py`, `src/classical.py`, `src/features.py`, `src/eda.py`, `src/evaluate.py`, `src/error_analysis.py`, `app.py`, `figures/`, and data/analysis documentation.
- **Person A:** `src/models.py`, `src/train.py`, neural checkpoint files under `results/weights/`, neural prediction exports, and `results/attention/`.
- **Shared by agreement:** `src/contracts.py`, `results/results.csv`, `README.md`, and final presentation material.

If the current repository layout differs, keep the actual layout and update this ownership table rather than moving working code without a reason.

---

## 6. Git and integration workflow

The project plan uses two work branches: `analysis` for Person B and `models` for Person A, with integration into `main` at handoff points.

### Normal work cycle

1. Each person commits only their owned files to their own branch.
2. Push the branch and open or communicate a concise handoff summary.
3. Before merging, state which files changed, what experiment was run, and where the outputs were saved.
4. Merge at the agreed checkpoint, run tests, and resolve contract conflicts together.
5. Do not commit large checkpoints or raw data unless the repository policy explicitly allows them; use the agreed artifact storage method where appropriate.

### Example integration commands

Run these from the repository root and adapt the remote/branch names to the actual repository:

```bash
# Refresh remote branch information
git fetch origin

# Update the integration branch
git checkout main
git pull --ff-only origin main

# Merge the completed analysis work
git merge --no-ff origin/analysis

# Merge the neural-model work after the handoff is ready
git merge --no-ff origin/models

# Run the test suite before pushing
pytest -q

git push origin main
```

If the team uses pull requests, create a PR for each branch instead of merging directly. Do not merge both branches blindly if they changed the same shared contract or result file; resolve that with both contributors.

---

## 7. Priority order if time becomes limited

Protect the complete end-to-end result rather than trying to run every optional experiment.

### Must finish

1. Verify the current Person B implementation and preserve the Logistic Regression reference result.
2. Complete the Person B → Person A data/embedding handoff.
3. Train at least LSTM and BiLSTM with valid saved predictions; complete additional baseline architectures where feasible.
4. Add at least one extension—preferably BiLSTM with attention—and make a fair comparison.
5. Complete normalization and random-vs-FastText comparisons for the best model(s).
6. Consolidate the results table and produce final error analysis.
7. Integrate and test the Gradio demo.
8. Freeze the README, slides, and backup demo recording.

### Do only if the must-finish list is secure

- Full model × embedding experiment grid;
- Word2Vec comparison;
- extensive hyperparameter search;
- MuRIL or IndicBERT fine-tuning;
- additional ensemble experiments.

### Explicit fallback decisions

- If neural models are late, retain Logistic Regression as the working baseline and finish at least one neural comparison.
- If GPU time is limited, prioritize LSTM, BiLSTM, and one attention model; defer optional transformer fine-tuning.
- If FastText integration is delayed, complete the random-embedding model comparison first and label the FastText runs as pending.
- If the neural demo is unstable, keep a working classical-model demo as the fallback and clearly state which model the demo uses.
- If labeled test targets are unavailable, report validation results only and say so explicitly.

---

## 8. Final definition of done

The project is ready to submit when all required items below are satisfied.

### Data and NLP pipeline

- [ ] Official split usage and label mapping are documented.
- [ ] Preprocessing can be saved, reloaded, and applied identically at inference.
- [ ] EDA figures and data summaries can be regenerated.
- [ ] Train-only fitting is respected for learned text artifacts.

### Experiments

- [ ] Classical baseline metrics and validation predictions are preserved.
- [ ] Planned neural baseline/extension runs are completed or explicitly identified as deferred.
- [ ] Random vs. FastText and normalization ablations are reported for matched configurations.
- [ ] Results table includes all completed runs and their seeds/settings.
- [ ] Best-model selection is based on validation evidence.

### Evaluation

- [ ] Accuracy, macro-F1, weighted-F1, per-class metrics, and confusion matrix are reported.
- [ ] Length and CMI analysis are included where the underlying measurements are valid.
- [ ] Misclassified examples are manually reviewed and annotated.
- [ ] Any unavailable test-label evaluation is disclosed.

### Demo and delivery

- [ ] Gradio demo loads the selected model and matching preprocessing metadata.
- [ ] At least 20 unseen inputs are smoke-tested.
- [ ] Backup screen recording exists.
- [ ] README setup/train/evaluate/demo instructions work from a clean environment.
- [ ] Tests pass after final integration.
- [ ] Slides and citations are complete; both contributors have rehearsed.

---

## 9. Immediate next actions — start here

The next work session should be limited to these actions, in this order:

1. **Person B:** run the current tests and regenerate the Logistic Regression validation result from its saved prediction file.
2. **Person B:** prepare a small handoff bundle containing the processed train/validation CSVs, `preprocessor.json`, vocabulary, FastText artifacts (if present), label mapping, and current sequence-length recommendation.
3. **Person A:** load that bundle and return a smoke-test report confirming row counts, vocabulary size, embedding dimension, maximum sequence length, and label mapping.
4. **Person A:** implement and smoke-test LSTM, then BiLSTM, with random embeddings and the agreed prediction schema.
5. **Person B:** run the evaluation code against those predictions and append the first neural rows to `results/results.csv`.
6. After these checks pass, proceed to the remaining neural models and extensions in Phases 2–4.

**The immediate milestone is not “train every model.” It is “one neural model can consume the shared data, train reproducibly, export a valid checkpoint and prediction file, and be evaluated by the existing analysis pipeline.”** Once this works, the rest of the model comparison becomes a repeatable experiment process.

---

## 10. Source documents used for this roadmap

This roadmap consolidates the supplied materials:

- `Pasted markdown.md` — current implementation summary and reported implementation status.
- `Hinglish Sentiment Analysis_ Project Plan - Claude.pdf` — project goals, dataset, baseline/extension scope, and presentation outline.
- `Hinglish Sentiment_ Step-by-Step Implementation Guide - Claude.pdf` — detailed NLP tasks, experiment steps, and three-week plan.
- `NLP course project ideas with RNN_LSTM - Claude.pdf` — two-person ownership, handoff contracts, branch workflow, and workload split.

Where the current implementation summary and planning guides differ, this roadmap treats the current summary as the description of what has already been built and the guides as the intended remaining scope. Any repository state or dataset detail that has not been independently inspected is marked for verification rather than assumed.
