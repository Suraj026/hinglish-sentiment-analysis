# Hinglish Sentiment Analysis

Person B owns the data, NLP analysis, classical baselines, evaluation,
error analysis, demo integration, figures, and delivery documentation on the
`analysis` branch. Person A owns neural models, training, weights, neural
predictions, and attention exports on the `models` branch.

## Shared contracts

- Labels are fixed: `negative=0`, `neutral=1`, `positive=2`.
- Use the official SemEval-2020 Task 9 train/validation/test splits.
- Use seed `42` for comparable experiments.
- H1/H2 split files use `uid,text_clean,label`.
- Prediction files use `uid,y_true,y_pred,p_neg,p_neu,p_pos`.
- Results rows use `model,embedding,normalized,accuracy,macro_f1,weighted_f1,seed`.
- Person B fits vocabularies, spelling maps, TF-IDF, and embeddings on the
  training split only.

## Person B handoffs

### H1 — quick data

`data/processed/train.csv`, `val.csv`, and `test.csv` contain the shared
columns after lowercase plus URL/mention replacement. Include
`H1_MANIFEST.json` with source files, row counts, label counts, seed,
preprocessing version, and checksums.

### H2 — final preprocessing and embeddings

Provide raw and normalized split CSVs, the serialized preprocessing
configuration/resources, `embeddings/fasttext.model`,
`embeddings/vocab.json`, and `H2_MANIFEST.json` with checksums, vocabulary
metadata, and the recommended 95th-percentile sequence length.

Every handoff message to Person A must include canonical paths, schema
validation status, row counts, preprocessing version, recommended max length,
embedding status, checksums, and known deviations from the reference.

## Reference implementation

The project takes conceptual guidance from
[`singhnivedita/SemEval2020-Task9`](https://github.com/singhnivedita/SemEval2020-Task9),
revision `7e2bb7576c35a122d65f574e0fc706451879413d`. Its V1/V2 preprocessing
notebooks, UID/text/label TSV layout, Hindi stopword-generation idea,
tokenizer/padding conventions, and baseline architecture diagrams are
references only. This project reimplements them in tested, reusable modules;
it does not copy notebook code line by line.

The reference is notebook-based and lacks the reusable modules, tests,
macro/weighted-F1 reporting, reproducible handoff manifests, and requested
normalization/embedding/error-analysis extensions. Verify dataset licensing
before redistributing raw data; cite the reference repository and the NITS
paper in the final presentation.

## Reproduction process

The commands below assume Windows PowerShell from the repository root.

### 1. Create the environment

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Alternatively, activate the environment for the current PowerShell session:

```powershell
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

### 2. Add the data

Place these headerless files in `data/`:

```text
data/train.tsv
data/validation.tsv
data/test.tsv
```

The train and validation files must contain `uid`, `text`, and `label`
columns in that order. The test file must contain `uid` and `text`. Do not
commit raw data unless its license allows redistribution. The processed
handoff files contain exactly:

```text
uid,text_clean,label
```

with labels `0`, `1`, and `2` corresponding to negative, neutral, and positive.
Keep the official split membership and UIDs unchanged.

### 3. Run the verification suite

```powershell
.venv\Scripts\python.exe -m pytest -q
```

The tests cover contracts, preprocessing, EDA helpers, TF-IDF baselines,
FastText utilities, evaluation, error analysis, the demo adapter, and handoff
manifest generation. The expected current result is 33 passing tests.

### 4. Reproduce the analysis

Start VS Code with the project virtual environment selected as the notebook
kernel. If Jupyter is installed, it can also be launched explicitly:

```powershell
.venv\Scripts\python.exe -m jupyter notebook
```

Run the notebooks in this order:

1. `notebooks/preprocessing_comparison.ipynb`
2. `notebooks/eda.ipynb`
3. `notebooks/classical_baseline.ipynb`
4. `notebooks/embeddings.ipynb`
5. `notebooks/evaluation.ipynb`
6. `notebooks/error_analysis.ipynb`

The notebooks call the reusable modules in `src/`; use the same `.venv`
interpreter as the test command. They produce:

- `data/processed/`: normalized `train.csv`, `val.csv`, and `test.csv`,
  `preprocessor.json`, and `H1_MANIFEST.json`;
- `embeddings/`: `fasttext.model`, `vocab.json`, and
  `nearest_neighbors.json`;
- `figures/`: class-balance, length, Zipf, confusion-matrix, and
  misclassification CSV/PNG outputs;
- `results/`: validation predictions, metrics, classification report, and
  append-only `results.csv`.

The classical baseline must run before `evaluation.ipynb` and
`error_analysis.ipynb`, because those notebooks consume
`results/predictions/model.csv`. Fit preprocessing resources, TF-IDF
vectorizers, and embeddings on the training split only.

Each notebook discovers the repository root automatically. It also limits
BLAS worker threads to avoid memory-related kernel crashes on constrained
machines. If running cells manually, execute the setup/import cell first and
restart the kernel before rerunning a notebook after an interrupted run.

Normalization removes URLs, user mentions, emoji and other symbols,
punctuation, hashtag markers, and literal missing-value tokens such as `nan`;
it lowercases text, collapses repeated characters, and applies the configured
spelling map while preserving sentiment-bearing words and negations. The
result is saved in `data/processed/*.csv`, and the exact configuration and
spelling map are saved in `data/processed/preprocessor.json`.

To reproduce all notebook code cells non-interactively:

```powershell
$env:OPENBLAS_NUM_THREADS = "1"
$env:OMP_NUM_THREADS = "1"
$env:MKL_NUM_THREADS = "1"
.venv\Scripts\python.exe -m jupyter nbconvert --to notebook --execute `
  --ExecutePreprocessor.kernel_name=python3 `
  --output-dir notebooks\executed notebooks\preprocessing_comparison.ipynb
```

For the normal workflow, execute the six notebooks in VS Code in the order
listed above; the output files remain in the repository's `data/processed/`,
`embeddings/`, `figures/`, and `results/` directories.

### 5. Run the Gradio demo

Before launching the demo, provide:

```text
results/weights/best_model.pkl
data/processed/preprocessor.json
```

These are model artifacts supplied by Person A or produced by a compatible
training workflow. Then run:

```powershell
.venv\Scripts\python.exe app.py
```

The demo loads the saved artifacts without retraining and returns probabilities
for `negative`, `neutral`, and `positive`. It fails explicitly when either
artifact is missing or incompatible.

### 6. Create a handoff manifest

After generating the agreed split files, create a manifest with the handoff
helpers:

```powershell
.venv\Scripts\python.exe -c "from pathlib import Path; from src.handoff import build_manifest, write_manifest; files = {name: Path('data/processed') / f'{name}.csv' for name in ('train', 'val', 'test')}; write_manifest(build_manifest(files, 'H1', seed=42), Path('data/processed/H1_MANIFEST.json'))"
```

The manifest records SHA-256 checksums, CSV columns, row counts, label counts,
the handoff name, and the experiment seed. Replace `H1` with `H2` when
producing the final preprocessing and embedding handoff, and include the
additional H2 artifacts described above.

All Git commits are intentionally manual; verify the relevant tests and
artifacts before committing a checkpoint.
