# Future result and model integration

SentiLokal – Malaysian E-Commerce Review Sentiment Analytics does not train,
balance, infer, or download a model during Stage 1. No result files are
bundled. The dashboard discovers real files under `results/` on each rerun.
Validation errors are displayed rather than replaced with example results.

## Historical prediction file

Save `results/predictions.csv` with columns:

- `review_id`: private canonical ID obtained through the owner-only local export
  below. Use the exact selected dataset: original `cmtid` values may repeat and
  the loader disambiguates these rows. Never join on comment text or row position.
- `predicted_sentiment`: actual model output, `0`/`1` or `Negative`/`Positive`.
- `confidence` (optional): actual probability for the selected class in [0, 1].

IDs must be unique and nonempty. Missing predictions stay missing. The explorer
reports coverage; prediction-only filtering excludes rows without a prediction.
Extra IDs are permitted because the source may include records excluded from the
binary dashboard. Files with no matching IDs are rejected. Uploading a different
dataset does not authorize a positional prediction join.

### Owner-only review-key export

Customer screens and CSV downloads intentionally exclude `review_id`,
`source_review_id`, and `product_id`. Do not add an owner export or source-schema
panel to the public Streamlit pages. An owner can prepare a private local key
file from the repository root without changing the original dataset:

```python
from pathlib import Path
from dashboard.data import load_dataset

dataset = load_dataset(Path("data/dataset_preprocessed_v3_clean.csv"))
keys = dataset.reviews[
    ["review_id", "product_id", "product_name", "review_text", "sentiment"]
].copy()
destination = Path("outputs/dashboard_review_keys.csv")
destination.parent.mkdir(parents=True, exist_ok=True)
keys.to_csv(destination, index=False, encoding="utf-8-sig")
```

`outputs/` and CSV files are ignored by Git. Keep this file owner-only and local;
it contains source identifiers and review text. The example reads the source and
writes a separate file, with no balancing, model training, preprocessing rerun,
or original CSV rewrite. Use the exact source dataset and its stable review IDs
when generating genuine predictions later. The customer-facing projection in
`dashboard/presentation.py` continues to exclude these identifiers even after
validated predictions are joined.

## Held-out evaluation

Create one `results/evaluation_<model-id>.json` per model. Required fields are
`model_id`, `model_name`, `test_set_id` (a stable identity for the exact held-out
samples), `split: "test"`, `held_out: true`, and an `evaluation_config` object
containing `averaging: "binary"`, `positive_label: 1`, `zero_division: 0`.
Record additional relevant settings in that object, such as decision threshold.
Do not describe training or whole-dataset predictions as a held-out evaluation.

Choose either input route:

1. Set `predictions_file` to a CSV relative to `results/`, containing nonempty,
   unique canonical `review_id`, `y_true`, `y_pred`. Metrics and the confusion
   matrix are calculated from these genuine test outputs with scikit-learn.
2. Supply recorded `metrics` (`accuracy`, `precision`, `recall`, `f1` as fractions
   from 0 to 1), `confusion_matrix` in `[[TN, FP], [FN, TP]]` order, and positive
   integer `n_samples`. Matrix totals and binary metrics must agree (within
   0.00005); save at least five decimal places. This validates consistency, not
   the experimental provenance, which the file author must establish.

Optional metadata such as `checkpoint`, `training_split`, `cross_validation`,
`training_run_id`, and limitations may be retained in the private result files.
The public page shows a small readable evaluation summary, not raw source or
artifact identifiers. Baseline
comparison requires the same test_set_id, sample count, and full evaluation
configuration. If both inputs include test review IDs, membership must agree.
Use the same evaluation configuration for all compared models. Scores from
unrelated papers or different test sets are not experiment results for this app.
Files alone cannot prove that a held-out set was never used during training.

## Optional trained-model adapter

`dashboard.model_loader.load_trained_model(manifest_path,
preprocessing_registry)` is an explicit integration API, unused by Stage 1.
No model checkpoint or trained hyperparameters are assumed. Export the actual
training artifacts, including local tokenizer and encoder directories (encoder
weights in safetensors format), and XGBoost JSON/UBJ. Pickle/joblib loading is
intentionally unsupported. All artifact paths must stay inside the manifest's
directory.

The JSON manifest must record `model_id`, exact `checkpoint`,
`preprocessing_id`, `tokenizer_path`, `encoder_path`, `classifier_path`,
`pooling` (`cls` or attention-mask `mean`), `max_length` (1–512), the actual
`decision_threshold` in (0, 1), and
`label_mapping: {"0": "Negative", "1": "Positive"}`. These must match training;
the adapter cannot recover missing training choices or independently authenticate
the checkpoint provenance declared in the manifest. The classifier dimensions
and class order are checked, but these checks do not prove the checkpoint is the
original one. No automatic substitute is
made for a missing checkpoint, tokenizer, preprocessing callable, or pooling rule.

Register the *existing* preprocessing function under its recorded ID and install
matching training versions of PyTorch, Transformers, XGBoost, and safetensors.
The loader uses local files only and disallows custom remote code. Cache resource
loading with `st.cache_resource` when adding an inference page, including the
manifest version and preprocessing identity in the cache key. The returned
`HybridModel.predict(texts, batch_size=16)` runs CPU batches and returns binary
labels and the genuine selected-class probabilities. It never fits a classifier.
Only integrate artifacts whose origin and training settings you trust.
