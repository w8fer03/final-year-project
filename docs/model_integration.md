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

## Future trained-model interface (disabled now)

The current UI always shows **Model prediction feature coming after training**.
Neither module calls a model loader or predictor. Supplying files to `models/`
does not enable a feature automatically, and predictions in uploaded CSV columns
are not trusted. No normalization pipeline is chosen from the several existing
preprocessing scripts: the approved training export will be authoritative.
Those original scripts and normalization resources are not modified here.

`dashboard.inference.PredictionService()` is unavailable by default. Calling it
without a trained predictor raises `ModelIntegrationError` with the pending
message; it never returns dummy labels. The future owner-only factory is:

```python
from dashboard.inference import load_prediction_service
from dashboard.model_loader import RegisteredPreprocessor

# These names must refer to the exact code/resources/settings exported by training.
profile = RegisteredPreprocessor(
    transform=approved_training_preprocess,
    files=approved_training_files,
    settings=approved_training_settings,
)
service = load_prediction_service(
    manifest_path,
    {recorded_preprocessing_id: profile},
)
results = service.predict_reviews(uploaded_dataset.records, batch_size=16)
# In a future UI, expose results using st.dataframe and st.download_button.
```

This is an integration example, not code that the current app executes. The
original `review_text` is passed to the registered callable only inside model
inference; it remains unchanged in the returned table. No star rating is needed.
Every input row, including duplicates, is retained in order. No balancing,
deduplication, review eligibility filtering, or fitting happens during inference.
If preprocessing cannot produce a string for a row, inference raises an error
instead of silently dropping it. Future policy for such reviews must be approved
and match the actual training pipeline.

### Training export contract

`dashboard.model_loader.read_model_manifest` requires a local owner-provided JSON
manifest with these recorded fields. Do not fill them with guesses before training:

| Field | Meaning |
| --- | --- |
| `schema_version` | `1`, the supported contract version |
| `model_id`, `checkpoint` | Actual trained run and DistilBERT checkpoint identities |
| `preprocessing_id` | Explicit registered approved preprocessing profile |
| `preprocessing_fingerprint` | Hash of the exact normalization code, resource files, and settings |
| `tokenizer_path`, `encoder_path`, `classifier_path` | Saved local tokenizer/encoder directories and XGBoost JSON/UBJ file |
| `artifact_sha256` | Exact relative filename-to-SHA-256 inventory of all tokenizer, encoder and classifier files |
| `max_length` | Training sequence length, within the DistilBERT limit |
| `tokenizer_settings` | Explicit `padding` (`longest`, `max_length`, or false), boolean `truncation` and `add_special_tokens`, and saved `padding_side` / `truncation_side` |
| `encoder_inputs` | Tensors passed during training: `input_ids`, optionally with `attention_mask` |
| `encoder_dtype`, `feature_dtype` | Actual encoder precision and classifier feature precision |
| `pooling` | Recorded first-token `cls` or attention-mask `mean` extraction |
| `label_mapping` | `{"0": "Negative", "1": "Positive"}` |
| `classification` | Actual objective, prediction method, and optional confidence policy |
| `library_versions` | Exact installed versions of torch, transformers, tokenizers, and xgboost used in training |

`preprocessing_fingerprint(files, settings)` is the shared export/validation
helper. `files` maps stable logical names to the actual pipeline code and all
approved Malay-English normalization maps, emoji maps, and other resources that
the callable uses. `settings` records the exact normalization/case policies and
other preprocessing choices. At integration time, `RegisteredPreprocessor` must
supply those same logical names, file bytes, settings, and the actual callable.
The adapter verifies the fingerprint before importing optional model libraries.
It never executes a Python file supplied by a model manifest.

The saved tokenizer's padding and truncation sides must match the manifest.
Tokenization uses the explicit saved settings, with PyTorch tensors and no token
type IDs (DistilBERT does not accept them). Attention masks are available for
mask-aware mean pooling, but are passed to the encoder only when recorded in
`encoder_inputs`. Encoder precision and conversion to the recorded feature dtype
are explicit. The classifier's feature dimension, objective and class order must
match the contract. Unsupported architectures, feature transformations or pooling
rules need a deliberate adapter; they are never approximated.

### Classification and optional confidence

`classification.method` must be either:

- `predict`: use the saved classifier's `predict`, with no added threshold.
- `probability_threshold`: use the actual training decision threshold, recorded
  as `classification.decision_threshold` in (0, 1).

`classification.objective` must match the saved model. The adapter supports
`binary:logistic` and label-only `binary:hinge`. Probability thresholding and
probability confidence require `binary:logistic` and a working `predict_proba`.
`classification.confidence` must explicitly be `none` or
`predicted_class_probability`. Probabilities must be finite binary distributions;
they are not claims of held-out accuracy or probability calibration. Confidence
is omitted entirely when unsupported, not filled with zeros, stars, or scores
from another classifier.

`HybridModel.predict` returns `PredictionBatch(labels, confidence=None)` with
binary integers and optional actual selected-class probabilities. The service
validates row counts and output values, then returns public review fields plus
`predicted_label` (0/1), `predicted_sentiment`, and optional `confidence`. CSV
prediction columns supplied by a caller are discarded. A future single-review
text box can call the same service with one `review_text` row.

### Activation after training

1. Finish and approve training separately, preserving the actual preprocessing
   source, normalization files, tokenizer, trained encoder, classifier and settings.
2. Save local artifacts, hashes, preprocessing fingerprint and matching library
   versions. Encoder weights must be safetensors; pickle/joblib and remote custom
   code are unsupported. All model artifacts must stay inside the manifest directory.
3. Register the exact callable and verified files/settings in trusted application
   code. Install the training versions of optional ML packages separately.
4. Verify identical embeddings and labels on a known training-export reference
   batch before connecting `load_prediction_service` to the upload UI. None of the
   current synthetic contract tests proves parity with unavailable trained weights.
5. Add genuine prediction display/download, retaining original text and separate
   rating-derived labels. Cache resources by manifest/artifact/preprocessing
   identity if needed; do not cache private uploaded text across sessions.
6. Publish only genuine held-out evaluation files through the independent
   evaluation contract above. Uploaded inference batches are never evaluation sets.

No genuine model parity or accuracy has been tested because trained artifacts do
not exist yet. Hash and configuration validation detect mismatches; the training
owner must still establish artifact provenance and approve the inference policy.
