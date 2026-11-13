# Final Version B model integration

The existing SentiLokal interface now imports the genuine eight-configuration evaluation exports and supports saved-model inference on Upload & Analyze. The yellow/Poppins design, radio section shortcuts, five-product mapping, research charts, and original review display are preserved.

**Validation is partial:** the actual supplied TF-IDF + XGBoost model loads and runs new text. The code-only kit supplies the reference implementation and saved tokenizers; these are now verified. The other three architecture interfaces have encoder tests with explicit doubles because their actual transformer weights are missing. This is not a claim that all four saved models have been validated.

## Local setup

Use Python 3.10 or newer, from the existing branch `codex/shopee-sentiment-dashboard`:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/check_version_b.py --research --check-preprocessing --smoke-tfidf
streamlit run app.py
```

On Windows use `.venv\Scripts\activate`. For a CPU-only PyTorch installation, use the official PyTorch installation instructions for your host before installing the requirements. No GPU is required by these adapters.

`requirements.txt` pins XGBoost 3.4.1 (the supplied JSON save version), scikit-learn 1.6.1 (the saved vectorizer metadata), and regex 2026.5.9 (the recorded preprocessing version). Actual TF-IDF inference was verified with those versions and joblib 1.5.3 on Python 3.12. Transformers is pinned to 4.57.6, as recorded in the supplied checkpoint config, and both saved tokenizers were exercised with tokenizers 0.22.2. PyTorch weights were not available and full transformer inference was not validated. Do not claim encoder compatibility from tokenizer checks or adapter doubles.

Provision owner-approved files under ignored `models/version_b/`, or set `SENTILOKAL_MODEL_DIR` to an approved local directory. Read-only mounts are supported. Files present in this workspace are not transferred by a Git push.

```text
models/version_b/
  xgb_tf_idf.json
  tfidf_vectorizer.joblib
  xgb_frozen_hybrid.json
  xgb_finetuned_hybrid.json
  version_B_model_ready.csv
  version_B_dashboard_metadata.csv
  cldr/
    ms.xml
    ms_derived.xml
  # Still required for transformer inference:
  frozen_encoder_manifest.json       # original frozen revision and checksums
  saved_tokenizer/                    # tokenizer files supplied in the code-only kit
  frozen_distilbert/                  # original frozen encoder weights and config
  fine_tuned_distilbert_80_20_full_training_only/
    config.json
    model.safetensors
    tokenizer_config.json
    tokenizer.json or vocab.txt       # plus all other saved tokenizer files
```

Weights, vectorizers, the new private CSVs, and local settings stay ignored by Git. Joblib deserialization is restricted to the owner-supplied vectorizer after its exact deployment SHA-256 is verified. The app does not accept user-uploaded models, execute remote model code, or download Hugging Face weights. Transformer loads use local files, safetensors, and `trust_remote_code=False`.

## Reference pipeline and remaining transformer requirements

The new `VersionB_Codex_IntegrationKit_CODE_ONLY.zip` supplies the actual `dashboard_app.py`, saved tokenizer files, checkpoint config, and identical deployment/preprocessing/metric exports. Their bytes were checked against the previously supplied files. The reference is retained as text in `resources/version_b/reference_dashboard_app.py.txt`; the application does not import or replace its UI. The derived `inference_profile.json` pins the reference and every supplied tokenizer/config file by SHA-256.

The reference establishes dynamic padding (`padding=True`), truncation at 256, fast tokenizers with right padding/truncation, CPU batches of four, float32 features, and **CLS extraction**. The frozen architecture uses `AutoModel.last_hidden_state[:,0,:]`. Both fine-tuned architectures load `AutoModelForSequenceClassification`; the hybrid uses `hidden_states[-1][:,0,:]`, while the standalone model uses softmax over the two logits. The adapter follows those paths and uses inference mode/eval on CPU. No pooling or tokenizer settings remain guessed.

The saved frozen tokenizer is a `BertTokenizerFast` and emits `token_type_ids` by default. Passing all its tensors directly to DistilBERT, as the reference does, is incompatible with DistilBERT's forward signature. The adapter deliberately requests only `input_ids` and `attention_mask`; tests verify that their token IDs/masks exactly match reference tokenization. The fine-tuned tokenizer already uses the DistilBERT input schema. An unrelated Mistral-regex heuristic warning can appear in Transformers 4.57.6; the supplied WordPiece tokenizer is retained unchanged, without an unrecorded regex correction.

Still needed for actual transformer inference:

1. `fine_tuned_distilbert_80_20_full_training_only/model.safetensors`, **541,317,368 bytes**, expected SHA-256 `6c97a6084994017eccfc96a2438a1c5fd29f87677c9c52be8178f59c6bfae8a1`. Its tokenizer/config files are already provisioned locally from the code-only kit.
2. The original frozen `distilbert/distilbert-base-multilingual-cased` encoder's `config.json` and `model.safetensors`, under `frozen_distilbert/`. The reference downloads a floating model ID and does not record the training revision. To avoid a silent replacement, provision the actual original encoder and record its revision/checksums in `frozen_encoder_manifest.json`.

The frozen manifest must contain `pretrained_model_id` equal to the deployment config, the owner-approved original `revision`, and `artifact_sha256` entries for `config.json` and `model.safetensors`. No invented revision or checksum is supplied. This is an explicit local provenance control in place of the reference's runtime network download.

```bash
python scripts/check_version_b.py --require-all
```

This checks file/config availability, not actual four-model inference parity. It currently exits nonzero because the transformer weights are absent. Research and rating-based upload exploration remain usable. After provisioning, verify cleaned text, tokens, embeddings/logits, scores and labels against a non-held-out batch from the original pipeline. Do not run held-out datasets again.

## Decision semantics and uploaded reviews

`0 = Negative`, `1 = Positive`. The backend verifies class ordering and extracts **P(Negative)**. A score greater than or equal to the architecture's full-precision threshold produces class 0; otherwise class 1. Thresholds come from the supplied deployment configuration, not held-out result selection:

| Architecture | Negative-score threshold |
| --- | --- |
| Frozen hybrid | `0.07502188896358628` |
| TF-IDF + XGBoost | `0.28` |
| Standalone fine-tuned DistilBERT | `0.060000000000000005` |
| Fine-tuned hybrid | `0.26` |

The configured default remains `finetuned_hybrid`, even while its artifacts are unavailable. The app never substitutes another model automatically. Select TF-IDF explicitly to use the model currently available locally. The default and thresholds are owner-recorded training-only CV choices; held-out scores do not tune them. Scores retain their model's native precision during the NumPy comparison, including float32 boundary behavior, to match the supplied reference exactly.

Select a model, upload CSV, and click **Analyze Reviews**. A manual input is under **Analyze one review**. Only original text enters the model. Ratings, existing labels, source IDs, and uploaded prediction columns are not features. Every original row, including duplicates and unrated rows, is processed in order, before display filters. Empty-after-preprocessing reviews reject the batch with an explicit error; no rows disappear silently.

Outputs add `predicted_label`, `predicted_sentiment`, `negative_score`, `prediction_model`, and `negative_threshold`. Rating-derived `sentiment` remains separate and blank without eligible ratings. Negative scores are uncalibrated, never presented as confidence percentages. Product/date/rating-label selections control the predicted summary; the explorer additionally supports text/star/prediction filters and exports all matching rows.

Uploaded content/results stay in session memory, never in a global text cache or project files. A file/model/artifact identity change invalidates old outputs. **Clear Uploaded Dataset** removes batch/manual results and controls. Only model resources are shared through `st.cache_resource`, keyed by artifact/resource identity.

## Exact Version B preprocessing

The original repository preprocessing scripts are unchanged. A byte-identical copy of the supplied `preprocessv3_cldr_ab.py` is vendored under `resources/version_b/`, alongside the exact normalization map, corpus emoji subset, CLDR manifest, and deployment config. Resource hashes are checked before import. Its guarded CLI is not run.

Inference calls the exact Version B function with case preserved and the same Version B normalization exclusions (`padu`, `mantap`, `hauk`, `baguss`), using the recorded regex grapheme backend. It performs no new eligibility filtering, balancing, deduplication, fitting, or training.

The observed TSV covers 341 canonical emoji sequences, not all future user text. For new corpora provision the two full official Malay CLDR XML files in `cldr/`. Their Git blob SHA-1 pins must match the supplied manifest:

- `ms.xml`: `5d04da416ab5e3e05a2262b5b2aff1bb773bcc90`
- `ms_derived.xml`: `dbf09d6c11b2904f7cfc97050ea0298210e917f5`

The files in this workspace were downloaded from Unicode's official repository and checked against these pins. A fresh host needs the same files. Fetch the pinned Git blobs through GitHub's `unicode-org/cldr` Git database API, decode the returned base64 content, verify the Git blob hash (`SHA1(b"blob " + byte_length + NUL + content)`), and install them under the names above. Do not rely on an unverified future `main` download. Without full files the app rejects an emoji outside the verified corpus subset rather than introducing an unrecorded name. There are no runtime network downloads.

Actual preprocessing parity on all 18,705 supplied original comments produced **zero differences** from `cleaned_comment`, both with the observed subset and the pinned full XML. This check does not classify or re-evaluate a held-out set.

## Research source and saved predictions

The legacy 19,295-review source/checksum remains the default. The final approved Version B source contains 18,705 rows (18,423 positive rating labels, 282 negative), with all 17 listing keys mapping to the existing five products. Its separate date metadata joins one-to-one on `source_row_id`. Namespace `version_b:` prevents collisions with legacy review keys. Neither review text nor row position is used for joins.

Switch explicitly using [deployment.md](deployment.md#version-b-final-research-source). Do not rename the Version B file to the legacy filename or alter the legacy checksum. Source hashes, exact row count, unique IDs, metadata ID-set equality, labels, and known product mapping are checked. Source-row IDs, listing/shop IDs, and cleaned text stay outside public tables/exports.

No whole-research prediction file was supplied or manufactured. Optional owner-generated `research_predictions.csv` in the private model directory requires `source_row_id` and genuine `predicted_sentiment` (`0`/`1` or Negative/Positive). A companion `research_predictions_manifest.json` must contain:

- `training_source_sha256`: the approved Version B source hash.
- `predictions_sha256`: the actual prediction file hash.
- `architecture`: one of the four deployment keys.
- `negative_threshold`: that architecture's exact deployment threshold.

IDs must be unique, nonempty, and a subset of the verified Version B source. Partial coverage remains partial. Legacy `results/predictions.csv` never crosses into the Version B namespace. The legacy source retains its original ID-based prediction contract: owners may obtain canonical `review_id` values locally with `dashboard.data.load_dataset(...).reviews`; customer exports intentionally omit them.

## Recorded held-out results

The exact supplied `day3_all_models_test_metrics.csv` and `day3_all_models_confusion.csv` are committed under `evaluation/version_b/`. They contain aggregate results, not private review rows or weights. `SENTILOKAL_EVALUATION_DIR` optionally selects a server directory containing the same two filenames.

The loader requires eight unique architecture/split pairs, finite metrics, integer counts, matching confusion/support totals, and aggregate arithmetic consistency. It **displays the recorded values**, including PR-AUC; it does not run held-out inference or reconstruct PR-AUC from a confusion matrix. Comparison tables/charts stay within 80:20 (3,762 tests, 61 actual negatives) or 70:30 (5,625 tests, 71 actual negatives). Per-review test membership was not supplied; exact membership is not inferred from counts. Provenance and the owner's held-out declaration remain separate from numerical consistency checks.

Macro F1 and minority negative F1 lead the metric cards, followed by positive/weighted F1, accuracy, negative precision/recall, and negative PR-AUC. A selected recorded confusion matrix and original aggregate CSV downloads are provided. No test-score winner is selected for deployment.

The older `dashboard/inference.py`, `dashboard/model_loader.py`, and `results/` API remain for compatibility, separate from this active Version B path. If the final aggregate files are absent, legacy recorded evaluation JSON is still supported. The current Version B path never uses the legacy option that recalculates metrics from test prediction rows.

## Verification and changed files

Run `python -m unittest discover -s tests -v`. Tests cover the eight genuine exports, malformed/missing results, safe research joins/provenance, exact cleaned-text parity, real saved-tokenizer parity (including 256-token truncation), threshold direction/ties/class ordering, all four adapter interfaces against the reviewed reference functions with explicit doubles, row/duplicate preservation, missing artifacts, resource caching, upload actions/filter/export, navigation, and existing research behavior. Real TF-IDF tests are separately named and skipped on hosts without private artifacts. Transformer doubles are not evidence of actual saved-model validation.

| Modified or added file | Purpose |
| --- | --- |
| `app.py` | Final evaluation rendering and explicit research profile; existing page layout retained |
| `dashboard/upload_page.py` | Connect existing upload exploration to explicit inference, filters/export, session clearing |
| `dashboard/prediction_controls.py` | Model availability/selection, batch/manual actions, prediction totals/distribution |
| `dashboard/prediction_backend.py` | Four architecture adapters, class-zero thresholds, verified local loading and resource cache |
| `dashboard/version_b.py` | Shared artifact locations, config semantics and checksum/path helpers |
| `dashboard/version_b_preprocessing.py` | Exact verified Version B preprocessing adapter and pinned CLDR handling |
| `dashboard/version_b_research.py` | Explicit source/metadata migration and provenance-bound saved prediction joins |
| `dashboard/final_evaluations.py` | Eight-configuration aggregate loading/validation, without test inference |
| `dashboard/artifacts.py` | Preserve the legacy API while allowing final-report deployments to skip legacy metric recalculation |
| `dashboard/performance_page.py` | Existing performance section extended with final split comparisons and metrics |
| `dashboard/inference.py` | Update obsolete pre-training message in the retained legacy service |
| `requirements.txt` | Inference dependencies; tested serialization/preprocessing pins |
| `.gitignore` | Allow only small public manifests/resources and two aggregate result CSVs; new data/weights ignored |
| `resources/version_b/deployment_config.json` | Unchanged supplied model configuration, thresholds and artifact hashes |
| `resources/version_b/preprocessv3_cldr_ab.py` | Unchanged supplied preprocessing source; existing scripts untouched |
| `resources/version_b/malay_normalization_map_v3.json` | Unchanged supplied training normalization map |
| `resources/version_b/CLDR_reference_manifest.json` | Unchanged supplied preprocessing provenance and CLDR pins |
| `resources/version_b/cldr_ms_observed_tts.tsv` | Unchanged supplied corpus emoji-name subset |
| `resources/version_b/inference_profile.json` | Reference-derived inference settings and saved tokenizer/config checksums |
| `resources/version_b/reference_dashboard_app.py.txt` | Unchanged supplied reference retained as text; its UI is not imported |
| `resources/version_b/research_manifest.json` | Approved Version B dataset/metadata hashes, row count, key namespace and eligibility |
| `evaluation/version_b/day3_all_models_test_metrics.csv` | Unchanged genuine eight-record metric export |
| `evaluation/version_b/day3_all_models_confusion.csv` | Unchanged genuine eight-record confusion export |
| `scripts/check_version_b.py` | Read-only availability, source/parity checks and optional new-text TF-IDF smoke |
| `tests/test_version_b.py` | Genuine aggregate, exact preprocessing and source/provenance tests |
| `tests/test_prediction_backend.py` | Reference adapter doubles plus separately named real TF-IDF/tokenizer checks |
| `tests/test_final_app.py` | Final evaluation UI, upload prediction/filter/export and explicit Version B profile checks |
| `tests/test_app.py`, `tests/test_upload_page.py`, `tests/test_deployment.py` | Preserve legacy regression isolation and update outdated pre-training expectations |
| `README.md`, `docs/dashboard.md`, `docs/deployment.md`, `docs/model_integration.md` | Current behavior, setup, source migration, limits and this file inventory |

The existing original datasets and preprocessing scripts are unchanged. No training, class balancing, threshold tuning, or held-out classifier evaluation was run.
