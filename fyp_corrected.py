# -*- coding: utf-8 -*-
"""
fyp_corrected.py

Corrections applied vs. the original fyp.py:

1. LABEL CONVENTION made explicit and VERIFIED at runtime instead of assumed.
   Your methodology chapter (3.4.1) states "negative = 0, positive = 1", but
   the dataset/CSV code encodes Negative -> 1, Positive -> 0. This script
   keeps the CSV/embeddings convention (re-extracting embeddings with a
   different encoding would be expensive) and checks, on load, that the
   majority label in y_train really is the label you expect to be Positive.
   If it doesn't match, the script stops with a clear message instead of
   silently mislabeling everything downstream. Update Chapter 3's wording
   to match this convention (or flip POSITIVE_LABEL/NEGATIVE_LABEL below).

2. The RandomizedSearchCV scorer now EXPLICITLY targets Negative-class F1
   (`pos_label=NEGATIVE_LABEL`) instead of relying on f1_score's default
   `pos_label=1`. Evidence suggested the default already happened to line
   up correctly, but making it explicit removes any doubt and survives a
   future relabeling.

3. Added an ABLATION: scale_pos_weight + XGBoost trained directly on the
   same DistilBERT embeddings, with NO SMOTE, tuned via the identical
   search procedure/grid/scorer/CV folds as the proposed hybrid. This
   isolates one specific question: is SMOTE-in-768-dim-embedding-space the
   reason the hybrid underperforms Baseline 1 on the Negative class, or is
   it something else? Comparing this ablation's test-set report against
   the SMOTE hybrid's report answers that directly.

4. Both embeddings-based models now print their cross-validated
   best_score_ (Negative-F1 on real, non-oversampled held-out folds) right
   next to their held-out test-set score, so you can see whether there's a
   generalization gap (CV score much higher than test score) or whether
   the ceiling is already low during tuning.

5. Confusion matrices now pass `labels=[POSITIVE_LABEL, NEGATIVE_LABEL]`
   explicitly, so the heatmap axes stay correctly ordered regardless of
   which numeric value you assign to which class.

6. Removed `eval_metric='aucpr'` from the hybrid's XGBClassifier — it had
   no effect without an `eval_set`/early stopping, so it wasn't doing
   anything and could give a false impression that it was.

7. Added `random_state` to both RandomizedSearchCV calls for reproducible
   hyperparameter sampling.

Everything else (data files, model choices, param grid, Baseline 2 logic)
is unchanged from your original script.
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV, train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import make_scorer, f1_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# 0. LABEL CONVENTION — single source of truth, used everywhere below.
#    Verified against the actual embeddings labels a few lines down.
# ---------------------------------------------------------------------------
POSITIVE_LABEL = 0
NEGATIVE_LABEL = 1

# ---------------------------------------------------------------------------
# 1. Load embeddings (DistilBERT feature extraction phase, done earlier)
# ---------------------------------------------------------------------------
X_train = np.load('X_train_80_embeddings.npy')
y_train = np.load('y_train_80.npy')
X_test_emb = np.load('X_test_20_embeddings.npy')
y_test_emb = np.load('y_test_20.npy')

train_counts = {int(v): int((y_train == v).sum()) for v in np.unique(y_train)}
print("y_train label counts:", train_counts)
majority_label = max(train_counts, key=train_counts.get)
minority_label = min(train_counts, key=train_counts.get)
assert majority_label == POSITIVE_LABEL and minority_label == NEGATIVE_LABEL, (
    f"Label convention mismatch! Majority label in y_train is {majority_label}, "
    f"but POSITIVE_LABEL is set to {POSITIVE_LABEL}. Flip POSITIVE_LABEL/"
    f"NEGATIVE_LABEL above to match your embeddings' actual encoding, then re-run."
)
print(f"Confirmed: label {POSITIVE_LABEL} = Positive (majority), "
      f"label {NEGATIVE_LABEL} = Negative (minority).")

# ---------------------------------------------------------------------------
# 2. Proposed Hybrid: SMOTE + XGBoost on DistilBERT embeddings
# ---------------------------------------------------------------------------
smote = SMOTE(random_state=RANDOM_STATE)
xgb_smote = xgb.XGBClassifier(objective='binary:logistic', random_state=RANDOM_STATE)

hybrid_pipeline = ImbPipeline([
    ('smote', smote),
    ('xgb', xgb_smote)
])

param_dist = {
    'xgb__max_depth': [4, 6, 8],
    'xgb__learning_rate': [0.05, 0.1, 0.2],
    'xgb__n_estimators': [100, 300],
    'xgb__colsample_bytree': [0.5, 0.7, 1.0],
    'xgb__subsample': [0.8, 1.0]
}

# Explicitly target Negative-class F1 — do not rely on f1_score's default
negative_f1_scorer = make_scorer(f1_score, pos_label=NEGATIVE_LABEL)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

hybrid_search = RandomizedSearchCV(
    hybrid_pipeline, param_dist, n_iter=15, scoring=negative_f1_scorer,
    cv=skf, n_jobs=-1, verbose=2, random_state=RANDOM_STATE
)

print("\nRunning SMOTE + XGBoost tuning (scoring = Negative-class F1)...")
hybrid_search.fit(X_train, y_train)
print(f"Best SMOTE Hybrid CV Negative-F1: {hybrid_search.best_score_:.4f}")
best_hybrid_model = hybrid_search.best_estimator_

# ---------------------------------------------------------------------------
# 3. ABLATION: scale_pos_weight + XGBoost on the SAME embeddings, no SMOTE
# ---------------------------------------------------------------------------
scale_weight_emb = train_counts[POSITIVE_LABEL] / train_counts[NEGATIVE_LABEL]
print(f"\nscale_pos_weight for embeddings ablation: {scale_weight_emb:.2f}")

xgb_weighted = xgb.XGBClassifier(
    objective='binary:logistic', random_state=RANDOM_STATE,
    scale_pos_weight=scale_weight_emb
)
weighted_pipeline = SkPipeline([('xgb', xgb_weighted)])

weighted_search = RandomizedSearchCV(
    weighted_pipeline, param_dist, n_iter=15, scoring=negative_f1_scorer,
    cv=skf, n_jobs=-1, verbose=2, random_state=RANDOM_STATE
)

print("Running scale_pos_weight + XGBoost tuning on embeddings (no SMOTE)...")
weighted_search.fit(X_train, y_train)
print(f"Best Weighted-Embedding CV Negative-F1: {weighted_search.best_score_:.4f}")
best_weighted_model = weighted_search.best_estimator_

# ---------------------------------------------------------------------------
# 4. Baseline 1: TF-IDF + XGBoost (scale_pos_weight)
# ---------------------------------------------------------------------------
df = pd.read_csv('dataset_v3_malay_clean5.csv')
X_text = df['cleaned_comment'].astype(str)
y = df['sentiment'].map({'Negative': NEGATIVE_LABEL, 'Positive': POSITIVE_LABEL}).values

X_train_text, X_test_text, y_train_tfidf, y_test_tfidf = train_test_split(
    X_text, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
)

tfidf = TfidfVectorizer(max_features=768)  # matched to DistilBERT dimensions for fair comparison
X_train_tfidf = tfidf.fit_transform(X_train_text)
X_test_tfidf = tfidf.transform(X_test_text)

scale_weight_tfidf = np.sum(y_train_tfidf == POSITIVE_LABEL) / np.sum(y_train_tfidf == NEGATIVE_LABEL)
baseline_xgb = xgb.XGBClassifier(
    scale_pos_weight=scale_weight_tfidf, max_depth=6, random_state=RANDOM_STATE
)
baseline_xgb.fit(X_train_tfidf, y_train_tfidf)
print("\nTF-IDF Baseline Trained.")

# ---------------------------------------------------------------------------
# 5. Baseline 2: Standalone multilingual transformer
# ---------------------------------------------------------------------------
from transformers import pipeline as hf_pipeline

distilbert_classifier = hf_pipeline(
    "sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment", device=0
)
print("Standalone DistilBERT Baseline Ready.")

# ---------------------------------------------------------------------------
# 6. Evaluation — all four models on their respective held-out test sets
# ---------------------------------------------------------------------------
# sklearn's classification_report assigns target_names to sorted label
# values in order, so the ordering below must match POSITIVE_LABEL's value.
target_names = ['Positive', 'Negative'] if POSITIVE_LABEL == 0 else ['Negative', 'Positive']

print("\n=== PROPOSED HYBRID (DistilBERT + SMOTE + XGBoost) ===")
hybrid_preds = best_hybrid_model.predict(X_test_emb)
print(classification_report(y_test_emb, hybrid_preds, target_names=target_names))
print(f"(CV Negative-F1 during tuning: {hybrid_search.best_score_:.4f})")

print("\n=== ABLATION (DistilBERT + scale_pos_weight + XGBoost, no SMOTE) ===")
weighted_preds = best_weighted_model.predict(X_test_emb)
print(classification_report(y_test_emb, weighted_preds, target_names=target_names))
print(f"(CV Negative-F1 during tuning: {weighted_search.best_score_:.4f})")

print("\n=== BASELINE 1 (TF-IDF + XGBoost) ===")
tfidf_preds = baseline_xgb.predict(X_test_tfidf)
print(classification_report(y_test_tfidf, tfidf_preds, target_names=target_names))

print("\n=== BASELINE 2 (Standalone Multilingual Transformer) ===")
db_preds_raw = distilbert_classifier(X_test_text.tolist(), truncation=True, max_length=128)
db_preds = [NEGATIVE_LABEL if int(p['label'].split(' ')[0]) <= 2 else POSITIVE_LABEL
            for p in db_preds_raw]
print(classification_report(y_test_tfidf, db_preds, target_names=target_names))

# ---------------------------------------------------------------------------
# 7. Confusion matrices — hybrid vs. ablation, side by side
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for ax, preds, title in [
    (axes[0], hybrid_preds, 'Hybrid (SMOTE) Confusion Matrix'),
    (axes[1], weighted_preds, 'Ablation (scale_pos_weight) Confusion Matrix'),
]:
    cm = confusion_matrix(y_test_emb, preds, labels=[POSITIVE_LABEL, NEGATIVE_LABEL])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Positive', 'Negative'], yticklabels=['Positive', 'Negative'], ax=ax)
    ax.set_ylabel('Actual')
    ax.set_xlabel('Predicted')
    ax.set_title(title)
plt.tight_layout()
plt.show()
