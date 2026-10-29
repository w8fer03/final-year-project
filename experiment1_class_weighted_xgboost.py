"""Experiment 1 of 5: class weighting for Shopee Malay/English/Manglish sentiment.

This is a controlled cross-validation experiment. The final test partition is
created, saved, and NEVER USED for selecting weights or thresholds.

Labels: 0 = Negative (minority), 1 = Positive (majority).

Examples:
  python experiment1_class_weighted_xgboost.py \
      --input supervisor_demo_thesis_columns.csv --features tfidf --test-size 0.20

  python experiment1_class_weighted_xgboost.py \
      --input supervisor_demo_thesis_columns.csv --features distilbert --test-size 0.20 \
      --batch-size 16 --max-length 256

For multilingual DistilBERT mode, first install:
  pip install torch transformers scikit-learn xgboost numpy scipy

The DistilBERT feature mode follows the proposal: distilbert-base-multilingual-cased
and a 768-dimensional CLS-token feature vector. This uses frozen pretrained
DistilBERT, not fine-tuned DistilBERT. TF-IDF mode is for the thesis baseline,
not a substitute for the proposed final hybrid model.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from xgboost import XGBClassifier

SEED = 42
MODEL_NAME = "distilbert-base-multilingual-cased"
THRESHOLDS = [round(x / 100, 2) for x in range(5, 96, 5)]
CONFIGS = {"unweighted": 0.0, "moderate_sqrt": 0.5, "full_ratio": 1.0}


def load_data(path: Path):
    texts, labels, source_rows = [], [], []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        missing = {"cleaned_comment", "sentiment"} - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing required columns: {sorted(missing)}")
        for src_row, entry in enumerate(reader, start=2):
            text = (entry.get("cleaned_comment") or "").strip()
            lab = (entry.get("sentiment") or "").strip()
            if text and lab in ("0", "1"):
                texts.append(text)
                labels.append(int(lab))
                source_rows.append(src_row)
    if not texts:
        raise ValueError("No valid reviews found")
    return np.asarray(texts, dtype=object), np.asarray(labels, dtype=np.int8), np.asarray(source_rows)


def save_csv(path: Path, data):
    if not data:
        return
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(data[0].keys()))
        writer.writeheader()
        writer.writerows(data)


def make_classifier(n_estimators: int):
    return XGBClassifier(
        objective="binary:logistic", eval_metric="logloss", tree_method="hist",
        n_estimators=n_estimators, max_depth=3, learning_rate=0.1,
        subsample=0.85, colsample_bytree=0.85, reg_lambda=1.0,
        n_jobs=4, random_state=SEED,
    )


def neg_probs(model, features):
    classes = list(model.classes_)
    return model.predict_proba(features)[:, classes.index(0)].astype(np.float64)


def scores(y_true, prob_neg, threshold):
    y_pred = np.where(prob_neg >= threshold, 0, 1)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    return {
        "threshold_for_P_negative": f"{threshold:.2f}",
        "n_actual_negative": int((y_true == 0).sum()),
        "n_actual_positive": int((y_true == 1).sum()),
        "n_pred_negative": int((y_pred == 0).sum()),
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 5),
        "macro_f1": round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 5),
        "negative_precision": round(float(precision_score(y_true, y_pred, pos_label=0, zero_division=0)), 5),
        "negative_recall": round(float(recall_score(y_true, y_pred, pos_label=0, zero_division=0)), 5),
        "negative_f1": round(float(f1_score(y_true, y_pred, pos_label=0, zero_division=0)), 5),
        "true_negative_reviews_detected": int(cm[0, 0]),
        "negative_reviews_missed": int(cm[0, 1]),
        "positive_reviews_flagged_negative": int(cm[1, 0]),
    }


def distilbert_embeddings(texts, cache_dir: Path, input_fingerprint: str,
                          batch_size: int, max_length: int):
    try:
        import torch
        from transformers import AutoTokenizer, AutoModel
    except ImportError as exc:
        raise SystemExit(
            "DistilBERT mode needs torch and transformers. Install them in your "
            "Colab/VS Code environment: pip install torch transformers"
        ) from exc
    cache_dir.mkdir(parents=True, exist_ok=True)
    fp = hashlib.sha256((input_fingerprint + MODEL_NAME + str(max_length)).encode()).hexdigest()[:16]
    path = cache_dir / f"distilbert_cls_{fp}.npy"
    if path.exists():
        embs = np.load(path)
        if embs.shape == (len(texts), 768):
            print("Reusing frozen multilingual DistilBERT CLS embeddings:", path, flush=True)
            return embs
    print("Loading multilingual DistilBERT (first use downloads pretrained model)...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    output = np.zeros((len(texts), 768), dtype=np.float32)
    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start:start + batch_size].tolist()
        inputs = tokenizer(batch_texts, padding=True, truncation=True,
                           max_length=max_length, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.inference_mode():
            # Proposal uses CLS token (position 0), not mean pooling.
            hidden = model(**inputs).last_hidden_state[:, 0, :]
        output[start:start + len(batch_texts)] = hidden.cpu().numpy().astype(np.float32)
        if start == 0 or (start // batch_size) % 50 == 0:
            print(f"Embedded {min(start + batch_size, len(texts)):,}/{len(texts):,} ({device})", flush=True)
    np.save(path, output)
    return output


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", default="supervisor_demo_thesis_columns.csv")
    ap.add_argument("--features", choices=["tfidf", "distilbert"], default="tfidf")
    ap.add_argument("--test-size", type=float, default=0.20, choices=[0.20, 0.30])
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--n-estimators", type=int, default=130)
    ap.add_argument("--max-features", type=int, default=5000)
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--output-dir", default=".")
    args = ap.parse_args()
    if args.folds < 3:
        ap.error("Use at least 3 folds")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    input_path = Path(args.input)
    input_fingerprint = hashlib.sha256(input_path.read_bytes()).hexdigest()
    texts, labels, source_rows = load_data(input_path)
    ind = np.arange(len(labels), dtype=np.int32)
    training_indices, test_indices = train_test_split(
        ind, test_size=args.test_size, stratify=labels, random_state=SEED
    )
    split_key = "80_20" if args.test_size == 0.20 else "70_30"
    prefix = f"experiment1_{args.features}_{split_key}"
    np.savez_compressed(output_dir / f"{prefix}_locked_split.npz",
                        train_indices=training_indices, test_indices=test_indices,
                        source_rows=source_rows, input_sha256=np.asarray(input_fingerprint))
    ntrain0 = int((labels[training_indices] == 0).sum())
    ntrain1 = int((labels[training_indices] == 1).sum())
    ratio = ntrain1 / ntrain0
    print(f"Source: {input_path.name}; SHA256={input_fingerprint[:16]}", flush=True)
    print(f"Total={len(labels):,} negative={(labels==0).sum():,} positive={(labels==1).sum():,}", flush=True)
    print(f"Training={len(training_indices):,} ({ntrain0:,} negative, {ntrain1:,} positive)", flush=True)
    print(f"LOCKED test={len(test_indices):,}; results are NOT viewed in this experiment.", flush=True)
    print(f"Weight ratio={ratio:.4f}; moderate={math.sqrt(ratio):.4f}; full={ratio:.4f}", flush=True)

    if args.features == "distilbert":
        embeddings = distilbert_embeddings(
            texts, output_dir / "embedding_cache", input_fingerprint,
            args.batch_size, args.max_length
        )
    else:
        embeddings = None

    cv = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=SEED)
    # Out-of-fold probability for EVERY training review and EVERY class-weight strategy.
    # These validation predictions are suitable for selecting the threshold and weight.
    oof = {key: np.full(len(training_indices), np.nan) for key in CONFIGS}
    fold_rows = []
    for k, (fold_train_pos, fold_valid_pos) in enumerate(cv.split(training_indices, labels[training_indices]), 1):
        fold_train = training_indices[fold_train_pos]
        fold_valid = training_indices[fold_valid_pos]
        train_y = labels[fold_train]
        val_y = labels[fold_valid]
        f_neg = int((train_y == 0).sum())
        f_pos = int((train_y == 1).sum())
        f_ratio = f_pos / f_neg
        if args.features == "tfidf":
            vectorizer = TfidfVectorizer(max_features=args.max_features, min_df=2,
                                         ngram_range=(1, 2), sublinear_tf=True)
            X_train = vectorizer.fit_transform(texts[fold_train].tolist())
            X_valid = vectorizer.transform(texts[fold_valid].tolist())
        else:
            X_train, X_valid = embeddings[fold_train], embeddings[fold_valid]
        print(f"Fold {k}/{args.folds}: {len(train_y):,} training, {len(val_y):,} validation; imbalance {f_ratio:.2f}:1", flush=True)
        for name, exponent in CONFIGS.items():
            neg_weight = f_ratio ** exponent
            sample_weights = np.where(train_y == 0, neg_weight, 1.0).astype(np.float32)
            model = make_classifier(args.n_estimators)
            model.fit(X_train, train_y, sample_weight=sample_weights)
            oof[name][fold_valid_pos] = neg_probs(model, X_valid)
            fold_rows.append({"fold": k, "configuration": name,
                              "train_negative": f_neg, "train_positive": f_pos,
                              "val_negative": int((val_y == 0).sum()),
                              "val_positive": int((val_y == 1).sum()),
                              "negative_sample_weight": round(float(neg_weight), 5),
                              "positive_sample_weight": 1.0})
        del X_train, X_valid

    ycv = labels[training_indices]
    threshold_rows = []
    for name, pneg in oof.items():
        if not np.all(np.isfinite(pneg)):
            raise RuntimeError("Some OOF predictions were not generated")
        for threshold in THRESHOLDS:
            threshold_rows.append({"configuration": name, **scores(ycv, pneg, threshold)})
    summary_rows = []
    for name in CONFIGS:
        group = [r for r in threshold_rows if r["configuration"] == name]
        # Select threshold only via CV OOF, never via held-out test.
        best = max(group, key=lambda r: (r["macro_f1"], r["negative_f1"], r["negative_recall"]))
        for label, selected in [("default_0.50", next(r for r in group if r["threshold_for_P_negative"] == "0.50")),
                                ("candidate_0.20", next(r for r in group if r["threshold_for_P_negative"] == "0.20")),
                                ("cv_selected", best)]:
            summary_rows.append({"configuration": name, "threshold_scenario": label,
                                 "negative_training_weight": round(float(ratio ** CONFIGS[name]), 5), **selected})

    save_csv(output_dir / f"{prefix}_cv_summary.csv", summary_rows)
    save_csv(output_dir / f"{prefix}_threshold_sweep.csv", threshold_rows)
    save_csv(output_dir / f"{prefix}_fold_details.csv", fold_rows)
    predictions = []
    for pos, idx in enumerate(training_indices):
        predictions.append({"dataset_index_zero_based": int(idx),
                            "source_csv_row": int(source_rows[idx]),
                            "actual_sentiment": int(labels[idx]),
                            **{f"P_negative_{name}": round(float(oof[name][pos]), 7) for name in CONFIGS}})
    save_csv(output_dir / f"{prefix}_cv_predictions.csv", predictions)
    best_cv = [row for row in summary_rows if row["threshold_scenario"] == "cv_selected"]
    winner = max(best_cv, key=lambda r: (r["macro_f1"], r["negative_f1"], r["negative_recall"]))
    summary = {
        "experiment": "Experiment 1: unweighted vs moderately and fully class-weighted XGBoost",
        "feature_type": args.features,
        "model_name": MODEL_NAME if args.features == "distilbert" else "TF-IDF baseline, not DistilBERT",
        "input_path": str(input_path), "input_sha256": input_fingerprint,
        "sentiment_mapping": {"0": "negative", "1": "positive"},
        "random_state": SEED, "test_size": args.test_size, "cv_folds": args.folds,
        "total_count": len(labels), "total_negative": int((labels == 0).sum()),
        "total_positive": int((labels == 1).sum()),
        "train_count": len(training_indices), "train_negative": ntrain0,
        "train_positive": ntrain1, "locked_test_count": len(test_indices),
        "test_never_used_for_selection": True,
        "threshold_probability_class": "negative (label 0)",
        "cv_best_by_macro_f1": winner,
        "warning": "Class-weight comparisons are CV estimates; holdout not used. TF-IDF mode is only a baseline."
    }
    with (output_dir / f"{prefix}_experiment_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print("\nOut-of-fold stratified CV comparison (same folds for all models):", flush=True)
    for row in summary_rows:
        print(f"  {row['configuration']:>13} {row['threshold_scenario']:>14} "
              f"t={row['threshold_for_P_negative']} macroF1={row['macro_f1']:.4f} "
              f"negF1={row['negative_f1']:.4f} "
              f"negRecall={row['negative_recall']:.4f} negPrecision={row['negative_precision']:.4f}", flush=True)
    print(f"\nCV-selected configuration: {winner['configuration']} "
          f"threshold={winner['threshold_for_P_negative']} "
          f"macroF1={winner['macro_f1']:.4f}", flush=True)
    print("Final test stayed LOCKED. Save/retain locked_split.npz for all five experiments.", flush=True)
    print("Output prefix:", output_dir / prefix, flush=True)


if __name__ == "__main__":
    main()
