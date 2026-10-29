import numpy as np
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import make_scorer, f1_score, precision_score, recall_score, average_precision_score

# 1. Load the extracted DistilBERT embeddings and target labels
X_train = np.load('X_train_80_embeddings.npy')
y_train = np.load('y_train_80.npy')

# 2. Dynamically calculate scale_pos_weight for the XGBoost loss function
# (Assuming Negative sentiment is mapped to 1, Positive to 0)
negative_count = np.sum(y_train == 1)
positive_count = np.sum(y_train == 0)
scale_weight = positive_count / negative_count
print(f"Class Imbalance Penalty (scale_pos_weight): {scale_weight:.2f}")

# 3. Initialize the XGBoost Classifier
xgb_model = xgb.XGBClassifier(
    objective='binary:logistic',
    scale_pos_weight=scale_weight,
    eval_metric='aucpr',
    random_state=42,
    max_depth=7,              # Increased depth to navigate 768 dimensions
    colsample_bytree=0.4,     # Forces trees to evaluate different subsets of features
    subsample=0.8,            # Randomly samples rows to prevent overfitting
    learning_rate=0.05,       # Slower, more calculated gradient descent
    n_estimators=300          # More trees to build a stronger consensus
)

# 4. Configure the 5-Fold Stratified Cross-Validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# 5. Define strict evaluation metrics isolated to the minority class
scoring = {
    'f1': make_scorer(f1_score),
    'pr_auc': make_scorer(average_precision_score),
    'recall': make_scorer(recall_score)
}

print("Executing 5-Fold Cross Validation...")
cv_results = cross_validate(xgb_model, X_train, y_train, cv=skf, scoring=scoring, n_jobs=-1)

# 6. Evaluate the stability of the training phase
print("\n--- Cross-Validation Results ---")
print(f"Mean CV PR-AUC:   {np.mean(cv_results['test_pr_auc']):.4f}")
print(f"Mean CV F1-Score: {np.mean(cv_results['test_f1']):.4f}")
print(f"Mean CV Recall:   {np.mean(cv_results['test_recall']):.4f}")