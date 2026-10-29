import numpy as np
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import make_scorer, f1_score

# 1. Load the Training Data
X_train = np.load('X_train_80_embeddings.npy')
y_train = np.load('y_train_80.npy')

# 2. Calculate Imbalance Penalty
negative_count = np.sum(y_train == 1)
positive_count = np.sum(y_train == 0)
scale_weight = positive_count / negative_count

# 3. Initialize the Base Classifier
xgb_model = xgb.XGBClassifier(
    objective='binary:logistic',
    scale_pos_weight=scale_weight,
    eval_metric='aucpr',
    random_state=42
)

# 4. Define the Tuning Grid specifically for High-Dimensional, Imbalanced Data
param_dist = {
    'max_depth': [4, 6, 8, 10],               # How deep the trees can grow
    'min_child_weight': [1, 3, 5, 7],         # Higher values prevent overfitting the 1.6% minority class
    'learning_rate': [0.01, 0.05, 0.1, 0.2],  # Step size during gradient descent
    'n_estimators': [100, 300, 500],          # Total number of trees
    'colsample_bytree': [0.3, 0.5, 0.7],      # Fraction of the 768 features used per tree (forces variety)
    'subsample': [0.6, 0.8, 1.0]              # Fraction of rows used per tree
}

# 5. Configure the Randomized Search
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
f1_scorer = make_scorer(f1_score)

random_search = RandomizedSearchCV(
    estimator=xgb_model,
    param_distributions=param_dist,
    n_iter=20,          # Tests 20 random combinations (increase to 50 if you have time/GPU)
    scoring=f1_scorer,  # Optimizes strictly for the F1-Score
    cv=skf,
    verbose=2,          # Prints progress updates
    random_state=42,
    n_jobs=-1           # Uses all available CPU cores
)

print("Commencing Hyperparameter Tuning across 5-Folds...")
random_search.fit(X_train, y_train)

# 6. Output the Optimal Configuration
print("\n--- Tuning Complete ---")
print(f"Best CV F1-Score: {random_search.best_score_:.4f}")
print("Optimal Hyperparameters:")
for param, value in random_search.best_params_.items():
    print(f" - {param}: {value}")