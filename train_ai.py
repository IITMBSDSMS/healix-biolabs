"""
train_ai.py
-----------
Trains breast cancer classification model on real data.

Priority:
  1. cleaned_genomic_data.csv  (output from clean_dataset.py)
  2. sklearn Wisconsin Breast Cancer dataset (real labeled fallback)

Saves:
  breast_model.pkl   - trained CalibratedClassifierCV(RandomForestClassifier)
  scaler.pkl         - fitted StandardScaler (for inference)
  model_meta.json    - feature count, columns, accuracy
"""

import os
import json
import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, RandomizedSearchCV
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, classification_report, brier_score_loss

TARGET_FEATURES = 20
CLEANED_DATA_PATH = "cleaned_genomic_data.csv"
MODEL_PATH = "breast_model.pkl"
SCALER_PATH = "scaler.pkl"
META_PATH = "model_meta.json"

# ─── 1. Load Dataset ──────────────────────────────────────────────────────────

def load_real_dataset():
    """Load cleaned_genomic_data.csv if available, else use sklearn breast cancer dataset."""
    if os.path.exists(CLEANED_DATA_PATH):
        print(f"✓ Found {CLEANED_DATA_PATH} — loading your real genomic data...")
        df = pd.read_csv(CLEANED_DATA_PATH)

        # Detect label column: a column named 'label', 'target', 'diagnosis', 'class', or 'y'
        label_candidates = [c for c in df.columns if c.strip().lower() in
                            ("label", "target", "diagnosis", "class", "y", "status", "group")]

        if label_candidates:
            label_col = label_candidates[0]
            print(f"  → Using '{label_col}' as label column")
            y = df[label_col]
            X = df.drop(columns=[label_col])
        else:
            # No obvious label found — use last column as target
            print("  → No label column found. Using last column as target.")
            y = df.iloc[:, -1]
            X = df.iloc[:, :-1]

        # Ensure y is binary (0/1) — map strings if needed
        if y.dtype == object:
            uniques = y.unique()
            mapping = {val: i for i, val in enumerate(sorted(uniques))}
            y = y.map(mapping)
            print(f"  → Mapped labels: {mapping}")

        y = pd.to_numeric(y, errors="coerce").fillna(0).astype(int)

        # Keep only numeric feature columns
        X = X.select_dtypes(include=[np.number])
        X = X.fillna(X.mean()).fillna(0)

        source = "cleaned_genomic_data.csv"

    else:
        print("⚠  cleaned_genomic_data.csv not found.")
        print("   Using sklearn Wisconsin Breast Cancer dataset (569 real samples, 30 features).")
        from sklearn.datasets import load_breast_cancer
        data = load_breast_cancer()
        X = pd.DataFrame(data.data, columns=data.feature_names)
        y = pd.Series(data.target)
        source = "sklearn_breast_cancer"

    return X, y, source


# ─── 2. Feature Engineering ───────────────────────────────────────────────────

def prepare_features(X, target_n=TARGET_FEATURES):
    """Ensure exactly target_n numeric features via duplication or slicing."""
    X = X.reset_index(drop=True)
    n_cols = X.shape[1]

    if n_cols == 0:
        raise ValueError("Dataset has zero usable numeric features after cleaning.")

    if n_cols < target_n:
        print(f"  → Expanding {n_cols} → {target_n} features (duplicating real columns)")
        original_cols = list(X.columns)
        i = 0
        while X.shape[1] < target_n:
            src = original_cols[i % len(original_cols)]
            X[f"dup_{i}"] = X[src]
            i += 1
    else:
        X = X.iloc[:, :target_n]
        print(f"  → Using first {target_n} of {n_cols} features")

    return X


# ─── 3. Scale ─────────────────────────────────────────────────────────────────

def scale_features(X_train, X_test):
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)
    return X_train_s, X_test_s, scaler


# ─── 4. Train ─────────────────────────────────────────────────────────────────

def train_model(X_train, y_train):
    print("\nOptimizing RandomForestClassifier...")
    
    # Define hyperparameter search space
    param_dist = {
        'n_estimators': [100, 200, 300, 500],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2', None]
    }
    
    rf = RandomForestClassifier(class_weight="balanced", random_state=42, n_jobs=-1)
    
    # Randomized Search for optimization
    search = RandomizedSearchCV(
        rf, param_distributions=param_dist, 
        n_iter=10, cv=3, scoring='accuracy', 
        random_state=42, n_jobs=-1
    )
    search.fit(X_train, y_train)
    
    best_rf = search.best_estimator_
    print(f"  → Best params: {search.best_params_}")
    
    print("Calibrating probability outputs...")
    # Calibrate the model for realistic probabilities
    calibrated_model = CalibratedClassifierCV(best_rf, cv=5, method='sigmoid')
    calibrated_model.fit(X_train, y_train)
    
    return calibrated_model


# ─── 5. Evaluate ──────────────────────────────────────────────────────────────

def evaluate_model(model, X_train, X_test, y_train, y_test):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    train_acc = accuracy_score(y_train, model.predict(X_train))
    test_acc  = accuracy_score(y_test,  y_pred)

    # Calculate calibration quality
    brier = brier_score_loss(y_test, y_prob)

    print(f"\n{'─'*45}")
    print(f"  Train Accuracy:   {train_acc*100:.2f}%")
    print(f"  Test  Accuracy:   {test_acc*100:.2f}%")
    print(f"  Brier Score:      {brier:.4f} (lower is better calibration)")
    print(f"{'─'*45}")

    print("\nClassification Report (test set):")
    print(classification_report(y_test, y_pred,
                                 target_names=["Benign", "Malignant"]))
    
    # Check probability diversity
    prob_range = np.ptp(y_prob)
    print(f"  → Probability spread: {prob_range:.4f} (diversity check)")
    if prob_range < 0.1:
        print("  ⚠ WARNING: Low probability spread detected.")

    return round(test_acc * 100, 2)


# ─── 6. Save ──────────────────────────────────────────────────────────────────

def save_artifacts(model, scaler, feature_cols, accuracy, source):
    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)

    meta = {
        "source": source,
        "n_features": len(feature_cols),
        "feature_names": list(feature_cols),
        "accuracy_pct": accuracy,
        "model_type": type(model).__name__
    }
    with open(META_PATH, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n✓ Saved: {MODEL_PATH}")
    print(f"✓ Saved: {SCALER_PATH}")
    print(f"✓ Saved: {META_PATH}")


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 45)
    print("   HEALIX AI MODEL TRAINING PIPELINE")
    print("=" * 45)

    X, y, source = load_real_dataset()

    print(f"\n  Dataset source:  {source}")
    print(f"  Raw shape:       {X.shape[0]} rows × {X.shape[1]} cols")
    print(f"  Class balance:   {dict(y.value_counts().sort_index())}")

    X = prepare_features(X, TARGET_FEATURES)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    X_train_s, X_test_s, scaler = scale_features(X_train, X_test)

    model = train_model(X_train_s, y_train)
    accuracy = evaluate_model(model, X_train_s, X_test_s, y_train, y_test)

    save_artifacts(model, scaler, X.columns, accuracy, source)

    print(f"\n✓ Training complete. Model accuracy: {accuracy}%")
    print("=" * 45)