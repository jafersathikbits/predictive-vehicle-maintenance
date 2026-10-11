# run_training.py
import json
import os

import joblib
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from src.config import (
    FEATURE_NAMES_PATH,
    INPUT_DEFAULTS_PATH,
    METRICS_PATH,
    MODEL_DIR,
    MODEL_PATH,
    NUMERIC_FEATURES,
    PROCESSED_DATA_PATH,
    RANDOM_STATE,
    SCALER_PATH,
    TARGET,
    TEST_SIZE,
)


def input_defaults(df: pd.DataFrame) -> dict:
    """Median / min / max of each numeric feature, used by the app for form defaults and ranges."""
    stats = {}
    for col in NUMERIC_FEATURES:
        if col == 'condition_score':
            continue
        s = df[col]
        stats[col] = {"min": float(s.min()), "max": float(s.max()), "median": float(s.median())}
    return stats


def train_and_evaluate():
    print("=== Phase 5–6: Modeling, Training & Evaluation ===")
    print("Loading processed data...")
    df = pd.read_csv(PROCESSED_DATA_PATH)
    feature_names = pd.read_csv(FEATURE_NAMES_PATH, header=None)[0].tolist()

    X = df[feature_names]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    X_train, X_test = X_train.copy(), X_test.copy()
    print(f"Train: {X_train.shape[0]} rows | Test: {X_test.shape[0]} rows")

    # Scale only numeric features (fit on train only to avoid leakage)
    scaler = StandardScaler()
    X_train[NUMERIC_FEATURES] = scaler.fit_transform(X_train[NUMERIC_FEATURES])
    X_test[NUMERIC_FEATURES] = scaler.transform(X_test[NUMERIC_FEATURES])

    # Models (with imbalance handling)
    models = {
        "XGBoost": XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.07,
            scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(),
            random_state=RANDOM_STATE,
            eval_metric='aucpr',
            n_jobs=-1,
            verbosity=0,
        ),
        "LightGBM": LGBMClassifier(
            n_estimators=300,
            max_depth=7,
            learning_rate=0.08,
            class_weight='balanced',
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=-1,
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            class_weight='balanced',
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }

    results = {}
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)

        y_proba = model.predict_proba(X_test)[:, 1]
        y_pred = (y_proba >= 0.5).astype(int)
        report = classification_report(y_test, y_pred, output_dict=True)

        results[name] = {
            "AUC": roc_auc_score(y_test, y_proba),
            "Recall_1": report['1']['recall'],
            "Precision_1": report['1']['precision'],
            "F1_1": report['1']['f1-score'],
            "Accuracy": report['accuracy'],
        }

        print(f"\n{name} Results:")
        print(classification_report(y_test, y_pred))
        print(f"ROC-AUC: {results[name]['AUC']:.4f}")
        print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

    # Pick the best model by F1 on the "needs maintenance" class, then AUC.
    # (Recall alone rewards a model that flags every vehicle.)
    best_name = max(results, key=lambda k: (results[k]['F1_1'], results[k]['AUC']))
    print(f"\nBest model: {best_name} (F1 = {results[best_name]['F1_1']:.4f}, "
          f"AUC = {results[best_name]['AUC']:.4f})")

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(models[best_name], MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    with open(METRICS_PATH, "w") as f:
        json.dump({
            "best_model": best_name,
            "test_rows": int(len(y_test)),
            "positive_rate": float(y.mean()),
            "results": results,
        }, f, indent=2)
    with open(INPUT_DEFAULTS_PATH, "w") as f:
        json.dump(input_defaults(df), f, indent=2)

    print(f"\nBest model saved → {MODEL_PATH}")
    print(f"Scaler saved      → {SCALER_PATH}")
    print(f"Metrics saved     → {METRICS_PATH}")
    print("\nPhase 5–6 completed. Model ready for app & explanation.")


if __name__ == "__main__":
    train_and_evaluate()
