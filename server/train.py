"""
╔══════════════════════════════════════════════════════════╗
║         ML Deployment Project — Model Training           ║
║  Trains, compares, selects, and saves the best model.    ║
╚══════════════════════════════════════════════════════════╝

Usage:
    python server/train.py --data path/to/dataset.csv --target target_column
    python server/train.py --demo   (uses Iris dataset for quick demo)
"""

import argparse
import os
import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from loguru import logger

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# ── Classifiers ──────────────────────────────────────────
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

# ── Metrics ──────────────────────────────────────────────
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    confusion_matrix, roc_auc_score
)

# ─────────────────────────────────────────────
#  PATHS
# ─────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

MODEL_PATH    = os.path.join(MODELS_DIR, "best_model.pkl")
METADATA_PATH = os.path.join(MODELS_DIR, "model_metadata.json")
PLOTS_DIR     = os.path.join(MODELS_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)


# ─────────────────────────────────────────────
#  1. DATA LOADING & PREPROCESSING
# ─────────────────────────────────────────────

def load_data(csv_path: str, target_col: str) -> tuple:
    """Load CSV, separate features from target, encode categoricals."""
    logger.info(f"Loading dataset from: {csv_path}")
    df = pd.read_csv(csv_path)
    logger.info(f"Shape: {df.shape}  |  Target: '{target_col}'")

    if target_col not in df.columns:
        raise ValueError(f"Column '{target_col}' not found. Available: {list(df.columns)}")

    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Encode categorical features
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    for col in cat_cols:
        X[col] = LabelEncoder().fit_transform(X[col].astype(str))

    # Encode target if needed
    le = None
    if y.dtype == object or str(y.dtype) == "category":
        le = LabelEncoder()
        y = le.fit_transform(y)
        logger.info(f"Classes: {list(le.classes_)}")
    else:
        y = y.values

    feature_names = list(X.columns)
    X = X.values.astype(float)

    classes = list(le.classes_) if le else sorted(list(set(y)))
    return X, y, feature_names, classes, le


def load_demo_data() -> tuple:
    """Load Iris dataset as a built-in demo."""
    logger.info("Using DEMO mode — Iris dataset")
    from sklearn.datasets import load_iris
    iris = load_iris()
    X, y = iris.data, iris.target
    feature_names = list(iris.feature_names)
    classes = list(iris.target_names)
    return X, y, feature_names, classes, None


# ─────────────────────────────────────────────
#  2. MODEL DEFINITIONS
# ─────────────────────────────────────────────

def get_models(n_classes: int) -> dict:
    """Return a dict of named pipelines to compare."""
    use_proba = True  # All models here support predict_proba

    models = {
        "Logistic Regression": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler",  StandardScaler()),
            ("clf",     LogisticRegression(max_iter=1000, random_state=42))
        ]),
        "SVM (RBF)": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler",  StandardScaler()),
            ("clf",     SVC(probability=True, random_state=42))
        ]),
        "K-Nearest Neighbors": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler",  StandardScaler()),
            ("clf",     KNeighborsClassifier(n_neighbors=5))
        ]),
        "Decision Tree": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("clf",     DecisionTreeClassifier(max_depth=8, random_state=42))
        ]),
        "Random Forest": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("clf",     RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1))
        ]),
        "Gradient Boosting": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("clf",     GradientBoostingClassifier(n_estimators=150, random_state=42))
        ]),
        "XGBoost": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("clf",     XGBClassifier(
                n_estimators=200,
                use_label_encoder=False,
                eval_metric="mlogloss" if n_classes > 2 else "logloss",
                random_state=42, verbosity=0
            ))
        ]),
    }
    return models


# ─────────────────────────────────────────────
#  3. TRAINING & COMPARISON
# ─────────────────────────────────────────────

def train_and_compare(X_train, X_test, y_train, y_test, n_classes: int) -> dict:
    """Train all models, evaluate, and return sorted results."""
    models   = get_models(n_classes)
    cv       = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results  = {}

    logger.info("\n{'─'*55}")
    logger.info(f"{'Model':<25} {'CV F1':>8} {'Test Acc':>10} {'Test F1':>10}")
    logger.info("─" * 55)

    for name, pipeline in models.items():
        # Cross-validated F1 on train set
        cv_f1 = cross_val_score(
            pipeline, X_train, y_train,
            cv=cv,
            scoring="f1_macro",
            n_jobs=-1
        ).mean()

        # Fit on full training set
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        acc    = accuracy_score(y_test, y_pred)
        f1     = f1_score(y_test, y_pred, average="macro")

        # ROC-AUC (only meaningful with proba)
        try:
            y_proba = pipeline.predict_proba(X_test)
            auc = roc_auc_score(
                y_test, y_proba,
                multi_class="ovr" if n_classes > 2 else "raise",
                average="macro"
            )
        except Exception:
            auc = None

        results[name] = {
            "pipeline": pipeline,
            "cv_f1":    round(cv_f1, 4),
            "test_acc": round(acc,   4),
            "test_f1":  round(f1,    4),
            "test_auc": round(auc, 4) if auc else "N/A",
            "y_pred":   y_pred,
        }

        logger.info(f"{name:<25} {cv_f1:>8.4f} {acc:>10.4f} {f1:>10.4f}")

    return results


# ─────────────────────────────────────────────
#  4. SELECT BEST MODEL
# ─────────────────────────────────────────────

def select_best(results: dict) -> tuple:
    """Select the model with highest CV F1-Macro score."""
    best_name = max(results, key=lambda k: results[k]["cv_f1"])
    logger.info(f"\n✅  Best model: {best_name}  (CV F1 = {results[best_name]['cv_f1']})")
    return best_name, results[best_name]


# ─────────────────────────────────────────────
#  5. PLOTS
# ─────────────────────────────────────────────

def plot_comparison(results: dict):
    """Bar chart comparing all models."""
    names  = list(results.keys())
    f1s    = [results[n]["test_f1"]  for n in names]
    accs   = [results[n]["test_acc"] for n in names]

    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(12, 5))
    bars1 = ax.bar(x - 0.2, accs, 0.35, label="Accuracy", color="#4A90D9", alpha=0.85)
    bars2 = ax.bar(x + 0.2, f1s,  0.35, label="F1-Macro", color="#E87040", alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=25, ha="right", fontsize=9)
    ax.set_ylim(0, 1.1)
    ax.set_title("Model Comparison — Accuracy & F1-Macro", fontsize=14, fontweight="bold")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    for bar in [*bars1, *bars2]:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=7)

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "model_comparison.png")
    plt.savefig(path, dpi=150)
    plt.close()
    logger.info(f"📊 Comparison chart → {path}")


def plot_confusion_matrix(y_test, y_pred, classes: list):
    """Heatmap confusion matrix for the best model."""
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(max(5, len(classes)), max(4, len(classes))))
    sns.heatmap(
        cm, annot=True, fmt="d",
        xticklabels=classes, yticklabels=classes,
        cmap="Blues", linewidths=0.5, ax=ax
    )
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("Actual",    fontsize=11)
    ax.set_title("Confusion Matrix — Best Model", fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "confusion_matrix.png")
    plt.savefig(path, dpi=150)
    plt.close()
    logger.info(f"📊 Confusion matrix → {path}")


# ─────────────────────────────────────────────
#  6. SAVE MODEL & METADATA
# ─────────────────────────────────────────────

def save_model(best_name: str, best: dict, feature_names: list, classes: list, le, target_col: str = ""):
    """Persist the best pipeline and metadata to disk."""
    joblib.dump(best["pipeline"], MODEL_PATH)
    logger.info(f"💾 Model saved → {MODEL_PATH}")

    metadata = {
        "best_model":    best_name,
        "feature_names": feature_names,
        "classes":       [str(c) for c in classes],
        "target_col":    target_col,
        "cv_f1":         best["cv_f1"],
        "test_accuracy": best["test_acc"],
        "test_f1_macro": best["test_f1"],
        "test_roc_auc":  str(best["test_auc"]),
        "label_encoded": le is not None,
    }
    with open(METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"📋 Metadata saved → {METADATA_PATH}")


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",   type=str, help="Path to CSV dataset")
    parser.add_argument("--target", type=str, help="Target column name")
    parser.add_argument("--demo",   action="store_true", help="Use Iris demo dataset")
    parser.add_argument("--test_size", type=float, default=0.2)
    args = parser.parse_args()

    # ── Load Data ──────────────────────────────
    if args.demo or not args.data:
        X, y, feature_names, classes, le = load_demo_data()
    else:
        X, y, feature_names, classes, le = load_data(args.data, args.target)

    n_classes = len(set(y))
    logger.info(f"Features: {len(feature_names)}  |  Classes: {n_classes}  |  Samples: {len(X)}")

    # ── Split ──────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=42, stratify=y
    )
    logger.info(f"Train: {len(X_train)}  |  Test: {len(X_test)}")

    # ── Train & Compare ────────────────────────
    results = train_and_compare(X_train, X_test, y_train, y_test, n_classes)

    # ── Select Best ────────────────────────────
    best_name, best = select_best(results)

    # ── Full Report ────────────────────────────
    logger.info("\n" + classification_report(
        y_test, best["y_pred"],
        target_names=[str(c) for c in classes]
    ))

    # ── Plots ──────────────────────────────────
    plot_comparison(results)
    plot_confusion_matrix(y_test, best["y_pred"], [str(c) for c in classes])

    # ── Save ───────────────────────────────────
    save_model(best_name, best, feature_names, classes, le, target_col=args.target or "")
    logger.success("\n🎉 Training complete! Run: uvicorn server.main:app --reload")


if __name__ == "__main__":
    main()
