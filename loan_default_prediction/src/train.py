"""
Loan Default Prediction — Training Pipeline
=============================================
Handles data loading, preprocessing, feature engineering, model training,
evaluation, and artifact saving.

Usage:
    python src/train.py          (run from loan_default_prediction/ directory)

TODOs are organized by phase — follow the guides in the todos/ folder:
    Phase 3: Feature Engineering  → todos/phase3_feature_engineering.md
    Phase 4: Model Training       → todos/phase4_model_training.md
    Phase 5: Hyperparameter Tuning→ todos/phase5_hyperparameter_tuning.md
    Phase 6: Evaluation & Plots   → todos/phase6_evaluation.md
"""

import logging
import warnings
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for saving figures
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import uniform, randint
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score, roc_auc_score, roc_curve,
    precision_recall_curve, average_precision_score,
)
from sklearn.model_selection import (
    cross_val_score, StratifiedKFold, RandomizedSearchCV, train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import seaborn as sns

warnings.filterwarnings("ignore")

# ── Logging ─────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

# ── Configuration ───────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent      # src/ → project/
DATA_PATH = BASE_DIR / "data" / "accepted_2007_to_2018Q4.csv.gz"
MODELS_DIR = BASE_DIR / "models"
FIGURES_DIR = BASE_DIR / "reports" / "figures"
RANDOM_STATE = 42
CV_FOLDS = 5
TARGET_COL = "loan_status"


# ═══════════════════════════════════════════════════════════════════════
#  HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════

def divider(title: str) -> None:
    """Print a section divider to the console."""
    log.info(f"\n{'=' * 60}\n  {title}\n{'=' * 60}")

def save_figure(fig, name: str) -> None:
    """Save a matplotlib figure as PNG to reports/figures/."""
    path = FIGURES_DIR / f"{name}.png"
    fig.savefig(path, bbox_inches="tight", dpi=150)
    log.info(f"  📊 Saved figure: {path.name}")

def save_model(model, name: str) -> None:
    """Save a model object as .pkl to models/."""
    path = MODELS_DIR / f"{name}.pkl"
    joblib.dump(model, path)
    log.info(f"  💾 Saved model: {path.name}")


# ═══════════════════════════════════════════════════════════════════════
#  PHASE 3 — FEATURE ENGINEERING & PREPROCESSING
#  Guide: todos/phase3_feature_engineering.md
# ═══════════════════════════════════════════════════════════════════════

def load_data() -> pd.DataFrame:
    """TODO 3.2 — Load accepted loans, filter to known outcomes, binarize target."""
    divider("1. LOADING DATA")

    log.info(f"  📂 Reading {DATA_PATH.name} ...")
    df = pd.read_csv(DATA_PATH, low_memory=False)
    log.info(f"     Raw shape: {df.shape[0]:,} rows × {df.shape[1]} cols")

    # Keep only definitive outcomes
    keep = ["Fully Paid", "Charged Off"]
    df = df[df[TARGET_COL].isin(keep)].copy()
    df[TARGET_COL] = df[TARGET_COL].map({"Fully Paid": 0, "Charged Off": 1})
    log.info(f"     After filtering: {len(df):,} rows")
    log.info(f"     Default rate: {df[TARGET_COL].mean():.2%}")

    return df


def drop_columns(df: pd.DataFrame) -> pd.DataFrame:
    """TODO 3.3 — Remove leakage columns, IDs, free text, and high-missing columns."""
    divider("2. DROPPING COLUMNS")

    # ── Leakage: post-origination data ──────────────────────────────────
    leakage = [
        "total_pymnt", "total_pymnt_inv", "total_rec_prncp", "total_rec_int",
        "total_rec_late_fee", "last_pymnt_d", "last_pymnt_amnt",
        "recoveries", "collection_recovery_fee",
        "funded_amnt", "funded_amnt_inv", "out_prncp", "out_prncp_inv",
        "last_credit_pull_d", "last_fico_range_high", "last_fico_range_low",
        "hardship_flag", "debt_settlement_flag", "settlement_status",
        "settlement_date", "settlement_amount", "settlement_percentage",
        "settlement_term", "payment_plan_start_date",
    ]

    # ── IDs and free text ───────────────────────────────────────────────
    ids_text = ["id", "member_id", "url", "desc", "emp_title", "title"]

    # Combine all columns to drop (only those that actually exist)
    to_drop = [c for c in leakage + ids_text if c in df.columns]
    df = df.drop(columns=to_drop)
    log.info(f"  🗑️  Dropped {len(to_drop)} leakage/ID/text columns")

    # ── High missing (>50%) ─────────────────────────────────────────────
    miss_pct = df.isnull().mean()
    high_miss = miss_pct[miss_pct > 0.50].index.tolist()
    high_miss = [c for c in high_miss if c != TARGET_COL]
    df = df.drop(columns=high_miss)
    log.info(f"  🗑️  Dropped {len(high_miss)} columns with >50% missing")
    if high_miss:
        log.info(f"     {high_miss[:10]}{'...' if len(high_miss) > 10 else ''}")

    # ── Zero-variance columns ───────────────────────────────────────────
    nunique = df.nunique()
    zero_var = nunique[nunique <= 1].index.tolist()
    zero_var = [c for c in zero_var if c != TARGET_COL]
    df = df.drop(columns=zero_var)
    log.info(f"  🗑️  Dropped {len(zero_var)} zero-variance columns")

    log.info(f"  ✅ Remaining: {df.shape[1]} columns")
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """TODO 3.4 — Create new features from existing columns using domain knowledge."""
    divider("3. FEATURE ENGINEERING")
    n_before = df.shape[1]

    # ── Parse term to numeric months ────────────────────────────────────
    # " 36 months" → 36, " 60 months" → 60
    if "term" in df.columns:
        df["term_months"] = (df["term"]
                             .astype(str)
                             .str.extract(r"(\d+)")[0]
                             .astype(float))
        df = df.drop(columns=["term"])
        log.info("  ✨ Created: term_months (from term string)")

    # ── Parse emp_length to numeric years ───────────────────────────────
    # "10+ years" → 10, "< 1 year" → 0, "2 years" → 2
    if "emp_length" in df.columns:
        emp_map = {"< 1 year": 0, "1 year": 1}
        for i in range(2, 10):
            emp_map[f"{i} years"] = i
        emp_map["10+ years"] = 10
        df["emp_years"] = df["emp_length"].map(emp_map)
        df = df.drop(columns=["emp_length"])
        log.info("  ✨ Created: emp_years (from emp_length string)")

    # ── Credit history length ───────────────────────────────────────────
    # How many years between earliest credit line and loan issue date
    if "issue_d" in df.columns and "earliest_cr_line" in df.columns:
        issue = pd.to_datetime(df["issue_d"], format="mixed", errors="coerce")
        earliest = pd.to_datetime(df["earliest_cr_line"], format="mixed", errors="coerce")
        df["credit_history_years"] = (issue - earliest).dt.days / 365.25
        df = df.drop(columns=["issue_d", "earliest_cr_line"])
        log.info("  ✨ Created: credit_history_years")
    else:
        # Drop raw date columns even if we can't compute the feature
        for col in ["issue_d", "earliest_cr_line"]:
            if col in df.columns:
                df = df.drop(columns=[col])

    # ── Handle int_rate if stored as string (e.g., "12.5%") ─────────────
    if "int_rate" in df.columns and df["int_rate"].dtype == "object":
        df["int_rate"] = df["int_rate"].str.replace("%", "", regex=False).astype(float)
        log.info("  🔧 Cleaned: int_rate (removed % sign)")

    # ── Handle revol_util if stored as string (e.g., "25.5%") ───────────
    if "revol_util" in df.columns and df["revol_util"].dtype == "object":
        df["revol_util"] = df["revol_util"].str.replace("%", "", regex=False).astype(float)
        log.info("  🔧 Cleaned: revol_util (removed % sign)")

    # ── Income-to-loan ratio ────────────────────────────────────────────
    # Higher ratio = borrower earns much more relative to what they're borrowing
    if "annual_inc" in df.columns and "loan_amnt" in df.columns:
        df["income_to_loan"] = df["annual_inc"] / df["loan_amnt"].replace(0, np.nan)
        log.info("  ✨ Created: income_to_loan")

    # ── Installment-to-income ratio (monthly burden) ────────────────────
    # What fraction of monthly income goes to this loan payment
    if "installment" in df.columns and "annual_inc" in df.columns:
        monthly_inc = df["annual_inc"] / 12
        df["installment_to_income"] = df["installment"] / monthly_inc.replace(0, np.nan)
        log.info("  ✨ Created: installment_to_income")

    n_after = df.shape[1]
    log.info(f"  ✅ Features: {n_before} → {n_after} ({n_after - n_before:+d})")
    return df


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """TODO 3.5 — Build a ColumnTransformer with separate pipelines for each type."""
    divider("5. BUILDING PREPROCESSOR")

    # Identify column types
    num_cols = X.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

    # Ordinal columns (have a natural order: A < B < C < ... < G)
    ordinal_cols = [c for c in ["grade", "sub_grade"] if c in cat_cols]
    nominal_cols = [c for c in cat_cols if c not in ordinal_cols]

    log.info(f"  Numeric cols:  {len(num_cols)}")
    log.info(f"  Ordinal cols:  {len(ordinal_cols)}  {ordinal_cols}")
    log.info(f"  Nominal cols:  {len(nominal_cols)}")

    # ── Numeric pipeline ────────────────────────────────────────────────
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    # ── Ordinal pipeline ────────────────────────────────────────────────
    grade_order = list("ABCDEFG")
    sub_grade_order = [f"{g}{n}" for g in "ABCDEFG" for n in range(1, 6)]

    ordinal_categories = []
    for col in ordinal_cols:
        if col == "grade":
            ordinal_categories.append(grade_order)
        elif col == "sub_grade":
            ordinal_categories.append(sub_grade_order)

    ordinal_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OrdinalEncoder(categories=ordinal_categories,
                                   handle_unknown="use_encoded_value",
                                   unknown_value=-1)),
    ])

    # ── Nominal pipeline ────────────────────────────────────────────────
    nominal_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False,
                                  max_categories=15)),
    ])

    # ── Combine ─────────────────────────────────────────────────────────
    transformers = [("num", numeric_pipe, num_cols)]
    if ordinal_cols:
        transformers.append(("ord", ordinal_pipe, ordinal_cols))
    if nominal_cols:
        transformers.append(("nom", nominal_pipe, nominal_cols))

    preprocessor = ColumnTransformer(transformers, remainder="drop")

    return preprocessor


def split_data(df: pd.DataFrame):
    """TODO 3.6 — Stratified 80/20 train-test split."""
    divider("4. TRAIN / TEST SPLIT")

    y = df[TARGET_COL]
    X = df.drop(columns=[TARGET_COL])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.20,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    log.info(f"  Train: {X_train.shape[0]:,} rows ({y_train.mean():.2%} default)")
    log.info(f"  Test:  {X_test.shape[0]:,} rows ({y_test.mean():.2%} default)")

    return X_train, X_test, y_train, y_test


# ═══════════════════════════════════════════════════════════════════════
#  PHASE 4 — MODEL TRAINING
#  Guide: todos/phase4_model_training.md
# ═══════════════════════════════════════════════════════════════════════

def evaluate_model(name: str, model, X_test, y_test):
    """TODO 4.2 — Standardized evaluation helper. Returns metrics dict, y_pred, y_prob."""
    log.info(f"\n  📈 Evaluating: {name}")

    y_pred = model.predict(X_test)

    # Some models support probability predictions (needed for ROC-AUC)
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    else:
        y_prob = y_pred.astype(float)

    metrics = {
        "model": name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_prob),
        "avg_precision": average_precision_score(y_test, y_prob),
    }

    log.info(f"     Accuracy:     {metrics['accuracy']:.4f}")
    log.info(f"     Precision:    {metrics['precision']:.4f}")
    log.info(f"     Recall:       {metrics['recall']:.4f}")
    log.info(f"     F1-score:     {metrics['f1']:.4f}")
    log.info(f"     ROC-AUC:      {metrics['roc_auc']:.4f}")
    log.info(f"     Avg Precision: {metrics['avg_precision']:.4f}")

    return metrics, y_pred, y_prob


def train_logistic_regression(preprocessor, X_train, y_train):
    """TODO 4.3 — Train a Logistic Regression with class weighting."""
    divider("6A. LOGISTIC REGRESSION (BASELINE)")

    pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            solver="lbfgs",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )),
    ])

    log.info("  🔄 Running 5-fold stratified cross-validation...")
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_val_score(pipe, X_train, y_train, cv=cv,
                             scoring="roc_auc", n_jobs=-1)
    log.info(f"     CV ROC-AUC: {scores.mean():.4f} ± {scores.std():.4f}")

    log.info("  🏋️ Fitting on full training set...")
    pipe.fit(X_train, y_train)

    return pipe


def train_random_forest(preprocessor, X_train, y_train):
    """TODO 4.4 — Train a Random Forest with class weighting."""
    divider("6B. RANDOM FOREST")

    pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_leaf=50,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )),
    ])

    log.info("  🔄 Running 5-fold stratified cross-validation...")
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_val_score(pipe, X_train, y_train, cv=cv,
                             scoring="roc_auc", n_jobs=-1)
    log.info(f"     CV ROC-AUC: {scores.mean():.4f} ± {scores.std():.4f}")

    log.info("  🏋️ Fitting on full training set...")
    pipe.fit(X_train, y_train)

    # Feature importances (from the tree-based model)
    feature_names = pipe.named_steps["preprocessor"].get_feature_names_out()
    importances = pipe.named_steps["classifier"].feature_importances_
    feat_imp = pd.Series(importances, index=feature_names).sort_values(ascending=False)
    log.info(f"\n  🌳 Top 10 features:")
    for name, imp in feat_imp.head(10).items():
        log.info(f"     {name:40s} {imp:.4f}")

    return pipe, feat_imp


def train_xgboost(preprocessor, X_train, y_train):
    """TODO 4.5 — Train an XGBoost classifier with class imbalance handling."""
    divider("6C. XGBOOST")

    # Calculate scale_pos_weight = n_negative / n_positive
    n_pos = (y_train == 1).sum()
    n_neg = (y_train == 0).sum()
    scale_weight = n_neg / n_pos
    log.info(f"  ⚖️  scale_pos_weight: {scale_weight:.2f} (neg/pos ratio)")

    pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_weight,
            eval_metric="auc",
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbosity=0,
        )),
    ])

    log.info("  🔄 Running 5-fold stratified cross-validation...")
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_val_score(pipe, X_train, y_train, cv=cv,
                             scoring="roc_auc", n_jobs=-1)
    log.info(f"     CV ROC-AUC: {scores.mean():.4f} ± {scores.std():.4f}")

    log.info("  🏋️ Fitting on full training set...")
    pipe.fit(X_train, y_train)

    # Feature importances
    feature_names = pipe.named_steps["preprocessor"].get_feature_names_out()
    importances = pipe.named_steps["classifier"].feature_importances_
    feat_imp = pd.Series(importances, index=feature_names).sort_values(ascending=False)
    log.info(f"\n  🚀 Top 10 features:")
    for name, imp in feat_imp.head(10).items():
        log.info(f"     {name:40s} {imp:.4f}")

    return pipe, feat_imp


# ═══════════════════════════════════════════════════════════════════════
#  PHASE 5 — HYPERPARAMETER TUNING
#  Guide: todos/phase5_hyperparameter_tuning.md
# ═══════════════════════════════════════════════════════════════════════

def tune_xgboost(preprocessor, X_train, y_train):
    """TODO 5.2 — Tune XGBoost hyperparameters using RandomizedSearchCV."""
    divider("7. HYPERPARAMETER TUNING (XGBOOST)")

    n_pos = (y_train == 1).sum()
    n_neg = (y_train == 0).sum()
    scale_weight = n_neg / n_pos

    # ── Base pipeline (preprocessor + untuned XGBoost) ──────────────────
    pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", XGBClassifier(
            scale_pos_weight=scale_weight,
            eval_metric="auc",
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbosity=0,
        )),
    ])

    # ── Search space ────────────────────────────────────────────────────
    # Prefix with "classifier__" because parameters are inside the Pipeline
    param_distributions = {
        "classifier__max_depth":        randint(3, 10),         # 3 to 9
        "classifier__learning_rate":    uniform(0.01, 0.19),    # 0.01 to 0.20
        "classifier__n_estimators":     randint(100, 501),      # 100 to 500
        "classifier__subsample":        uniform(0.6, 0.3),      # 0.6 to 0.9
        "classifier__colsample_bytree": uniform(0.6, 0.3),      # 0.6 to 0.9
        "classifier__min_child_weight": randint(1, 11),         # 1 to 10
        "classifier__gamma":           uniform(0, 0.5),         # 0 to 0.5
    }

    # ── RandomizedSearchCV ──────────────────────────────────────────────
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)

    search = RandomizedSearchCV(
        pipe,
        param_distributions,
        n_iter=30,              # 30 random combinations (practical for large data)
        scoring="roc_auc",
        cv=cv,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=1,
        return_train_score=True,
    )

    log.info(f"  🔍 Searching 30 random combinations × 3 folds = 90 fits")
    log.info(f"     This will take a while on large data... ☕")
    search.fit(X_train, y_train)

    # ── Results ─────────────────────────────────────────────────────────
    log.info(f"\n  🏆 Best ROC-AUC (CV): {search.best_score_:.4f}")
    log.info(f"  🎯 Best parameters:")
    for param, value in search.best_params_.items():
        clean_name = param.replace("classifier__", "")
        log.info(f"     {clean_name:25s} = {value}")

    # Check for overfitting: train score vs test score
    results_df = pd.DataFrame(search.cv_results_)
    best_idx = search.best_index_
    train_score = results_df.loc[best_idx, "mean_train_score"]
    test_score = results_df.loc[best_idx, "mean_test_score"]
    log.info(f"\n  📊 Train AUC: {train_score:.4f}  |  Test AUC: {test_score:.4f}")
    if train_score - test_score > 0.05:
        log.info(f"  ⚠️  Gap > 5% — possible overfitting!")
    else:
        log.info(f"  ✅ Minimal gap — good generalization")

    return search.best_estimator_


# ═══════════════════════════════════════════════════════════════════════
#  PHASE 6 — EVALUATION & VISUALIZATION
#  Guide: todos/phase6_evaluation.md
# ═══════════════════════════════════════════════════════════════════════

def plot_roc_curves(results_list, y_test):
    """TODO 6.1 — Plot ROC curves for all models on one figure."""
    fig, ax = plt.subplots(figsize=(8, 8))

    for name, y_prob in results_list:
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc = roc_auc_score(y_test, y_prob)
        ax.plot(fpr, tpr, linewidth=2, label=f"{name} (AUC = {auc:.3f})")

    # Diagonal = random classifier (AUC = 0.5)
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random (AUC = 0.500)")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate (Recall)", fontsize=12)
    ax.set_title("ROC Curves — Model Comparison", fontsize=14)
    ax.legend(loc="lower right", fontsize=11)
    ax.set_aspect("equal")
    plt.tight_layout()
    save_figure(fig, "roc_curves")
    plt.close(fig)


def plot_precision_recall_curves(results_list, y_test):
    """TODO 6.2 — Plot Precision-Recall curves for all models."""
    fig, ax = plt.subplots(figsize=(8, 8))

    for name, y_prob in results_list:
        precision, recall, _ = precision_recall_curve(y_test, y_prob)
        ap = average_precision_score(y_test, y_prob)
        ax.plot(recall, precision, linewidth=2, label=f"{name} (AP = {ap:.3f})")

    # Baseline = proportion of positives (random classifier)
    baseline = y_test.mean()
    ax.axhline(baseline, color="k", linestyle="--", linewidth=1,
               label=f"Random (AP = {baseline:.3f})")

    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title("Precision-Recall Curves — Model Comparison", fontsize=14)
    ax.legend(loc="upper right", fontsize=11)
    plt.tight_layout()
    save_figure(fig, "precision_recall_curves")
    plt.close(fig)


def plot_confusion_matrix(name, y_test, y_pred):
    """TODO 6.3 — Plot a confusion matrix heatmap."""
    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt=",d", cmap="Blues", ax=ax,
                xticklabels=["Fully Paid", "Charged Off"],
                yticklabels=["Fully Paid", "Charged Off"])
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("Actual", fontsize=12)
    ax.set_title(f"Confusion Matrix — {name}", fontsize=14)
    plt.tight_layout()
    save_figure(fig, f"confusion_matrix_{name.lower().replace(' ', '_').replace('(', '').replace(')', '')}")
    plt.close(fig)


def plot_feature_importances(name, feat_imp, top_n=20):
    """TODO 6.4 — Plot top N feature importances (horizontal bar chart)."""
    top = feat_imp.head(top_n).sort_values()

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(top.index, top.values, color="steelblue", edgecolor="black")
    ax.set_xlabel("Importance", fontsize=12)
    ax.set_title(f"Top {top_n} Feature Importances — {name}", fontsize=14)
    plt.tight_layout()
    save_figure(fig, f"feature_importances_{name.lower().replace(' ', '_').replace('(', '').replace(')', '')}")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════
#  MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════════

def main():
    """
    Run the full training pipeline end-to-end.

    Order of operations (matches the phase guides):
        Phase 3: load → drop → engineer → split → preprocess
        Phase 4: train LR → train RF → train XGB → compare
        Phase 5: tune XGB → save best model
        Phase 6: plot ROC → plot PR → plot confusion → plot importances
    """
    divider("LOAN DEFAULT PREDICTION — TRAINING PIPELINE")

    # Ensure output directories exist
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # ── Phase 3: Feature Engineering ────────────────────────────────────
    df = load_data()
    df = drop_columns(df)
    df = engineer_features(df)
    X_train, X_test, y_train, y_test = split_data(df)
    preprocessor = build_preprocessor(X_train)

    # ── Phase 4: Model Training ─────────────────────────────────────────
    lr_pipe = train_logistic_regression(preprocessor, X_train, y_train)
    lr_metrics, lr_pred, lr_prob = evaluate_model(
        "Logistic Regression", lr_pipe, X_test, y_test)

    rf_pipe, rf_feat_imp = train_random_forest(preprocessor, X_train, y_train)
    rf_metrics, rf_pred, rf_prob = evaluate_model(
        "Random Forest", rf_pipe, X_test, y_test)

    xgb_pipe, xgb_feat_imp = train_xgboost(preprocessor, X_train, y_train)
    xgb_metrics, xgb_pred, xgb_prob = evaluate_model(
        "XGBoost", xgb_pipe, X_test, y_test)

    # ── Model Comparison ────────────────────────────────────────────────
    divider("MODEL COMPARISON")
    results = pd.DataFrame([lr_metrics, rf_metrics, xgb_metrics])
    results = results.set_index("model")
    log.info("\n" + results.round(4).to_string())

    best_name = results["roc_auc"].idxmax()
    log.info(f"\n  🏆 Best model by ROC-AUC: {best_name}")

    # ── Phase 5: Hyperparameter Tuning ──────────────────────────────────
    best_model = tune_xgboost(preprocessor, X_train, y_train)
    best_metrics, best_pred, best_prob = evaluate_model(
        "XGBoost (Tuned)", best_model, X_test, y_test)

    # ── Save the best model ─────────────────────────────────────────────
    save_model(best_model, "xgboost_tuned")

    all_results = pd.DataFrame([lr_metrics, rf_metrics, xgb_metrics, best_metrics])
    all_results = all_results.set_index("model").round(4)
    all_results.to_csv(MODELS_DIR / "results.csv")
    log.info(f"\n  📊 Results saved to models/results.csv")

    # ── Classification Report (Best Model) ──────────────────────────────
    divider("CLASSIFICATION REPORT (BEST MODEL)")
    report = classification_report(
        y_test, best_pred,
        target_names=["Fully Paid", "Charged Off"],
    )
    log.info(f"\n{report}")

    # ── Phase 6: Evaluation & Plots ─────────────────────────────────────
    divider("GENERATING FIGURES")

    results_for_plot = [
        ("Logistic Regression", lr_prob),
        ("Random Forest", rf_prob),
        ("XGBoost (Tuned)", best_prob),
    ]

    plot_roc_curves(results_for_plot, y_test)
    plot_precision_recall_curves(results_for_plot, y_test)
    plot_confusion_matrix("XGBoost (Tuned)", y_test, best_pred)
    plot_feature_importances("XGBoost", xgb_feat_imp)

    log.info(f"\n  ✅ All figures saved to {FIGURES_DIR}")
    log.info("\n✅ Pipeline complete!")


if __name__ == "__main__":
    main()
