"""
Train regression models on the Ames Housing dataset.

This script handles:
  1. Data loading & cleaning
  2. Preprocessing pipeline (imputation + encoding)
  3. Training Linear Regression and Random Forest models
  4. Cross-validation evaluation
  5. Saving trained models (.pkl) and comparisons figures (.pdf)

Usage:
    cd house_price_prediction
    python src/train.py
"""

import logging
import warnings
from pathlib import Path

import joblib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import make_column_transformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder

# ---------------------------------------------------------------------------
# Logging Setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s"
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Evaluation Helpers
# ---------------------------------------------------------------------------
def get_performance_emoji(r2_score: float) -> str:
    """Return emoji based on R² score."""
    if r2_score >= 0.95:
        return "🔥"  # Excellent
    elif r2_score >= 0.90:
        return "⭐"  # Great
    elif r2_score >= 0.80:
        return "👍"  # Good
    elif r2_score >= 0.70:
        return "📊"  # Okay
    else:
        return "⚠️"  # Poor


def evaluate_model(scores: np.ndarray, model_name: str) -> None:
    """Log detailed model performance evaluation."""
    mean_score = scores.mean()
    std_score = scores.std()
    min_score = scores.min()
    max_score = scores.max()
    emoji = get_performance_emoji(mean_score)

    logger.info(f"\n  {emoji} {model_name} Performance:")
    logger.info(f"     Mean R²:     {mean_score:.4f}")
    logger.info(f"     Std Dev:     {std_score:.4f}")
    logger.info(f"     Min R²:      {min_score:.4f}")
    logger.info(f"     Max R²:      {max_score:.4f}")

    # Performance assessment
    if mean_score >= 0.95:
        assessment = "Excellent! 🎯"
    elif mean_score >= 0.90:
        assessment = "Great performance! 🚀"
    elif mean_score >= 0.80:
        assessment = "Good fit! ✓"
    elif mean_score >= 0.70:
        assessment = "Acceptable, but could improve"
    else:
        assessment = "Poor performance, needs tuning"

    logger.info(f"     Assessment: {assessment}\n")


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "ames-housing.csv"
MODELS_DIR = BASE_DIR / "models"
FIGURES_DIR = BASE_DIR / "reports" / "figures"

RANDOM_STATE = 42
CV_FOLDS = 5
RF_N_ESTIMATORS = 200

# Use non-interactive backend so plt.show() is never needed
matplotlib.use("Agg")
warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def save_figure(fig, filename: str) -> None:
    """Save a matplotlib figure to the figures directory as PDF."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    filepath = FIGURES_DIR / filename
    fig.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  📊 Saved {filepath.relative_to(BASE_DIR)}")


def save_model(model, filename: str) -> None:
    """Persist a trained sklearn model with joblib."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    filepath = MODELS_DIR / filename
    joblib.dump(model, filepath)
    logger.info(f"  💾 Saved {filepath.relative_to(BASE_DIR)}")


def divider(title: str = "") -> None:
    """Print a formatted section divider."""
    width = 75
    if title:
        logger.info("\n" + "=" * width)
        logger.info(f"  {title}")
        logger.info("=" * width)
    else:
        logger.info("\n" + "=" * width)


# ---------------------------------------------------------------------------
# 1. Load & Clean
# ---------------------------------------------------------------------------
def load_data() -> tuple[pd.DataFrame, pd.Series]:
    """Load the Ames Housing CSV and return (X, y)."""
    divider("🔍 LOADING DATA")
    df = pd.read_csv(DATA_PATH)

    # Standardise column names
    df.columns = df.columns.str.strip().str.replace(" ", "", regex=False)

    # Separate target
    target = "SalePrice"
    X = df.drop(columns=target)
    y = df[target]

    # Drop irrelevant identifiers
    X = X.drop(columns=["Order", "PID"])

    # Strip whitespace from categorical values
    str_cols = X.select_dtypes(include=["object", "string"]).columns
    X[str_cols] = X[str_cols].apply(lambda c: c.str.strip())

    logger.info(f"  ✓ Loaded {len(X):,} samples, {X.shape[1]} features")
    logger.info(f"  ✓ Target variable: {target} (mean: ${y.mean():,.0f})")
    return X, y



# ---------------------------------------------------------------------------
# 3. Build Preprocessing Pipeline
# ---------------------------------------------------------------------------
def build_preprocessor(X: pd.DataFrame):
    """Return a ColumnTransformer and feature name lists."""
    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns
    categorical_features = X.select_dtypes(include=["object", "string"]).columns

    numeric_imputer = SimpleImputer(strategy="median")

    categorical_pipeline = make_pipeline(
        SimpleImputer(strategy="most_frequent"),
        OneHotEncoder(handle_unknown="ignore", sparse_output=False),
    )

    preprocessor = make_column_transformer(
        (numeric_imputer, numeric_features),
        (categorical_pipeline, categorical_features),
    )
    
    logger.info(f"\n  ✓ Preprocessor built:")
    logger.info(f"     • {len(numeric_features)} numeric features → median imputation")
    logger.info(f"     • {len(categorical_features)} categorical features → one-hot encoding\n")
    
    return preprocessor


# ---------------------------------------------------------------------------
# 4. Train & Evaluate
# ---------------------------------------------------------------------------
def train_linear_regression(preprocessor, X, y) -> object:
    divider("🔗 LINEAR REGRESSION")
    model = make_pipeline(preprocessor, LinearRegression())

    scores = cross_val_score(model, X, y, cv=CV_FOLDS, scoring="r2")
    evaluate_model(scores, "Linear Regression")

    model.fit(X, y)
    save_model(model, "linear_regression.pkl")
    return model, scores


def train_random_forest(preprocessor, X, y_log) -> object:
    divider("🌲 RANDOM FOREST (log-transformed target)")
    model_rf = make_pipeline(
        preprocessor,
        RandomForestRegressor(
            n_estimators=RF_N_ESTIMATORS,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    )

    scores = cross_val_score(model_rf, X, y_log, cv=CV_FOLDS, scoring="r2")
    evaluate_model(scores, "Random Forest")

    model_rf.fit(X, y_log)
    save_model(model_rf, "random_forest.pkl")

    # Quick sanity check — inverse-transform a few predictions
    y_pred = np.expm1(model_rf.predict(X[:5]))
    logger.info(f"  🔮 Sample predictions (original scale):")
    logger.info(f"     {[f'${p:,.0f}' for p in y_pred[:5]]}\n")
    
    return model_rf, scores


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    logger.info("\n🏠 AMES HOUSING PRICE PREDICTION")
    logger.info("=" * 75)
    
    X, y = load_data()
    y_log = np.log1p(y)

    preprocessor = build_preprocessor(X)
    model_lr, scores_lr = train_linear_regression(preprocessor, X, y)
    model_rf, scores_rf = train_random_forest(preprocessor, X, y_log)

    # Final comparison
    divider("🏆 MODEL COMPARISON")
    logger.info(f"  Linear Regression:  {scores_lr.mean():.4f} (±{scores_lr.std():.4f})")
    logger.info(f"  Random Forest:      {scores_rf.mean():.4f} (±{scores_rf.std():.4f})")
    
    if scores_rf.mean() > scores_lr.mean():
        logger.info(f"\n  🎯 Winner: Random Forest by {(scores_rf.mean() - scores_lr.mean()):.4f}")
    else:
        logger.info(f"\n  🎯 Winner: Linear Regression by {(scores_lr.mean() - scores_rf.mean()):.4f}")

    divider("✅ TRAINING COMPLETE")
    logger.info("  📁 Output Locations:")
    logger.info("     • Models:  models/")
    logger.info("     • Figures: reports/figures/")
    logger.info("")


if __name__ == "__main__":
    main()