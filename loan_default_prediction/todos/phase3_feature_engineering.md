# 🎨 Phase 3 — Feature Engineering & Preprocessing

> **Where:** `src/train.py` (create this file)  
> **Prereq:** [Phase 2 — EDA](phase2_eda.md) completed — you need the findings to decide what to engineer  
> **Next:** [Phase 4 — Model Training](phase4_model_training.md)

---

## 🧒 What Are We Doing Here? (The Big Picture)

EDA was the detective work. Now you're the **blacksmith** — forging raw materials
(raw columns) into tools the model can actually use (clean, engineered features).

Models don't understand strings like "RENT" or dates like "Jan-2015". They need
numbers. Feature engineering is the art of converting domain knowledge into numbers
that help the model learn patterns.

**This phase happens in `src/train.py`** so the preprocessing is reproducible and
doesn't depend on running 30 notebook cells in order.

---

## ✅ TODO 3.1 — Create `src/train.py` Scaffold

**Purpose:** Following the project convention (same as `house_price_prediction/src/train.py`),
create a standalone training script that handles everything from data loading to model saving.

**Create the file `loan_default_prediction/src/train.py` with this structure:**

```python
"""
Loan Default Prediction — Training Pipeline
=============================================
Handles data loading, preprocessing, feature engineering, model training,
evaluation, and artifact saving.

Usage:
    python src/train.py          (run from loan_default_prediction/ directory)
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
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score, roc_auc_score, roc_curve,
    precision_recall_curve, average_precision_score,
)
from sklearn.model_selection import (
    cross_val_score, StratifiedKFold, RandomizedSearchCV,
)
from sklearn.pipeline import Pipeline, make_pipeline
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


# ── Helper functions ────────────────────────────────────────────────────
def divider(title: str) -> None:
    log.info(f"\n{'=' * 60}\n  {title}\n{'=' * 60}")

def save_figure(fig, name: str) -> None:
    path = FIGURES_DIR / f"{name}.png"
    fig.savefig(path, bbox_inches="tight", dpi=150)
    log.info(f"  📊 Saved figure: {path.name}")

def save_model(model, name: str) -> None:
    path = MODELS_DIR / f"{name}.pkl"
    joblib.dump(model, path)
    log.info(f"  💾 Saved model: {path.name}")


# ── Data loading & preprocessing functions go below ─────────────────────

def main():
    divider("LOAN DEFAULT PREDICTION — TRAINING PIPELINE")
    # TODO: Fill in subsequent steps
    pass

if __name__ == "__main__":
    main()
```

**Key design patterns:**

| Pattern | Why |
|---------|-----|
| `Path(__file__).resolve().parent.parent` | Finds project root relative to script location |
| `matplotlib.use("Agg")` | Saves figures without needing a display (works on servers) |
| `logging` instead of `print` | Professional practice; can redirect to file |
| `if __name__ == "__main__"` | Allows importing functions without running the script |

- [ ] Create `loan_default_prediction/src/train.py` with the scaffold above
- [ ] Run `python src/train.py` from `loan_default_prediction/` — should print the divider and exit cleanly

---

## ✅ TODO 3.2 — `load_data()` Function — Load & Filter

**Purpose:** Encapsulate all data loading logic in one function. This function reads
the CSV, binarizes the target, and returns a clean DataFrame.

**Add to `train.py`:**

```python
def load_data() -> pd.DataFrame:
    """Load accepted loans, filter to known outcomes, binarize target."""
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
```

**Functions to learn:**

| Function | Purpose |
|----------|---------|
| `pd.read_csv(path, low_memory=False)` | Read large CSV without mixed-type warnings |
| `df[col].isin([...])` | Boolean mask: True where value is in the list |
| `df[col].map({...})` | Replace values using a dictionary |
| `.copy()` | Avoid SettedWithCopyWarning by making an explicit copy |

- [ ] Add `load_data()` to train.py
- [ ] Call it in `main()`: `df = load_data()`
- [ ] Test: `python src/train.py` — should print row counts

---

## ✅ TODO 3.3 — `drop_columns()` Function — Remove Leakage, IDs, Junk

**Purpose:** Remove columns that would cause data leakage, columns that are useless
(IDs, URLs, free text), and columns with excessive missing data.

**Add to `train.py`:**

```python
def drop_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Remove leakage columns, IDs, free text, and high-missing columns."""
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
    # Don't drop the target!
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
```

**Why each group is dropped:**

| Group | Why | How to identify |
|-------|-----|-----------------|
| Leakage | Contains post-outcome info → model cheats | Domain knowledge: "would I know this at loan application time?" |
| IDs | Unique per row → no pattern to learn | Column like `id`, `member_id` |
| Free text | Too complex for tabular models (NLP needed) | String columns with high cardinality |
| High missing | >50% gaps → imputation would be mostly fictional | `df.isnull().mean() > 0.50` |
| Zero variance | Same value for all rows → carries no information | `df.nunique() <= 1` |

- [ ] Add `drop_columns()` to train.py
- [ ] Call in `main()`: `df = drop_columns(df)`
- [ ] Test: print remaining column count

---

## ✅ TODO 3.4 — `engineer_features()` Function — Create New Features

**Purpose:** Combine existing columns into new, more informative features. This is
where **domain knowledge** meets **data science**. Good features can matter more than
choosing the right algorithm.

**Add to `train.py`:**

```python
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create new features from existing columns using domain knowledge."""
    divider("3. FEATURE ENGINEERING")
    n_before = df.shape[1]

    # ── Parse term to numeric months ────────────────────────────────────
    # "36 months" → 36, " 60 months" → 60
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
    elif "issue_d" in df.columns:
        # drop raw date columns even if we can't compute the feature
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
```

**Key functions to learn for feature engineering:**

| Function | What It Does | Example |
|----------|-------------|---------|
| `str.extract(r"(\d+)")` | Extract a number pattern from string | "36 months" → "36" |
| `col.map({...})` | Map values via dictionary | "< 1 year" → 0 |
| `pd.to_datetime(col)` | Parse strings to datetime | "Jan-2015" → 2015-01-01 |
| `(date1 - date2).dt.days` | Difference in days between two dates | 3650 (= ~10 years) |
| `col.replace(0, np.nan)` | Prevents division by zero | 0 → NaN instead of infinity |

**Feature engineering ideas and their intuition:**

| New Feature | Formula | Why It Helps |
|-------------|---------|-------------|
| `term_months` | parse "36 months" → 36 | Numeric version of categorical term |
| `emp_years` | parse "10+ years" → 10 | Employment stability is a risk signal |
| `credit_history_years` | issue_date − earliest_credit_line | Longer history = more stable borrower |
| `income_to_loan` | annual_income / loan_amount | Can borrower "afford" this loan? |
| `installment_to_income` | monthly_payment / (annual_income/12) | Monthly burden ratio |

**Note:** `dti` (debt-to-income ratio) already exists in the dataset — no need to re-create it.

- [ ] Add `engineer_features()` to train.py
- [ ] Call in `main()`: `df = engineer_features(df)`
- [ ] Test: verify new columns are created and raw date/string columns are dropped

---

## ✅ TODO 3.5 — `build_preprocessor()` Function — Sklearn Pipeline

**Purpose:** Build a reusable preprocessing pipeline using scikit-learn's
`ColumnTransformer`. This is the **professional** way to preprocess data — it
ensures the same transformations are applied to train AND test data, preventing
data leakage from preprocessing.

**Why not just do `X.fillna(0)` and `pd.get_dummies()`?**
- Those operate on the **entire** dataset → information leaks from test into train.
- They don't produce a reusable object → can't apply to new data in production.
- `ColumnTransformer` fits on training data only, then transforms both train & test consistently.

**Add to `train.py`:**

```python
def build_preprocessor(X: pd.DataFrame):
    """Build a ColumnTransformer that handles numeric and categorical features."""
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
    # grade: A (best) → G (worst); sub_grade: A1 → G5
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
    preprocessor = ColumnTransformer([
        ("num", numeric_pipe, num_cols),
        ("ord", ordinal_pipe, ordinal_cols),
        ("nom", nominal_pipe, nominal_cols),
    ], remainder="drop")  # drop any column not listed → safeguard

    return preprocessor
```

**Pipeline building blocks explained:**

| Component | What It Does | When to Use |
|-----------|-------------|-------------|
| `SimpleImputer(strategy="median")` | Fills NaN with the median of that column | Numeric columns with missing data |
| `SimpleImputer(strategy="most_frequent")` | Fills NaN with the most common value | Categorical columns with missing data |
| `StandardScaler()` | Transforms to mean=0, std=1 | Required for Logistic Regression, optional for trees |
| `OrdinalEncoder` | Converts A→0, B→1, C→2... preserving order | Features with natural ranking (grade) |
| `OneHotEncoder` | Creates binary columns per category | Features without natural order (purpose, state) |
| `ColumnTransformer` | Applies different pipelines to different column groups | The glue that holds it all together |

**Why `Pipeline` and not manual transforms?**

```
❌ Manual approach:
   imputer.fit(X_train)         → X_train_imputed = imputer.transform(X_train)
   scaler.fit(X_train_imputed)  → X_train_scaled = scaler.transform(X_train_imputed)
   # Easy to mess up order, forget to transform test set, etc.

✅ Pipeline approach:
   pipe.fit(X_train)     → pipe.transform(X_test)
   # One call does everything in the right order. Can't mess up.
```

- [ ] Add `build_preprocessor()` to train.py
- [ ] Read it carefully and understand each sub-pipeline
- [ ] Note: `max_categories=15` in OneHotEncoder caps the number of dummy columns per feature

---

## ✅ TODO 3.6 — Train/Test Split

**Purpose:** Split data into a training set (model learns from this) and a test set
(model is evaluated on this — never seen during training).

**Why time-based split?** In the real world, you'd train on historical loans and
predict future loans. A random split lets future data "leak" into training.

**Add to `train.py`:**

```python
def split_data(df: pd.DataFrame):
    """Split into features (X) and target (y), then train/test."""
    divider("4. TRAIN / TEST SPLIT")

    y = df[TARGET_COL]
    X = df.drop(columns=[TARGET_COL])

    # ── Option A: Time-based split (more realistic) ─────────────────────
    # If credit_history_years was derived from issue_d, we've already dropped
    # the raw date. Use a stratified random split instead.

    # ── Option B: Stratified random split ───────────────────────────────
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.20,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    log.info(f"  Train: {X_train.shape[0]:,} rows ({y_train.mean():.2%} default)")
    log.info(f"  Test:  {X_test.shape[0]:,} rows ({y_test.mean():.2%} default)")

    return X_train, X_test, y_train, y_test
```

**Key functions:**

| Function | What It Does |
|----------|-------------|
| `train_test_split(X, y, test_size=0.20)` | 80% train, 20% test |
| `stratify=y` | Ensures same default rate in train and test |
| `random_state=42` | Makes the split reproducible |

- [ ] Add `split_data()` to train.py
- [ ] Call in `main()`: `X_train, X_test, y_train, y_test = split_data(df)`
- [ ] Verify train and test have the same default rate (prints should match)

---

## 🎯 Phase 3 Checklist

When all TODOs above are done, you should have:

- [ ] `src/train.py` created with proper scaffold
- [ ] `load_data()` → loads CSV, binarizes target
- [ ] `drop_columns()` → removes leakage, IDs, high-missing, zero-variance
- [ ] `engineer_features()` → creates term_months, emp_years, credit_history_years, income_to_loan, installment_to_income
- [ ] `build_preprocessor()` → sklearn ColumnTransformer with numeric/ordinal/nominal pipelines
- [ ] `split_data()` → stratified 80/20 split
- [ ] Script runs end-to-end without error
- [ ] Ready for [Phase 4 — Model Training](phase4_model_training.md)!

---

## 📚 Concepts to Remember

| Concept | Explanation |
|---------|-------------|
| **Feature engineering** | Creating new columns from existing data using domain knowledge |
| **ColumnTransformer** | Applies different preprocessing to different column groups |
| **Pipeline** | Chains multiple steps (impute → scale → encode) into one object |
| **Ordinal encoding** | Preserves natural order: A=0, B=1, C=2 (for grades) |
| **One-Hot encoding** | Creates binary columns: [RENT, OWN, MORTGAGE] → [1,0,0], [0,1,0], [0,0,1] |
| **StandardScaler** | $z = \frac{x - \mu}{\sigma}$ — centers to mean=0, std=1 |
| **Data leakage from preprocessing** | If you fit imputer on full data then split → test set influenced the imputation |
| **`remainder="drop"`** | ColumnTransformer drops any column not explicitly listed — safety net |
