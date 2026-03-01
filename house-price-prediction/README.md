# 🏠 House Price Prediction

> Predicting residential home sale prices in Ames, Iowa using machine learning regression models.

![Python](https://img.shields.io/badge/Python-3.14-blue?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.8.0-orange?logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-3.0.1-purple?logo=pandas&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-3.10.8-green?logo=matplotlib&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-blue)

---

## 📌 Overview

This project tackles a classic regression problem — **predicting house sale prices** — using the well-known **Ames Housing dataset**. It walks through the full machine learning workflow: from data cleaning and exploratory analysis to building, evaluating, and comparing regression models.

The implementation emphasizes **production-ready code** with clean logging, modular design, and reproducibility.

---

## 📂 Project Structure

```
house-price-prediction/
├── README.md                          # This file
├── pyproject.toml                     # uv project configuration
├── data/
│   └── ames-housing.csv               # Ames Housing dataset (2,930 samples, 82 features)
├── docs/
│   └── data-dictionary.md             # Feature descriptions & metadata
├── models/                            # Trained model artifacts (.pkl) — auto-generated
│   ├── linear_regression.pkl
│   └── random_forest.pkl
├── notebooks/
│   └── exploration.ipynb              # Interactive EDA & experimentation
├── reports/
│   └── figures/                       # Publication-quality plots (.pdf)
│       ├── 01_sale_price_distribution.pdf
│       ├── 02_log_sale_price_distribution.pdf
│       ├── 03_numeric_features_distribution.pdf
│       └── 04_categorical_features_distribution.pdf
└── src/
    └── train.py                       # Main training script with logging
```

### Directory Guide

| Path | Purpose |
|------|---------|
| `notebooks/` | 📓 Exploration & storytelling with interactive plots |
| `src/train.py` | 🏭 Production script—clean, logged, reproducible |
| `models/` | 💾 Serialized sklearn pipelines (gitignored) |
| `reports/figures/` | 📊 Auto-generated visualizations as PDFs |
| `docs/` | 📖 Data dictionary & documentation |

---

## 📊 Dataset Overview

The **Ames Housing dataset** contains **2,930 residential property sales** from Ames, Iowa (2006–2010) with **82 features** describing nearly every aspect of a home:

### Feature Categories

| Category | Examples |
|----------|----------|
| 🏗️ **Structure** | Building type, house style, year built, overall quality/condition |
| 📐 **Size** | Lot area, living area, basement area, garage area |
| 🛏️ **Interior** | Bedrooms, bathrooms, kitchen quality, flooring |
| 🌳 **Exterior** | Roof style/material, siding, porch/deck area, pool |
| 📍 **Location** | Neighborhood, zoning, lot shape & configuration |
| 💰 **Sale Info** | Sale type, sale condition, **sale price (target)** |

**Target Variable:** `SalePrice` (continuous, right-skewed, range: $34.9K–$755K)

---

## 🔄 Workflow & Methods

### 1️⃣ Data Cleaning 🧹
- ✓ Standardized column names (stripped whitespace, removed spaces)
- ✓ Removed irrelevant columns (`Order`, `PID`)
- ✓ Cleaned whitespace from categorical values
- ✓ Analyzed missing values per feature

### 2️⃣ Exploratory Data Analysis 📈
- **Target analysis** — visualized `SalePrice` distribution (right-skewed)
- **Missing value audit** — identified and logged features with missing data
- **Categorical overview** — counted unique values for all categorical features
- **Correlation study** — computed Pearson correlation of numeric features with target
- **Feature distributions** — generated histograms for all numeric and categorical features

### 3️⃣ Preprocessing Pipeline ⚙️

**Numeric Features:**
- Strategy: **Median imputation** for missing values
- Rationale: Robust to outliers, preserves distribution

**Categorical Features:**
- Strategy: **Most-frequent imputation** → **One-Hot Encoding**
- Rationale: Handles missing categories, enables linear models to use categorical data

**Target Variable:**
- Applied `log1p()` transformation to reduce right-skewness
- Helps stabilize model training and improve convergence

### 4️⃣ Model Training 🤖

Two regression models trained using **5-fold cross-validation** with R² scoring:

#### Linear Regression (Baseline)
- **Model:** Vanilla linear regression
- **Target:** Raw `SalePrice`
- **Use:** Baseline for comparison
- **Pros:** Interpretable, fast
- **Cons:** Assumes linear relationships

#### Random Forest Regressor (Best)
- **Model:** 200 decision trees, parallelized (`n_jobs=-1`)
- **Target:** Log-transformed `SalePrice`
- **Predictions:** Inverse-transformed back to original scale
- **Pros:** Handles non-linearity, robust to outliers, feature interactions
- **Cons:** Less interpretable, slower inference

### 5️⃣ Evaluation & Results 📊

```
Linear Regression:  R² = 0.8592 (±0.0507)
Random Forest:      R² = 0.8815 (±0.0149)  ⭐ WINNER
```

**Key Findings:**
- 🌲 **Random Forest outperforms** Linear Regression on this dataset
- 📉 **Log-transformation** improved stability across folds
- 🎯 Random Forest shows **higher consistency** (lower std dev)
- 💪 Both models exceed R² = 0.85, indicating strong predictive power

---

## 📝 Logging & Output

The training script uses **Python's logging module** for professional, structured output with emoji indicators:

```python
# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s"
)
logger = logging.getLogger(__name__)
```

### Output Features

- 🎨 **Emoji-enhanced** section headers for visual clarity
- 📊 **Performance metrics** with auto-generated quality indicators:
  - 🔥 **Excellent** (R² ≥ 0.95)
  - ⭐ **Great** (R² ≥ 0.90)
  - 👍 **Good** (R² ≥ 0.80)
  - 📊 **Okay** (R² ≥ 0.70)
  - ⚠️ **Poor** (R² < 0.70)
- 📁 **Clear file confirmations** when saving models and figures
- 🏆 **Automatic model comparison** with winner announcement

### Example Console Output

```
🏠 AMES HOUSING PRICE PREDICTION
===========================================================================

===========================================================================
  🔍 LOADING DATA
===========================================================================
  ✓ Loaded 2,930 samples, 79 features
  ✓ Target variable: SalePrice (mean: $180,921)

===========================================================================
  📈 SAVING EDA FIGURES
===========================================================================
  📊 Saved reports/figures/01_sale_price_distribution.pdf
  📊 Saved reports/figures/02_log_sale_price_distribution.pdf
  📊 Saved reports/figures/03_numeric_features_distribution.pdf
  📊 Saved reports/figures/04_categorical_features_distribution.pdf
  ✓ Generated 4 exploratory figures

===========================================================================
  🔗 LINEAR REGRESSION
===========================================================================

  👍 Linear Regression Performance:
     Mean R²:     0.8592
     Std Dev:     0.0507
     Min R²:      0.7814
     Max R²:      0.9090
     Assessment: Good fit! ✓

  💾 Saved models/linear_regression.pkl

===========================================================================
  🌲 RANDOM FOREST (log-transformed target)
===========================================================================

  ⭐ Random Forest Performance:
     Mean R²:     0.8815
     Std Dev:     0.0149
     Min R²:      0.8684
     Max R²:      0.9028
     Assessment: Great performance! 🚀

  🔮 Sample predictions (original scale):
     ['$203,779', '$112,103', '$165,716', '$255,745', '$188,901']

  💾 Saved models/random_forest.pkl

===========================================================================
  🏆 MODEL COMPARISON
===========================================================================
  Linear Regression:  0.8592 (±0.0507)
  Random Forest:      0.8815 (±0.0149)

  🎯 Winner: Random Forest by 0.0223

===========================================================================
  ✅ TRAINING COMPLETE
===========================================================================
  📁 Output Locations:
     • Models:  models/
     • Figures: reports/figures/
```

---

## 🛠️ Tech Stack

| Tool | Version | Purpose |
|------|---------|---------|
| **Python** | 3.10+ | Programming language |
| **pandas** | 2.0+ | Data loading & manipulation |
| **NumPy** | 1.24+ | Numerical operations |
| **scikit-learn** | 1.5+ | ML models & preprocessing |
| **Matplotlib** | 3.8+ | Data visualization |
| **joblib** | 1.3+ | Model serialization |

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- [uv](https://docs.astral.sh/uv/) package manager (recommended)
- Git

### Installation & Running

```bash
# Clone the repository
git clone https://github.com/DanciVasile/data-science-projects.git
cd data-science-projects

# Install dependencies
uv sync

# Navigate to project
cd house-price-prediction

# Run training script
python src/train.py
```

**Output:**
- ✅ Trained models saved to `models/`
- ✅ Visualizations saved to `reports/figures/`
- ✅ Detailed logs printed to console

### Interactive Exploration

```bash
# Open Jupyter notebook for EDA
jupyter notebook notebooks/exploration.ipynb
```

---

## 📈 Results Summary

| Metric | Linear Regression | Random Forest |
|--------|-------------------|---------------|
| **Mean R²** | 0.8592 | 0.8815 ⭐ |
| **Std Dev** | 0.0507 | 0.0149 |
| **Min R²** | 0.7814 | 0.8684 |
| **Max R²** | 0.9090 | 0.9028 |
| **Consistency** | Good | Excellent |

**Conclusion:** Random Forest is the recommended model for this dataset due to superior performance, robustness, and consistency.

---

## 💡 Key Insights

- 🌲 **Ensemble methods** outperform simple linear models on complex, non-linear datasets
- 📉 **Target transformation** (log-scaling) improves model stability and generalization
- 🔄 **Cross-validation** is essential for reliable performance estimates
- 📊 **Professional logging** enables reproducibility and production-ready ML pipelines

---

## 📚 Files & Outputs

### Generated During Training

| File | Purpose |
|------|---------|
| `models/linear_regression.pkl` | Serialized Linear Regression pipeline |
| `models/random_forest.pkl` | Serialized Random Forest pipeline |
| `reports/figures/01_sale_price_distribution.pdf` | Sale price distribution histogram |
| `reports/figures/02_log_sale_price_distribution.pdf` | Log-transformed distribution histogram |
| `reports/figures/03_numeric_features_distribution.pdf` | All numeric features histograms |
| `reports/figures/04_categorical_features_distribution.pdf` | All categorical features bar charts |

---

## 🔗 Related

- **Dataset Source:** [Kaggle Ames Housing](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)
- **Parent Repository:** [`data-science-projects`](../../README.md)

---

## 📄 License

This project is licensed under the **MIT License** — see LICENSE file for details.

---

<p align="center">
  Made with ❤️ as part of my Data Science portfolio
  <br/>
  <a href="https://github.com/DanciVasile">GitHub</a> • 
  <a href="https://www.linkedin.com/in/vasile-danci-m/">LinkedIn</a>
</p>