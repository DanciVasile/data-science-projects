# 🏠 House Price Prediction

## 📌 Overview

This project tackles a classic regression problem — **predicting house sale prices** — using the well-known **Ames Housing dataset**. It walks through the full data science workflow: from data cleaning and exploratory analysis to building, evaluating, and comparing machine learning models.

## 📂 Project Structure

```
house-price-prediction/
├── main.py                        # Full pipeline (EDA → Preprocessing → Modeling)
├── README.md
└── dataset/
    ├── ames-housing.csv           # Ames Housing dataset (~2,930 observations, 82 features)
    └── data-categories.txt        # Feature descriptions and category mappings
```

## 📊 Dataset

The **Ames Housing dataset** contains **2,930 residential property sales** from Ames, Iowa, with **82 features** describing nearly every aspect of a home:

| Feature Category | Examples |
|---|---|
| 🏗️ Structure | Building type, house style, year built, overall quality |
| 📐 Size | Lot area, living area, basement SF, garage area |
| 🛏️ Rooms | Bedrooms, bathrooms, kitchen quality, total rooms |
| 🌳 Exterior | Roof style, exterior material, porch/deck area, pool |
| 📍 Location | Neighborhood, zoning, lot configuration |
| 💰 Sale Info | Sale type, sale condition, **Sale Price (target)** |

## 🔍 Workflow

### 1. Data Cleaning 🧹
- Standardized column names (stripped whitespace, removed spaces)
- Removed irrelevant columns (`Order`, `PID`)
- Cleaned whitespace from categorical feature values

### 2. Exploratory Data Analysis 📈
- **Target distribution** — visualized `SalePrice` distribution (right-skewed)
- **Missing values audit** — identified features with missing data
- **Categorical analysis** — counted unique values and plotted distributions
- **Correlation analysis** — computed Pearson correlation of all numeric features with `SalePrice`
- **Feature histograms** — plotted distributions for all numeric and categorical features

### 3. Feature Engineering ⚙️
- **Log transformation** on the target variable (`log1p`) to reduce skewness and stabilize model training
- **Numeric imputation** — filled missing values using the **median** strategy
- **Categorical encoding** — applied **most-frequent imputation** followed by **One-Hot Encoding**
- Built a unified `ColumnTransformer` preprocessing pipeline

### 4. Modeling 🤖
Two regression models were trained and evaluated using **5-fold cross-validation** with R² scoring:

| Model | Target | Strategy |
|---|---|---|
| **Linear Regression** | Raw `SalePrice` | Baseline model |
| **Random Forest Regressor** | Log-transformed `SalePrice` | 200 estimators, parallelized (`n_jobs=-1`) |

### 5. Evaluation & Results 📉
- Cross-validated **R² scores** were compared across both models
- Predictions from the log-transformed model were **inverse-transformed** (`expm1`) back to the original dollar scale

## ✅ Key Takeaways

- 🌲 **Random Forest > Linear Regression** for this dataset — it better handles non-linear relationships and high-cardinality categorical features
- 📉 **Log-transforming the target** helped stabilize training and improve consistency across folds
- 🔄 **Cross-validation** confirmed Random Forest delivers more robust, consistent performance with fewer drastic dips

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| **pandas** | Data loading & manipulation |
| **NumPy** | Numerical operations & log transform |
| **Matplotlib** | Data visualization |
| **scikit-learn** | Preprocessing, pipelines & models |


