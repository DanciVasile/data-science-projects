# 🎨 Phase 8 — Documentation & Polish

> **Where:** `loan_default_prediction/README.md`, `docs/data-dictionary.md`  
> **Prereq:** [Phase 7 — Gradio](phase7_gradio.md) completed  
> **Next:** You're done! 🎉

---

## 🧒 What Are We Doing Here? (The Big Picture)

The project works. Now make it **presentable**. A recruiter visiting your GitHub
spends ~30 seconds scanning the README before deciding if your project is worth
a deeper look. A clear, well-structured README with results, figures, and a live
demo link is what separates "I followed a tutorial" from "I built this."

---

## ✅ TODO 8.1 — Data Dictionary

**Purpose:** Document what each column means. This shows *domain understanding* —
you didn't just throw data at a model; you understood the business context.

**Create `loan_default_prediction/docs/data-dictionary.md`:**

```markdown
# Loan Default Prediction — Data Dictionary

**Source:** LendingClub accepted loans (2007–2018 Q4)

## Key Features Used in the Model

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| `loan_amnt` | numeric | Loan amount requested by the borrower | 15000 |
| `term` | categorical | Loan term in months | "36 months", "60 months" |
| `int_rate` | numeric | Interest rate (%) | 12.5 |
| `installment` | numeric | Monthly payment amount ($) | 432.56 |
| `grade` | ordinal | Loan grade assigned by LendingClub (A=best → G=worst) | "C" |
| `sub_grade` | ordinal | Loan sub-grade (A1–G5) | "C3" |
| `emp_length` | categorical | Employment length | "10+ years", "< 1 year" |
| `home_ownership` | categorical | Borrower's housing status | "RENT", "OWN", "MORTGAGE" |
| `annual_inc` | numeric | Self-reported annual income ($) | 65000 |
| `verification_status` | categorical | Whether income was verified by LendingClub | "Verified", "Not Verified" |
| `purpose` | categorical | Reason for the loan | "debt_consolidation" |
| `dti` | numeric | Debt-to-income ratio (%) — existing debt payments / income | 15.3 |
| `earliest_cr_line` | date | Date of borrower's earliest credit line | "Jan-1990" |
| `open_acc` | numeric | Number of open credit lines | 12 |
| `pub_rec` | numeric | Number of derogatory public records | 0 |
| `revol_bal` | numeric | Total revolving balance ($) | 8532 |
| `revol_util` | numeric | Revolving credit utilization (%) | 45.2 |
| `total_acc` | numeric | Total number of credit lines | 25 |
| `issue_d` | date | Date the loan was issued | "Jan-2015" |

## Target Variable

| Column | Type | Description |
|--------|------|-------------|
| `loan_status` | binary | **0** = Fully Paid, **1** = Charged Off (default) |

## Engineered Features

| Column | Formula | Purpose |
|--------|---------|---------|
| `term_months` | parse "36 months" → 36 | Numeric version of term |
| `emp_years` | parse "10+ years" → 10 | Numeric employment length |
| `credit_history_years` | (issue_date − earliest_credit_line) / 365.25 | Borrower's credit history length |
| `income_to_loan` | annual_inc / loan_amnt | Can borrower afford this loan? |
| `installment_to_income` | installment / (annual_inc / 12) | Monthly payment burden |

## Dropped Columns (and Why)

| Category | Columns | Reason |
|----------|---------|--------|
| Leakage | total_pymnt, recoveries, last_pymnt_d, ... | Post-origination data — model would "cheat" |
| IDs | id, member_id, url | Unique identifiers — no predictive value |
| Free text | desc, emp_title, title | Requires NLP — out of scope for tabular model |
| High missing | (columns with >50% null) | Too many gaps to impute reliably |
```

- [ ] Create `docs/data-dictionary.md`
- [ ] Verify all features used in the model are documented
- [ ] Verify engineered features are explained

---

## ✅ TODO 8.2 — Write the Project README

**Purpose:** The "front page" of your project. Follow the pattern from
`house_price_prediction/README.md` but adapt for classification.

**Overwrite `loan_default_prediction/README.md` with:**

```markdown
# 🏦 Loan Default Prediction

> Binary classification model predicting whether a LendingClub loan will
> **default** (Charged Off) or be **Fully Paid**, using borrower and loan
> characteristics available at origination time.

---

## 📁 Project Structure

```
loan_default_prediction/
├── app.py                  # Gradio dashboard for interactive predictions
├── data/
│   └── accepted_2007_to_2018Q4.csv.gz
├── docs/
│   └── data-dictionary.md  # Column definitions and feature documentation
├── models/
│   ├── xgboost_tuned.pkl   # Best trained model (Pipeline: preprocessor + XGBoost)
│   └── results.csv         # Model comparison metrics
├── notebooks/
│   └── exploration.ipynb   # Exploratory Data Analysis
├── reports/
│   └── figures/            # ROC curve, confusion matrix, feature importances
├── src/
│   └── train.py            # Reproducible training pipeline
└── todos/                  # Learning guide (phase-by-phase TODOs)
```

## 📊 Dataset

- **Source:** [LendingClub](https://www.lendingclub.com/) accepted loans (2007–2018 Q4)
- **Size:** ~2.2M loans × 150 columns (raw), filtered to ~X rows with known outcomes
- **Target:** `loan_status` → 0 (Fully Paid) / 1 (Charged Off)
- **Class balance:** ~80% Fully Paid / ~20% Charged Off

## 🔬 Workflow

### 1. Exploratory Data Analysis (`notebooks/exploration.ipynb`)
- Visualized class imbalance, feature distributions, and temporal trends
- Identified and flagged **data leakage** columns (post-origination features)
- Analyzed default rates by grade, term, purpose, home ownership
- Mapped missing data and determined drop/impute thresholds

### 2. Feature Engineering & Preprocessing (`src/train.py`)
- Dropped leakage, ID, free-text, and high-missing columns
- Engineered 5 new features from domain knowledge
- Built sklearn Pipeline with ColumnTransformer:
  - Numeric: median imputation → standard scaling
  - Ordinal (grade): ordinal encoding (A=0 → G=6)
  - Nominal: mode imputation → one-hot encoding

### 3. Model Training
- **Logistic Regression** — baseline with `class_weight="balanced"`
- **Random Forest** — 200 trees, balanced class weights
- **XGBoost** — gradient boosting with `scale_pos_weight`
- All models evaluated with 5-fold stratified cross-validation

### 4. Hyperparameter Tuning
- RandomizedSearchCV (50 iterations × 3 folds) on XGBoost
- Tuned: max_depth, learning_rate, n_estimators, subsample, colsample_bytree

### 5. Deployment
- Interactive Gradio dashboard with risk assessment UI

## 📈 Results

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|-------|----------|-----------|--------|-----|---------|
| Logistic Regression | ... | ... | ... | ... | ... |
| Random Forest | ... | ... | ... | ... | ... |
| XGBoost (Default) | ... | ... | ... | ... | ... |
| **XGBoost (Tuned)** | **...** | **...** | **...** | **...** | **...** |

> Fill in actual numbers from `models/results.csv` after training.

## 🛠️ Tech Stack

- **Language:** Python 3.14
- **ML:** scikit-learn, XGBoost
- **Data:** pandas, NumPy
- **Visualization:** matplotlib, seaborn
- **Dashboard:** Gradio
- **Environment:** uv (dependency management)

## 🚀 Getting Started

```bash
# Clone and setup
git clone https://github.com/DanciVasile/data-science-projects.git
cd data-science-projects
./init.ps1  # creates .venv, installs dependencies, registers kernel

# Run EDA notebook
cd loan_default_prediction
jupyter notebook notebooks/exploration.ipynb

# Train models
python src/train.py

# Launch dashboard
python app.py
```

## 📝 Key Decisions

1. **Accepted loans only** — rejected loans lack outcome data and have different features
2. **Fully Paid vs Charged Off** — dropped in-progress statuses for clean binary signal
3. **Class weighting over SMOTE** — with 400k+ defaults, synthetic oversampling is unnecessary
4. **Stratified random split** — 80% train / 20% test with preserved class proportions
5. **Leakage prevention** — removed all post-origination features to ensure realistic evaluation
```

- [ ] Write the README using the template above
- [ ] Fill in actual metric numbers from `models/results.csv`
- [ ] Add a Gradio screenshot if possible
- [ ] Verify all file paths in the structure diagram are accurate

---

## ✅ TODO 8.3 — Update the Root README

**Purpose:** Mark the loan default project as "Done" in the main repo README.

**In the root `README.md`, update the project table:**

Change:
```
| 2 | [💳 Loan Default Prediction](loan_default_prediction/) | Binary classification on loan repayment data. | 🔜 Coming soon |
```
To:
```
| 2 | [💳 Loan Default Prediction](loan_default_prediction/) | Binary classification on LendingClub data. XGBoost + Gradio dashboard. | ✅ Complete |
```

- [ ] Update the root README
- [ ] Verify the link works

---

## ✅ TODO 8.4 — Final Verification Checklist

**Purpose:** Run through everything one more time to make sure nothing is broken.

**File checklist:**

| File | Exists? | Runs? |
|------|---------|-------|
| `notebooks/exploration.ipynb` | [ ] | [ ] Restart & Run All without errors |
| `src/train.py` | [ ] | [ ] `python src/train.py` completes |
| `app.py` | [ ] | [ ] `python app.py` loads |
| `models/xgboost_tuned.pkl` | [ ] | [ ] Generated by train.py |
| `models/results.csv` | [ ] | [ ] Generated by train.py |
| `reports/figures/roc_curves.png` | [ ] | [ ] Generated by train.py |
| `reports/figures/confusion_matrix_*.png` | [ ] | [ ] Generated by train.py |
| `reports/figures/feature_importances_*.png` | [ ] | [ ] Generated by train.py |
| `docs/data-dictionary.md` | [ ] | N/A (static file) |
| `README.md` | [ ] | N/A (static file) |

**Code quality:**

- [ ] No hardcoded absolute paths (all use `Path(__file__).resolve()`)
- [ ] No leakage columns in the feature set
- [ ] `random_state=42` used everywhere for reproducibility
- [ ] Comments explain the "why", not just the "what"
- [ ] Notebook runs top-to-bottom without manual intervention

**Git:**

- [ ] `git add .`
- [ ] `git commit -m "feat: loan default prediction — EDA, training, evaluation, Gradio dashboard"`
- [ ] `git push`

---

## 🎯 Phase 8 Checklist

When all TODOs above are done, you should have:

- [ ] `docs/data-dictionary.md` — all features documented
- [ ] `README.md` — professional project description with results
- [ ] Root README updated
- [ ] All files verified to exist and run
- [ ] Code pushed to GitHub
- [ ] **PROJECT COMPLETE!** 🎉

---

## 📚 Concepts to Remember

| Concept | Explanation |
|---------|-------------|
| **README.md** | The first file a recruiter reads — it's your project's resume |
| **Data dictionary** | Documents what each column means — shows domain expertise |
| **Reproducibility** | Anyone can clone your repo, run `python src/train.py`, and get the same results |
| **End-to-end** | From raw data → EDA → model → deployment → documentation |
| **Portfolio project** | Should demonstrate technical skill, domain knowledge, and communication ability |
