# Loan Default Prediction — Data Dictionary

**Source:** LendingClub accepted loans (2007–2018 Q4)  
**Download:** [Kaggle — LendingClub Accepted Loans](https://www.kaggle.com/datasets/wordsforthewise/lending-club)

---

## Key Features Used in the Model

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| `loan_amnt` | numeric | Loan amount requested by the borrower | 15000 |
| `term` | categorical | Loan term in months | " 36 months", " 60 months" |
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
| `fico_range_low` | numeric | Lower bound of borrower's FICO score at origination | 670 |
| `fico_range_high` | numeric | Upper bound of borrower's FICO score at origination | 674 |
| `issue_d` | date | Date the loan was issued | "Jan-2015" |

---

## Target Variable

| Column | Type | Values | Description |
|--------|------|--------|-------------|
| `loan_status` | binary | 0 = Fully Paid, 1 = Charged Off | Whether the borrower defaulted |

**Original values (before binarization):**
- "Fully Paid" → **0** (good outcome)
- "Charged Off" → **1** (default — the event we predict)
- "Current", "Late", "In Grace Period", "Default" → **dropped** (no definitive outcome)

---

## Engineered Features

Created in `src/train.py → engineer_features()`:

| Column | Formula | Purpose |
|--------|---------|---------|
| `term_months` | parse " 36 months" → 36 | Numeric version of term (replaces string column) |
| `emp_years` | parse "10+ years" → 10, "< 1 year" → 0 | Numeric employment length |
| `credit_history_years` | (issue_date − earliest_credit_line) / 365.25 | Borrower's credit history length in years |
| `income_to_loan` | annual_inc / loan_amnt | Can the borrower afford this loan? Higher = safer |
| `installment_to_income` | installment / (annual_inc / 12) | Monthly payment burden ratio |

---

## Dropped Columns (and Why)

| Category | Example Columns | Reason |
|----------|----------------|--------|
| **Leakage** | total_pymnt, total_rec_prncp, recoveries, last_pymnt_amnt, funded_amnt, out_prncp, last_fico_range_high/low | Post-origination data — the model would "cheat" by using information not available at loan application time |
| **IDs** | id, member_id, url | Unique identifiers — no predictive value |
| **Free text** | desc, emp_title, title | Requires NLP — out of scope for tabular model |
| **High missing (>50%)** | mths_since_last_delinq, mths_since_last_record, annual_inc_joint, dti_joint, etc. | Too many gaps to impute reliably |
| **Zero variance** | (columns with only 1 unique value) | Carry no information |

---

## Preprocessing Pipeline

Built in `src/train.py → build_preprocessor()`:

| Feature Type | Pipeline | Details |
|-------------|----------|---------|
| **Numeric** (59 cols) | `SimpleImputer(median)` → `StandardScaler()` | Fill missing with median, scale to mean=0, std=1 |
| **Ordinal** (2 cols: grade, sub_grade) | `SimpleImputer(mode)` → `OrdinalEncoder()` | A=0, B=1, ..., G=6; preserves natural order |
| **Nominal** (8 cols) | `SimpleImputer(mode)` → `OneHotEncoder(max_categories=15)` | Binary columns per category, capped at 15 |

---

## Dataset Statistics

| Metric | Value |
|--------|-------|
| Raw rows | ~2,260,701 |
| Raw columns | 151 |
| After filtering (Fully Paid + Charged Off only) | ~1,345,310 |
| After column dropping | ~69 columns |
| After feature engineering | ~70 columns |
| Default rate | ~19.96% |
| Train set (80%) | ~1,076,248 rows |
| Test set (20%) | ~269,062 rows |
