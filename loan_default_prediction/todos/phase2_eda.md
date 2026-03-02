# 🎨 Phase 2 — Exploratory Data Analysis (EDA)

> **Where:** `notebooks/exploration.ipynb` — Sections 4 through 10  
> **Prereq:** [Phase 1 — Config & Loading](phase1_config_and_loading.md) completed  
> **Next:** [Phase 3 — Feature Engineering](phase3_feature_engineering.md)

---

## 🧒 What Are We Doing Here? (The Big Picture)

You've loaded 2 million loan records. Before building any model, you need to
**understand** the data — like a detective studying the crime scene before forming
a theory. EDA answers questions like:

- How many loans default? (class balance)
- Which features differ most between defaulters and non-defaulters?
- How much data is missing, and where?
- Are there features that would "cheat" (leak future info)?
- Are there patterns over time?

We use `df_sample` (200k rows) for plots to keep things fast, and `df` (full data)
for aggregated statistics that don't require plotting.

---

## ✅ TODO 2.1 — Target Variable Distribution

**Purpose:** See how balanced/imbalanced the classes are. This affects **everything**:
which metrics to use, whether to apply class weights, how to interpret accuracy.

**What to do:** Run the existing Target Variable cells (Section 4). Since you set
`TASK = "classification"`, the conditional branch will produce bar charts.

**But first — update Cell 13 (X/y split) to work with our binarized target:**

```python
# Separate features (X) and target (y)
X = df_sample.drop(columns=[TARGET_COL])
y = df_sample[TARGET_COL]

print(f"Features shape: {X.shape}")
print(f"Target: '{TARGET_COL}'  |  dtype: {y.dtype}")
print(f"Class distribution:\n{y.value_counts().to_string()}")
print(f"Default rate: {y.mean():.2%}")
```

**What to look for:**
- **Class imbalance ratio** — if it's 80/20, that's moderate imbalance.
  A model that always predicts "Fully Paid" would be 80% accurate but useless.
- This is why we'll use **precision, recall, F1, ROC-AUC** instead of just accuracy.

**Key functions:**

| Function | What It Does |
|----------|-------------|
| `y.value_counts()` | Count per class |
| `y.value_counts(normalize=True)` | Proportion per class (sums to 1.0) |
| `y.mean()` | For binary 0/1, this equals the proportion of 1s (default rate) |

- [ ] Update the X/y split cell to use `df_sample`
- [ ] Run the target distribution cell (bar charts)
- [ ] Note down: exact default rate percentage
- [ ] Note down: exact count of each class

---

## ✅ TODO 2.2 — Missing Values Analysis

**Purpose:** Know which columns are missing data, how much, and decide what to do
about it (drop column, impute, or leave for the pipeline to handle).

**What to do:** Run the existing Missing Values cell (Section 5, Cell 16).

**But add a visual heatmap too — create a NEW cell:**

```python
# ── Missing values heatmap ──────────────────────────────────────────────
missing = df_sample.isnull().mean().sort_values(ascending=False)
missing_cols = missing[missing > 0]

# Group by severity
high_missing   = missing_cols[missing_cols > 0.50]  # >50% missing → DROP
medium_missing = missing_cols[(missing_cols > 0.05) & (missing_cols <= 0.50)]
low_missing    = missing_cols[missing_cols <= 0.05]

print(f"🔴 High missing (>50% → drop):     {len(high_missing)} columns")
print(f"🟡 Medium missing (5-50% → impute): {len(medium_missing)} columns")
print(f"🟢 Low missing (<5% → minor):       {len(low_missing)} columns")

if not high_missing.empty:
    print(f"\n🔴 Columns to DROP (>50% missing):")
    print(high_missing.to_string())

# Visual: top 30 columns by missing %
fig, ax = plt.subplots(figsize=(10, 8))
missing_cols.head(30).plot(kind="barh", ax=ax, color="salmon", edgecolor="black")
ax.set_xlabel("Fraction Missing")
ax.set_title("Top 30 Columns by Missing Data")
ax.invert_yaxis()
plt.tight_layout()
plt.show()
```

**Decision framework for missing data:**

| Missing % | Action | Why |
|-----------|--------|-----|
| >50% | Drop the column | More gaps than data — imputing would be mostly fiction |
| 5–50% | Impute (median for numeric, mode for categorical) | Enough data to estimate reasonable fill values |
| <5% | Impute or let the pipeline handle it | Minor impact |
| 0% | Nothing to do | 🎉 |

- [ ] Run the existing missing values cell
- [ ] Add and run the heatmap cell above
- [ ] Write down which columns have >50% missing (these will be dropped in Phase 3)
- [ ] Write down columns with 5-50% missing (these need imputation strategy)

---

## ✅ TODO 2.3 — Identify and Flag Leakage Columns

**Purpose:** This is the **#1 mistake** in loan default projects. Some columns contain
information that's only available **after** the loan outcome is known. Using them would
be like predicting yesterday's weather using today's newspaper.

**Add a NEW cell for leakage detection:**

```python
# ── Leakage detection ───────────────────────────────────────────────────
# These columns contain post-origination information — they wouldn't be
# available at the time a loan application is submitted.
LEAKAGE_COLS = [
    # Payment history (only known after payments start)
    "total_pymnt", "total_pymnt_inv", "total_rec_prncp", "total_rec_int",
    "total_rec_late_fee", "last_pymnt_d", "last_pymnt_amnt",
    # Recovery (only known after charge-off)
    "recoveries", "collection_recovery_fee",
    # Post-funding
    "funded_amnt", "funded_amnt_inv", "out_prncp", "out_prncp_inv",
    # Credit pulls after origination
    "last_credit_pull_d", "last_fico_range_high", "last_fico_range_low",
    # Policy code & hardship (system/post-origination)
    "hardship_flag", "debt_settlement_flag", "settlement_status",
    "settlement_date", "settlement_amount", "settlement_percentage",
    "settlement_term", "payment_plan_start_date",
]

# Check which ones actually exist in our dataset
found_leakage = [c for c in LEAKAGE_COLS if c in X.columns]
not_found = [c for c in LEAKAGE_COLS if c not in X.columns]

print(f"⚠️  Leakage columns FOUND in dataset: {len(found_leakage)}")
for col in found_leakage:
    corr_with_target = df_sample[col].corr(y) if df_sample[col].dtype != "object" else "N/A (categorical)"
    print(f"   • {col:30s}  corr w/ target: {corr_with_target}")

print(f"\n✅ Listed but not in dataset: {len(not_found)}")
```

**Why this matters:**
If you include `total_pymnt` (total payments received), the model will learn:
"high payments → Fully Paid" — that's **trivially true** and gives you 99% accuracy
that's completely useless in production (you won't know total payments when deciding
to approve a loan).

**Rule of thumb:** Ask yourself *"Would I know this value at the moment the borrower
submits their application?"* If no → it's leakage → drop it.

- [ ] Add the leakage detection cell
- [ ] Run it and review the list
- [ ] **Do NOT drop them yet** — just document them. We drop in Phase 3 (cleaning).
- [ ] If you see any suspiciously high correlations (>0.5), that confirms leakage

---

## ✅ TODO 2.4 — Data Cleaning (Section 6)

**Purpose:** Basic cleaning before visualization. We don't do heavy preprocessing here
(that's Phase 3) — just enough to make the EDA work.

**Update the cleaning cell (Cell 18):**

```python
# ── Basic cleaning for EDA ──────────────────────────────────────────────
# Drop ID and free-text columns (useless for analysis)
ID_COLS = ["id", "member_id", "url"]
TEXT_COLS = ["desc", "emp_title", "title"]
DROP_FOR_EDA = [c for c in ID_COLS + TEXT_COLS + found_leakage if c in X.columns]

X = X.drop(columns=DROP_FOR_EDA, errors="ignore")
print(f"Dropped {len(DROP_FOR_EDA)} columns (IDs + text + leakage)")
print(f"Remaining features: {X.shape[1]}")

# Drop columns that are >50% missing
high_miss_cols = [c for c in X.columns if X[c].isnull().mean() > 0.50]
X = X.drop(columns=high_miss_cols)
print(f"Dropped {len(high_miss_cols)} high-missing columns (>50%)")
print(f"Final features for EDA: {X.shape[1]}")

# Drop duplicate rows
n_dup = X.duplicated().sum()
if n_dup > 0:
    X = X.drop_duplicates()
    y = y.loc[X.index]  # keep y aligned
print(f"Duplicates removed: {n_dup}")
```

**Key functions:**

| Function | What It Does |
|----------|-------------|
| `df.drop(columns=[...], errors="ignore")` | Drop columns, skip if not found |
| `df.isnull().mean()` | Fraction missing per column (0.0 to 1.0) |
| `df.duplicated().sum()` | Count exact duplicate rows |
| `df.drop_duplicates()` | Remove duplicate rows, keep first occurrence |
| `y.loc[X.index]` | Realign target after dropping rows from X |

- [ ] Replace the existing cleaning cell with the code above
- [ ] Run and note how many columns remain after cleaning
- [ ] Verify `X` and `y` have the same number of rows

---

## ✅ TODO 2.5 — Univariate Analysis (Section 7)

**Purpose:** Look at the distribution of each feature **individually**. This reveals:
- Skewed distributions (may need log transform)
- Zero-variance columns (useless → drop)
- Suspicious values (negative income? 999 as placeholder?)

**What to do:** Run the existing Cells 20-22 (feature type separation + histograms +
categorical bar charts). They should work with the cleaned `X`.

**What to look for in the numeric histograms:**

| Pattern | What It Means | Action |
|---------|--------------|--------|
| Right-skewed (long right tail) | Income, loan amounts etc. | Consider `np.log1p()` transform |
| Spike at zero | Missing data coded as 0 | Investigate — may need recoding |
| Bimodal (two humps) | Two sub-populations | Potentially useful feature |
| Uniform/flat | Low discriminative power | May drop later |

**What to look for in categorical bar charts:**

| Pattern | What It Means | Action |
|---------|--------------|--------|
| One category dominates (>95%) | Near-zero variance | Consider dropping |
| Many categories (>20 unique) | High cardinality | Group rare categories into "Other" |
| Ordinal ordering (A < B < C) | Natural ranking | Use `OrdinalEncoder` not `OneHotEncoder` |

- [ ] Run Cell 20 — note counts of numeric vs categorical features
- [ ] Run Cell 21 — scan histograms for skew, spikes, anomalies
- [ ] Run Cell 22 — check which categoricals were skipped (high cardinality)
- [ ] Write down: which numeric features are heavily skewed?
- [ ] Write down: which categoricals should use ordinal encoding? (hint: `grade`, `sub_grade`)

---

## ✅ TODO 2.6 — Bivariate Analysis: Default Rate by Category

**Purpose:** This is the **most insightful** analysis for classification. Instead of
generic scatter plots, we want to see: *"What is the default rate for each grade?
Each purpose? Each home ownership type?"*

**Add a NEW cell with loan-specific bivariate analysis:**

```python
# ── Default rate by key categorical features ────────────────────────────
KEY_CATS = ["grade", "sub_grade", "home_ownership", "verification_status",
            "purpose", "term", "application_type", "initial_list_status"]
KEY_CATS = [c for c in KEY_CATS if c in df_sample.columns]

fig, axes = plt.subplots(len(KEY_CATS), 1, figsize=(12, 5 * len(KEY_CATS)))
if len(KEY_CATS) == 1:
    axes = [axes]

for i, col in enumerate(KEY_CATS):
    rates = df_sample.groupby(col)[TARGET_COL].agg(["mean", "count"])
    rates.columns = ["default_rate", "n_loans"]
    rates = rates.sort_values("default_rate", ascending=True)

    bars = axes[i].barh(rates.index.astype(str), rates["default_rate"],
                        color="steelblue", edgecolor="black")
    axes[i].set_xlabel("Default Rate")
    axes[i].set_title(f"Default Rate by {col}  (n categories = {len(rates)})",
                      fontsize=12, fontweight="bold")
    # Add count annotations
    for bar, count in zip(bars, rates["n_loans"]):
        axes[i].text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                     f"n={count:,.0f}", va="center", fontsize=8)

plt.tight_layout()
plt.show()
```

**What to look for:**
- `grade`: Should show a clear gradient (A = low default, G = high default).
  If it does, grade is a powerful predictor.
- `term`: 60-month loans typically default more than 36-month.
- `purpose`: Certain loan purposes (small business) default more.
- `home_ownership`: RENT vs OWN vs MORTGAGE differences.

**Key functions:**

| Function | What It Does |
|----------|-------------|
| `df.groupby("col")["target"].mean()` | Default rate per category |
| `df.groupby("col")["target"].agg(["mean", "count"])` | Rate + sample size |

- [ ] Add the default-rate-by-category cell
- [ ] Run and identify the top 3 most predictive categorical features
- [ ] Note: does `grade` show a clear risk gradient? (it should!)

---

## ✅ TODO 2.7 — Bivariate Analysis: Correlation with Target

**Purpose:** Find which numeric features are most correlated with default/no-default.

**What to do:** Run the existing correlation cells (Cells 23-25). Update Cell 26
(top features vs target) since we're doing classification, not regression.

**Replace the strip plot cell with box plots (more informative for classification):**

```python
# ── Box plots: top numeric features by default class ────────────────────
top_n = 6
top_features = target_corr.abs().sort_values(ascending=False).head(top_n).index.tolist()

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

for i, col in enumerate(top_features):
    # Use full sample data, not just X (which had leakage cols removed)
    data_for_plot = df_sample[[col, TARGET_COL]].dropna()
    sns.boxplot(x=TARGET_COL, y=col, data=data_for_plot, ax=axes[i],
                palette=["#2ecc71", "#e74c3c"])
    axes[i].set_title(f"{col} by Default Status", fontsize=11)
    axes[i].set_xticklabels(["Fully Paid", "Charged Off"])

plt.suptitle(f"Top {top_n} Correlated Features by Default Status", fontsize=14)
plt.tight_layout()
plt.show()
```

- [ ] Run correlation matrix heatmap (Cell 23)
- [ ] Run top correlations with target (Cell 24)
- [ ] Add/update the box plot cell above
- [ ] Write down: top 5 features most correlated with default
- [ ] Write down: any features that are highly correlated with each other (>0.9)? → multicollinearity

---

## ✅ TODO 2.8 — Temporal Analysis (NEW — Loan-Specific)

**Purpose:** See how default rates change over time. This is critical because:
1. It reveals **data drift** (newer loans behave differently from older ones)
2. It helps decide the **train/test split strategy** (time-based vs random)
3. Recruiters love temporal analysis — it shows domain understanding

**Add a NEW cell:**

```python
# ── Temporal analysis: default rate over time ───────────────────────────
# Parse issue_d (loan issue date) to datetime
if "issue_d" in df_sample.columns:
    df_sample["issue_dt"] = pd.to_datetime(df_sample["issue_d"], format="mixed", errors="coerce")

    # Monthly default rate
    monthly = (df_sample
               .groupby(df_sample["issue_dt"].dt.to_period("Q"))
               .agg(default_rate=(TARGET_COL, "mean"),
                    n_loans=(TARGET_COL, "count"))
               .reset_index())
    monthly["issue_dt"] = monthly["issue_dt"].astype(str)

    fig, ax1 = plt.subplots(figsize=(14, 5))
    color1, color2 = "#e74c3c", "#3498db"

    ax1.plot(monthly["issue_dt"], monthly["default_rate"],
             color=color1, marker="o", linewidth=2, label="Default Rate")
    ax1.set_ylabel("Default Rate", color=color1)
    ax1.tick_params(axis="y", labelcolor=color1)
    ax1.set_xlabel("Quarter")
    plt.xticks(rotation=45)

    ax2 = ax1.twinx()
    ax2.bar(monthly["issue_dt"], monthly["n_loans"],
            alpha=0.3, color=color2, label="Loan Volume")
    ax2.set_ylabel("Number of Loans", color=color2)
    ax2.tick_params(axis="y", labelcolor=color2)

    fig.suptitle("Default Rate & Loan Volume Over Time", fontsize=14)
    fig.legend(loc="upper left", bbox_to_anchor=(0.1, 0.95))
    plt.tight_layout()
    plt.show()
else:
    print("⚠️ Column 'issue_d' not found — skipping temporal analysis")
```

**Key functions:**

| Function | What It Does |
|----------|-------------|
| `pd.to_datetime(col, format="mixed")` | Parse date strings to datetime objects |
| `dt.to_period("Q")` | Group by quarter (M = month, Y = year) |
| `ax.twinx()` | Add a second y-axis (for overlaying two different scales) |

- [ ] Add the temporal analysis cell
- [ ] Run and observe: does default rate change over time?
- [ ] Note: is there a trend? seasonal pattern? sharp changes?
- [ ] Decision: if the pattern changes significantly → use time-based train/test split

---

## ✅ TODO 2.9 — Outlier Detection (Section 9)

**Purpose:** Extreme values can distort model training (especially linear models).
Not all outliers are bad — some are just natural variation. The IQR method flags
them so you can decide.

**What to do:** Run the existing outlier report cell (Cell 30). It uses the
IQR method (Interquartile Range):
- Q1 = 25th percentile, Q3 = 75th percentile
- IQR = Q3 − Q1
- Outlier if value < Q1 − 1.5×IQR or > Q3 + 1.5×IQR

**What to look for:**

| Outlier % | Interpretation | Action |
|-----------|---------------|--------|
| < 1% | Normal tail | Usually leave alone |
| 1–5% | Moderate outliers | Cap (winsorize) or leave for tree models |
| > 5% | Heavy outliers or skewed distribution | Log-transform or cap at 1st/99th percentile |

**Key insight:** Tree-based models (Random Forest, XGBoost) are **robust** to outliers.
Linear models (Logistic Regression) are **sensitive**. Since we're training both,
we'll handle outliers in the preprocessing pipeline (Phase 3).

- [ ] Run the outlier detection cell
- [ ] Note: which features have >5% outliers?
- [ ] Note: are the "outliers" real extreme values or just skewed distributions?

---

## ✅ TODO 2.10 — Fill in the EDA Summary (Section 10)

**Purpose:** Document everything you discovered. This is your "detective's report."
Future-you (and recruiters reviewing your portfolio) will thank you.

**Fill in the markdown cell with your actual findings:**

```markdown
## 💡 10 · EDA Summary

### Dataset Overview
- **Rows:** [your number] (after filtering to Fully Paid & Charged Off)
- **Columns:** [your number] (after dropping leakage, IDs, high-missing)
- **Task:** Binary classification (loan default prediction)

### Data Quality Findings
- [X] columns had >50% missing values → dropped
- [X] leakage columns identified and removed (post-origination data)
- [X] ID and free-text columns dropped
- Memory reduced from ~X GB to ~X GB via dtype optimization

### Target Variable Observations
- Class imbalance: ~80% Fully Paid / ~20% Charged Off
- Imbalance strategy: class weighting (no SMOTE needed — minority class is large)

### Key Feature Insights
- `grade` / `sub_grade`: Clear risk gradient (A=safe → G=risky)
- `term`: 60-month loans default more than 36-month
- `int_rate`: Strong positive correlation with default
- `annual_inc`: Higher income → fewer defaults
- [fill in your other findings]

### Correlations Worth Investigating
- [top 3 correlated features with target]
- [any multicollinear pairs (r > 0.9)]

### Temporal Patterns
- Default rate trend: [increasing / decreasing / stable]
- Recommendation: [time-based / random] train-test split

### Recommended Preprocessing Steps
- [ ] Drop leakage columns (listed in TODO 2.3)
- [ ] Drop columns with >50% missing
- [ ] Impute remaining missing: median (numeric), mode (categorical)
- [ ] Ordinal encode: grade, sub_grade
- [ ] One-hot encode: home_ownership, verification_status, purpose, term
- [ ] Standard scale: all numeric features
- [ ] Engineer new features (see Phase 3)
- [ ] Apply class weights during training (see Phase 4)
```

- [ ] Fill in all the `[placeholders]` with your actual numbers
- [ ] Review: does your summary tell a clear story?
- [ ] Save the notebook

---

## 🎯 Phase 2 Checklist

When all TODOs above are done, you should have:

- [ ] Target distribution visualized, class imbalance documented
- [ ] Missing values mapped and categorized (drop / impute / ignore)
- [ ] Leakage columns identified (critical!)
- [ ] Basic cleaning done (IDs, text, high-missing dropped)
- [ ] Univariate distributions examined (skew, zero-variance, anomalies)
- [ ] Default rates by key categories plotted
- [ ] Correlations with target computed
- [ ] Temporal trends analyzed
- [ ] Outliers flagged
- [ ] EDA Summary filled in with real findings
- [ ] Ready for [Phase 3 — Feature Engineering](phase3_feature_engineering.md)!

---

## 📚 Concepts to Remember

| Concept | Explanation |
|---------|-------------|
| **Univariate analysis** | Looking at one variable at a time (histograms, bar charts) |
| **Bivariate analysis** | Looking at two variables together (scatter, box plots, default rate by group) |
| **Multivariate analysis** | Looking at many variables together (correlation matrix, pairplot) |
| **Class imbalance** | When one class is much more common than the other |
| **Data leakage** | Using future information to predict the past — invalidates the model |
| **IQR (Interquartile Range)** | Q3 − Q1; used to define "normal" range for outlier detection |
| **Correlation** | Linear relationship strength (−1 to +1); 0 = no linear relationship |
| **Multicollinearity** | When two features are highly correlated with each other (redundant information) |
