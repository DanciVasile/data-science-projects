# 🎨 Phase 1 — Configuration & Data Loading

> **Where:** `notebooks/exploration.ipynb` — Cells 4 (Configuration) and 6 (Load Data)  
> **Prereq:** None — this is where it all begins  
> **Next:** [Phase 2 — EDA](phase2_eda.md)

---

## 🧒 What Are We Doing Here? (The Big Picture)

Imagine you're about to paint a huge mural. Before you touch a brush, you need to:
1. Set up your easel (configure paths, constants)
2. Squeeze paint onto the palette (load data into memory)
3. Sketch the outline (first look at shape, types, sample rows)

That's Phase 1. We're **not** analyzing anything yet — just getting the data into Python
and making sure it looks right.

---

## ✅ TODO 1.1 — Update the Configuration Cell

**Purpose:** Tell the notebook what dataset we're using, what column is the target, and
what kind of ML task this is.

**What to change in the Configuration cell (Cell 5):**

```python
# CHANGE THESE THREE LINES:
TARGET_COL = "loan_status"          # was "target" — this is the actual column name in LendingClub data
TASK = "classification"              # was "regression" — we're predicting default YES/NO
DATA_FILE = "accepted_2007_to_2018Q4.csv.gz"  # already correct, just verify
```

**Why `loan_status`?**  
In the LendingClub dataset, the column `loan_status` contains values like "Fully Paid",
"Charged Off", "Current", etc. We'll later convert this to a binary 0/1.

**Why `classification` not `regression`?**  
- **Regression** = predict a continuous number (e.g., house price = $245,000)
- **Classification** = predict a category (e.g., default = Yes or No)

Loan default is a **binary classification** problem: the loan either defaults or it doesn't.

- [ ] Change `TARGET_COL` from `"target"` to `"loan_status"`
- [ ] Change `TASK` from `"regression"` to `"classification"`
- [ ] Verify `DATA_FILE` is `"accepted_2007_to_2018Q4.csv.gz"`
- [ ] Run the cell and confirm it prints `exists: True`

---

## ✅ TODO 1.2 — Load Data with Memory Optimization

**Purpose:** The raw CSV has ~2.2 million rows × 150 columns. Loading it naively with
`pd.read_csv()` can use 5–8 GB of RAM. We need to be smart about it.

**What to do:** Replace the simple `pd.read_csv(DATA_PATH)` in Cell 7 with a
memory-optimized version.

**Key functions to learn:**

| Function | What It Does | Why We Need It |
|----------|-------------|----------------|
| `pd.read_csv(path, low_memory=False)` | Reads the full CSV without dtype warnings | Prevents pandas from guessing dtypes per chunk |
| `df.memory_usage(deep=True)` | Shows memory per column in bytes | Identifies the biggest memory hogs |
| `pd.to_numeric(col, downcast="float")` | Converts float64 → float32 (halves memory) | Most ML doesn't need float64 precision |
| `col.astype("category")` | Converts string column to category dtype | Strings like "RENT" repeated 2M times waste memory; categories store them once |

**Code to use:**

```python
# Load with low_memory=False to avoid dtype warnings on large files
df_raw = pd.read_csv(DATA_PATH, low_memory=False)
print(f"Raw memory: {df_raw.memory_usage(deep=True).sum() / 1024**2:.0f} MB")

# ── Memory reduction ────────────────────────────────────────────────────
def reduce_memory(df):
    """Downcast numeric types and convert low-cardinality strings to categories."""
    for col in df.columns:
        col_type = df[col].dtype
        if col_type == "object":
            if df[col].nunique() < 1000:           # low cardinality → category
                df[col] = df[col].astype("category")
        elif str(col_type).startswith("float"):
            df[col] = pd.to_numeric(df[col], downcast="float")
        elif str(col_type).startswith("int"):
            df[col] = pd.to_numeric(df[col], downcast="unsigned"
                                    if df[col].min() >= 0 else "signed")
    return df

df = reduce_memory(df_raw.copy())
print(f"Optimized memory: {df.memory_usage(deep=True).sum() / 1024**2:.0f} MB")
df.head()
```

- [ ] Replace `df = pd.read_csv(DATA_PATH)` with the memory-optimized version above
- [ ] Run and verify memory drops significantly (expect ~60-70% reduction)
- [ ] Check `df.head()` output looks correct (no mangled values)

---

## ✅ TODO 1.3 — Binarize the Target Column

**Purpose:** The `loan_status` column has many values:
- `"Fully Paid"` — borrower paid back ✅
- `"Charged Off"` — borrower stopped paying, loan written off ❌
- `"Current"` — loan still active (no outcome yet) ❓
- `"Late (31-120 days)"`, `"In Grace Period"`, `"Default"` — in-progress ❓

We only want loans with a **known outcome**: Fully Paid or Charged Off.
Everything else is ambiguous and would add noise to the model.

**Key functions to learn:**

| Function | What It Does | Why We Need It |
|----------|-------------|----------------|
| `df["col"].value_counts()` | Counts occurrences of each unique value | See the class distribution before filtering |
| `df[df["col"].isin([...])]` | Filters rows to only those matching a list | Keep only Fully Paid and Charged Off |
| `df["col"].map({...})` | Replaces values using a dictionary | Convert "Fully Paid" → 0, "Charged Off" → 1 |

**Code to use — add a NEW cell right after loading:**

```python
# ── Binarize target ─────────────────────────────────────────────────────
# Show all loan statuses before filtering
print("📊 loan_status value counts (before filtering):")
print(df["loan_status"].value_counts().to_string())
print(f"\nTotal rows: {len(df):,}")

# Keep only loans with a definitive outcome
KEEP_STATUSES = ["Fully Paid", "Charged Off"]
df = df[df["loan_status"].isin(KEEP_STATUSES)].copy()

# Map to binary: 0 = Fully Paid (good), 1 = Charged Off (default)
df["loan_status"] = df["loan_status"].map({"Fully Paid": 0, "Charged Off": 1})

print(f"\n✅ After filtering: {len(df):,} rows")
print(f"   Fully Paid (0): {(df['loan_status'] == 0).sum():,}")
print(f"   Charged Off (1): {(df['loan_status'] == 1).sum():,}")
print(f"   Default rate:    {df['loan_status'].mean():.2%}")
```

**What to expect:** ~80% Fully Paid, ~20% Charged Off. This imbalance is normal for
credit data and we'll handle it in Phase 4 with class weighting.

- [ ] Add a new cell after the data loading cell
- [ ] Paste and run the binarization code
- [ ] Verify both classes are present and the default rate is ~20%
- [ ] Note the exact class counts for your EDA Summary later

---

## ✅ TODO 1.4 — Create a Stratified Sample for EDA

**Purpose:** Plotting 2 million points is slow and makes scatter plots unreadable.
We'll create a ~200k sample that preserves the class balance (same % of defaults)
for all visualization work. Full data is kept for model training later.

**Key functions to learn:**

| Function | What It Does | Why We Need It |
|----------|-------------|----------------|
| `df.sample(n=..., random_state=42)` | Random sample of n rows | Faster EDA, reproducible results |
| `df.groupby("col").apply(...)` | Apply function per group | Sample proportionally from each class |
| `sklearn.model_selection.train_test_split(..., stratify=y)` | Stratified split | Maintains class proportions in sample |

**Code to use — add after binarization:**

```python
# ── Stratified sample for EDA (keeps class proportions) ─────────────────
from sklearn.model_selection import train_test_split

SAMPLE_SIZE = 200_000
RANDOM_STATE = 42

if len(df) > SAMPLE_SIZE:
    df_sample, _ = train_test_split(
        df,
        train_size=SAMPLE_SIZE,
        stratify=df["loan_status"],
        random_state=RANDOM_STATE,
    )
    print(f"📊 EDA sample: {len(df_sample):,} rows "
          f"(from {len(df):,} total)")
else:
    df_sample = df.copy()
    print(f"📊 Dataset small enough, using all {len(df):,} rows")

print(f"   Default rate in sample: {df_sample['loan_status'].mean():.2%}")
print(f"   Default rate in full:   {df['loan_status'].mean():.2%}")
```

**Why "stratified"?** If you randomly sample, you might accidentally get 25% defaults
instead of 20%. Stratified sampling guarantees the proportions match the original data.

- [ ] Add the stratified sampling cell
- [ ] Verify sample default rate matches full data default rate
- [ ] From this point forward, use `df_sample` for plots and `df` for training

---

## ✅ TODO 1.5 — Run the Data Quality Summary Card

**Purpose:** Get a quick overview of the dataset: how big it is, what types of columns
it has, how much is missing, any duplicates.

**What to do:** Run the existing Data Quality Summary cell (Cell 8) — it should work
as-is since the data is loaded.

- [ ] Run the Data Quality Summary cell
- [ ] Note down: total rows, total columns, missing %, memory usage
- [ ] Run `df.info()` cell — scan for columns with unexpected dtypes
- [ ] Run `df.describe()` — look for suspicious min/max values (e.g., negative income)
- [ ] Run `df.describe(include="object")` — check cardinality of string columns

---

## 🎯 Phase 1 Checklist

When all TODOs above are done, you should have:

- [ ] Configuration updated (`TARGET_COL`, `TASK`)
- [ ] Data loaded with reduced memory
- [ ] Target binarized to 0/1 (Fully Paid / Charged Off)
- [ ] 200k stratified sample ready in `df_sample`
- [ ] Data quality card printed
- [ ] You can move to [Phase 2 — EDA](phase2_eda.md)!

---

## 📚 Concepts to Remember

| Concept | Explanation |
|---------|-------------|
| **Binary classification** | Predicting one of two outcomes (yes/no, 0/1, default/no default) |
| **Target variable** | The column we're trying to predict (`loan_status` → 0 or 1) |
| **Features** | All other columns used as inputs to the model |
| **Stratified sampling** | Sampling that preserves the proportion of each class |
| **dtype downcasting** | Converting float64 → float32, int64 → int16 etc. to save memory |
| **Category dtype** | Pandas stores repeated strings efficiently (like an enum) |
| **Data leakage** | Using information that wouldn't be available at prediction time — we'll handle this in Phase 3 |
