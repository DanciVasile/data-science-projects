"""
Generates a starter exploration.ipynb EDA template for data science projects.

Usage:
    cd into any folder, then run:
        python <path-to>/create_notebook.py
    Or pass an output directory:
        python create_notebook.py notebooks

Conventions used (consistent with train.py across projects):
    X  — feature matrix (pd.DataFrame)
    y  — target vector (pd.Series)
    df — raw dataframe before splitting
    TARGET_COL           — name of the target column (string)
    TASK                 — "regression" or "classification"
    numeric_features     — list of numeric column names
    categorical_features — list of categorical column names
"""

from datetime import date

import nbformat
import os
import sys

AUTHOR = "Vasile-Marian Danci"


def build_template(project_name: str = "Project"):
    nb = nbformat.v4.new_notebook()

    nb.metadata.kernelspec = {
        "display_name": "Data Science Projects",
        "language": "python",
        "name": "data-science-projects",
    }

    cells = []

    today = date.today().strftime("%Y-%m-%d")

    # ── Header ─────────────────────────────────────────────
    cells.append(nbformat.v4.new_markdown_cell(
        f"# 🔬 {project_name} — Exploratory Data Analysis\n"
        f"\n"
        f"**Author:** {AUTHOR}  \n"
        f"**Date:** {today}  \n"
        f"\n"
        f"---\n"
        f"\n"
        f"### 🎯 Objective\n"
        f"\n"
        f"> Describe the goal of this analysis in one or two sentences."
    ))

    # ── 1. Imports ─────────────────────────────────────────
    cells.append(nbformat.v4.new_markdown_cell(
        "---\n"
        "## 📦 1 · Imports\n"
        "\n"
        "Import all required packages here. Keep standard-library, third-party, "
        "and local imports separated."
    ))

    cells.append(nbformat.v4.new_code_cell(
        "import math\n"
        "import warnings\n"
        "from pathlib import Path\n"
        "warnings.filterwarnings(\"ignore\")\n"
        "\n"
        "import numpy as np\n"
        "import pandas as pd\n"
        "\n"
        "# Visualisation — swap with plotly / altair / any library you prefer\n"
        "import matplotlib.pyplot as plt\n"
        "import seaborn as sns\n"
        "\n"
        "sns.set_theme(style=\"whitegrid\", palette=\"muted\")\n"
        "plt.rcParams[\"figure.figsize\"] = (12, 6)\n"
        "\n"
        "pd.set_option(\"display.max_columns\", None)\n"
        "pd.set_option(\"display.max_rows\", 100)"
    ))

    # ── 2. Configuration ──────────────────────────────────
    cells.append(nbformat.v4.new_markdown_cell(
        "---\n"
        "## ⚙️ 2 · Configuration\n"
        "\n"
        "Define all configurable parameters (paths, constants, column names) in "
        "one place so the notebook is easy to adapt across projects."
    ))

    cells.append(nbformat.v4.new_code_cell(
        "# ── Resolve project directory automatically ─────────────────────────\n"
        "# Works in VS Code, JupyterLab, and classic Jupyter Notebook.\n"
        "_nb_path = globals().get(\"__vsc_ipynb_file__\")  # VS Code injects this\n"
        "if _nb_path:\n"
        "    PROJECT_DIR = Path(_nb_path).resolve().parent.parent  # notebooks/ → project/\n"
        "else:\n"
        "    # Browser Jupyter sets CWD to the notebook's directory.\n"
        "    _cwd = Path.cwd()\n"
        "    PROJECT_DIR = next(\n"
        "        (p for p in [_cwd, *_cwd.parents]\n"
        "         if (p / \"data\").is_dir() and (p / \"notebooks\").is_dir()),\n"
        "        _cwd,\n"
        "    )\n"
        "\n"
        "# ── Dataset ─────────────────────────────────────────────────────────\n"
        "# Just set the filename — the full path is resolved automatically.\n"
        "DATA_FILE = \"dataset.csv\"  # TODO: replace with your dataset filename\n"
        "DATA_PATH = PROJECT_DIR / \"data\" / DATA_FILE\n"
        "\n"
        "# TODO: Set the name of the target column in your dataset\n"
        "TARGET_COL = \"target\"\n"
        "\n"
        "# Task type — drives conditional behaviour throughout the notebook:\n"
        "#   \"regression\"      → histograms, scatter plots, correlation analysis\n"
        "#   \"classification\"  → bar charts, box plots per class, class balance checks\n"
        "TASK = \"regression\"  # or \"classification\"\n"
        "\n"
        "print(f\"📁 Project dir: {PROJECT_DIR}\")\n"
        "print(f\"📄 Data path:   {DATA_PATH}  (exists: {DATA_PATH.exists()})\")\n"
    ))

    # ── 3. Load Data ───────────────────────────────────────
    cells.append(nbformat.v4.new_markdown_cell(
        "---\n"
        "## 📂 3 · Load Data\n"
        "\n"
        "Load the raw dataset and take a first look at its shape, types, and "
        "sample rows."
    ))

    cells.append(nbformat.v4.new_code_cell(
        "df = pd.read_csv(DATA_PATH)\n"
        "df.head()"
    ))

    cells.append(nbformat.v4.new_code_cell(
        "# --- Data quality summary card ---\n"
        "n_rows, n_cols = df.shape\n"
        "dtypes_breakdown = df.dtypes.value_counts().to_dict()\n"
        "total_missing = df.isnull().sum().sum()\n"
        "total_cells = n_rows * n_cols\n"
        "missing_pct = (total_missing / total_cells * 100)\n"
        "n_duplicates = df.duplicated().sum()\n"
        "mem_mb = df.memory_usage(deep=True).sum() / 1024**2\n"
        "\n"
        "print(\"=\" * 50)\n"
        "print(\"  📋 DATA QUALITY SUMMARY\")\n"
        "print(\"=\" * 50)\n"
        "print(f\"  Rows:            {n_rows:,}\")\n"
        "print(f\"  Columns:         {n_cols:,}\")\n"
        "print(f\"  Dtypes:          {dtypes_breakdown}\")\n"
        "print(f\"  Missing values:  {total_missing:,} ({missing_pct:.2f}%)\")\n"
        "print(f\"  Duplicate rows:  {n_duplicates:,}\")\n"
        "print(f\"  Memory usage:    {mem_mb:.2f} MB\")\n"
        "print(\"=\" * 50)"
    ))

    cells.append(nbformat.v4.new_code_cell(
        "df.info()"
    ))

    cells.append(nbformat.v4.new_code_cell(
        "df.describe()"
    ))

    cells.append(nbformat.v4.new_code_cell(
        "df.describe(include=\"object\")"
    ))

    # ── 4. Target Variable ────────────────────────────────
    cells.append(nbformat.v4.new_markdown_cell(
        "---\n"
        "## 🎯 4 · Target Variable\n"
        "\n"
        "Separate features (`X`) and target (`y`) early.  \n"
        "By convention in ML, **`X`** denotes the feature matrix and **`y`** "
        "denotes the target vector — this comes from the statistical notation "
        "$y = f(X) + \\varepsilon$ and is the standard used by scikit-learn, "
        "XGBoost, LightGBM, and virtually every ML library."
    ))

    cells.append(nbformat.v4.new_code_cell(
        "# Separate features (X) and target (y)\n"
        "# By ML convention: X = features, y = target\n"
        "X = df.drop(columns=[TARGET_COL])\n"
        "y = df[TARGET_COL]\n"
        "\n"
        "print(f\"Features shape: {X.shape}\")\n"
        "print(f\"Target: '{TARGET_COL}'  |  dtype: {y.dtype}  |  \"\n"
        "      f\"mean: {y.mean():.2f}  |  median: {y.median():.2f}\")"
    ))

    cells.append(nbformat.v4.new_code_cell(
        "# Target distribution — conditional on TASK\n"
        "fig, axes = plt.subplots(1, 2, figsize=(14, 5))\n"
        "\n"
        "if TASK == \"regression\":\n"
        "    y.hist(bins=30, edgecolor=\"black\", ax=axes[0])\n"
        "    axes[0].set_title(f\"{TARGET_COL} — Distribution\")\n"
        "    axes[0].set_xlabel(TARGET_COL)\n"
        "    axes[0].set_ylabel(\"Count\")\n"
        "    axes[1].boxplot(y.dropna(), vert=True)\n"
        "    axes[1].set_title(f\"{TARGET_COL} — Box Plot\")\n"
        "else:  # classification\n"
        "    counts = y.value_counts().sort_index()\n"
        "    counts.plot(kind=\"bar\", edgecolor=\"black\", ax=axes[0])\n"
        "    axes[0].set_title(f\"{TARGET_COL} — Class Distribution\")\n"
        "    axes[0].set_xlabel(TARGET_COL)\n"
        "    axes[0].set_ylabel(\"Count\")\n"
        "    axes[0].tick_params(axis=\"x\", rotation=0)\n"
        "    # Class balance as percentage\n"
        "    pct = (counts / counts.sum() * 100).round(1)\n"
        "    pct.plot(kind=\"bar\", edgecolor=\"black\", ax=axes[1])\n"
        "    axes[1].set_title(f\"{TARGET_COL} — Class Balance (%)\")\n"
        "    axes[1].set_ylabel(\"%\")\n"
        "    axes[1].tick_params(axis=\"x\", rotation=0)\n"
        "\n"
        "plt.tight_layout()\n"
        "plt.show()"
    ))

    # ── 5. Missing Values ─────────────────────────────────
    cells.append(nbformat.v4.new_markdown_cell(
        "---\n"
        "## 🕳️ 5 · Missing Values"
    ))

    cells.append(nbformat.v4.new_code_cell(
        "missing = X.isnull().sum()\n"
        "missing = missing[missing > 0].sort_values(ascending=False)\n"
        "missing_pct = (missing / len(X) * 100).round(2)\n"
        "\n"
        "if missing.empty:\n"
        "    print(\"No missing values 🎉\")\n"
        "else:\n"
        "    print(pd.DataFrame({\"count\": missing, \"% of total\": missing_pct}).to_string())"
    ))

    # ── 6. Data Cleaning ──────────────────────────────────
    cells.append(nbformat.v4.new_markdown_cell(
        "---\n"
        "## 🧹 6 · Data Cleaning\n"
        "\n"
        "Handle missing values, fix dtypes, remove duplicates, drop irrelevant "
        "columns."
    ))

    cells.append(nbformat.v4.new_code_cell(
        "# Drop duplicates\n"
        "n_dup = X.duplicated().sum()\n"
        "print(f\"Duplicates found: {n_dup}\")\n"
        "X = X.drop_duplicates()\n"
        "\n"
        "# TODO: Drop irrelevant columns (IDs, row numbers, etc.)\n"
        "# X = X.drop(columns=[\"id\"], errors=\"ignore\")\n"
        "\n"
        "# TODO: Handle missing values\n"
        "# X[\"col\"] = X[\"col\"].fillna(X[\"col\"].median())\n"
        "\n"
        "# TODO: Fix data types\n"
        "# X[\"col\"] = X[\"col\"].astype(\"category\")"
    ))

    # ── 7. EDA — Univariate ───────────────────────────────
    cells.append(nbformat.v4.new_markdown_cell(
        "---\n"
        "## 📊 7 · Exploratory Data Analysis — Univariate\n"
        "\n"
        "Distribution of individual features. Separate numeric from categorical "
        "using `select_dtypes` — a standard pandas pattern."
    ))

    cells.append(nbformat.v4.new_code_cell(
        "# Separate feature types — standard naming used across notebooks & train.py\n"
        "numeric_features = X.select_dtypes(include=[\"int64\", \"float64\"]).columns\n"
        "categorical_features = X.select_dtypes(include=[\"object\", \"category\"]).columns\n"
        "\n"
        "print(f\"Numeric features:     {len(numeric_features)}\")\n"
        "print(f\"Categorical features: {len(categorical_features)}\")"
    ))

    cells.append(nbformat.v4.new_code_cell(
        "# Numeric distributions — 3 per row\n"
        "COLS_PER_ROW = 3\n"
        "n_num = len(numeric_features)\n"
        "n_rows = math.ceil(n_num / COLS_PER_ROW)\n"
        "\n"
        "fig, axes = plt.subplots(n_rows, COLS_PER_ROW, figsize=(5 * COLS_PER_ROW, 4 * n_rows))\n"
        "axes = axes.flatten()\n"
        "\n"
        "for i, col in enumerate(numeric_features):\n"
        "    X[col].hist(bins=30, edgecolor=\"black\", ax=axes[i])\n"
        "    axes[i].set_title(col, fontsize=10)\n"
        "    axes[i].tick_params(labelsize=8)\n"
        "\n"
        "# Hide unused subplots\n"
        "for j in range(n_num, len(axes)):\n"
        "    axes[j].set_visible(False)\n"
        "\n"
        "fig.suptitle(\"Numeric Feature Distributions\", fontsize=14, y=1.01)\n"
        "plt.tight_layout()\n"
        "plt.show()"
    ))

    cells.append(nbformat.v4.new_code_cell(
        "# Categorical value counts — 2 per row (skip high-cardinality columns)\n"
        "HIGH_CARD_THRESHOLD = 20\n"
        "\n"
        "plot_cats = [c for c in categorical_features if X[c].nunique() <= HIGH_CARD_THRESHOLD]\n"
        "skipped = [c for c in categorical_features if X[c].nunique() > HIGH_CARD_THRESHOLD]\n"
        "if skipped:\n"
        "    print(f\"⚠️  Skipping high-cardinality columns: {', '.join(skipped)}\")\n"
        "\n"
        "CAT_COLS_PER_ROW = 2\n"
        "n_cats = len(plot_cats)\n"
        "n_cat_rows = math.ceil(n_cats / CAT_COLS_PER_ROW)\n"
        "\n"
        "fig, axes = plt.subplots(n_cat_rows, CAT_COLS_PER_ROW,\n"
        "                         figsize=(7 * CAT_COLS_PER_ROW, 4 * n_cat_rows))\n"
        "axes = np.array(axes).flatten()\n"
        "\n"
        "for i, col in enumerate(plot_cats):\n"
        "    counts = X[col].value_counts().head(15)\n"
        "    counts.sort_values().plot(kind=\"barh\", ax=axes[i])\n"
        "    axes[i].set_title(col, fontsize=10)\n"
        "    axes[i].set_xlabel(\"Count\")\n"
        "    axes[i].tick_params(labelsize=8)\n"
        "\n"
        "for j in range(n_cats, len(axes)):\n"
        "    axes[j].set_visible(False)\n"
        "\n"
        "fig.suptitle(\"Categorical Feature Distributions\", fontsize=14, y=1.01)\n"
        "plt.tight_layout()\n"
        "plt.show()"
    ))

    # ── 8. EDA — Bivariate / Multivariate ─────────────────
    cells.append(nbformat.v4.new_markdown_cell(
        "---\n"
        "## 🔗 8 · Exploratory Data Analysis — Bivariate / Multivariate"
    ))

    cells.append(nbformat.v4.new_code_cell(
        "# Correlation matrix (numeric features only)\n"
        "corr = X[numeric_features].corr()\n"
        "\n"
        "fig, ax = plt.subplots(figsize=(14, 10))\n"
        "mask = np.triu(np.ones_like(corr, dtype=bool))\n"
        "sns.heatmap(corr, mask=mask, annot=False, cmap=\"coolwarm\",\n"
        "            center=0, square=True, linewidths=0.5, ax=ax)\n"
        "ax.set_title(\"Correlation Matrix\")\n"
        "plt.tight_layout()\n"
        "plt.show()"
    ))

    cells.append(nbformat.v4.new_code_cell(
        "# Top correlations with target\n"
        "target_corr = X[numeric_features].corrwith(y).sort_values(ascending=False)\n"
        "print(\"Top positive correlations with target:\")\n"
        "print(target_corr.head(10).to_string())\n"
        "print(\"\\nTop negative correlations with target:\")\n"
        "print(target_corr.tail(5).to_string())"
    ))

    cells.append(nbformat.v4.new_code_cell(
        "# Target vs top numeric features — grid layout\n"
        "top_n = 5\n"
        "top_features = target_corr.abs().sort_values(ascending=False).head(top_n).index.tolist()\n"
        "\n"
        "BIV_COLS = 3\n"
        "biv_rows = math.ceil(len(top_features) / BIV_COLS)\n"
        "fig, axes = plt.subplots(biv_rows, BIV_COLS, figsize=(6 * BIV_COLS, 5 * biv_rows))\n"
        "axes = np.array(axes).flatten()\n"
        "\n"
        "for i, col in enumerate(top_features):\n"
        "    if TASK == \"regression\":\n"
        "        axes[i].scatter(X[col], y, alpha=0.3, edgecolors=\"k\", linewidths=0.3)\n"
        "        axes[i].set_xlabel(col)\n"
        "        axes[i].set_ylabel(TARGET_COL)\n"
        "        axes[i].set_title(f\"{col} vs {TARGET_COL}\")\n"
        "    else:  # classification\n"
        "        sns.stripplot(x=y, y=X[col], ax=axes[i], alpha=0.3, jitter=True)\n"
        "        axes[i].set_title(f\"{col} by {TARGET_COL} class\")\n"
        "\n"
        "for j in range(len(top_features), len(axes)):\n"
        "    axes[j].set_visible(False)\n"
        "\n"
        "fig.suptitle(f\"Top {top_n} Features vs {TARGET_COL}\", fontsize=14, y=1.01)\n"
        "plt.tight_layout()\n"
        "plt.show()"
    ))

    cells.append(nbformat.v4.new_code_cell(
        "# Target vs categorical features — mean target per category (2 per row)\n"
        "CAT_BIV_COLS = 2\n"
        "cat_biv_rows = math.ceil(len(plot_cats) / CAT_BIV_COLS)\n"
        "\n"
        "fig, axes = plt.subplots(cat_biv_rows, CAT_BIV_COLS,\n"
        "                         figsize=(7 * CAT_BIV_COLS, 4 * cat_biv_rows))\n"
        "axes = np.array(axes).flatten()\n"
        "\n"
        "for i, col in enumerate(plot_cats):\n"
        "    means = df.groupby(col)[TARGET_COL].mean().sort_values(ascending=True)\n"
        "    means.plot(kind=\"barh\", ax=axes[i])\n"
        "    axes[i].set_title(f\"Mean {TARGET_COL} by {col}\", fontsize=10)\n"
        "    axes[i].set_xlabel(f\"Mean {TARGET_COL}\")\n"
        "    axes[i].tick_params(labelsize=8)\n"
        "\n"
        "for j in range(len(plot_cats), len(axes)):\n"
        "    axes[j].set_visible(False)\n"
        "\n"
        "fig.suptitle(f\"Mean {TARGET_COL} by Category\", fontsize=14, y=1.01)\n"
        "plt.tight_layout()\n"
        "plt.show()"
    ))

    cells.append(nbformat.v4.new_code_cell(
        "# Pairplot — top 5 numeric features most correlated with target\n"
        "pair_cols = top_features + [TARGET_COL]\n"
        "pair_df = df[pair_cols].dropna()\n"
        "\n"
        "g = sns.pairplot(pair_df, corner=True, plot_kws={\"alpha\": 0.3, \"s\": 10})\n"
        "g.figure.suptitle(\"Pairplot — Top 5 Correlated Features\", y=1.02)\n"
        "plt.show()"
    ))

    # ── 9. Outlier Detection ──────────────────────────────
    cells.append(nbformat.v4.new_markdown_cell(
        "---\n"
        "## 🚨 9 · Outlier Detection"
    ))

    cells.append(nbformat.v4.new_code_cell(
        "# IQR-based outlier summary\n"
        "def outlier_report(dataframe, cols):\n"
        "    records = []\n"
        "    for col in cols:\n"
        "        Q1 = dataframe[col].quantile(0.25)\n"
        "        Q3 = dataframe[col].quantile(0.75)\n"
        "        IQR = Q3 - Q1\n"
        "        lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR\n"
        "        n_out = ((dataframe[col] < lower) | (dataframe[col] > upper)).sum()\n"
        "        records.append({\"feature\": col, \"n_outliers\": n_out,\n"
        "                        \"% outliers\": round(n_out / len(dataframe) * 100, 2)})\n"
        "    return pd.DataFrame(records).sort_values(\"n_outliers\", ascending=False)\n"
        "\n"
        "outlier_report(X, numeric_features)"
    ))

    # ── 10. EDA Summary ──────────────────────────────────
    cells.append(nbformat.v4.new_markdown_cell(
        "---\n"
        "## 💡 10 · EDA Summary\n"
        "\n"
        "### Dataset Overview\n"
        "- **Rows:** *...*\n"
        "- **Columns:** *...*\n"
        "- **Task:** *regression / classification*\n"
        "\n"
        "### Data Quality Findings\n"
        "- *e.g. X columns have >50% missing values*\n"
        "- *e.g. N duplicate rows removed*\n"
        "- *e.g. columns A, B have inconsistent dtypes*\n"
        "\n"
        "### Target Variable Observations\n"
        "- *e.g. right-skewed distribution → consider log transform*\n"
        "- *e.g. class imbalance: 90/10 split → consider SMOTE or class weights*\n"
        "\n"
        "### Key Feature Insights\n"
        "- *e.g. Feature X has high cardinality (500+ unique values)*\n"
        "- *e.g. Feature Y shows clear separation between classes*\n"
        "\n"
        "### Correlations Worth Investigating\n"
        "- *e.g. Feature A and B are highly correlated (r=0.95) → possible multicollinearity*\n"
        "- *e.g. Feature C has the strongest relationship with the target*\n"
        "\n"
        "### Recommended Preprocessing Steps\n"
        "- [ ] *e.g. Drop columns with >60% missing*\n"
        "- [ ] *e.g. Impute column X with median*\n"
        "- [ ] *e.g. Log-transform target*\n"
        "- [ ] *e.g. One-hot encode low-cardinality categoricals*\n"
        "- [ ] *e.g. Move final pipeline to `src/train.py`*"
    ))

    nb.cells = cells
    return nb


def main():
    filename = "exploration.ipynb"

    # Allow an optional output directory as argument
    if len(sys.argv) > 1:
        out_dir = sys.argv[1]
        os.makedirs(out_dir, exist_ok=True)
        filepath = os.path.join(out_dir, filename)
    else:
        filepath = filename

    # Derive project name from ancestor folder: loan_default_prediction → Loan Default Prediction
    project_dir = os.path.basename(os.path.dirname(os.path.abspath(filepath)))
    # Walk up once more if we're inside a "notebooks" subfolder
    if project_dir.lower() == "notebooks":
        project_dir = os.path.basename(
            os.path.dirname(os.path.dirname(os.path.abspath(filepath)))
        )
    project_name = project_dir.replace("_", " ").replace("-", " ").title()

    if os.path.exists(filepath):
        answer = input(f"⚠️  '{filepath}' already exists. Overwrite? [y/N]: ")
        if answer.lower() != "y":
            print("Aborted.")
            return

    nb = build_template(project_name=project_name)
    with open(filepath, "w", encoding="utf-8") as f:
        nbformat.write(nb, f)

    print(f"✅ Created '{filepath}'")


if __name__ == "__main__":
    main()
