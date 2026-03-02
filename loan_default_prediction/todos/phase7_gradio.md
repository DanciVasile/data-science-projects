# 🎨 Phase 7 — Gradio Dashboard

> **Where:** `loan_default_prediction/app.py` (new file)  
> **Prereq:** [Phase 6 — Evaluation](phase6_evaluation.md) completed — need saved model  
> **Next:** [Phase 8 — Documentation](phase8_documentation.md)

---

## 🧒 What Are We Doing Here? (The Big Picture)

You've built a working ML model. Now make it **interactive** so anyone — recruiters,
hiring managers, non-technical stakeholders — can play with it. A Gradio dashboard
turns your model from a `.pkl` file into a live product.

Gradio is a Python library that creates web apps with pure Python — no HTML/CSS/JS
needed. You write Python functions, and Gradio renders them as a beautiful web
interface with input controls and output displays.

**This is what makes a portfolio project "end-to-end."** Most candidates stop at
model training. You'll have a deployable, interactive application.

---

## ✅ TODO 7.1 — Verify Gradio Installation

**Purpose:** Gradio is already in `pyproject.toml` (`gradio ≥6.8.0`). Verify it's
available.

```powershell
# From the workspace root:
uv run python -c "import gradio; print(gradio.__version__)"
```

If not installed: `uv pip install gradio`

- [ ] Verify Gradio is installed and prints a version number (6.8.0+)

---

## ✅ TODO 7.2 — Create `app.py` with Basic Structure

**Purpose:** Build the app in layers. Start with the skeleton, then add features.

**Create `loan_default_prediction/app.py`:**

```python
"""
Loan Default Prediction — Gradio Dashboard
===========================================
Interactive web app for predicting loan default risk.

Usage:
    python app.py      (run from loan_default_prediction/ directory)
    # or: gradio app.py
"""

import gradio as gr
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# ── Configuration ───────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "xgboost_tuned.pkl"
RESULTS_PATH = BASE_DIR / "models" / "results.csv"

# ── Load model ──────────────────────────────────────────────────────────
if MODEL_PATH.exists():
    model = joblib.load(MODEL_PATH)
else:
    model = None  # app still loads — shows error on predict
```

**Key Gradio concepts:**

| Concept | What It Does |
|---------|-------------|
| `gr.Interface()` | Quick one-function-in, one-output app |
| `gr.Blocks()` | Full layout control (rows, columns, tabs) |
| `gr.Number()` | Numeric input field |
| `gr.Slider()` | Slider for numeric ranges |
| `gr.Dropdown()` | Single choice from a list |
| `gr.Label()` | Show prediction with confidence scores |
| `gr.DataFrame()` | Display a pandas DataFrame |
| `gr.Image()` | Display a saved figure |

- [ ] Create `loan_default_prediction/app.py` with the skeleton above
- [ ] Test: `python app.py` — should run without errors (no UI yet)

---

## ✅ TODO 7.3 — Build the Prediction Function

**Purpose:** Gradio works by wrapping a plain Python function. You define the
function, and Gradio creates the input/output UI automatically.

**Add to `app.py`:**

```python
def predict_default(
    loan_amnt: float,
    term_months: int,
    int_rate: float,
    installment: float,
    grade: str,
    sub_grade: str,
    annual_inc: float,
    emp_years: int,
    home_ownership: str,
    verification_status: str,
    purpose: str,
    dti: float,
) -> dict:
    """Take loan parameters and return default probability."""

    # ── Build input DataFrame ───────────────────────────────────────────
    # Must match the columns the model was trained on
    input_data = pd.DataFrame([{
        "loan_amnt": loan_amnt,
        "term_months": term_months,
        "int_rate": int_rate,
        "installment": installment,
        "grade": grade,
        "sub_grade": sub_grade,
        "emp_years": emp_years,
        "home_ownership": home_ownership,
        "annual_inc": annual_inc,
        "verification_status": verification_status,
        "purpose": purpose,
        "dti": dti,
        # Engineered features
        "income_to_loan": annual_inc / max(loan_amnt, 1),
        "installment_to_income": installment / max(annual_inc / 12, 1),
    }])

    # ── Align columns with model expectations ───────────────────────────
    # The model's Pipeline preprocessor expects specific column names.
    # Fill any missing columns with NaN — the imputer handles them.
    try:
        expected_cols = model.named_steps["preprocessor"].feature_names_in_
        for col in expected_cols:
            if col not in input_data.columns:
                input_data[col] = np.nan
        input_data = input_data[expected_cols]
    except AttributeError:
        pass  # model doesn't have named_steps — use columns as-is

    # ── Predict ─────────────────────────────────────────────────────────
    proba = model.predict_proba(input_data)[0][1]  # probability of default

    return {
        "Fully Paid": float(1 - proba),
        "Default": float(proba),
    }
```

**How Gradio differs from Streamlit:**

| Streamlit (old) | Gradio (new) | Explanation |
|-----------------|-------------|-------------|
| Reruns entire script on interaction | Calls one function | Gradio only re-executes your function |
| `st.sidebar.number_input()` | `gr.Number()` | Input widget |
| `st.sidebar.selectbox()` | `gr.Dropdown()` | Dropdown choice |
| `st.sidebar.slider()` | `gr.Slider()` | Range slider |
| `st.metric("Label", value)` | `gr.Label()` | Output display |
| `st.cache_resource` | Not needed | Model loaded once at module level |

- [ ] Add the `predict_default()` function to app.py
- [ ] The function takes numbers/strings as input, returns a dictionary

---

## ✅ TODO 7.4 — Build the UI with `gr.Blocks`

**Purpose:** Create the full dashboard layout with input controls, prediction
output, and a performance tab.

**Add to `app.py`:**

```python
# ── Build the Gradio UI ─────────────────────────────────────────────────
with gr.Blocks(
    title="🏦 Loan Default Predictor",
    theme=gr.themes.Soft(),
) as demo:

    gr.Markdown("# 🏦 Loan Default Prediction")
    gr.Markdown(
        "Predict whether a loan will **default** based on borrower characteristics. "
        "Adjust the inputs and click **Predict** to see the result."
    )

    with gr.Row():
        # ── LEFT COLUMN: Inputs ─────────────────────────────────────────
        with gr.Column(scale=1):
            gr.Markdown("### 📋 Loan Parameters")

            loan_amnt = gr.Number(
                label="Loan Amount ($)", value=15000, minimum=500, maximum=40000
            )
            term_months = gr.Dropdown(
                label="Term (months)", choices=[36, 60], value=36
            )
            int_rate = gr.Slider(
                label="Interest Rate (%)", minimum=5.0, maximum=30.0, value=12.0, step=0.5
            )
            installment = gr.Number(
                label="Monthly Installment ($)", value=400.0, minimum=20.0, maximum=1500.0
            )
            grade = gr.Dropdown(
                label="Grade", choices=list("ABCDEFG"), value="C"
            )
            sub_grade = gr.Dropdown(
                label="Sub Grade",
                choices=[f"{g}{n}" for g in "ABCDEFG" for n in range(1, 6)],
                value="C3",
            )

            gr.Markdown("### 👤 Borrower Information")

            annual_inc = gr.Number(
                label="Annual Income ($)", value=65000, minimum=10000, maximum=500000
            )
            emp_years = gr.Slider(
                label="Employment Length (years)", minimum=0, maximum=10, value=5, step=1
            )
            home_ownership = gr.Dropdown(
                label="Home Ownership",
                choices=["RENT", "OWN", "MORTGAGE", "OTHER"],
                value="RENT",
            )
            verification_status = gr.Dropdown(
                label="Income Verification",
                choices=["Not Verified", "Source Verified", "Verified"],
                value="Not Verified",
            )
            purpose = gr.Dropdown(
                label="Loan Purpose",
                choices=[
                    "debt_consolidation", "credit_card", "home_improvement",
                    "major_purchase", "small_business", "car", "medical",
                    "moving", "vacation", "house", "wedding",
                    "renewable_energy", "educational", "other",
                ],
                value="debt_consolidation",
            )
            dti = gr.Slider(
                label="Debt-to-Income Ratio", minimum=0.0, maximum=50.0, value=15.0, step=0.5
            )

            predict_btn = gr.Button("🔮 Predict", variant="primary")

        # ── RIGHT COLUMN: Output ────────────────────────────────────────
        with gr.Column(scale=1):
            gr.Markdown("### 🎯 Prediction")
            prediction_label = gr.Label(label="Default Risk")

    # ── Wire the button to the function ─────────────────────────────────
    predict_btn.click(
        fn=predict_default,
        inputs=[
            loan_amnt, term_months, int_rate, installment,
            grade, sub_grade, annual_inc, emp_years,
            home_ownership, verification_status, purpose, dti,
        ],
        outputs=prediction_label,
    )
```

**Gradio layout building blocks:**

| Component | What It Does |
|-----------|-------------|
| `gr.Blocks()` | Top-level container — gives you full layout control |
| `gr.Row()` | Arrange children side by side |
| `gr.Column(scale=N)` | Stacked layout; `scale` controls relative width |
| `gr.Tab("name")` | Create a tab within a `gr.Tabs()` |
| `gr.Markdown("# ...")` | Render Markdown text |
| `gr.Button()` | Clickable button — `.click()` wires it to a function |

- [ ] Add the `gr.Blocks` UI to app.py
- [ ] Test: `python app.py` — should open a browser with the form
- [ ] Verify: click **Predict** and see a label with "Fully Paid" vs "Default" probabilities

---

## ✅ TODO 7.5 — Add Model Performance Tab

**Purpose:** Show the model's metrics and figures so users understand how reliable
the predictions are. Transparency builds trust.

**Add inside the `with gr.Blocks(...) as demo:` block, after the prediction row:**

```python
    # ── Model Performance Section ───────────────────────────────────────
    gr.Markdown("---")

    with gr.Tabs():
        with gr.Tab("📊 Model Performance"):
            if RESULTS_PATH.exists():
                results_df = pd.read_csv(RESULTS_PATH, index_col=0)
                gr.DataFrame(value=results_df, label="Model Comparison")
            else:
                gr.Markdown(
                    "⚠️ Run `python src/train.py` first to generate `models/results.csv`."
                )

            # Show saved figures
            figures_dir = BASE_DIR / "reports" / "figures"
            figure_files = [
                "roc_curves.png",
                "precision_recall_curves.png",
                "confusion_matrix_xgboost_tuned.png",
                "feature_importances_xgboost.png",
            ]
            with gr.Row():
                for fig_name in figure_files:
                    fig_path = figures_dir / fig_name
                    if fig_path.exists():
                        gr.Image(
                            value=str(fig_path),
                            label=fig_name.replace("_", " ").replace(".png", "").title(),
                            show_download_button=False,
                        )

        with gr.Tab("ℹ️ About"):
            gr.Markdown("""
**Loan Default Prediction** uses machine learning to assess the risk of a loan
defaulting based on borrower characteristics and loan parameters.

- **Dataset:** LendingClub accepted loans (2007–2018)
- **Model:** XGBoost with tuned hyperparameters
- **Evaluation:** ROC-AUC, Precision, Recall, F1-score

Built by **Vasile-Marian Danci** as an end-to-end data science portfolio project.
            """)
```

**Gradio output components:**

| Component | What It Does |
|-----------|-------------|
| `gr.Label()` | Classification result with confidence bars |
| `gr.DataFrame()` | Interactive table for pandas DataFrames |
| `gr.Image()` | Display a saved figure (PNG, JPG, SVG) |
| `gr.Markdown()` | Render formatted text |
| `gr.Tabs()` / `gr.Tab()` | Tabbed sections |

- [ ] Add tabs and model performance display inside the Blocks context
- [ ] Test: all tabs should render (figures only appear after running `train.py`)

---

## ✅ TODO 7.6 — Launch & Final Testing

**Purpose:** End-to-end smoke test of the dashboard.

**Add at the bottom of `app.py`:**

```python
# ── Launch ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    demo.launch()
```

**Run the app:**

```powershell
# From loan_default_prediction/ directory:
python app.py
# Or: gradio app.py   (auto-reloads on file changes — great for development)
```

**Test scenarios:**

| Scenario | Expected Result |
|----------|----------------|
| High income ($200k), low loan ($5k), Grade A | Low risk, ~0–10% default probability |
| Low income ($30k), high loan ($35k), Grade G, 60 months | High risk, >50% default probability |
| Average everything | Medium risk, ~15–25% |

- [ ] Test all three scenarios above
- [ ] Verify the label colors change (green for Fully Paid, red for Default)
- [ ] Take a screenshot for the README
- [ ] (Optional) Deploy to Hugging Face Spaces for a free live URL

---

## 🎯 Phase 7 Checklist

When all TODOs above are done, you should have:

- [ ] `app.py` created with input controls, prediction output, and performance tabs
- [ ] Dashboard loads the saved model and makes predictions
- [ ] `gr.Label` visually communicates default probability with confidence bars
- [ ] Model performance metrics and figures displayed in tabs
- [ ] Three test scenarios pass with reasonable results
- [ ] Ready for [Phase 8 — Documentation](phase8_documentation.md)!

---

## 📚 Concepts to Remember

| Concept | Explanation |
|---------|-------------|
| **Gradio** | Python library for building ML web apps — no frontend code needed |
| **`gr.Blocks()`** | Full-control layout with rows, columns, tabs |
| **`gr.Interface()`** | Quick single-function wrapper (simpler but less flexible) |
| **Feature alignment** | Model expects same columns in same order as training data |
| **`predict_proba`** | Returns probability [P(class=0), P(class=1)] instead of hard 0/1 |
| **Risk bucketing** | Converting continuous probability into categories (Low/Medium/High) |
| **Hugging Face Spaces** | Free hosting for Gradio apps — deploy from a GitHub repo |
