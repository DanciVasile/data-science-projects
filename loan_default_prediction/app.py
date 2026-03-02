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
FIGURES_DIR = BASE_DIR / "reports" / "figures"

# ── Load model ──────────────────────────────────────────────────────────
if MODEL_PATH.exists():
    model = joblib.load(MODEL_PATH)
else:
    model = None


# ── Prediction function ────────────────────────────────────────────────
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
    if model is None:
        return {"Error": "Model not found. Run `python src/train.py` first."}

    # ── Build input DataFrame ───────────────────────────────────────────
    # Must match the columns the model was trained on
    input_data = pd.DataFrame([{
        "loan_amnt": loan_amnt,
        "term_months": float(term_months),
        "int_rate": int_rate,
        "installment": installment,
        "grade": grade,
        "sub_grade": sub_grade,
        "emp_years": float(emp_years),
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


# ── Build the Gradio UI ─────────────────────────────────────────────────
with gr.Blocks(
    title="🏦 Loan Default Predictor",
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
                label="Interest Rate (%)", minimum=5.0, maximum=30.0,
                value=12.0, step=0.5
            )
            installment = gr.Number(
                label="Monthly Installment ($)", value=400.0,
                minimum=20.0, maximum=1500.0
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
                label="Annual Income ($)", value=65000,
                minimum=10000, maximum=500000
            )
            emp_years = gr.Slider(
                label="Employment Length (years)",
                minimum=0, maximum=10, value=5, step=1
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
                label="Debt-to-Income Ratio",
                minimum=0.0, maximum=50.0, value=15.0, step=0.5
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

    # ── Model Performance Section ───────────────────────────────────────
    gr.Markdown("---")

    with gr.Tabs():
        with gr.Tab("📊 Model Performance"):
            if RESULTS_PATH.exists():
                results_df = pd.read_csv(RESULTS_PATH, index_col=0)
                gr.DataFrame(value=results_df, label="Model Comparison")
            else:
                gr.Markdown(
                    "⚠️ Run `python src/train.py` first to generate "
                    "`models/results.csv`."
                )

            # Show saved figures
            figure_files = [
                "roc_curves.png",
                "precision_recall_curves.png",
                "confusion_matrix_xgboost_tuned.png",
                "feature_importances_xgboost.png",
            ]
            with gr.Row():
                for fig_name in figure_files:
                    fig_path = FIGURES_DIR / fig_name
                    if fig_path.exists():
                        gr.Image(
                            value=str(fig_path),
                            label=fig_name.replace("_", " ").replace(
                                ".png", ""
                            ).title(),
                        )

        with gr.Tab("ℹ️ About"):
            gr.Markdown("""
**Loan Default Prediction** uses machine learning to assess the risk of a loan
defaulting based on borrower characteristics and loan parameters.

- **Dataset:** LendingClub accepted loans (2007–2018), ~2.2M rows × 150 columns
- **Model:** XGBoost with tuned hyperparameters via RandomizedSearchCV
- **Evaluation:** ROC-AUC, Precision, Recall, F1-score
- **Preprocessing:** ColumnTransformer with ordinal/nominal/numeric pipelines

Built by **Vasile-Marian Danci** as an end-to-end data science portfolio project.
            """)


# ── Launch ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft())
