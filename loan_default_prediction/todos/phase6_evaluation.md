# 🎨 Phase 6 — Model Evaluation & Visualization

> **Where:** `src/train.py` (add plotting functions)  
> **Prereq:** [Phase 5 — Hyperparameter Tuning](phase5_hyperparameter_tuning.md) completed  
> **Next:** [Phase 7 — Gradio Dashboard](phase7_gradio.md)

---

## 🧒 What Are We Doing Here? (The Big Picture)

You've trained and tuned models — now prove they work. This is the "show your work"
phase that separates a portfolio project from a tutorial copy-paste.

A number like "ROC-AUC = 0.72" means little without context. Plots make metrics
**visual and intuitive** for recruiters and stakeholders.

---

## ✅ TODO 6.1 — ROC Curve

**Purpose:** Visualize the trade-off between True Positive Rate (catching defaults)
and False Positive Rate (false alarms) at every possible threshold.

**What is a threshold?** The model outputs a probability (e.g., 0.73). If the
threshold is 0.5, any loan with probability > 0.5 is classified as "default."
Lower threshold → catch more defaults, but more false alarms.

**Add to `train.py`:**

```python
def plot_roc_curves(results_list, y_test):
    """Plot ROC curves for all models on one figure."""
    fig, ax = plt.subplots(figsize=(8, 8))

    for name, y_prob in results_list:
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc = roc_auc_score(y_test, y_prob)
        ax.plot(fpr, tpr, linewidth=2, label=f"{name} (AUC = {auc:.3f})")

    # Diagonal = random classifier (AUC = 0.5)
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random (AUC = 0.500)")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate (Recall)", fontsize=12)
    ax.set_title("ROC Curves — Model Comparison", fontsize=14)
    ax.legend(loc="lower right", fontsize=11)
    ax.set_aspect("equal")
    plt.tight_layout()
    save_figure(fig, "roc_curves")
    plt.close(fig)
```

**How to read an ROC curve:**
```
TPR (y-axis)
 1.0 ┌──────────────────────┐
     │         ╭────────────│  ← Good model (hugs top-left corner)
     │       ╱              │
     │     ╱                │
 0.5 │   ╱   ╱ ← Random    │
     │  ╱  ╱    (diagonal)  │
     │╱ ╱                   │
 0.0 └──────────────────────┘
    0.0        0.5        1.0
         FPR (x-axis)
```

**Key functions:**

| Function | Returns |
|----------|---------|
| `roc_curve(y_true, y_prob)` | `fpr, tpr, thresholds` — the curve coordinates |
| `roc_auc_score(y_true, y_prob)` | Single number: area under the curve |

- [ ] Add `plot_roc_curves()` to train.py
- [ ] Call with all models: `plot_roc_curves([("LR", lr_prob), ("RF", rf_prob), ("XGBoost", best_prob)], y_test)`
- [ ] Verify `reports/figures/roc_curves.png` is created
- [ ] Compare: which model's curve hugs the top-left corner best?

---

## ✅ TODO 6.2 — Precision-Recall Curve

**Purpose:** More informative than ROC for imbalanced datasets. Shows the trade-off
between precision (how many flagged loans actually default) and recall (how many
defaults you catch).

**Why better than ROC for imbalanced data?** ROC's FPR denominator is the *negative*
class (Fully Paid, ~80%). Even many false alarms barely move FPR. Precision uses the
*predictions* as denominator, which is more honest.

**Add to `train.py`:**

```python
def plot_precision_recall_curves(results_list, y_test):
    """Plot Precision-Recall curves for all models."""
    fig, ax = plt.subplots(figsize=(8, 8))

    for name, y_prob in results_list:
        precision, recall, _ = precision_recall_curve(y_test, y_prob)
        ap = average_precision_score(y_test, y_prob)
        ax.plot(recall, precision, linewidth=2, label=f"{name} (AP = {ap:.3f})")

    # Baseline = proportion of positives (random classifier)
    baseline = y_test.mean()
    ax.axhline(baseline, color="k", linestyle="--", linewidth=1,
               label=f"Random (AP = {baseline:.3f})")

    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title("Precision-Recall Curves — Model Comparison", fontsize=14)
    ax.legend(loc="upper right", fontsize=11)
    plt.tight_layout()
    save_figure(fig, "precision_recall_curves")
    plt.close(fig)
```

**How to read a PR curve:**
- **Top-right corner** = perfect (precision=1, recall=1)
- Higher curve = better model
- The "knee" of the curve is usually the practical operating point

- [ ] Add `plot_precision_recall_curves()` to train.py
- [ ] Call and save the figure
- [ ] Verify `reports/figures/precision_recall_curves.png` is created

---

## ✅ TODO 6.3 — Confusion Matrix

**Purpose:** A 2×2 table showing exactly where the model gets it right and wrong.

```
                  Predicted
                  Paid    Default
Actual  Paid   [ TN     │  FP  ]   ← FP = "false alarm" (flagged but didn't default)
        Default[ FN     │  TP  ]   ← FN = "missed default" (didn't flag but defaulted)
```

**Add to `train.py`:**

```python
def plot_confusion_matrix(name, y_test, y_pred):
    """Plot a confusion matrix heatmap."""
    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt=",d", cmap="Blues", ax=ax,
                xticklabels=["Fully Paid", "Charged Off"],
                yticklabels=["Fully Paid", "Charged Off"])
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("Actual", fontsize=12)
    ax.set_title(f"Confusion Matrix — {name}", fontsize=14)
    plt.tight_layout()
    save_figure(fig, f"confusion_matrix_{name.lower().replace(' ', '_').replace('(', '').replace(')', '')}")
    plt.close(fig)
```

**How to read it:**

| Cell | Name | Meaning | Good if... |
|------|------|---------|-----------|
| Top-left | True Negative (TN) | Correctly predicted Fully Paid | High |
| Top-right | False Positive (FP) | Predicted Default but actually Paid | Low |
| Bottom-left | False Negative (FN) | Predicted Paid but actually Defaulted | Low (this is costly!) |
| Bottom-right | True Positive (TP) | Correctly predicted Default | High |

**For a bank:**
- FN (missed default) costs the entire unpaid loan balance
- FP (false alarm) only costs a rejected good customer
- So FN is **much worse** than FP → we want high recall

- [ ] Add `plot_confusion_matrix()` to train.py
- [ ] Call for the best model: `plot_confusion_matrix("XGBoost Tuned", y_test, best_pred)`
- [ ] Verify the figure is saved
- [ ] Analyze: how many defaults did the model miss (bottom-left cell)?

---

## ✅ TODO 6.4 — Feature Importance Bar Chart

**Purpose:** Show which features the model relies on most. This is the "explainability"
that stakeholders and recruiters care about.

**Add to `train.py`:**

```python
def plot_feature_importances(name, feat_imp, top_n=20):
    """Plot a horizontal bar chart of top feature importances."""
    top = feat_imp.head(top_n).sort_values()

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(top.index, top.values, color="steelblue", edgecolor="black")
    ax.set_xlabel("Importance", fontsize=12)
    ax.set_title(f"Top {top_n} Feature Importances — {name}", fontsize=14)
    plt.tight_layout()
    save_figure(fig, f"feature_importances_{name.lower().replace(' ', '_').replace('(', '').replace(')', '')}")
    plt.close(fig)
```

- [ ] Add `plot_feature_importances()` to train.py
- [ ] Call with XGBoost or RF importances
- [ ] Verify the figure is saved
- [ ] Note: are engineered features (income_to_loan, credit_history_years) in the top 20?

---

## ✅ TODO 6.5 — Classification Report

**Purpose:** A one-shot summary of precision, recall, F1 per class. Built into sklearn.

**Add to `main()` after evaluation:**

```python
    # ── Detailed classification report ──────────────────────────────────
    divider("CLASSIFICATION REPORT (BEST MODEL)")
    report = classification_report(
        y_test, best_pred,
        target_names=["Fully Paid", "Charged Off"],
    )
    log.info(f"\n{report}")
```

**Sample output:**
```
              precision    recall  f1-score   support
  Fully Paid       0.89      0.75      0.81    160000
 Charged Off       0.42      0.67      0.52     40000
    accuracy                           0.73    200000
   macro avg       0.66      0.71      0.67    200000
weighted avg       0.80      0.73      0.75    200000
```

**How to read it:**
- **Fully Paid precision = 0.89** → 89% of loans predicted as "Fully Paid" actually were
- **Charged Off recall = 0.67** → We caught 67% of actual defaults
- **macro avg** → Simple average of both classes (treats them equally)
- **weighted avg** → Average weighted by class size (affected by imbalance)

- [ ] Add classification report to `main()`
- [ ] Run and review per-class metrics
- [ ] Key question: is Charged Off recall acceptable? (>0.60 is decent for LendingClub)

---

## ✅ TODO 6.6 — Generate All Figures in `main()`

**Purpose:** Wire everything together so `python src/train.py` generates all figures.

**Add to the end of `main()`:**

```python
    # ── Generate all evaluation figures ─────────────────────────────────
    divider("GENERATING FIGURES")

    results_for_plot = [
        ("Logistic Regression", lr_prob),
        ("Random Forest", rf_prob),
        ("XGBoost (Tuned)", best_prob),
    ]

    plot_roc_curves(results_for_plot, y_test)
    plot_precision_recall_curves(results_for_plot, y_test)
    plot_confusion_matrix("XGBoost (Tuned)", y_test, best_pred)
    plot_feature_importances("XGBoost", xgb_feat_imp)

    log.info(f"\n  ✅ All figures saved to {FIGURES_DIR}")
```

- [ ] Add the figure generation block to `main()`
- [ ] Run the full pipeline end-to-end: `python src/train.py`
- [ ] Verify all 4 PNGs exist in `reports/figures/`

---

## 🎯 Phase 6 Checklist

When all TODOs above are done, you should have:

- [ ] `reports/figures/roc_curves.png` — all models compared
- [ ] `reports/figures/precision_recall_curves.png` — imbalance-aware comparison
- [ ] `reports/figures/confusion_matrix_xgboost_tuned.png` — error breakdown
- [ ] `reports/figures/feature_importances_xgboost.png` — top 20 features
- [ ] Classification report printed in terminal
- [ ] `models/results.csv` — all metrics in a table
- [ ] Ready for [Phase 7 — Gradio Dashboard](phase7_gradio.md)!

---

## 📚 Concepts to Remember

| Concept | Explanation |
|---------|-------------|
| **ROC Curve** | Plots TPR vs FPR at every threshold. AUC = overall ranking quality |
| **Precision-Recall Curve** | Better for imbalanced data. Area under = Average Precision |
| **Confusion Matrix** | 2×2 table of TN, FP, FN, TP |
| **True Positive (TP)** | Correctly predicted the positive class (default) |
| **False Negative (FN)** | Missed a positive (predicted "Paid" but actually defaulted) — most costly |
| **Threshold** | Probability cutoff for classification. Default is 0.5 but can be tuned |
| **Feature Importance** | How much each feature contributed to splits in tree models |
| **Classification Report** | Per-class precision, recall, F1 + averages |
