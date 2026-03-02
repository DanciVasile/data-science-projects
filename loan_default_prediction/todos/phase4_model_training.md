# 🎨 Phase 4 — Model Training

> **Where:** `src/train.py` (continue adding functions)  
> **Prereq:** [Phase 3 — Feature Engineering](phase3_feature_engineering.md) completed  
> **Next:** [Phase 5 — Hyperparameter Tuning](phase5_hyperparameter_tuning.md)

---

## 🧒 What Are We Doing Here? (The Big Picture)

Now the fun part — training actual machine learning models! We'll train three
models of increasing complexity:

1. **Logistic Regression** — the baseline. Simple, fast, interpretable. If this works
   well, you might not need anything fancier.
2. **Random Forest** — an ensemble of decision trees. Handles non-linear relationships,
   doesn't need feature scaling (but we do it anyway for consistency).
3. **XGBoost** — the competition-winning algorithm. Builds trees sequentially, each one
   correcting the mistakes of the previous. State-of-the-art for tabular data.

Think of it like cooking: you start with a simple recipe (logistic regression),
then try a more complex one (random forest), then the gourmet version (XGBoost).
You compare all three to see which is best.

---

## ✅ TODO 4.1 — Understand Class Imbalance Handling

**Purpose:** ~80% of loans are Fully Paid, ~20% are Charged Off. If the model just
predicts "Fully Paid" for everyone, it gets 80% accuracy — but catches 0% of defaults.
That's useless for a bank.

**How we handle it — `class_weight="balanced"`:**

Instead of treating every sample equally, we tell the model:
*"Mistakes on Charged Off loans cost MORE than mistakes on Fully Paid loans."*

The formula:
$$w_c = \frac{n_{samples}}{n_{classes} \times n_{samples\_in\_class\_c}}$$

For our data (~80/20 split):
- Weight for Fully Paid (majority): ~0.63
- Weight for Charged Off (minority): ~2.5

This means the model penalizes itself **4× more** for missing a default.

**Why NOT SMOTE?**
- SMOTE (Synthetic Minority Over-sampling) creates **fake** minority samples.
- With 400k+ defaults already, the minority class is huge. SMOTE is unnecessary.
- SMOTE on 2M rows is extremely slow and memory-intensive.
- Class weighting achieves the same effect with zero overhead.

- [ ] Read and understand the class weighting formula above
- [ ] Understand why SMOTE is overkill here

---

## ✅ TODO 4.2 — `evaluate_model()` Helper Function

**Purpose:** Standardized evaluation that we'll reuse for every model.

**Add to `train.py`:**

```python
def evaluate_model(name: str, model, X_test, y_test):
    """Evaluate a trained model and return a metrics dictionary."""
    log.info(f"\n  📈 Evaluating: {name}")

    y_pred = model.predict(X_test)

    # Some models support probability predictions (needed for ROC-AUC)
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]  # probability of class 1 (default)
    else:
        y_prob = y_pred.astype(float)

    metrics = {
        "model": name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_prob),
        "avg_precision": average_precision_score(y_test, y_prob),
    }

    log.info(f"     Accuracy:     {metrics['accuracy']:.4f}")
    log.info(f"     Precision:    {metrics['precision']:.4f}")
    log.info(f"     Recall:       {metrics['recall']:.4f}")
    log.info(f"     F1-score:     {metrics['f1']:.4f}")
    log.info(f"     ROC-AUC:      {metrics['roc_auc']:.4f}")
    log.info(f"     Avg Precision: {metrics['avg_precision']:.4f}")

    return metrics, y_pred, y_prob
```

**Metric cheat sheet — what each metric tells you:**

| Metric | Formula (intuitive) | What It Answers | Good Value |
|--------|-------------------|-----------------|------------|
| **Accuracy** | correct / total | Overall correctness | >0.80 (but misleading with imbalance!) |
| **Precision** | true defaults / predicted defaults | "Of loans I flagged as risky, how many actually defaulted?" | Higher = fewer false alarms |
| **Recall** | true defaults / actual defaults | "Of all loans that defaulted, how many did I catch?" | Higher = fewer missed defaults |
| **F1-score** | $\frac{2 \times P \times R}{P + R}$ | Harmonic mean of precision & recall | Balance between P and R |
| **ROC-AUC** | Area under the ROC curve | Model's ability to rank defaults higher than non-defaults | 0.5 = random, 1.0 = perfect |
| **Avg Precision** | Area under precision-recall curve | Like ROC-AUC but better for imbalanced data | Higher is better |

**Which metric matters most for a bank?**
- **Recall** — a missed default costs the bank the entire loan amount.
- **ROC-AUC** — overall ranking quality.
- For your portfolio, report **all** metrics; discuss the trade-offs in the README.

- [ ] Add `evaluate_model()` to train.py
- [ ] Study the metric table — understand each one

---

## ✅ TODO 4.3 — Train Logistic Regression (Baseline)

**Purpose:** A simple, fast baseline model. If it achieves reasonable performance,
you know the features have signal. If it doesn't, the problem might be very non-linear
(which is fine — tree models handle that).

**Add to `train.py`:**

```python
def train_logistic_regression(preprocessor, X_train, y_train):
    """Train a Logistic Regression with class weighting."""
    divider("6A. LOGISTIC REGRESSION (BASELINE)")

    # Build full pipeline: preprocessor → classifier
    pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(
            class_weight="balanced",  # handle imbalance
            max_iter=1000,            # ensure convergence
            solver="lbfgs",           # good general-purpose solver
            random_state=RANDOM_STATE,
            n_jobs=-1,                # use all CPU cores
        )),
    ])

    # Cross-validation — estimates performance on unseen data
    log.info("  🔄 Running 5-fold stratified cross-validation...")
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_val_score(pipe, X_train, y_train, cv=cv,
                             scoring="roc_auc", n_jobs=-1)
    log.info(f"     CV ROC-AUC: {scores.mean():.4f} ± {scores.std():.4f}")

    # Final fit on full training set
    log.info("  🏋️ Fitting on full training set...")
    pipe.fit(X_train, y_train)

    return pipe
```

**Key concepts:**

| Concept | Explanation |
|---------|-------------|
| `Pipeline([("preprocessor", ...), ("classifier", ...)])` | Chains preprocessing + model into one object — `fit()` and `predict()` on raw data |
| `class_weight="balanced"` | Automatically computes weights inversely proportional to class frequency |
| `max_iter=1000` | Maximum iterations for the optimization algorithm to converge |
| `solver="lbfgs"` | Limited-memory BFGS — efficient for large datasets |
| `StratifiedKFold` | Splits data into K folds preserving class proportions in each fold |
| `cross_val_score` | Trains K models, each tested on a different fold — gives a reliable estimate |
| `n_jobs=-1` | Use all available CPU cores for parallelism |

**Cross-validation illustrated:**
```
Fold 1: [TEST | TRAIN | TRAIN | TRAIN | TRAIN] → score_1
Fold 2: [TRAIN | TEST | TRAIN | TRAIN | TRAIN] → score_2
Fold 3: [TRAIN | TRAIN | TEST | TRAIN | TRAIN] → score_3
Fold 4: [TRAIN | TRAIN | TRAIN | TEST | TRAIN] → score_4
Fold 5: [TRAIN | TRAIN | TRAIN | TRAIN | TEST] → score_5
Final estimate = mean(score_1 ... score_5) ± std
```

- [ ] Add `train_logistic_regression()` to train.py
- [ ] Call in `main()`: `lr_pipe = train_logistic_regression(preprocessor, X_train, y_train)`
- [ ] Evaluate: `lr_metrics, lr_pred, lr_prob = evaluate_model("Logistic Regression", lr_pipe, X_test, y_test)`
- [ ] Note: CV is slow on 2M rows. Expect 5-15 minutes.

---

## ✅ TODO 4.4 — Train Random Forest

**Purpose:** Random Forest builds many decision trees (a "forest"), each trained on a
random subset of data and features, then averages their predictions. It captures
non-linear relationships that logistic regression misses.

**Add to `train.py`:**

```python
def train_random_forest(preprocessor, X_train, y_train):
    """Train a Random Forest with class weighting."""
    divider("6B. RANDOM FOREST")

    pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=200,         # number of trees
            max_depth=15,             # limit tree depth to prevent overfitting
            min_samples_leaf=50,      # each leaf needs ≥50 samples (regularization)
            class_weight="balanced",  # handle imbalance
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )),
    ])

    log.info("  🔄 Running 5-fold stratified cross-validation...")
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_val_score(pipe, X_train, y_train, cv=cv,
                             scoring="roc_auc", n_jobs=-1)
    log.info(f"     CV ROC-AUC: {scores.mean():.4f} ± {scores.std():.4f}")

    log.info("  🏋️ Fitting on full training set...")
    pipe.fit(X_train, y_train)

    # Feature importances (from the tree-based model)
    feature_names = pipe.named_steps["preprocessor"].get_feature_names_out()
    importances = pipe.named_steps["classifier"].feature_importances_
    feat_imp = pd.Series(importances, index=feature_names).sort_values(ascending=False)
    log.info(f"\n  🌳 Top 10 features:")
    for name, imp in feat_imp.head(10).items():
        log.info(f"     {name:40s} {imp:.4f}")

    return pipe, feat_imp
```

**Hyperparameter intuition:**

| Parameter | What It Means | Low Value | High Value |
|-----------|--------------|-----------|------------|
| `n_estimators=200` | Number of trees in the forest | More variance, less stable | More stable, slower |
| `max_depth=15` | How deep each tree can grow | Underfitting (too simple) | Overfitting (memorizes data) |
| `min_samples_leaf=50` | Minimum samples per leaf node | Overfitting (too specific) | Underfitting (too general) |

**Feature importance:** Random Forest can tell you which features mattered most.
This is invaluable for understanding the model and explaining it to stakeholders.

**How feature importance works:** Each time a feature is used to split a node,
it measures how much the split reduces impurity (Gini). Features that reduce
impurity the most are "more important."

- [ ] Add `train_random_forest()` to train.py
- [ ] Call in `main()` and evaluate
- [ ] Compare CV ROC-AUC with Logistic Regression
- [ ] Note: which features are in the top 10 importance list?

---

## ✅ TODO 4.5 — Train XGBoost

**Purpose:** XGBoost (eXtreme Gradient Boosting) builds trees **sequentially** — each
new tree focuses on the errors of the previous ones. This is called "boosting."
It's consistently the top performer on structured/tabular data.

**Add to `train.py`:**

```python
def train_xgboost(preprocessor, X_train, y_train):
    """Train an XGBoost classifier with class imbalance handling."""
    divider("6C. XGBOOST")

    # Calculate scale_pos_weight = n_negative / n_positive
    n_pos = (y_train == 1).sum()
    n_neg = (y_train == 0).sum()
    scale_weight = n_neg / n_pos
    log.info(f"  ⚖️  scale_pos_weight: {scale_weight:.2f} (neg/pos ratio)")

    pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", XGBClassifier(
            n_estimators=300,           # number of boosting rounds
            max_depth=6,                # shallower than RF — boosting compensates
            learning_rate=0.1,          # step size — lower = more rounds needed
            subsample=0.8,              # use 80% of data per tree (regularization)
            colsample_bytree=0.8,       # use 80% of features per tree
            scale_pos_weight=scale_weight,  # handle imbalance
            eval_metric="auc",          # optimize for AUC during training
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbosity=0,               # suppress XGBoost warnings
        )),
    ])

    log.info("  🔄 Running 5-fold stratified cross-validation...")
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_val_score(pipe, X_train, y_train, cv=cv,
                             scoring="roc_auc", n_jobs=-1)
    log.info(f"     CV ROC-AUC: {scores.mean():.4f} ± {scores.std():.4f}")

    log.info("  🏋️ Fitting on full training set...")
    pipe.fit(X_train, y_train)

    # Feature importances
    feature_names = pipe.named_steps["preprocessor"].get_feature_names_out()
    importances = pipe.named_steps["classifier"].feature_importances_
    feat_imp = pd.Series(importances, index=feature_names).sort_values(ascending=False)
    log.info(f"\n  🚀 Top 10 features:")
    for name, imp in feat_imp.head(10).items():
        log.info(f"     {name:40s} {imp:.4f}")

    return pipe, feat_imp
```

**XGBoost key ideas:**

| Concept | Analogy |
|---------|---------|
| **Boosting** | Each student corrects the previous student's wrong answers |
| **Learning rate** | How much each new tree adjusts — smaller = more cautious, needs more trees |
| **Subsampling** | Each tree only sees a random 80% of data — reduces overfitting |
| **scale_pos_weight** | XGBoost's version of `class_weight="balanced"` |

**Random Forest vs XGBoost:**

| Aspect | Random Forest | XGBoost |
|--------|--------------|---------|
| Tree building | Independent (parallel) | Sequential (each corrects previous) |
| Overfitting | Harder to overfit | Can overfit if not tuned |
| Training speed | Fast (parallel) | Slower (sequential) |
| Typical performance | Good | Usually better |
| Interpretability | Feature importances | Feature importances (built-in) |

- [ ] Add `train_xgboost()` to train.py
- [ ] Call in `main()` and evaluate
- [ ] Compare CV ROC-AUC across all three models
- [ ] XGBoost should typically perform best

---

## ✅ TODO 4.6 — Model Comparison Table

**Purpose:** Show all models side by side — this is what goes in your README.

**Add to `main()` after all three models are trained and evaluated:**

```python
    # ── Model Comparison ────────────────────────────────────────────────
    divider("MODEL COMPARISON")
    results = pd.DataFrame([lr_metrics, rf_metrics, xgb_metrics])
    results = results.set_index("model")
    log.info("\n" + results.round(4).to_string())

    # Identify best model by ROC-AUC
    best_name = results["roc_auc"].idxmax()
    log.info(f"\n  🏆 Best model by ROC-AUC: {best_name}")
```

- [ ] Add comparison code to `main()`
- [ ] Run the full pipeline: `python src/train.py`
- [ ] Document which model wins and by how much

---

## 🎯 Phase 4 Checklist

When all TODOs above are done, you should have:

- [ ] `evaluate_model()` helper written and tested
- [ ] Logistic Regression trained with `class_weight="balanced"`
- [ ] Random Forest trained with feature importances extracted
- [ ] XGBoost trained with `scale_pos_weight`
- [ ] All three models compared in a table
- [ ] You know which model is best by ROC-AUC
- [ ] Ready for [Phase 5 — Hyperparameter Tuning](phase5_hyperparameter_tuning.md)!

---

## 📚 Concepts to Remember

| Concept | Explanation |
|---------|-------------|
| **Logistic Regression** | Linear model that outputs probability via sigmoid function: $P(y=1) = \frac{1}{1+e^{-z}}$ |
| **Random Forest** | Ensemble of independent decision trees → average their votes |
| **XGBoost** | Gradient-boosted trees → each tree corrects previous errors |
| **Cross-validation** | Train/test multiple times on different folds → reliable performance estimate |
| **Class weighting** | Make mistakes on the minority class more "expensive" |
| **Feature importance** | How much each feature contributed to the model's decisions |
| **ROC-AUC** | Area under receiver operating characteristic curve — 0.5 = random, 1.0 = perfect |
| **Precision vs Recall trade-off** | Catching more defaults (recall ↑) means more false alarms (precision ↓) |
