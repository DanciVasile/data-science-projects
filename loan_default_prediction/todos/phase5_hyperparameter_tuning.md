# 🎨 Phase 5 — Hyperparameter Tuning

> **Where:** `src/train.py` (add tuning function)  
> **Prereq:** [Phase 4 — Model Training](phase4_model_training.md) completed  
> **Next:** [Phase 6 — Evaluation](phase6_evaluation.md)

---

## 🧒 What Are We Doing Here? (The Big Picture)

In Phase 4, we trained models with manually chosen hyperparameters (like
`max_depth=6`, `learning_rate=0.1`). These were educated guesses — but maybe
`max_depth=8` and `learning_rate=0.05` would be better?

**Hyperparameter tuning** is like adjusting the knobs on an oven:
- Temperature too low → bread is undercooked (underfitting)
- Temperature too high → bread burns (overfitting)
- Just right → golden brown (good generalization)

We automate this by trying many combinations and letting cross-validation tell us
which one performs best.

---

## ✅ TODO 5.1 — Understand the Search Space

**Purpose:** Know which hyperparameters to tune and what range of values to try.
Don't tune everything — focus on the parameters that matter most.

**XGBoost parameters to tune (ranked by impact):**

| Parameter | What It Controls | Search Range | Default |
|-----------|-----------------|-------------|---------|
| `max_depth` | Tree depth (complexity) | 3, 5, 7, 9 | 6 |
| `learning_rate` | Step size per boosting round | 0.01, 0.05, 0.1, 0.2 | 0.1 |
| `n_estimators` | Number of boosting rounds | 100, 200, 300, 500 | 300 |
| `subsample` | Fraction of data per tree | 0.6, 0.7, 0.8, 0.9 | 0.8 |
| `colsample_bytree` | Fraction of features per tree | 0.6, 0.7, 0.8, 0.9 | 0.8 |
| `min_child_weight` | Min sum of weights in a leaf | 1, 3, 5, 10 | 1 |
| `gamma` | Minimum loss reduction for split | 0, 0.1, 0.3, 0.5 | 0 |

**Why RandomizedSearchCV instead of GridSearchCV?**
- **GridSearchCV** tries every combination: 4 × 4 × 4 × 4 × 4 × 4 × 4 = 16,384 combos × 3 folds = 49,152 model fits. On 2M rows, that's weeks.
- **RandomizedSearchCV** tries N random combinations (e.g., 50). Much faster, and research shows it finds near-optimal results with far fewer trials.

- [ ] Read the parameter table and understand what each one controls
- [ ] Understand why GridSearch is impractical for large datasets

---

## ✅ TODO 5.2 — Implement Hyperparameter Tuning

**Purpose:** Find the best hyperparameters for the best-performing model (likely XGBoost).

**Add to `train.py`:**

```python
from scipy.stats import uniform, randint

def tune_xgboost(preprocessor, X_train, y_train):
    """Tune XGBoost hyperparameters using RandomizedSearchCV."""
    divider("7. HYPERPARAMETER TUNING (XGBOOST)")

    n_pos = (y_train == 1).sum()
    n_neg = (y_train == 0).sum()
    scale_weight = n_neg / n_pos

    # ── Base pipeline (preprocessor + untuned XGBoost) ──────────────────
    pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", XGBClassifier(
            scale_pos_weight=scale_weight,
            eval_metric="auc",
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbosity=0,
        )),
    ])

    # ── Search space ────────────────────────────────────────────────────
    # Prefix with "classifier__" because parameters are inside the Pipeline
    param_distributions = {
        "classifier__max_depth":        randint(3, 10),         # 3 to 9
        "classifier__learning_rate":    uniform(0.01, 0.19),    # 0.01 to 0.20
        "classifier__n_estimators":     randint(100, 501),      # 100 to 500
        "classifier__subsample":        uniform(0.6, 0.3),      # 0.6 to 0.9
        "classifier__colsample_bytree": uniform(0.6, 0.3),      # 0.6 to 0.9
        "classifier__min_child_weight": randint(1, 11),         # 1 to 10
        "classifier__gamma":           uniform(0, 0.5),         # 0 to 0.5
    }

    # ── RandomizedSearchCV ──────────────────────────────────────────────
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)

    search = RandomizedSearchCV(
        pipe,
        param_distributions,
        n_iter=30,              # try 30 random combinations (practical for 1M+ rows)
        scoring="roc_auc",      # optimize for ROC-AUC
        cv=cv,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=1,              # show progress
        return_train_score=True,
    )

    log.info(f"  🔍 Searching 50 random combinations × 3 folds = 150 fits")
    log.info(f"     This will take a while on 2M rows... ☕")
    search.fit(X_train, y_train)

    # ── Results ─────────────────────────────────────────────────────────
    log.info(f"\n  🏆 Best ROC-AUC (CV): {search.best_score_:.4f}")
    log.info(f"  🎯 Best parameters:")
    for param, value in search.best_params_.items():
        clean_name = param.replace("classifier__", "")
        log.info(f"     {clean_name:25s} = {value}")

    # Check for overfitting: train score vs test score
    results = pd.DataFrame(search.cv_results_)
    best_idx = search.best_index_
    train_score = results.loc[best_idx, "mean_train_score"]
    test_score = results.loc[best_idx, "mean_test_score"]
    log.info(f"\n  📊 Train AUC: {train_score:.4f}  |  Test AUC: {test_score:.4f}")
    if train_score - test_score > 0.05:
        log.info(f"  ⚠️  Gap > 5% — possible overfitting!")
    else:
        log.info(f"  ✅ Minimal gap — good generalization")

    return search.best_estimator_
```

**Key functions to learn:**

| Function | What It Does |
|----------|-------------|
| `RandomizedSearchCV(estimator, param_distributions, n_iter=30)` | Try 30 random combos from the distribution |
| `scipy.stats.randint(low, high)` | Random integer from [low, high) |
| `scipy.stats.uniform(loc, scale)` | Random float from [loc, loc+scale) |
| `search.best_score_` | Best cross-validation score found |
| `search.best_params_` | Parameters that achieved the best score |
| `search.best_estimator_` | The fully fitted model with best params |

**Why `"classifier__max_depth"` with the prefix?**
When tuning inside a `Pipeline`, sklearn needs to know which step the parameter
belongs to. The naming convention is `stepname__parametername`.

- [ ] Add `tune_xgboost()` to train.py
- [ ] Add `from scipy.stats import uniform, randint` to imports
- [ ] Call in `main()`: `best_model = tune_xgboost(preprocessor, X_train, y_train)`
- [ ] Evaluate: `best_metrics, best_pred, best_prob = evaluate_model("XGBoost (Tuned)", best_model, X_test, y_test)`
- [ ] Compare tuned vs untuned XGBoost ROC-AUC — is there improvement?

---

## ✅ TODO 5.3 — Save the Best Model

**Purpose:** Persist the trained model to disk so it can be loaded later
(for the Gradio dashboard in Phase 7, or for production deployment).

**Add to `main()` after tuning:**

```python
    # ── Save the best model ─────────────────────────────────────────────
    save_model(best_model, "xgboost_tuned")

    # Also save the results for the README
    all_results = pd.DataFrame([lr_metrics, rf_metrics, xgb_metrics, best_metrics])
    all_results = all_results.set_index("model").round(4)
    all_results.to_csv(MODELS_DIR / "results.csv")
    log.info(f"\n  📊 Results saved to models/results.csv")
```

**What gets saved?**
`joblib.dump()` serializes the **entire Pipeline** — preprocessor + model — into
a single `.pkl` file. To predict on new data later:

```python
import joblib
model = joblib.load("models/xgboost_tuned.pkl")
prediction = model.predict(new_data_df)  # raw data in → prediction out
```

The pipeline handles preprocessing internally — no need to manually impute,
scale, encode. That's the beauty of sklearn Pipelines.

- [ ] Add model saving code to `main()`
- [ ] Run the full pipeline and verify `models/xgboost_tuned.pkl` is created
- [ ] Verify `models/results.csv` is created

---

## ✅ TODO 5.4 — (Optional) Bayesian Optimization with Optuna

**Purpose:** Even smarter than random search. Optuna uses past trial results to
guide where to search next (Bayesian optimization). Not required, but impressive
for a portfolio.

**Install:** Already available if you have `pip install optuna`

```python
# Optional advanced tuning — add as a separate function
import optuna

def tune_with_optuna(preprocessor, X_train, y_train, n_trials=50):
    """Bayesian hyperparameter optimization using Optuna."""

    X_processed = preprocessor.fit_transform(X_train)

    n_pos = (y_train == 1).sum()
    n_neg = (y_train == 0).sum()
    scale_weight = n_neg / n_pos

    def objective(trial):
        params = {
            "max_depth": trial.suggest_int("max_depth", 3, 9),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "subsample": trial.suggest_float("subsample", 0.6, 0.9),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 0.9),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "gamma": trial.suggest_float("gamma", 0, 0.5),
            "scale_pos_weight": scale_weight,
            "eval_metric": "auc",
            "random_state": RANDOM_STATE,
            "n_jobs": -1,
            "verbosity": 0,
        }

        model = XGBClassifier(**params)
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
        scores = cross_val_score(model, X_processed, y_train,
                                 cv=cv, scoring="roc_auc", n_jobs=-1)
        return scores.mean()

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    log.info(f"  🏆 Best trial AUC: {study.best_value:.4f}")
    log.info(f"  🎯 Best params: {study.best_params}")

    return study
```

- [ ] (Optional) Install optuna: `uv pip install optuna`
- [ ] (Optional) Add the Optuna function
- [ ] (Optional) Compare Optuna result with RandomizedSearchCV result

---

## 🎯 Phase 5 Checklist

When all TODOs above are done, you should have:

- [ ] Understood hyperparameter search spaces
- [ ] `tune_xgboost()` implemented with RandomizedSearchCV
- [ ] Best model saved to `models/xgboost_tuned.pkl`
- [ ] Results comparison saved to `models/results.csv`
- [ ] Tuned model shows improvement over default parameters
- [ ] (Optional) Bayesian optimization explored
- [ ] Ready for [Phase 6 — Evaluation](phase6_evaluation.md)!

---

## 📚 Concepts to Remember

| Concept | Explanation |
|---------|-------------|
| **Hyperparameter** | A setting chosen BEFORE training (not learned from data) |
| **GridSearchCV** | Exhaustive search over a parameter grid — exact but slow |
| **RandomizedSearchCV** | Random sampling from distributions — fast, nearly as good |
| **Bayesian optimization** | Uses past results to guide the search — smart and efficient |
| **n_iter** | Number of random combinations to try (higher = better but slower) |
| **Overfitting check** | Compare train score vs CV score — large gap = overfitting |
| **`classifier__param`** | Naming convention for parameters inside a sklearn Pipeline |
| **joblib.dump** | Serialize a Python object to a binary file (.pkl) |
