"""
Step 5-7: Model training, comparison, hyperparameter tuning, and
explainability for the Mental Health Score Prediction project.

Run: python3 src/train.py
Outputs:
  - models/model.pkl              (best tuned model, full pipeline)
  - models/metrics.json           (comparison table + tuning results)
  - models/feature_importance.json
"""

import json
import time
import warnings
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor

from preprocessing import TARGET, build_preprocessor, load_clean_data

warnings.filterwarnings("ignore")

DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "dataset.csv")
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")


def evaluate(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    return {"MAE": round(mae, 3), "MSE": round(mse, 3), "RMSE": round(rmse, 3), "R2": round(r2, 4)}


def main():
    df = load_clean_data(DATA_PATH)
    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    preprocessor = build_preprocessor()

    candidates = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(random_state=42),
        "Lasso Regression": Lasso(random_state=42),
        "Decision Tree": DecisionTreeRegressor(random_state=42, max_depth=6),
        "Random Forest": RandomForestRegressor(random_state=42, n_estimators=200),
        "Gradient Boosting": GradientBoostingRegressor(random_state=42),
    }

    results = {}
    fitted_pipelines = {}
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    print("=" * 70)
    print("BASELINE MODEL COMPARISON (test set)")
    print("=" * 70)
    for name, model in candidates.items():
        pipe = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)
        metrics = evaluate(y_test, preds)
        cv_scores = cross_val_score(pipe, X_train, y_train, cv=kf, scoring="r2")
        metrics["CV_R2_mean"] = round(cv_scores.mean(), 4)
        metrics["CV_R2_std"] = round(cv_scores.std(), 4)
        results[name] = metrics
        fitted_pipelines[name] = pipe
        print(f"{name:20s} -> {metrics}")

    comparison_df = pd.DataFrame(results).T.sort_values("R2", ascending=False)
    print("\nComparison table (sorted by test R2):\n", comparison_df)

    best_name = comparison_df.index[0]
    print(f"\nBest baseline model: {best_name}")

    # ---- Hyperparameter tuning on the strongest model ----
    print("\n" + "=" * 70)
    print(f"HYPERPARAMETER TUNING: {best_name}")
    print("=" * 70)

    baseline_score = results[best_name]["R2"]

    if best_name == "Random Forest":
        param_grid = {
            "model__n_estimators": [150, 250, 350],
            "model__max_depth": [None, 8, 12],
            "model__min_samples_leaf": [1, 3, 5],
        }
        base_model = RandomForestRegressor(random_state=42)
    elif best_name == "Gradient Boosting":
        param_grid = {
            "model__n_estimators": [100, 200, 300],
            "model__learning_rate": [0.03, 0.05, 0.1],
            "model__max_depth": [2, 3, 4],
        }
        base_model = GradientBoostingRegressor(random_state=42)
    elif best_name == "Ridge Regression":
        param_grid = {"model__alpha": [0.1, 1.0, 5.0, 10.0, 50.0]}
        base_model = Ridge(random_state=42)
    elif best_name == "Lasso Regression":
        param_grid = {"model__alpha": [0.001, 0.01, 0.1, 1.0]}
        base_model = Lasso(random_state=42)
    elif best_name == "Decision Tree":
        param_grid = {"model__max_depth": [4, 6, 8, 10], "model__min_samples_leaf": [1, 3, 5]}
        base_model = DecisionTreeRegressor(random_state=42)
    else:
        param_grid = {}
        base_model = LinearRegression()

    tuned_pipe = Pipeline(steps=[("preprocessor", preprocessor), ("model", base_model)])

    if param_grid:
        t0 = time.time()
        grid = GridSearchCV(tuned_pipe, param_grid, cv=kf, scoring="r2", n_jobs=-1)
        grid.fit(X_train, y_train)
        tuned_pipe = grid.best_estimator_
        print(f"Best params: {grid.best_params_}  (search took {time.time()-t0:.1f}s)")
    else:
        tuned_pipe.fit(X_train, y_train)

    tuned_preds = tuned_pipe.predict(X_test)
    tuned_metrics = evaluate(y_test, tuned_preds)
    print(f"Tuned {best_name} test metrics: {tuned_metrics}")
    print(f"Improvement in R2: {tuned_metrics['R2'] - baseline_score:+.4f}")

    # ---- Feature importance (best final model) ----
    feature_names = tuned_pipe.named_steps["preprocessor"].get_feature_names_out()
    final_model = tuned_pipe.named_steps["model"]
    importance_dict = {}
    if hasattr(final_model, "feature_importances_"):
        importances = final_model.feature_importances_
        importance_dict = dict(sorted(
            zip(feature_names, importances.tolist()), key=lambda x: -x[1]
        ))
    elif hasattr(final_model, "coef_"):
        coefs = final_model.coef_
        importance_dict = dict(sorted(
            zip(feature_names, np.abs(coefs).tolist()), key=lambda x: -x[1]
        ))

    print("\nTop 10 feature importances:")
    for k, v in list(importance_dict.items())[:10]:
        print(f"  {k}: {v:.4f}")

    # ---- Save everything ----
    joblib.dump(tuned_pipe, f"{MODELS_DIR}/model.pkl")

    all_metrics = {
        "baseline_comparison": results,
        "best_baseline_model": best_name,
        "tuned_metrics": tuned_metrics,
        "baseline_r2_of_best": baseline_score,
        "feature_importance": importance_dict,
        "test_actual": y_test.tolist(),
        "test_predicted": tuned_preds.tolist(),
    }
    with open(f"{MODELS_DIR}/metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=2)

    print(f"\nSaved model to {MODELS_DIR}/model.pkl")
    print(f"Saved metrics to {MODELS_DIR}/metrics.json")


if __name__ == "__main__":
    main()
