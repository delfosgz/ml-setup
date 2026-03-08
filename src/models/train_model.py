"""
src/models/train_model.py
──────────────────────────
Trains XGBoost and logs everything to MLflow.
Rules:
  - Reads ONLY from data/processed/ and configs/
  - Hyperparameters come exclusively from configs/baseline.yaml
  - What gets logged to MLflow is the FULL pipeline:
      Pipeline([preprocessor, (rfe?), model])
    so production inference receives interim-level data and the pipeline
    handles all value transformations internally.
  - Visualization functions are imported from src/visualization/plots.py
"""

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import yaml
from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    log_loss,
    average_precision_score,
)
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.features.build_features import build_preprocessor
from src.visualization.plots import (
    plot_confusion_matrix,
    plot_roc_curve,
    plot_precision_recall,
    plot_feature_importance,
    plot_training_history,
)


def compute_metrics(y_true, y_pred, y_proba) -> dict:
    return {
        "accuracy":          round(accuracy_score(y_true, y_pred),            4),
        "roc_auc":           round(roc_auc_score(y_true, y_proba),            4),
        "f1":                round(f1_score(y_true, y_pred),                  4),
        "precision":         round(precision_score(y_true, y_pred),           4),
        "recall":            round(recall_score(y_true, y_pred),              4),
        "log_loss":          round(log_loss(y_true, y_proba),                 4),
        "avg_precision":     round(average_precision_score(y_true, y_proba),  4),
    }


def main():
    with open("configs/baseline.yaml") as f:
        cfg = yaml.safe_load(f)

    INTERIM   = Path(cfg["paths"]["interim"])
    PROCESSED = Path(cfg["paths"]["processed"])
    REPORTS   = Path(cfg["paths"]["reports"]) / "figures"
    REPORTS.mkdir(parents=True, exist_ok=True)

    # ── Load preprocessed data ────────────────────────────────────────────────
    print("Loading preprocessed data from data/processed/...")
    X_train = pd.read_parquet(PROCESSED / "X_train.parquet").values
    X_val   = pd.read_parquet(PROCESSED / "X_val.parquet").values
    y_train = pd.read_parquet(PROCESSED / "y_train.parquet")["target"].values
    y_val   = pd.read_parquet(PROCESSED / "y_val.parquet")["target"].values

    # ── Optional: apply RFE selector ─────────────────────────────────────────
    rfe_selector = None
    if cfg["rfe"]["enabled"]:
        rfe_path = PROCESSED / "rfe_selector.pkl"
        if rfe_path.exists():
            rfe_selector = joblib.load(rfe_path)
            X_train = rfe_selector.transform(X_train)
            X_val   = rfe_selector.transform(X_val)
            print(f"  RFE applied → shape: {X_train.shape}")

    # ── Train XGBoost with early stopping ─────────────────────────────────────
    params      = cfg["model"]["params"].copy()
    early_stop  = params.pop("early_stopping_rounds", 50)
    n_estimators = params.pop("n_estimators", 500)

    evals_result = {}
    model = XGBClassifier(
        n_estimators          = n_estimators,
        early_stopping_rounds = early_stop,
        **params,
    )
    model.fit(
        X_train, y_train,
        eval_set      = [(X_train, y_train), (X_val, y_val)],
        verbose       = 100,
        callbacks     = [_EvalsDictCallback(evals_result)],
    )
    best_iter = model.best_iteration
    print(f"\nBest iteration: {best_iter}")

    # ── Evaluate ──────────────────────────────────────────────────────────────
    val_pred  = model.predict(X_val)
    val_proba = model.predict_proba(X_val)[:, 1]
    metrics   = compute_metrics(y_val, val_pred, val_proba)

    print("\nValidation metrics:")
    for k, v in metrics.items():
        print(f"  {k:20s}: {v}")

    # ── Build FULL production pipeline (what goes to MLflow) ──────────────────
    # This pipeline takes INTERIM-level data (structurally processed, no value
    # transformations) and produces predictions end-to-end.
    print("\nBuilding full production pipeline...")
    train_interim = pd.read_parquet(INTERIM / "train_split.parquet")
    drop_cols     = cfg["features"].get("drop_cols", [])
    train_interim = train_interim.drop(columns=drop_cols, errors="ignore")
    target_col    = cfg["target"]
    X_interim     = train_interim.drop(columns=[target_col], errors="ignore")

    prod_params = {k: v for k, v in cfg["model"]["params"].items()
                   if k != "early_stopping_rounds"}
    prod_params["n_estimators"] = best_iter

    steps = [("preprocessor", build_preprocessor(cfg))]
    if rfe_selector:
        steps.append(("rfe", rfe_selector))
    steps.append(("model", XGBClassifier(**prod_params)))

    production_pipeline = Pipeline(steps)
    production_pipeline.fit(X_interim, y_train[:len(X_interim)])

    # ── MLflow run ────────────────────────────────────────────────────────────
    mlflow.set_experiment(cfg["experiment_name"])

    with mlflow.start_run() as run:
        print(f"\nMLflow run: {run.info.run_id}")

        # Log config and metrics
        mlflow.log_params(cfg["model"]["params"])
        mlflow.log_param("encoding",    cfg["features"]["encoding"])
        mlflow.log_param("rfe_enabled", cfg["rfe"]["enabled"])
        mlflow.log_param("best_iteration", best_iter)
        mlflow.log_metrics(metrics)

        # Log plots as artifacts (via src/visualization/plots.py)
        plot_confusion_matrix(y_val, val_pred,  log_to_mlflow=True)
        plot_roc_curve(y_val, val_proba,        log_to_mlflow=True)
        plot_precision_recall(y_val, val_proba, log_to_mlflow=True)
        plot_training_history(
            {"train": evals_result.get("validation_0", {}),
             "val":   evals_result.get("validation_1", {})},
            log_to_mlflow=True,
        )

        # Feature importance (best effort — uses preprocessed feature indices)
        n_features = X_train.shape[1]
        plot_feature_importance(
            model,
            feature_names=[f"feature_{i}" for i in range(n_features)],
            log_to_mlflow=True,
        )

        # Save plots to reports/figures/ for human review
        plot_confusion_matrix(y_val, val_pred).savefig(
            REPORTS / "confusion_matrix.png", dpi=150, bbox_inches="tight"
        )
        plot_roc_curve(y_val, val_proba).savefig(
            REPORTS / "roc_curve.png", dpi=150, bbox_inches="tight"
        )

        # Log the full production pipeline
        mlflow.sklearn.log_model(production_pipeline, "production_pipeline")

        print(f"\nRun complete. View at: mlflow ui → http://localhost:5000")
        print(f"Run ID: {run.info.run_id}")

    plt_close_all()


# ── XGBoost callback to capture evals_result ─────────────────────────────────
from xgboost.callback import TrainingCallback
import matplotlib.pyplot as plt


class _EvalsDictCallback(TrainingCallback):
    def __init__(self, evals_result: dict):
        self._store = evals_result

    def after_iteration(self, model, epoch, evals_log):
        for data, metric in evals_log.items():
            if data not in self._store:
                self._store[data] = {}
            for m, v in metric.items():
                if m not in self._store[data]:
                    self._store[data][m] = []
                self._store[data][m].append(v[-1])
        return False


def plt_close_all():
    plt.close("all")


if __name__ == "__main__":
    main()