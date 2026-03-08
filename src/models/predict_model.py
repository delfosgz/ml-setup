"""
src/models/predict_model.py
────────────────────────────
Loads the full production pipeline from MLflow and generates the Kaggle submission.
Rules:
  - Loads INTERIM test data (structurally processed, pre-value transformations)
  - The MLflow pipeline handles all remaining transformations end-to-end
  - Writes submission.csv to data/raw/ (Kaggle format)
  - Writes final evaluation plots to reports/figures/
"""

import mlflow
import mlflow.sklearn
import pandas as pd
import yaml
from pathlib import Path

from src.visualization.plots import plot_confusion_matrix, plot_roc_curve
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score,
    precision_score, recall_score, log_loss,
)


def main(run_id: str):
    with open("configs/baseline.yaml") as f:
        cfg = yaml.safe_load(f)

    INTERIM  = Path(cfg["paths"]["interim"])
    RAW      = Path(cfg["paths"]["raw"])
    REPORTS  = Path(cfg["paths"]["reports"]) / "figures"
    REPORTS.mkdir(parents=True, exist_ok=True)

    # ── Load production pipeline from MLflow ─────────────────────────────────
    print(f"Loading pipeline from MLflow run: {run_id}")
    pipeline = mlflow.sklearn.load_model(f"runs:/{run_id}/production_pipeline")

    # ── Load interim test data (pipeline handles all value transformations) ───
    print("Loading interim test data...")
    test = pd.read_parquet(INTERIM / "test_original.parquet")
    drop_cols = cfg["features"].get("drop_cols", [])
    test = test.drop(columns=drop_cols, errors="ignore")

    passenger_ids = pd.read_parquet(INTERIM / "test_original.parquet")["PassengerId"]
    X_test = test.drop(columns=["PassengerId", cfg["target"]], errors="ignore")

    # ── Generate predictions ──────────────────────────────────────────────────
    print("Generating predictions...")
    predictions = pipeline.predict(X_test).astype(bool)

    # ── Submission file ───────────────────────────────────────────────────────
    submission = pd.DataFrame({
        "PassengerId": passenger_ids,
        "Transported": predictions,
    })
    submission_path = RAW / "submission.csv"
    submission.to_csv(submission_path, index=False)
    print(f"Submission saved to {submission_path} — shape: {submission.shape}")
    print(f"  Transported=True:  {predictions.sum()} ({predictions.mean():.1%})")
    print(f"  Transported=False: {(~predictions).sum()} ({(~predictions).mean():.1%})")

    # ── Optional: evaluate on validation set and export final plots ───────────
    val_path = INTERIM / "val_split.parquet"
    if val_path.exists():
        print("\nGenerating final evaluation plots on val set...")
        val = pd.read_parquet(val_path)
        val = val.drop(columns=drop_cols, errors="ignore")
        y_val     = val[cfg["target"]].astype(int).values
        X_val     = val.drop(columns=[cfg["target"]], errors="ignore")
        X_val     = X_val.drop(columns=["PassengerId"], errors="ignore")

        val_pred  = pipeline.predict(X_val)
        val_proba = pipeline.predict_proba(X_val)[:, 1]

        print("  Final val metrics:")
        print(f"    accuracy:  {accuracy_score(y_val, val_pred):.4f}")
        print(f"    roc_auc:   {roc_auc_score(y_val, val_proba):.4f}")
        print(f"    f1:        {f1_score(y_val, val_pred):.4f}")
        print(f"    precision: {precision_score(y_val, val_pred):.4f}")
        print(f"    recall:    {recall_score(y_val, val_pred):.4f}")
        print(f"    log_loss:  {log_loss(y_val, val_proba):.4f}")

        plot_confusion_matrix(y_val, val_pred, title="Final Confusion Matrix").savefig(
            REPORTS / "final_confusion_matrix.png", dpi=150, bbox_inches="tight"
        )
        plot_roc_curve(y_val, val_proba, title="Final ROC Curve").savefig(
            REPORTS / "final_roc_curve.png", dpi=150, bbox_inches="tight"
        )
        print(f"  Plots saved to {REPORTS}/")


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python -m src.models.predict_model <RUN_ID>")
        print("       Get RUN_ID from: mlflow ui → http://localhost:5000")
        sys.exit(1)
    main(run_id=sys.argv[1])