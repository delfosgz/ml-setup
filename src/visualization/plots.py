"""
src/visualization/plots.py
───────────────────────────
Reusable plot functions. Each function:
  - Receives data / model as input
  - Returns a matplotlib Figure
  - Optionally logs the figure as an MLflow artifact via mlflow.log_figure()
Never reads from data/ directly. Never writes to disk independently.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    PrecisionRecallDisplay,
)


def plot_confusion_matrix(
    y_true, y_pred, *, title="Confusion Matrix", log_to_mlflow=False, run=None
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, ax=ax, colorbar=False,
        display_labels=["Not Transported", "Transported"],
    )
    ax.set_title(title)
    fig.tight_layout()
    if log_to_mlflow:
        import mlflow
        mlflow.log_figure(fig, "confusion_matrix.png")
    return fig


def plot_roc_curve(
    y_true, y_proba, *, title="ROC Curve", log_to_mlflow=False
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(5, 4))
    RocCurveDisplay.from_predictions(y_true, y_proba, ax=ax)
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_title(title)
    fig.tight_layout()
    if log_to_mlflow:
        import mlflow
        mlflow.log_figure(fig, "roc_curve.png")
    return fig


def plot_precision_recall(
    y_true, y_proba, *, title="Precision-Recall Curve", log_to_mlflow=False
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(5, 4))
    PrecisionRecallDisplay.from_predictions(y_true, y_proba, ax=ax)
    ax.set_title(title)
    fig.tight_layout()
    if log_to_mlflow:
        import mlflow
        mlflow.log_figure(fig, "precision_recall_curve.png")
    return fig


def plot_feature_importance(
    model, feature_names=None, top_n=20, *, title="Feature Importance", log_to_mlflow=False
) -> plt.Figure:
    importance = model.feature_importances_
    if feature_names is None:
        feature_names = [f"f{i}" for i in range(len(importance))]

    indices = np.argsort(importance)[-top_n:]
    fig, ax = plt.subplots(figsize=(7, max(4, top_n * 0.35)))
    ax.barh(
        range(len(indices)),
        importance[indices],
        align="center",
        color="steelblue",
    )
    ax.set_yticks(range(len(indices)))
    ax.set_yticklabels([feature_names[i] for i in indices])
    ax.set_xlabel("Importance")
    ax.set_title(title)
    fig.tight_layout()
    if log_to_mlflow:
        import mlflow
        mlflow.log_figure(fig, "feature_importance.png")
    return fig


def plot_training_history(
    evals_result: dict, *, title="Training History", log_to_mlflow=False
) -> plt.Figure:
    """Plots XGBoost training vs validation loss over iterations."""
    fig, ax = plt.subplots(figsize=(7, 4))
    for split, metrics in evals_result.items():
        for metric, values in metrics.items():
            ax.plot(values, label=f"{split} {metric}")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    if log_to_mlflow:
        import mlflow
        mlflow.log_figure(fig, "training_history.png")
    return fig


def plot_null_heatmap(df: pd.DataFrame, *, title="Null Heatmap") -> plt.Figure:
    """
    EDA helper — call from notebooks/, export to reports/figures/.
    Shows null percentage per column as a heatmap.
    """
    null_pct = df.isnull().mean().sort_values(ascending=False)
    null_pct = null_pct[null_pct > 0]
    if null_pct.empty:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No nulls found", ha="center")
        return fig
    fig, ax = plt.subplots(figsize=(8, max(3, len(null_pct) * 0.4)))
    ax.barh(null_pct.index, null_pct.values * 100, color="salmon")
    ax.set_xlabel("Null %")
    ax.set_title(title)
    ax.axvline(60, color="red", linestyle="--", lw=1, label="drop threshold (60%)")
    ax.legend()
    fig.tight_layout()
    return fig