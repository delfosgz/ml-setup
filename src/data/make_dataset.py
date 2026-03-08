"""
src/data/make_dataset.py
────────────────────────
STRUCTURAL operations only — no feature creation, no value transformations.

What lives here:
  - Replace null-like strings with np.nan — needed here ONLY to correctly
    compute null ratios for the column-drop threshold decision.
    The same replacement also lives in NullStringReplacer (pipeline step 1)
    to handle any null-like strings that arrive at inference time.
  - Drop columns above null threshold (structural decision, decided on train)
  - Drop rows above null threshold (train only — NEVER applied to test)
  - Split train.csv → train_split / val_split
  - Save to data/interim/

What does NOT live here:
  - Cabin decomposition → CabinTransformer in the sklearn pipeline
  - Any other column derivation → sklearn pipeline transformers
  - group_size → requires full dataset context, cannot be reproduced at
    inference time, therefore discarded entirely.
"""

import numpy as np
import pandas as pd
import yaml
from pathlib import Path
from sklearn.model_selection import train_test_split

from src.utils import replace_string_nulls


def get_null_columns_to_drop(df: pd.DataFrame, threshold: float) -> list:
    """Return columns whose null ratio exceeds threshold."""
    null_ratio = df.isnull().mean()
    return null_ratio[null_ratio > threshold].index.tolist()


def drop_high_null_rows(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Drop rows (train only) where null ratio per row exceeds threshold."""
    row_null_ratio = df.isnull().mean(axis=1)
    n_before = len(df)
    df = df[row_null_ratio <= threshold].reset_index(drop=True)
    n_dropped = n_before - len(df)
    if n_dropped > 0:
        print(f"  Dropped {n_dropped} rows with null ratio > {threshold:.0%}")
    return df


def main():
    with open("configs/baseline.yaml") as f:
        cfg = yaml.safe_load(f)

    RAW     = Path(cfg["paths"]["raw"])
    INTERIM = Path(cfg["paths"]["interim"])
    INTERIM.mkdir(parents=True, exist_ok=True)

    null_cfg   = cfg["null_handling"]
    col_thresh = null_cfg["drop_column_threshold"]
    row_thresh = null_cfg["drop_row_threshold"]

    # ── Load ──────────────────────────────────────────────────────────────────
    print("Loading raw data...")
    train_raw = pd.read_csv(RAW / "train.csv")
    test_raw  = pd.read_csv(RAW / "test.csv")

    # ── Replace null-like strings ────────────────────────────────────────────
    print("Replacing null-like strings...")
    train_proc = replace_string_nulls(train_raw.copy())
    test_proc  = replace_string_nulls(test_raw.copy())

    # ── Column null threshold (decided on train, applied to both) ────────────
    cols_to_drop = get_null_columns_to_drop(train_proc, col_thresh)
    if cols_to_drop:
        print(f"  Dropping columns (null > {col_thresh:.0%}): {cols_to_drop}")
        train_proc = train_proc.drop(columns=cols_to_drop)
        test_proc  = test_proc.drop(columns=[c for c in cols_to_drop if c in test_proc.columns])

    # ── Row null threshold (TRAIN ONLY — never applied to test) ──────────────
    train_proc = drop_high_null_rows(train_proc, row_thresh)

    # ── Split ─────────────────────────────────────────────────────────────────
    target   = cfg["target"]
    stratify = train_proc[target] if cfg["split"]["stratify"] else None

    train_split, val_split = train_test_split(
        train_proc,
        test_size    = cfg["split"]["val_size"],
        random_state = cfg["random_seed"],
        stratify     = stratify,
    )

    print(f"\nSplit sizes → train: {train_split.shape} | val: {val_split.shape} | test: {test_proc.shape}")

    # ── Save to interim/ ──────────────────────────────────────────────────────
    train_split.reset_index(drop=True).to_parquet(INTERIM / "train_split.parquet",   index=False)
    val_split.reset_index(drop=True).to_parquet(  INTERIM / "val_split.parquet",     index=False)
    test_proc.reset_index(drop=True).to_parquet(  INTERIM / "test_original.parquet", index=False)

    print(f"Saved to {INTERIM}/")


if __name__ == "__main__":
    main()