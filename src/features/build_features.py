"""
src/features/build_features.py
────────────────────────────────
VALUE transformations and feature creation. Reads from data/interim/, writes to data/processed/.

Rules:
  - All feature creation lives inside sklearn Transformers (inference-safe)
  - Pipeline is fit ONLY on train_split. Val and test use transform only.
  - Encoding strategy (ordinal / onehot / target) controlled via configs/
  - RFE optional, toggled via configs/
  - Golden rule: every transformer must be applicable to a single row in production.
  - group_size is NOT created here — it requires full dataset context and
    cannot be reproduced at inference time.

Pipeline order:
  1. NullStringReplacer   → catches "None", "null", "", etc. → np.nan
  2. DropColumnsTransformer → removes Name
  3. GroupIdTransformer    → extracts group_id, drops PassengerId
  4. CabinTransformer      → decomposes Cabin, drops Cabin
  5. TotalSpendTransformer → engineers total_spend
  6. ColumnTransformer     → impute + encode/scale
"""

import joblib
import numpy as np
import pandas as pd
import yaml
from pathlib import Path

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import RFE
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from src.utils import NULL_STRINGS


# ── Transformer: replace null-like strings with np.nan ───────────────────────
class NullStringReplacer(BaseEstimator, TransformerMixin):
    """
    Replaces null-like strings ("None", "null", "", "unknown", etc.) with np.nan.
    Uses NULL_STRINGS from src/utils.py — single source of truth shared with
    make_dataset.py. Must be the first step in the pipeline.
    Inference-safe: operates row by row on string columns.
    """
    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for col in X.select_dtypes(include="object").columns:
            X[col] = X[col].apply(
                lambda x: np.nan
                if isinstance(x, str) and x.strip().lower() in NULL_STRINGS
                else x
            )
        return X


# ── Transformer: decompose Cabin → cabin_deck, cabin_num, cabin_side ─────────
class CabinTransformer(BaseEstimator, TransformerMixin):
    """
    Decomposes raw Cabin string ('B/12/P') into three features.
    Inference-safe: operates on a single row, no external context needed.
    Drops the original Cabin column.
    """
    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        if "Cabin" in X.columns:
            cabin = X["Cabin"].astype(str).str.split("/", expand=True)
            X["cabin_deck"] = cabin[0].replace("nan", np.nan)
            X["cabin_num"]  = pd.to_numeric(cabin[1] if cabin.shape[1] > 1 else np.nan, errors="coerce")
            X["cabin_side"] = cabin[2].replace("nan", np.nan) if cabin.shape[1] > 2 else np.nan
            X = X.drop(columns=["Cabin"])
        return X


# ── Transformer: extract group_id from PassengerId ───────────────────────────
class GroupIdTransformer(BaseEstimator, TransformerMixin):
    """
    Extracts group_id from PassengerId ('0012_01' → '0012').
    Inference-safe: derived from a single row value, no dataset context needed.
    Drops PassengerId after extraction.

    NOTE: group_size is NOT extracted here. It requires counting occurrences
    across the full dataset — impossible at single-row inference.
    """
    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        if "PassengerId" in X.columns:
            X["group_id"] = X["PassengerId"].astype(str).str.split("_").str[0]
            X = X.drop(columns=["PassengerId"])
        return X


# ── Transformer: total_spend = sum of all luxury service columns ─────────────
SPEND_COLS = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]


class TotalSpendTransformer(BaseEstimator, TransformerMixin):
    """
    Engineers total_spend from luxury service columns.
    Inference-safe: operates on a single row.
    """
    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        present = [c for c in SPEND_COLS if c in X.columns]
        X["total_spend"] = X[present].fillna(0).sum(axis=1)
        return X


# ── Transformer: drop Name (identifier, not predictive) ──────────────────────
class DropColumnsTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, cols: list):
        self.cols = cols

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X.drop(columns=[c for c in self.cols if c in X.columns], errors="ignore")


# ── Build unfitted preprocessor ───────────────────────────────────────────────
def build_preprocessor(cfg: dict) -> Pipeline:
    """
    Builds a full unfitted preprocessing pipeline from configs/baseline.yaml.
    Order:
      1. Drop non-predictive identifiers (Name)
      2. Extract group_id from PassengerId
      3. Decompose Cabin into cabin_deck, cabin_num, cabin_side
      4. Engineer total_spend
      5. Impute + encode/scale via ColumnTransformer
    """
    feat         = cfg["features"]
    num_cols     = feat["numerical_cols"]
    cat_cols     = feat["categorical_cols"]
    encoding     = feat.get("encoding", "ordinal")
    num_strategy = feat.get("numerical_imputer", "median")
    num_fill     = feat.get("numerical_fill_value", 0)
    cat_strategy = feat.get("categorical_imputer", "most_frequent")
    cat_fill     = feat.get("categorical_fill_value", "missing")

    # Numerical pipeline
    num_pipe = Pipeline([
        ("imputer", SimpleImputer(
            strategy   = num_strategy,
            fill_value = num_fill if num_strategy == "constant" else None,
        )),
        ("scaler", StandardScaler()),
    ])

    # Categorical pipeline
    cat_imputer = SimpleImputer(
        strategy   = cat_strategy,
        fill_value = cat_fill if cat_strategy == "constant" else None,
    )
    if encoding == "onehot":
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    elif encoding == "target":
        try:
            from sklearn.preprocessing import TargetEncoder
            encoder = TargetEncoder(random_state=cfg["random_seed"])
        except ImportError:
            print("  TargetEncoder requires sklearn >= 1.3. Falling back to OrdinalEncoder.")
            encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    else:
        encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)

    cat_pipe = Pipeline([("imputer", cat_imputer), ("encoder", encoder)])

    col_transformer = ColumnTransformer(
        transformers=[
            ("num", num_pipe, num_cols),
            ("cat", cat_pipe, cat_cols),
        ],
        remainder="drop",
    )

    return Pipeline([
        ("replace_null_strings", NullStringReplacer()),
        ("drop_non_predictive",  DropColumnsTransformer(["Name"])),
        ("extract_group_id",     GroupIdTransformer()),
        ("decompose_cabin",      CabinTransformer()),
        ("add_total_spend",      TotalSpendTransformer()),
        ("col_transformer",      col_transformer),
    ])


def main():
    with open("configs/baseline.yaml") as f:
        cfg = yaml.safe_load(f)

    INTERIM   = Path(cfg["paths"]["interim"])
    PROCESSED = Path(cfg["paths"]["processed"])
    PROCESSED.mkdir(parents=True, exist_ok=True)

    target    = cfg["target"]
    drop_cols = cfg["features"].get("drop_cols", [])
    rfe_cfg   = cfg["rfe"]

    # ── Load interim splits ───────────────────────────────────────────────────
    print("Loading interim data...")
    train = pd.read_parquet(INTERIM / "train_split.parquet")
    val   = pd.read_parquet(INTERIM / "val_split.parquet")
    test  = pd.read_parquet(INTERIM / "test_original.parquet")

    # Drop YAML-configured identifier columns
    for df in [train, val, test]:
        df.drop(columns=drop_cols, errors="ignore", inplace=True)

    y_train = train[target].astype(int).values
    y_val   = val[target].astype(int).values
    X_train = train.drop(columns=[target], errors="ignore")
    X_val   = val.drop(columns=[target],   errors="ignore")
    X_test  = test.drop(columns=[target, "PassengerId"], errors="ignore")

    # ── Fit preprocessor on train only ───────────────────────────────────────
    print("Fitting preprocessor on train_split...")
    preprocessor = build_preprocessor(cfg)

    if cfg["features"].get("encoding") == "target":
        X_train_t = preprocessor.fit_transform(X_train, y_train)
    else:
        X_train_t = preprocessor.fit_transform(X_train)

    X_val_t  = preprocessor.transform(X_val)
    X_test_t = preprocessor.transform(X_test)

    print(f"  Shapes → train: {X_train_t.shape} | val: {X_val_t.shape} | test: {X_test_t.shape}")

    # ── Optional RFE (fit on train only) ─────────────────────────────────────
    rfe_selector = None
    if rfe_cfg["enabled"]:
        print(f"Fitting RFE (n_features={rfe_cfg['n_features_to_select']})...")
        rfe_selector = RFE(
            estimator            = XGBClassifier(n_estimators=100, random_state=cfg["random_seed"], tree_method="hist", eval_metric="logloss"),
            n_features_to_select = rfe_cfg["n_features_to_select"],
            step                 = rfe_cfg["step"],
        )
        X_train_t = rfe_selector.fit_transform(X_train_t, y_train)
        X_val_t   = rfe_selector.transform(X_val_t)
        X_test_t  = rfe_selector.transform(X_test_t)
        print(f"  After RFE → train: {X_train_t.shape}")

    # ── Save to processed/ ────────────────────────────────────────────────────
    pd.DataFrame(X_train_t).to_parquet(PROCESSED / "X_train.parquet", index=False)
    pd.DataFrame(X_val_t).to_parquet(  PROCESSED / "X_val.parquet",   index=False)
    pd.DataFrame(X_test_t).to_parquet( PROCESSED / "X_test.parquet",  index=False)
    pd.DataFrame({"target": y_train}).to_parquet(PROCESSED / "y_train.parquet", index=False)
    pd.DataFrame({"target": y_val}).to_parquet(  PROCESSED / "y_val.parquet",   index=False)

    joblib.dump(preprocessor, PROCESSED / "preprocessor.pkl")
    if rfe_selector:
        joblib.dump(rfe_selector, PROCESSED / "rfe_selector.pkl")

    print(f"Saved to {PROCESSED}/")


if __name__ == "__main__":
    main()