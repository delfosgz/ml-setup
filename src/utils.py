"""
src/utils.py
─────────────
Shared utilities used across src/data/ and src/features/.
Changes here affect both make_dataset.py and the NullStringReplacer pipeline step.
"""

import numpy as np
import pandas as pd


# Canonical set of null-like strings.
# Used in:
#   - make_dataset.py → to correctly compute null ratios before column-drop decision
#   - NullStringReplacer (build_features.py) → to handle null strings at inference
NULL_STRINGS = {
    "none", "null", "nan", "na", "n/a", "not a value",
    "", " ", "missing", "not available", "unknown", "?", "undefined"
}


def replace_string_nulls(df: pd.DataFrame) -> pd.DataFrame:
    """Replace null-like strings in object columns with np.nan."""
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].apply(
            lambda x: np.nan
            if isinstance(x, str) and x.strip().lower() in NULL_STRINGS
            else x
        )
    return df