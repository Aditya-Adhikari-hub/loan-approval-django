import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import (
    CATEGORICAL_FEATURES, DATA_PATH, FEATURES, NUMERIC_FEATURES,
    POSITIVE_LABEL, RANDOM_STATE, TARGET, TEST_SIZE,
)


def load_data(path=DATA_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


def validate_data(df: pd.DataFrame) -> dict:
    required = set(FEATURES) | {TARGET}
    missing_cols = required - set(df.columns)
    if missing_cols:
        raise ValueError(f"Dataset is missing columns: {sorted(missing_cols)}")
    labels = set(df[TARGET].unique())
    if labels != {"Approved", "Denied"}:
        raise ValueError(f"Unexpected target labels: {labels}")
    return {
        "rows": int(len(df)),
        "missing_values": int(df[FEATURES + [TARGET]].isnull().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "class_counts": {k: int(v) for k, v in df[TARGET].value_counts().items()},
    }


def make_xy(df: pd.DataFrame):
    X = df[FEATURES].copy()
    y = (df[TARGET] == POSITIVE_LABEL).astype(int)
    return X, y


def split_data(X, y):
    """Stratified split, done BEFORE any fitting or resampling."""
    return train_test_split(X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE)


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        [
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ],
        verbose_feature_names_out=False,
    )