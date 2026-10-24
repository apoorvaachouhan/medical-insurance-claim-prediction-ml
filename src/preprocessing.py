from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import CATEGORICAL_FEATURES, NUMERIC_FEATURES


def build_preprocessor() -> ColumnTransformer:
    """Imputation -> scaling (numeric) and imputation -> one-hot (categorical)."""
    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer(
        [("num", numeric, NUMERIC_FEATURES), ("cat", categorical, CATEGORICAL_FEATURES)],
        sparse_threshold=0,
    )


def build_pipeline(estimator) -> Pipeline:
    """preprocess -> SelectKBest -> model (k is tuned by RandomizedSearchCV)."""
    return Pipeline([
        ("preprocess", build_preprocessor()),
        ("select", SelectKBest(score_func=f_regression, k="all")),
        ("model", estimator),
    ])
