import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import DATA_PATH, FEATURES, RANDOM_STATE, TARGET, TEST_SIZE


def load_data(path=DATA_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Download insurance.csv (Kaggle: 'Medical Cost "
            "Personal Datasets') and place it in the data/ folder."
        )
    df = pd.read_csv(path)
    missing = set(FEATURES + [TARGET]) - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing columns: {sorted(missing)}")
    return df


def split_data(df: pd.DataFrame | None = None):
    """Deterministic train/test split (same split in train.py and explain.py)."""
    df = load_data() if df is None else df
    X, y = df[FEATURES], df[TARGET]
    return train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)
