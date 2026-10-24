import functools

import joblib
import pandas as pd

from src.config import CATEGORY_VALUES, FEATURES, MODEL_PATH


@functools.lru_cache(maxsize=1)
def load_model():
    return joblib.load(MODEL_PATH)


def validate(payload: dict) -> dict:
    """Validate raw input and return a clean dict; raises ValueError with a clear message."""
    if not isinstance(payload, dict):
        raise ValueError("Request body must be a JSON object")
    missing = [f for f in FEATURES if f not in payload]
    if missing:
        raise ValueError(f"Missing fields: {missing}")
    try:
        age, bmi, children = int(payload["age"]), float(payload["bmi"]), int(payload["children"])
    except (TypeError, ValueError):
        raise ValueError("age and children must be integers, bmi must be a number")
    if not 18 <= age <= 100:
        raise ValueError("age must be between 18 and 100")
    if not 10 <= bmi <= 60:
        raise ValueError("bmi must be between 10 and 60")
    if not 0 <= children <= 10:
        raise ValueError("children must be between 0 and 10")
    clean = {"age": age, "bmi": bmi, "children": children}
    for col, allowed in CATEGORY_VALUES.items():
        value = str(payload[col]).strip().lower()
        if value not in allowed:
            raise ValueError(f"{col} must be one of {allowed}")
        clean[col] = value
    return clean


def predict_charge(payload: dict) -> float:
    row = pd.DataFrame([validate(payload)], columns=FEATURES)
    return float(load_model().predict(row)[0])
