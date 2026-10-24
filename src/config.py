from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "insurance.csv"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"
MODEL_PATH = MODEL_DIR / "champion_model.joblib"
METRICS_PATH = MODEL_DIR / "champion_metrics.json"

TARGET = "charges"
NUMERIC_FEATURES = ["age", "bmi", "children"]
CATEGORICAL_FEATURES = ["sex", "smoker", "region"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

CATEGORY_VALUES = {
    "sex": ["female", "male"],
    "smoker": ["no", "yes"],
    "region": ["northeast", "northwest", "southeast", "southwest"],
}

RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5
N_ITER = 15  # RandomizedSearchCV iterations per model

# MLflow: a SQLite backend is needed for the Model Registry
MLFLOW_TRACKING_URI = f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}"
MLFLOW_EXPERIMENT = "medical-insurance-claim-prediction"
REGISTERED_MODEL_NAME = "medical-insurance-claim-model"
