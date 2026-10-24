"""Train, tune and compare 9 regression models; track everything in MLflow;
register the champion in the MLflow Model Registry and save it for serving.

Run from the project root:  python -m src.train
"""
import json

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.tracking import MlflowClient
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import ParameterGrid, RandomizedSearchCV

from src import config
from src.data import load_data, split_data
from src.models import get_models
from src.preprocessing import build_pipeline


def evaluate(model, X, y) -> dict:
    pred = model.predict(X)
    return {
        "rmse": float(np.sqrt(mean_squared_error(y, pred))),
        "mae": float(mean_absolute_error(y, pred)),
        "r2": float(r2_score(y, pred)),
    }


def main():
    config.MODEL_DIR.mkdir(exist_ok=True)
    config.REPORT_DIR.mkdir(exist_ok=True)

    X_train, X_test, y_train, y_test = split_data(load_data())
    print(f"Train: {X_train.shape}  Test: {X_test.shape}")

    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(config.MLFLOW_EXPERIMENT)

    results = []
    with mlflow.start_run(run_name="model-comparison"):
        mlflow.log_params({"test_size": config.TEST_SIZE, "cv_folds": config.CV_FOLDS,
                           "n_iter": config.N_ITER, "random_state": config.RANDOM_STATE})
        for name, (estimator, space) in get_models().items():
            print(f"Tuning {name} ...")
            search = RandomizedSearchCV(
                build_pipeline(estimator), space,
                n_iter=min(config.N_ITER, len(ParameterGrid(space))),
                cv=config.CV_FOLDS, scoring="neg_root_mean_squared_error",
                random_state=config.RANDOM_STATE, n_jobs=-1,
            )
            with mlflow.start_run(run_name=name, nested=True) as run:
                search.fit(X_train, y_train)
                best = search.best_estimator_
                test = evaluate(best, X_test, y_test)
                cv_rmse = float(-search.best_score_)

                mlflow.log_param("algorithm", name)
                mlflow.log_params(search.best_params_)
                mlflow.log_metrics({"cv_rmse": cv_rmse, "test_rmse": test["rmse"],
                                    "test_mae": test["mae"], "test_r2": test["r2"]})
                model_info = mlflow.sklearn.log_model(best, artifact_path="model", serialization_format="cloudpickle")

            results.append({"model": name, "run_id": run.info.run_id, "cv_rmse": cv_rmse,
                            "test_rmse": test["rmse"], "test_mae": test["mae"],
                            "test_r2": test["r2"], "model_uri": model_info.model_uri, "estimator": best})
            print(f"   CV RMSE {cv_rmse:,.0f} | test RMSE {test['rmse']:,.0f} | test R2 {test['r2']:.3f}")

    # Champion = lowest cross-validated RMSE (the test set is never used to choose)
    champion = min(results, key=lambda r: r["cv_rmse"])
    print(f"\nChampion: {champion['model']} (test R2 = {champion['test_r2']:.3f})")

    table = pd.DataFrame([{k: v for k, v in r.items() if k not in ("estimator", "run_id", "model_uri")}
                          for r in results]).sort_values("cv_rmse")
    table.to_csv(config.REPORT_DIR / "model_comparison.csv", index=False)
    print(table.to_string(index=False))

    # Model Registry: register the champion's logged model and mark it "production"
    client = MlflowClient()
    version = mlflow.register_model(champion["model_uri"], config.REGISTERED_MODEL_NAME)
    client.set_registered_model_alias(config.REGISTERED_MODEL_NAME, "production", version.version)
    print(f"Registered '{config.REGISTERED_MODEL_NAME}' version {version.version} (alias: production)")

    # Local copy used by the Flask / Streamlit apps
    joblib.dump(champion["estimator"], config.MODEL_PATH)
    config.METRICS_PATH.write_text(json.dumps({
        "model": champion["model"], "cv_rmse": champion["cv_rmse"], "test_rmse": champion["test_rmse"],
        "test_mae": champion["test_mae"], "test_r2": champion["test_r2"],
        "registry_version": str(version.version)}, indent=2))
    print(f"Saved {config.MODEL_PATH}")


if __name__ == "__main__":
    main()
