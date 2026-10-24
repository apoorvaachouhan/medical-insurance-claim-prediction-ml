from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.config import RANDOM_STATE

K_GRID = [5, 8, "all"]  # number of features kept by SelectKBest


def get_models() -> dict:
    """name -> (estimator, hyperparameter search space) for 9 regression algorithms."""
    rs = RANDOM_STATE
    return {
        "Linear Regression": (LinearRegression(), {"select__k": K_GRID}),
        "Ridge": (Ridge(random_state=rs), {"select__k": K_GRID, "model__alpha": [0.01, 0.1, 1, 10, 100]}),
        "SVR": (SVR(), {"select__k": K_GRID, "model__C": [1, 10, 100, 1000],
                        "model__kernel": ["rbf", "linear"], "model__epsilon": [0.1, 1, 10]}),
        "Decision Tree": (DecisionTreeRegressor(random_state=rs), {
            "select__k": K_GRID, "model__max_depth": [3, 5, 7, 10, None],
            "model__min_samples_split": [2, 5, 10], "model__min_samples_leaf": [1, 2, 5]}),
        "KNN": (KNeighborsRegressor(), {"select__k": K_GRID, "model__n_neighbors": [3, 5, 7, 9, 15],
                                         "model__weights": ["uniform", "distance"]}),
        "Random Forest": (RandomForestRegressor(random_state=rs, n_jobs=-1), {
            "select__k": K_GRID, "model__n_estimators": [100, 200, 300],
            "model__max_depth": [4, 6, 8, None], "model__min_samples_leaf": [1, 2, 5]}),
        "AdaBoost": (AdaBoostRegressor(random_state=rs), {
            "select__k": K_GRID, "model__n_estimators": [50, 100, 200],
            "model__learning_rate": [0.01, 0.1, 0.5, 1.0]}),
        "Gradient Boosting": (GradientBoostingRegressor(random_state=rs), {
            "select__k": K_GRID, "model__n_estimators": [100, 200, 300],
            "model__learning_rate": [0.01, 0.05, 0.1], "model__max_depth": [2, 3, 4],
            "model__subsample": [0.7, 0.85, 1.0]}),
        "XGBoost": (XGBRegressor(random_state=rs, n_jobs=-1), {
            "select__k": K_GRID, "model__n_estimators": [100, 200, 300],
            "model__learning_rate": [0.01, 0.05, 0.1], "model__max_depth": [2, 3, 4],
            "model__subsample": [0.7, 0.85, 1.0]}),
    }
