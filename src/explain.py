"""SHAP analysis of the champion model.  Run:  python -m src.explain"""
import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import shap

from src import config
from src.data import split_data


def main():
    config.REPORT_DIR.mkdir(exist_ok=True)
    pipe = joblib.load(config.MODEL_PATH)
    _, X_test, _, _ = split_data()

    pre, sel, model = pipe.named_steps["preprocess"], pipe.named_steps["select"], pipe.named_steps["model"]
    names = pre.get_feature_names_out()[sel.get_support()]
    X = pd.DataFrame(sel.transform(pre.transform(X_test)), columns=names)

    try:
        values = shap.TreeExplainer(model).shap_values(X)
    except Exception:  # non-tree champion -> model-agnostic explainer
        values = shap.Explainer(model.predict, X)(X).values

    for kind, fname in [("dot", "shap_summary.png"), ("bar", "shap_importance.png")]:
        plt.figure()
        shap.summary_plot(values, X, plot_type=kind, show=False)
        plt.tight_layout()
        plt.savefig(config.REPORT_DIR / fname, dpi=150)
        plt.close()
    print(f"Saved SHAP plots to {config.REPORT_DIR}")


if __name__ == "__main__":
    main()
