"""Streamlit UI.  Run:  streamlit run app/streamlit_app.py"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st

from src.config import CATEGORY_VALUES, METRICS_PATH
from src.predict import predict_charge

st.set_page_config(page_title="Medical Insurance Claim Predictor", page_icon="🏥")
st.title("🏥 Medical Insurance Claim Predictor")
st.caption("Estimate annual medical insurance charges from a few personal details.")

col1, col2 = st.columns(2)
age = col1.number_input("Age", 18, 100, 30)
bmi = col2.number_input("BMI", 10.0, 60.0, 25.0, step=0.1)
children = col1.number_input("Children", 0, 10, 0)
sex = col2.selectbox("Sex", CATEGORY_VALUES["sex"])
smoker = col1.selectbox("Smoker", CATEGORY_VALUES["smoker"])
region = col2.selectbox("Region", CATEGORY_VALUES["region"])

if st.button("Predict charges", type="primary"):
    charge = predict_charge({"age": age, "bmi": bmi, "children": children,
                             "sex": sex, "smoker": smoker, "region": region})
    st.success(f"Estimated annual charges: **{charge:,.2f}**")

if METRICS_PATH.exists():
    info = json.loads(METRICS_PATH.read_text())
    with st.expander("About the model"):
        st.write(f"Algorithm: **{info['model']}**  |  Test R²: **{info['test_r2']:.3f}**  "
                 f"|  Test RMSE: **{info['test_rmse']:,.0f}**")
