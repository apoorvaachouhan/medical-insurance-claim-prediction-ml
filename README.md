# End-to-End Medical Insurance Claim Prediction with MLOps

Predicts annual medical insurance charges from age, sex, BMI, children, smoker status and region.

**Live demo:** https://medical-insurance-claim-prediction-ml.onrender.com/
(hosted on Render's free plan, so the first load after a quiet period can take about a minute)

**Pipeline:** preprocessing (imputation, scaling, one-hot) -> `SelectKBest` -> 9 regression models tuned with
`RandomizedSearchCV` -> experiment tracking + Model Registry in **MLflow** -> **SHAP** explanations ->
**Streamlit** UI + **Flask** API -> **Docker** -> **Render**.

## Project structure
```
data/insurance.csv        # you add this (see below)
src/                      # config, data, preprocessing, models, train, explain, predict
app/api.py                # Flask API  (/health, /model-info, /predict)
app/streamlit_app.py      # Streamlit UI
reports/                  # model comparison table and SHAP plots
Dockerfile, docker-compose.yml
requirements.txt          # training + serving     requirements-serve.txt  # serving only (Docker)
```

## 1. Setup
```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```
Download **insurance.csv** (Kaggle: "Medical Cost Personal Datasets") into `data/`.
Columns: `age, sex, bmi, children, smoker, region, charges`.

## 2. Train, track and register (run from the project root)
```bash
python -m src.train        # tunes 9 models, logs to MLflow, registers the champion, saves models/champion_model.joblib
python -m src.explain      # SHAP plots -> reports/shap_summary.png, reports/shap_importance.png
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5001     # http://127.0.0.1:5001
```
Outputs: `reports/model_comparison.csv`, `models/champion_metrics.json`.

## 3. Run the apps locally
```bash
streamlit run app/streamlit_app.py        # http://localhost:8501
python app/api.py                         # http://localhost:5000
curl -X POST localhost:5000/predict -H "Content-Type: application/json" \
  -d '{"age":40,"sex":"male","bmi":31.5,"children":2,"smoker":"yes","region":"southeast"}'
```

## 4. Docker
`requirements-serve.txt` pins the library versions used for training, so the saved model loads
identically in the container.
```bash
docker compose up --build -d              # Streamlit :8501, Flask API :5000
```

## 5. Deploy on Render (free)
1. Push this repository to GitHub (the trained `models/` folder must be included).
2. On [render.com](https://render.com), sign in with GitHub, then **New +** -> **Web Service**.
3. Pick this repository. Set **Language** to **Docker**, **Branch** to `main`, **Instance Type** to **Free**.
4. Click **Create Web Service**. Render builds the `Dockerfile` and publishes the app at an `onrender.com` address.

The `Dockerfile` reads Render's `PORT` variable, so no extra settings are needed. Free instances sleep after
15 minutes without traffic and take about a minute to wake up.
