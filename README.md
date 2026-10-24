# End-to-End Medical Insurance Claim Prediction with MLOps

Predicts annual medical insurance charges from age, sex, BMI, children, smoker status and region.

**Pipeline:** preprocessing (imputation, scaling, one-hot) -> `SelectKBest` -> 9 regression models tuned with
`RandomizedSearchCV` -> experiment tracking + Model Registry in **MLflow** -> **SHAP** explanations ->
**Flask** API + **Streamlit** UI -> **Docker** -> **AWS EC2**.

## Project structure
```
data/insurance.csv        # you add this (see below)
src/                      # config, data, preprocessing, models, train, explain, predict
app/api.py                # Flask API  (/health, /model-info, /predict)
app/streamlit_app.py      # Streamlit UI
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
mlflow ui --backend-store-uri sqlite:///mlflow.db     # http://127.0.0.1:5000  (compare runs, see the registry)
```
Outputs: `reports/model_comparison.csv`, `models/champion_metrics.json`.

## 3. Run the apps locally
```bash
streamlit run app/streamlit_app.py        # http://localhost:8501
python app/api.py                         # http://localhost:5000
curl -X POST localhost:5000/predict -H "Content-Type: application/json" \
  -d '{"age":40,"sex":"male","bmi":31.5,"children":2,"smoker":"yes","region":"southeast"}'
```
(Stop the MLflow UI first, or run it on another port with `--port 5001`.)

## 4. Docker
Pin the versions you trained with so the saved model loads identically in the container:
```bash
pip freeze | grep -i -E "^(pandas|numpy|scikit-learn|xgboost|joblib|flask|gunicorn|streamlit)==" > requirements-serve.txt
docker compose up --build -d              # Streamlit :8501, Flask API :5000
```

## 5. Deploy on AWS EC2
1. Launch an **Ubuntu** EC2 instance (t2.micro/t3.micro is enough) and download the key pair.
2. In the instance's **Security Group**, allow inbound TCP **22** (your IP), **8501** and **5000**.
3. Connect and install Docker:
   ```bash
   ssh -i key.pem ubuntu@<EC2_PUBLIC_IP>
   sudo apt update && sudo apt install -y docker.io docker-compose-v2 git
   sudo usermod -aG docker ubuntu && newgrp docker
   ```
4. Get the code onto the instance (the trained `models/` folder must be included; it is not git-ignored):
   `git clone <your-repo-url> && cd <repo>` (or `scp -r`).
5. `docker compose up --build -d`, then open `http://<EC2_PUBLIC_IP>:8501`.
