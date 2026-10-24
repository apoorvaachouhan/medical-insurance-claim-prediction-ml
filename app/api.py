"""Flask inference API.  Run:  python app/api.py   (or gunicorn app.api:app)"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from flask import Flask, jsonify, request

from src.config import METRICS_PATH
from src.predict import predict_charge

app = Flask(__name__)


@app.get("/health")
def health():
    return jsonify(status="ok")


@app.get("/model-info")
def model_info():
    if METRICS_PATH.exists():
        return jsonify(json.loads(METRICS_PATH.read_text()))
    return jsonify(error="metrics file not found"), 404


@app.post("/predict")
def predict():
    try:
        charge = predict_charge(request.get_json(force=True, silent=True))
    except ValueError as e:
        return jsonify(error=str(e)), 400
    return jsonify(predicted_charge=round(charge, 2))


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
