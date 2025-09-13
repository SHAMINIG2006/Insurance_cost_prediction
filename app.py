# app.py
from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
import os

app = Flask(__name__)

MODEL_PATH = 'model_xgb.pkl'
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("Model file not found. Run: python train_model.py")

model = joblib.load(MODEL_PATH)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Collect form fields
    age = int(request.form.get('age', 30))
    sex = request.form.get('sex', 'female')
    bmi = float(request.form.get('bmi', 25.0))
    children = int(request.form.get('children', 0))
    smoker = request.form.get('smoker', 'no')
    region = request.form.get('region', 'southeast')

    df = pd.DataFrame([{
        'age': age,
        'sex': sex,
        'bmi': bmi,
        'children': children,
        'smoker': smoker,
        'region': region
    }])

    pred = model.predict(df)[0]
    pred_rounded = round(float(pred), 2)

    return render_template('predict.html', prediction=pred_rounded)

# JSON API for AJAX
@app.route('/api/predict', methods=['POST'])
def api_predict():
    data = request.get_json()
    df = pd.DataFrame([data])
    pred = model.predict(df)[0]
    return jsonify({'prediction': round(float(pred), 2)})

if __name__ == '__main__':
    app.run(debug=True)
