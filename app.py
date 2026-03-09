from flask import Flask, render_template_string, request
import pandas as pd
import joblib
import os

app = Flask(__name__)

# Load models
mlr_model = joblib.load('mlr_model.pkl')
slr_model = joblib.load('slr_model.pkl')
pr_model = joblib.load('pr_model.pkl')
poly_converter = joblib.load('poly_converter.pkl')
log_reg_model = joblib.load('log_reg_model.pkl')
log_reg_threshold = joblib.load('log_reg_threshold.pkl')
knn_clf_model = joblib.load('knn_clf_model.pkl')
knn_clf_scaler = joblib.load('knn_clf_scaler.pkl')
knn_clf_threshold = joblib.load('knn_clf_threshold.pkl')

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Car Price Predictor</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Inter', sans-serif;
            background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
            color: #e0e0e0;
            min-height: 100vh;
        }
        .navbar {
            background: rgba(255,255,255,0.05);
            backdrop-filter: blur(10px);
            border-bottom: 1px solid rgba(255,255,255,0.1);
            padding: 16px 40px;
            display: flex;
            align-items: center;
            gap: 12px;
        }
        .navbar h1 {
            font-size: 1.4rem;
            font-weight: 600;
            background: linear-gradient(90deg, #00d2ff, #7b2ff7);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .container {
            max-width: 1100px;
            margin: 40px auto;
            padding: 0 20px;
        }
        .tabs {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            margin-bottom: 30px;
        }
        .tab {
            padding: 10px 20px;
            border-radius: 25px;
            background: rgba(255,255,255,0.06);
            border: 1px solid rgba(255,255,255,0.1);
            color: #aaa;
            text-decoration: none;
            font-size: 0.85rem;
            font-weight: 500;
            transition: all 0.3s;
        }
        .tab:hover { background: rgba(255,255,255,0.12); color: #fff; }
        .tab.active {
            background: linear-gradient(135deg, #7b2ff7, #00d2ff);
            color: #fff;
            border-color: transparent;
        }
        .card {
            background: rgba(255,255,255,0.06);
            backdrop-filter: blur(12px);
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 16px;
            padding: 36px;
        }
        .card h2 {
            font-size: 1.5rem;
            margin-bottom: 6px;
            color: #fff;
        }
        .card .subtitle {
            color: #888;
            font-size: 0.9rem;
            margin-bottom: 28px;
        }
        .form-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
        }
        .form-group { display: flex; flex-direction: column; gap: 6px; }
        .form-group.full { grid-column: 1 / -1; }
        label {
            font-size: 0.8rem;
            font-weight: 500;
            color: #aaa;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        input[type="number"], input[type="text"] {
            padding: 12px 16px;
            border-radius: 10px;
            border: 1px solid rgba(255,255,255,0.12);
            background: rgba(255,255,255,0.04);
            color: #fff;
            font-size: 1rem;
            font-family: 'Inter', sans-serif;
            outline: none;
            transition: border 0.3s;
        }
        input:focus { border-color: #7b2ff7; }
        .btn {
            margin-top: 10px;
            padding: 14px 36px;
            border: none;
            border-radius: 12px;
            background: linear-gradient(135deg, #7b2ff7, #00d2ff);
            color: #fff;
            font-size: 1rem;
            font-weight: 600;
            cursor: pointer;
            transition: transform 0.2s, box-shadow 0.3s;
            width: 100%;
        }
        .btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 30px rgba(123, 47, 247, 0.4);
        }
        .result-card {
            margin-top: 30px;
            background: linear-gradient(135deg, rgba(123,47,247,0.15), rgba(0,210,255,0.10));
            border: 1px solid rgba(123,47,247,0.3);
            border-radius: 16px;
            padding: 32px;
            text-align: center;
            animation: fadeIn 0.5s ease;
        }
        @keyframes fadeIn { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }
        .result-card .label { font-size: 0.9rem; color: #aaa; margin-bottom: 8px; }
        .result-card .price {
            font-size: 2.8rem;
            font-weight: 700;
            background: linear-gradient(90deg, #00d2ff, #7b2ff7);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .result-card .classification {
            font-size: 2rem;
            font-weight: 700;
        }
        .result-card .prob { color: #888; margin-top: 8px; font-size: 0.95rem; }
        .warning {
            margin-top: 12px;
            padding: 12px;
            background: rgba(243, 156, 18, 0.15);
            border: 1px solid rgba(243, 156, 18, 0.3);
            border-radius: 8px;
            color: #f39c12;
            font-size: 0.85rem;
        }
        @media (max-width: 600px) {
            .form-grid { grid-template-columns: 1fr; }
            .tabs { gap: 6px; }
            .tab { font-size: 0.75rem; padding: 8px 14px; }
        }
    </style>
</head>
<body>
    <nav class="navbar">
        <span style="font-size:1.6rem;">🚗</span>
        <h1>Car Price Predictor</h1>
    </nav>
    <div class="container">
        <div class="tabs">
            <a href="/?model=mlr" class="tab {{ 'active' if model=='mlr' }}">Multiple Linear Regression</a>
            <a href="/?model=slr" class="tab {{ 'active' if model=='slr' }}">Simple Linear Regression</a>
            <a href="/?model=poly" class="tab {{ 'active' if model=='poly' }}">Polynomial Regression</a>
            <a href="/?model=knn_clf" class="tab {{ 'active' if model=='knn_clf' }}">KNN Classification</a>
            <a href="/?model=logistic" class="tab {{ 'active' if model=='logistic' }}">Logistic Regression</a>
        </div>

        {% if model == 'mlr' %}
        <div class="card">
            <h2>Multiple Linear Regression</h2>
            <p class="subtitle">Predict selling price based on multiple features</p>
            <form method="POST" action="/?model=mlr">
                <div class="form-grid">
                    <div class="form-group"><label>Year of Manufacture</label><input type="number" name="year" value="{{ values.year or 2015 }}" min="1990" max="2026"></div>
                    <div class="form-group"><label>Kilometers Driven</label><input type="number" name="km_driven" value="{{ values.km_driven or 50000 }}" min="0"></div>
                    <div class="form-group"><label>Mileage (km/ltr/kg)</label><input type="number" name="mileage" value="{{ values.mileage or 20.0 }}" step="0.5" min="0"></div>
                    <div class="form-group"><label>Engine Capacity (CC)</label><input type="number" name="engine" value="{{ values.engine or 1197 }}" min="500"></div>
                    <div class="form-group"><label>Max Power (bhp)</label><input type="number" name="max_power" value="{{ values.max_power or 74.0 }}" step="1" min="20"></div>
                    <div class="form-group"><label>Number of Seats</label><input type="number" name="seats" value="{{ values.seats or 5 }}" min="2" max="14"></div>
                    <div class="form-group full"><button class="btn" type="submit">Predict Price</button></div>
                </div>
            </form>
            {% if prediction is not none %}
            <div class="result-card">
                <div class="label">Predicted Selling Price</div>
                <div class="price">₹ {{ prediction }}</div>
                {% if warning %}<div class="warning">{{ warning }}</div>{% endif %}
            </div>
            {% endif %}
        </div>

        {% elif model == 'slr' %}
        <div class="card">
            <h2>Simple Linear Regression</h2>
            <p class="subtitle">Predict selling price based only on Kilometers Driven</p>
            <form method="POST" action="/?model=slr">
                <div class="form-grid">
                    <div class="form-group"><label>Kilometers Driven</label><input type="number" name="km_driven" value="{{ values.km_driven or 50000 }}" min="0"></div>
                    <div class="form-group full"><button class="btn" type="submit">Predict Price</button></div>
                </div>
            </form>
            {% if prediction is not none %}
            <div class="result-card">
                <div class="label">Predicted Selling Price</div>
                <div class="price">₹ {{ prediction }}</div>
                {% if warning %}<div class="warning">{{ warning }}</div>{% endif %}
            </div>
            {% endif %}
        </div>

        {% elif model == 'poly' %}
        <div class="card">
            <h2>Polynomial Regression</h2>
            <p class="subtitle">Predict selling price using a degree-2 polynomial on Kilometers Driven</p>
            <form method="POST" action="/?model=poly">
                <div class="form-grid">
                    <div class="form-group"><label>Kilometers Driven</label><input type="number" name="km_driven" value="{{ values.km_driven or 50000 }}" min="0"></div>
                    <div class="form-group full"><button class="btn" type="submit">Predict Price</button></div>
                </div>
            </form>
            {% if prediction is not none %}
            <div class="result-card">
                <div class="label">Predicted Selling Price</div>
                <div class="price">₹ {{ prediction }}</div>
            </div>
            {% endif %}
        </div>

        {% elif model == 'knn_clf' %}
        <div class="card">
            <h2>KNN Classification</h2>
            <p class="subtitle">Classify if the car is High Value (above ₹ {{ knn_threshold }}) or Standard Value using KNN</p>
            <form method="POST" action="/?model=knn_clf">
                <div class="form-grid">
                    <div class="form-group"><label>Year of Manufacture</label><input type="number" name="year" value="{{ values.year or 2015 }}" min="1990" max="2026"></div>
                    <div class="form-group"><label>Kilometers Driven</label><input type="number" name="km_driven" value="{{ values.km_driven or 50000 }}" min="0"></div>
                    <div class="form-group"><label>Mileage (km/ltr/kg)</label><input type="number" name="mileage" value="{{ values.mileage or 20.0 }}" step="0.5" min="0"></div>
                    <div class="form-group"><label>Engine Capacity (CC)</label><input type="number" name="engine" value="{{ values.engine or 1197 }}" min="500"></div>
                    <div class="form-group"><label>Max Power (bhp)</label><input type="number" name="max_power" value="{{ values.max_power or 74.0 }}" step="1" min="20"></div>
                    <div class="form-group"><label>Number of Seats</label><input type="number" name="seats" value="{{ values.seats or 5 }}" min="2" max="14"></div>
                    <div class="form-group full"><button class="btn" type="submit">Classify Car (KNN)</button></div>
                </div>
            </form>
            {% if classification is not none %}
            <div class="result-card">
                <div class="label">KNN Classification Result</div>
                <div class="classification" style="color: {{ 'limegreen' if classification == 'High Value Car' else '#f39c12' }}">{{ classification }}</div>
                <div class="prob">{{ prob_text }}</div>
                <div class="prob">Based on median threshold of ₹ {{ knn_threshold }}</div>
            </div>
            {% endif %}
        </div>

        {% elif model == 'logistic' %}
        <div class="card">
            <h2>Logistic Regression (Classification)</h2>
            <p class="subtitle">Classify if the car is High Value (above ₹ {{ threshold }}) or Standard Value</p>
            <form method="POST" action="/?model=logistic">
                <div class="form-grid">
                    <div class="form-group"><label>Year of Manufacture</label><input type="number" name="year" value="{{ values.year or 2015 }}" min="1990" max="2026"></div>
                    <div class="form-group"><label>Kilometers Driven</label><input type="number" name="km_driven" value="{{ values.km_driven or 50000 }}" min="0"></div>
                    <div class="form-group"><label>Mileage (km/ltr/kg)</label><input type="number" name="mileage" value="{{ values.mileage or 20.0 }}" step="0.5" min="0"></div>
                    <div class="form-group"><label>Engine Capacity (CC)</label><input type="number" name="engine" value="{{ values.engine or 1197 }}" min="500"></div>
                    <div class="form-group"><label>Max Power (bhp)</label><input type="number" name="max_power" value="{{ values.max_power or 74.0 }}" step="1" min="20"></div>
                    <div class="form-group"><label>Number of Seats</label><input type="number" name="seats" value="{{ values.seats or 5 }}" min="2" max="14"></div>
                    <div class="form-group full"><button class="btn" type="submit">Classify Car Value</button></div>
                </div>
            </form>
            {% if classification is not none %}
            <div class="result-card">
                <div class="label">Classification Result</div>
                <div class="classification" style="color: {{ 'limegreen' if classification == 'High Value Car' else '#f39c12' }}">{{ classification }}</div>
                <div class="prob">{{ prob_text }}</div>
                <div class="prob">Based on median threshold of ₹ {{ threshold }}</div>
            </div>
            {% endif %}
        </div>
        {% endif %}
    </div>
</body>
</html>
"""

@app.route('/', methods=['GET', 'POST'])
def index():
    model = request.args.get('model', 'mlr')
    prediction = None
    classification = None
    prob_text = None
    warning = None
    values = {}
    threshold = f"{log_reg_threshold:,.2f}"

    if request.method == 'POST':
        values = request.form.to_dict()

        try:
            if model == 'mlr':
                input_data = pd.DataFrame({
                    'year': [float(values['year'])],
                    'km_driven': [float(values['km_driven'])],
                    'mileage(km/ltr/kg)': [float(values['mileage'])],
                    'engine': [float(values['engine'])],
                    'max_power': [float(values['max_power'])],
                    'seats': [float(values['seats'])]
                })
                pred = mlr_model.predict(input_data)[0]
                prediction = f"{pred:,.2f}"
                if pred < 0:
                    warning = "The model predicted a negative value — input may be outside the training data range."

            elif model == 'slr':
                input_data = pd.DataFrame({'km_driven': [float(values['km_driven'])]})
                pred = slr_model.predict(input_data)[0]
                prediction = f"{pred:,.2f}"
                if pred < 0:
                    warning = "The model predicted a negative value — input may be outside the training data range."

            elif model == 'poly':
                input_data = pd.DataFrame({'km_driven': [float(values['km_driven'])]})
                X_poly = poly_converter.transform(input_data)
                pred = pr_model.predict(X_poly)[0]
                prediction = f"{pred:,.2f}"

            elif model == 'knn_clf':
                input_data = pd.DataFrame({
                    'year': [float(values['year'])],
                    'km_driven': [float(values['km_driven'])],
                    'mileage(km/ltr/kg)': [float(values['mileage'])],
                    'engine': [float(values['engine'])],
                    'max_power': [float(values['max_power'])],
                    'seats': [float(values['seats'])]
                })
                scaled_input = knn_clf_scaler.transform(input_data)
                pred = knn_clf_model.predict(scaled_input)[0]
                probability = knn_clf_model.predict_proba(scaled_input)[0]
                if pred == 1:
                    classification = "High Value Car"
                    prob_text = f"Probability: {probability[1]*100:.1f}%"
                else:
                    classification = "Standard Value Car"
                    prob_text = f"Probability: {probability[0]*100:.1f}%"

            elif model == 'logistic':
                input_data = pd.DataFrame({
                    'year': [float(values['year'])],
                    'km_driven': [float(values['km_driven'])],
                    'mileage(km/ltr/kg)': [float(values['mileage'])],
                    'engine': [float(values['engine'])],
                    'max_power': [float(values['max_power'])],
                    'seats': [float(values['seats'])]
                })
                pred = log_reg_model.predict(input_data)[0]
                probability = log_reg_model.predict_proba(input_data)[0]
                if pred == 1:
                    classification = "High Value Car"
                    prob_text = f"Probability: {probability[1]*100:.1f}%"
                else:
                    classification = "Standard Value Car"
                    prob_text = f"Probability: {probability[0]*100:.1f}%"

        except Exception as e:
            warning = f"Prediction Error: {e}"

    return render_template_string(HTML_TEMPLATE,
                                  model=model,
                                  prediction=prediction,
                                  classification=classification,
                                  prob_text=prob_text,
                                  warning=warning,
                                  values=values,
                                  threshold=threshold,
                                  knn_threshold=f"{knn_clf_threshold:,.2f}")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
