########################################
# IMPORTS FOR DATA + MODEL TRAINING
########################################

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

import joblib
import os


MODEL_FILENAME = "amazon_fraud_default.joblib"



########################################
# TRAINING SECTION
# Only runs if file executed manually
########################################

def train_model():
    print("Training fraud model...")

    np.random.seed(42)
    n_samples = 5000

    # Feature generation
    num_disputed_tx_last_6m = np.random.poisson(lam=0.5, size=n_samples)
    device_account_mismatch_freq = np.random.poisson(lam=0.3, size=n_samples)
    chargeback_ratio_last_12m = np.random.beta(a=1.0, b=20.0, size=n_samples)

    # Target variable
    logit = (
        -3.0
        + 0.6 * num_disputed_tx_last_6m
        + 0.8 * device_account_mismatch_freq
        + 4.0 * chargeback_ratio_last_12m
    )

    prob_fraud = 1 / (1 + np.exp(-logit))
    fraud_driven_default = np.random.binomial(1, prob_fraud, size=n_samples)

    # Build DataFrame
    df = pd.DataFrame({
        "num_disputed_tx_last_6m": num_disputed_tx_last_6m,
        "device_account_mismatch_freq": device_account_mismatch_freq,
        "chargeback_ratio_last_12m": chargeback_ratio_last_12m,
        "fraud_driven_default": fraud_driven_default
    })

    # Save Excel (optional)
    df.to_excel("amazon_fraud_default.xlsx", index=False)

    # Train-test split
    X = df[[
        "num_disputed_tx_last_6m",
        "device_account_mismatch_freq",
        "chargeback_ratio_last_12m"
    ]]

    y = df["fraud_driven_default"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # Logistic Regression
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    print("\nTRAIN RESULTS:")
    print(classification_report(y_train, model.predict(X_train)))

    print("\nTEST RESULTS:")
    print(classification_report(y_test, model.predict(X_test)))

    # Save model
    joblib.dump(model, MODEL_FILENAME)
    print(f"\nModel saved as {MODEL_FILENAME}")



########################################
# FLASK UI + API SECTION
########################################

from flask import Flask, request, jsonify, render_template_string

app = Flask(__name__)

# Load model ONLY if exists (so Flask doesn’t crash on first deployment)
if os.path.exists(MODEL_FILENAME):
    model = joblib.load(MODEL_FILENAME)
else:
    model = None
    print("⚠ No model found. Run amazon_fraud.py manually to train the model.")



########################################
# HTML UI with Bootstrap
########################################

HTML_TEMPLATE= """
<!DOCTYPE html>
<html>
<head>
    <title>Amazon Fraud-driven Default Predictor</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet">

    <style>
        body { background-color: #f2f3f5; }
        .container-box {
            max-width: 650px;
            margin: 40px auto;
            background: white;
            padding: 30px;
            border-radius: 12px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        }
        .amazon-title {
            font-size: 28px;
            font-weight: bold;
            color: #232F3E;
            text-align: center;
            margin-bottom: 25px;
        }
        .btn-amazon {
            background-color: #FF9900;
            color: black;
            font-weight: bold;
        }
        .btn-amazon:hover { background-color: #e68a00; }

        .slider-value {
            font-weight: bold;
            color: #333;
        }
    </style>

    <script>
        function updateValue(id, outputId, divisor=1) {
            var val = document.getElementById(id).value;
            document.getElementById(outputId).innerHTML = (val/divisor).toFixed(divisor == 1 ? 0 : 2);
        }
    </script>
</head>

<body>

<div class="container-box">
    <div class="amazon-title">Amazon Fraud-driven Default Predictor</div>

    <form action="/predict_form" method="post">

        <!-- Disputed Transactions -->
        <label class="form-label">Disputed Transactions (last 6 months):
            <span class="slider-value" id="disp_val">0</span>
        </label>
        <input type="range" min="0" max="10" value="0" step="1"
               name="disputes" id="disputes"
               class="form-range"
               oninput="updateValue('disputes', 'disp_val')">

        <!-- Mismatch Frequency -->
        <label class="form-label mt-4">Device–Account Mismatch Frequency:
            <span class="slider-value" id="mm_val">0</span>
        </label>
        <input type="range" min="0" max="10" value="0" step="1"
               name="mismatch" id="mismatch"
               class="form-range"
               oninput="updateValue('mismatch', 'mm_val')">

        <!-- Chargeback Ratio -->
        <label class="form-label mt-4">Chargeback Ratio (0 to 1):
            <span class="slider-value" id="ratio_val">0.00</span>
        </label>
        <input type="range" min="0" max="100" value="0" step="1"
               name="ratio_raw" id="ratio_raw"
               class="form-range"
               oninput="updateValue('ratio_raw', 'ratio_val', 100)">

        <!-- Hidden field converts 0–100 slider to 0–1 ratio -->
        <input type="hidden" name="ratio" id="ratio">

        <script>
            // Sync hidden ratio field before submitting
            document.querySelector("form").onsubmit = function() {
                var raw = document.getElementById("ratio_raw").value;
                document.getElementById("ratio").value = (raw / 100).toFixed(4);
            }
        </script>

        <button type="submit" class="btn btn-amazon mt-4 w-100">Predict Fraud Risk</button>
    </form>

    {% if predicted is not none %}
    <div class="mt-4 p-3 border rounded"
         style="background-color: {% if predicted == 1 %}#ffe6e6{% else %}#e6ffe6{% endif %};">

        <h5>Prediction:
            {% if predicted == 1 %}
                <span class="badge bg-danger">High Fraud Risk</span>
            {% else %}
                <span class="badge bg-success">Low Fraud Risk</span>
            {% endif %}
        </h5>

        <p class="mt-3 mb-1"><strong>Probability of Fraud-driven Default:</strong></p>

        <div class="progress">
            <div class="progress-bar {% if predicted == 1 %}bg-danger{% else %}bg-success{% endif %}"
                 role="progressbar"
                 style="width: {{ probability * 100 }}%;">
                {{ (probability * 100) | round(2) }}%
            </div>
        </div>

    </div>
    {% endif %}
</div>

</body>
</html>
"""


########################################
# ROUTES
########################################

@app.route("/", methods=["GET"])
def home():
    return render_template_string(HTML_TEMPLATE, predicted=None)


@app.route("/predict_form", methods=["POST"])
def predict_form():
    disputes = float(request.form.get("disputes"))
    mismatch = float(request.form.get("mismatch"))
    ratio = float(request.form.get("ratio"))

    X = np.array([[disputes, mismatch, ratio]])
    prediction = int(model.predict(X)[0])
    probability = float(model.predict_proba(X)[0][1])

    return render_template_string(
        HTML_TEMPLATE,
        predicted=prediction,
        probability=probability
    )


@app.route("/predict")
def predict_api():
    """JSON API endpoint"""
    disputes = float(request.args.get("disputes", 0))
    mismatch = float(request.args.get("mismatch", 0))
    ratio = float(request.args.get("ratio", 0))

    X = np.array([[disputes, mismatch, ratio]])
    pred = int(model.predict(X)[0])
    prob = float(model.predict_proba(X)[0][1])

    return jsonify({
        "prediction": pred,
        "probability_fraud": prob
    })



########################################
# RUN TRAINING ONLY WHEN EXECUTED LOCALLY
########################################

if __name__ == "__main__":
    train_model()
    app.run(debug=True)
