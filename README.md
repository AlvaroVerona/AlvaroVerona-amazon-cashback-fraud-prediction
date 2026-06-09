# AlvaroVerona-amazon-cashback-fraud-prediction
Predictive analytics system for detecting fraud in Amazon cashback programs. Implements a linear regression model to estimate risk scores from transactional features. Deployed on PythonAnywhere with an interactive HTML interface that allows users to adjust input parameters and generate real-time fraud probability predictions.

# Amazon Cashback - Fraud Prediction Model

This project consists of a Machine Learning model designed to detect potential fraud within Amazon's Cashback system. It includes data analysis, model training, and the deployment of a basic web API using Flask.

## Project Structure

* **`amazon_fraud.py`**: The main Python script containing the logic for data analysis, preprocessing, and training the predictive model.
* **`amazon_fraud_default.joblib`**: The trained and serialized model, ready to be loaded for real-time predictions.
* **`amazon_fraud_default.xlsx`**: The dataset used for testing and demonstrating the model's functionality.
* **`flask_app.py`**: A web application developed with Flask (configured to run on PythonAnywhere) that exposes the model to receive inputs and return fraud risk assessments.

## Technologies Used

* **Python 3**
* **Pandas / Openpyxl** (for Excel data handling)
* **Scikit-Learn** (or the specific library used for your model)
* **Joblib** (for model serialization and loading)
* **Flask** (for the web API)
* **PythonAnywhere** (development environment and hosting)

## Model Performance
*(Optional: You can replace this text with a brief statement about your results, for example: "The model identifies anomalous patterns based on variables such as cashback amount, user behavior history, and transaction frequency.")*

## How to Run the API Locally

1. Clone this repository:
   ```bash
   git clone [https://github.com/AlvaroVerona/amazon-cashback-fraud-prediction.git](https://github.com/AlvaroVerona/amazon-cashback-fraud-prediction.git)
