# AlvaroVerona-amazon-cashback-fraud-prediction
Predictive analytics system for detecting fraud in Amazon cashback programs. Implements a linear regression model to estimate risk scores from transactional features. Deployed on PythonAnywhere with an interactive HTML interface that allows users to adjust input parameters and generate real-time fraud probability predictions.

## Project Structure

* **`amazon_fraud.py`**: The main Python script containing the logic for data analysis, preprocessing, and training the predictive model.
* **`amazon_fraud_default.joblib`**: The trained and serialized model, ready to be loaded for real-time predictions.
* **`amazon_fraud_default.xlsx`**: The dataset used for testing and demonstrating the model's functionality.
* **`flask_app.py`**: A web application developed with Flask (configured to run on PythonAnywhere) that exposes the model to receive inputs and return fraud risk assessments.

## Technologies Used

* **Python 3**
* **Pandas / Openpyxl** (for Excel data handling)
* **Scikit-Learn** (specifically using `LinearRegression` for risk scoring)
* **Joblib** (for model serialization and loading)
* **Flask** (for the web API)
* **PythonAnywhere** (development environment and hosting)

## Model Performance

The project utilizes a Linear Regression model to establish a continuous fraud risk score based on the dataset's features (such as transaction amounts and cashback metrics). This allows the system to flag transactions that exceed a specific risk threshold.

## How to Run the API Locally

1. Clone this repository:
   ```bash
   git clone [https://github.com/AlvaroVerona/amazon-cashback-fraud-prediction.git](https://github.com/AlvaroVerona/amazon-cashback-fraud-prediction.git)

## You can visualize the tool and interact with it in the following link: https://varoverona.pythonanywhere.com/
