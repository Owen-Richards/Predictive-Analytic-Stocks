# Predictive-Analytic-Stocks
# Stock Price Prediction Service

This service provides a Flask-based API for training machine learning models to predict future stock prices using historical data fetched from Yahoo Finance.

## Features

- **Data Fetching:** Utilizes `yfinance` to fetch historical stock data.
- **Preprocessing:** Cleans and preprocesses data, calculates technical indicators (e.g., RSI, MACD, Bollinger Bands).
- **Model Training:** Trains an XGBoost regressor model using GridSearchCV for hyperparameter tuning.
- **Model Persistence:** Saves the best model to disk using pickle for future predictions.
- **Prediction:** Provides an API endpoint to predict future stock prices based on the trained model.
- **Error Handling:** Includes robust error handling and logging for various exceptions during data fetching, preprocessing, training, and prediction.

## API Endpoints

### Train Model

- **Endpoint:** `/train/<stock_ticker>`
- **Method:** POST
- **Body:** JSON with optional `historical_years` parameter (default: 10)
- **Description:** Trains a model for the specified stock ticker using historical data.

### Predict Future Prices

- **Endpoint:** `/predict`
- **Method:** GET
- **Parameters:** `stock_ticker` (required)
- **Description:** Predicts future stock prices for the specified stock ticker using the pre-trained model.

## Getting Started

1. Clone the repository.
2. Install dependencies: `pip install -r requirements.txt`
3. Start the Flask server: `python stock_predictor.py`
4. Use API endpoints to train models and predict future stock prices.

## Requirements

- Python 3.6+
- Flask
- numpy
- pandas
- scikit-learn
- xgboost
- yfinance

## Usage

Ensure Flask is running (`stock_predictor.py`), then use tools like `curl` or Postman to interact with the API endpoints.

### Example Usage

#### Train Model

```bash
curl -X POST -H "Content-Type: application/json" -d '{"historical_years": 5}' http://localhost:5000/train/AAPL
