# Import the necessary libraries
import os
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
import yfinance as yf
from datetime import datetime, timedelta
import logging
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
import pickle
import time
from requests.exceptions import ConnectionError, Timeout, HTTPError
from sklearn.impute import SimpleImputer

# Initialize Flask application
app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Function to fetch historical data from Yahoo Finance
def fetch_data(stock_ticker, start_date, end_date, max_retries=3):
    retries = 0
    while retries < max_retries:
        try:
            # Fetch data from Yahoo Finance
            data = yf.download(stock_ticker, start=start_date, end=end_date, progress=False)
            if data.empty:
                raise ValueError(f"No data available for {stock_ticker} between {start_date} and {end_date}")
            if not all(col in data.columns for col in ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']):
                raise ValueError("Missing required columns in data")
            return data
        except (ConnectionError, Timeout, HTTPError) as ce:
            retries += 1
            logging.error(f"Connection or timeout error fetching data for {stock_ticker}: {ce}")
            if retries == max_retries:
                raise ValueError(f"Failed to fetch data after {max_retries} retries")
            time.sleep(1)  # Add a delay before retrying, if needed
        except ValueError as ve:
            retries += 1
            logging.error(f"No data available for {stock_ticker} between {start_date} and {end_date}")
            raise ve
        except Exception as e:
            retries += 1
            logging.error(f"Error fetching data for {stock_ticker}: {e}")
            if retries == max_retries:
                raise ValueError(f"Failed to fetch data after {max_retries} retries")
            time.sleep(1)  # Add a delay before retrying, if needed
    # If all retries fail
    raise ValueError(f"Failed to fetch data for {stock_ticker} after {max_retries} retries")

def preprocess_data(data):
    try:
        # Check for NaN values in critical columns
        if data[['Close', 'High', 'Low']].isnull().any().any():
            raise ValueError("NaN values present in critical columns (Close, High, Low)")

        # Interpolate NaN values
        data = data.interpolate(method='linear')

        # Convert index to datetime if needed
        data.index = pd.to_datetime(data.index)

        # Calculate technical indicators
        data['MA_20'] = data['Close'].rolling(window=20).mean()
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        RS = gain / loss
        data['RSI'] = 100 - (100 / (1 + RS))
        exp1 = data['Close'].ewm(span=12, adjust=False).mean()
        exp2 = data['Close'].ewm(span=26, adjust=False).mean()
        data['MACD'] = exp1 - exp2
        data['MACD_signal'] = data['MACD'].ewm(span=9, adjust=False).mean()
        data['BBANDS_middle'] = data['Close'].rolling(window=20).mean()
        std_dev = data['Close'].rolling(window=20).std()
        data['BBANDS_upper'] = data['BBANDS_middle'] + 2 * std_dev
        data['BBANDS_lower'] = data['BBANDS_middle'] - 2 * std_dev
        data['TR'] = np.maximum(np.maximum(data['High'] - data['Low'], abs(data['High'] - data['Close'].shift())), abs(data['Low'] - data['Close'].shift()))
        data['ATR'] = data['TR'].rolling(window=14).mean()
        low_min = data['Low'].rolling(window=14).min()
        high_max = data['High'].rolling(window=14).max()
        data['STOCH_slowk'] = 100 * (data['Close'] - low_min) / (high_max - low_min)
        data['STOCH_slowd'] = data['STOCH_slowk'].rolling(window=3).mean()
        data['EMA_50'] = data['Close'].ewm(span=50, adjust=False).mean()
        
        # Handling ADX calculation
        if 'ADX' in data.columns:
            data['ADX'].fillna(method='ffill', inplace=True)  # Forward fill NaN values in ADX
        else:
            data['ADX'] = calculate_adx(data)  # Calculate ADX if not already present

        # Handling CCI calculation
        if 'CCI' in data.columns:
            data['CCI'].fillna(method='bfill', inplace=True)  # Backward fill NaN values in CCI
        else:
            data['CCI'] = calculate_cci(data)  # Calculate CCI if not already present

        return data

    except ValueError as ve:
        logging.error(f"Error in data preprocessing: {ve}")
        raise ValueError("Error preprocessing data")
    except Exception as e:
        logging.error(f"Unexpected error in data preprocessing: {e}")
        raise ValueError("Error preprocessing data due to an unexpected error")

def calculate_adx(data, n=14):
    try:
        # Calculate ADX
        high_low = data['High'] - data['Low']
        high_close = np.abs(data['High'] - data['Close'].shift())
        low_close = np.abs(data['Low'] - data['Close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        plus_dm = np.where((data['High'] - data['High'].shift()) > (data['Low'].shift() - data['Low']), np.maximum(data['High'] - data['High'].shift(), 0), 0)
        minus_dm = np.where((data['Low'].shift() - data['Low']) > (data['High'] - data['High'].shift()), np.maximum(data['Low'].shift() - data['Low'], 0), 0)
        atr = tr.rolling(n).mean()
        plus_di = 100 * (pd.Series(plus_dm).rolling(n).sum() / atr)
        minus_di = 100 * (pd.Series(minus_dm).rolling(n).sum() / atr)
        dx = 100 * np.abs((plus_di - minus_di) / (plus_di + minus_di))
        adx = pd.Series(dx).rolling(n).mean()
        return adx
    except Exception as e:
        logging.error(f"Error calculating ADX: {e}")
        raise ValueError("Error calculating ADX")

def calculate_cci(data, n=20):
    try:
        # Calculate CCI
        TP = (data['High'] + data['Low'] + data['Close']) / 3
        SMA = TP.rolling(window=n, min_periods=1).mean()
        MAD = np.abs(TP - SMA).rolling(window=n, min_periods=1).mean()
        CCI = (TP - SMA) / (0.015 * MAD)
        return CCI
    except Exception as e:
        logging.error(f"Error calculating CCI: {e}")
        raise ValueError("Error calculating CCI")

# Function to train the model
def train_model(stock_ticker, historical_years=10):
    try:
        # Convert historical_years to integer if it's passed as a string
        historical_years = int(historical_years)

        # Fetch historical data from Yahoo Finance
        end_date = datetime.today().strftime('%Y-%m-%d')
        start_date = (datetime.today() - timedelta(days=historical_years * 365)).strftime('%Y-%m-%d')

        logging.info(f"Fetching historical data for {stock_ticker} from {start_date} to {end_date}")
        data = fetch_data(stock_ticker, start_date, end_date)

        # Preprocess data
        data = preprocess_data(data)

        # Ensure the data is aligned properly
        if 'Close' not in data.columns:
            raise ValueError("Missing 'Close' column after preprocessing")

        # Drop rows with NaN values in critical columns
        critical_columns = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
        data = data.dropna(subset=critical_columns)

        # Split data into features and target
        X = data.drop(['Close'], axis=1)
        y = data['Close']

        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train the XGBoost model
        model = XGBRegressor()
        model.fit(X_train, y_train)

        # Save the model
        model_filename = f"{stock_ticker}_model.pkl"
        with open(model_filename, 'wb') as file:
            pickle.dump(model, file)

        logging.info(f"Model trained and saved as {model_filename}")

        return model_filename

    except Exception as e:
        logging.error(f"Error training model: {e}")
        raise ValueError("Error training model")

# Function to load the trained model
def load_model(model_filename):
    try:
        # Load the model from file
        with open(model_filename, 'rb') as file:
            model = pickle.load(file)
        return model
    except FileNotFoundError:
        logging.error(f"Model file {model_filename} not found")
        raise ValueError("Model file not found")
    except Exception as e:
        logging.error(f"Error loading model: {e}")
        raise ValueError("Error loading model")

# Function to predict future stock prices
def predict_future_prices(stock_ticker, periods=30):
    try:
        # Load the trained model
        model_filename = f"{stock_ticker}_model.pkl"
        with open(model_filename, 'rb') as model_file:
            model = pickle.load(model_file)

        # Fetch the most recent data point to predict future prices
        end_date = datetime.today().strftime('%Y-%m-%d')
        start_date = (datetime.today() - timedelta(days=365)).strftime('%Y-%m-%d')
        data = fetch_data(stock_ticker, start_date, end_date)

        # Preprocess the most recent data point
        data = preprocess_data(data)

        # Ensure data has the same features as during training
        # Assuming the number of features is consistent with training
        latest_data_point = data.iloc[-1].values.reshape(1, -1)

        # Predict future prices
        future_dates = pd.date_range(end_date, periods=periods + 1)[1:]
        future_predictions = []
        for i in range(periods):
            next_prediction = model.predict(latest_data_point)
            future_predictions.append(next_prediction[0])
            # Update latest_data_point for the next prediction step
            latest_data_point = np.append(latest_data_point[:, 1:], [[next_prediction]], axis=1)

        # Format the predictions
        future_predictions = pd.Series(future_predictions, index=future_dates)

        return future_predictions.to_dict()

    except Exception as e:
        logging.error(f"Error predicting future prices: {e}")
        raise ValueError("Error predicting future prices")


# Route to train the model
@app.route('/train_model', methods=['POST'])
def train_model_route():
    try:
        # Get stock ticker and historical years from request body
        req_data = request.get_json()
        stock_ticker = req_data.get('stock_ticker')
        historical_years = req_data.get('historical_years', 10)

        # Train the model
        model_filename = train_model(stock_ticker, historical_years)

        return jsonify({
            "message": f"Model trained successfully for {stock_ticker}",
            "model_filename": model_filename
        }), 200

    except ValueError as ve:
        return jsonify({"error": str(ve)}), 400
    except Exception as e:
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500

# Route to predict future prices
@app.route('/predict_future', methods=['POST'])
def predict_future_route():
    try:
        # Get stock ticker and periods from request body
        req_data = request.get_json()
        stock_ticker = req_data.get('stock_ticker')
        periods = req_data.get('periods', 30)

        # Predict future prices
        future_predictions = predict_future_prices(stock_ticker, periods)

        return jsonify({
            "stock_ticker": stock_ticker,
            "future_predictions": future_predictions
        }), 200

    except ValueError as ve:
        return jsonify({"error": str(ve)}), 400
    except Exception as e:
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True)
