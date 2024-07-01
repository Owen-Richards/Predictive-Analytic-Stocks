import os
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
import yfinance as yf
from datetime import datetime, timedelta
import logging
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV
import pickle
import time

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)  # Adjust logging level as needed

def date_to_timestamp(date_str):
    # Convert date string to datetime object
    date_obj = datetime.strptime(date_str, '%Y-%m-%d')
    # Convert datetime object to Unix timestamp
    timestamp = int(time.mktime(date_obj.timetuple()))
    return timestamp

def fetch_data(stock_ticker, start_date, end_date, max_retries=3):
    retries = 0
    while retries < max_retries:
        try:
            # Fetch data from Yahoo Finance
            data = yf.download(stock_ticker, start=start_date, end=end_date)
            if data.empty:
                raise ValueError(f"No data available for {stock_ticker} between {start_date} and {end_date}")
            if not all(col in data.columns for col in ['High', 'Low', 'Close']):
                raise ValueError("Missing required columns in data")
            return data
        except (ConnectionError, TimeoutError) as ce:
            retries += 1
            logging.error(f"Connection or timeout error fetching data for {stock_ticker}: {ce}")
            if retries == max_retries:
                raise ValueError(f"Failed to fetch data after {max_retries} retries")
            time.sleep(1)  # Add a delay before retrying, if needed
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
        if data.isnull().values.any():
            data = data.interpolate(method='linear')  # Use linear interpolation for missing values

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
        data['ADX'] = calculate_adx(data)
        data['CCI'] = calculate_cci(data)

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

def train_model(stock_ticker, historical_years=10):
    try:
        # Convert historical_years to integer if it's passed as a string
        historical_years = int(historical_years)

        # Fetch historical data from Yahoo Finance
        end_date = datetime.today()
        start_date = end_date - timedelta(days=historical_years * 365)

        logging.info(f"Fetching historical data for {stock_ticker} from {start_date} to {end_date}")
        data = fetch_data(stock_ticker, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))

        if data.empty:
            raise ValueError(f"No data available for training {stock_ticker} between {start_date} and {end_date}")

        # Preprocess data
        logging.info("Preprocessing data...")
        data = preprocess_data(data)

        # Drop NaN values
        data.dropna(inplace=True)

        # Define features and target
        X = data[['Close', 'MA_20', 'RSI', 'MACD', 'MACD_signal', 'BBANDS_middle', 'BBANDS_upper', 'BBANDS_lower', 'ATR', 'STOCH_slowk', 'STOCH_slowd', 'EMA_50', 'ADX', 'CCI']]
        y = data['Close'].shift(-1).dropna()

        if len(X) == 0 or len(y) == 0:
            raise ValueError(f"No data available for training {stock_ticker}")

        # Split data into training and testing sets
        train_size = int(len(X) * 0.8)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]

        # Initialize XGBoost model
        model = XGBRegressor(objective='reg:squarederror', n_jobs=-1)

        # Define hyperparameters for GridSearchCV
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 4, 5]
        }

        # Perform Grid Search CV to find the best model
        logging.info("Performing GridSearchCV for hyperparameter tuning...")
        grid_search = GridSearchCV(estimator=model, param_grid=param_grid, scoring='neg_mean_squared_error', cv=5, verbose=1, n_jobs=-1)
        grid_search.fit(X_train, y_train)

        # Get the best model from Grid Search
        best_model = grid_search.best_estimator_

        # Save the best model to disk
        model_filename = f"{stock_ticker}_model.pkl"
        with open(model_filename, 'wb') as model_file:
            pickle.dump((best_model, X.columns), model_file)

        logging.info(f"Saved model as {model_filename}")

    except Exception as e:
        logging.error(f"Error training model for {stock_ticker}: {e}")
        raise ValueError(f"Error training model for {stock_ticker}: {e}")

@app.route('/train/<stock_ticker>', methods=['POST'])
def train(stock_ticker):
    try:
        if request.method == 'POST':
            data = request.get_json()
            historical_years = data.get('historical_years', 10)

            # Train model
            train_model(stock_ticker, historical_years)

            return jsonify({"message": f"Model trained successfully for {stock_ticker}"}), 200

    except Exception as e:
        logging.error(f"Error training model for {stock_ticker}: {e}")
        return jsonify({"error": f"Error training model for {stock_ticker}"}), 500


def predict_future(stock_ticker, years=2):
    try:
        # Load the pre-trained model
        model_filename = f"{stock_ticker}_model.pkl"
        if not os.path.exists(model_filename):
            raise ValueError(f"Model file {model_filename} not found for {stock_ticker}")

        with open(model_filename, 'rb') as model_file:
            model, X_columns = pickle.load(model_file)

        # Fetch future stock data for prediction
        end_date = datetime.today()
        start_date = end_date + timedelta(days=1)  # Start from the next day after today

        future_dates = pd.date_range(start=start_date, periods=8 * years, freq='Q')
        future_data = pd.DataFrame(index=future_dates)

        logging.info(f"Fetching future data for {stock_ticker} from {start_date} to {future_dates[-1]}")
        future_data = fetch_data(stock_ticker, start_date.strftime('%Y-%m-%d'), future_dates[-1].strftime('%Y-%m-%d'))

        # Preprocess future data
        future_data = preprocess_data(future_data)

        # Make predictions
        future_predictions = []
        for index, date in enumerate(future_dates):
            if date in future_data.index:
                future_data_slice = future_data.loc[date:date]
                X_future = future_data_slice[X_columns]
                prediction = model.predict(X_future)
                future_predictions.append({
                    "date": date.strftime('%Y-%m-%d'),
                    "predicted_price": float(prediction[0])
                })

        return future_predictions

    except Exception as e:
        logging.error(f"Error predicting future prices for {stock_ticker}: {e}")
        raise ValueError(f"Error predicting future prices for {stock_ticker}")

@app.route('/predict', methods=['GET'])
def predict():
    try:
        stock_ticker = request.args.get('stock_ticker')
        if not stock_ticker:
            return jsonify({"error": "Missing stock_ticker parameter"}), 400

        # Predict future prices
        future_predictions = predict_future(stock_ticker)

        return jsonify({"predictions": future_predictions}), 200

    except Exception as e:
        logging.error(f"Error predicting future prices: {e}")
        return jsonify({"error": f"Error predicting future prices: {e}"}), 500

if __name__ == '__main__':
    app.run(debug=True)
