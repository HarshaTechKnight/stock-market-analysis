import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Fetch stock data
def fetch_stock_data(ticker, start_date, end_date):
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    return stock_data

# Preprocess data
def preprocess_data(stock_data):
    stock_data['Next_Close'] = stock_data['Close'].shift(-1)
    stock_data.dropna(inplace=True)
    X = stock_data[['Open', 'High', 'Low', 'Close', 'Volume']]
    y = stock_data['Next_Close']
    return train_test_split(X, y, test_size=0.2, random_state=42)

# Train and evaluate model
def train_model(X_train, X_test, y_train, y_test):
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"Mean Squared Error: {mse}")
    print(f"R-squared: {r2}")
    return model, y_pred

# Visualize results
def plot_results(y_test, y_pred, ticker):
    plt.figure(figsize=(10, 6))
    plt.plot(y_test.index, y_test, label='Actual Prices', color='blue')
    plt.plot(y_test.index, y_pred, label='Predicted Prices', color='red', linestyle='--')
    plt.title(f'{ticker} Stock Price Prediction')
    plt.xlabel('Date')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.show()

# Main function
def main():
    ticker = "AAPL"
    start_date = "2020-01-01"
    end_date = "2023-01-01"
    
    stock_data = fetch_stock_data(ticker, start_date, end_date)
    X_train, X_test, y_train, y_test = preprocess_data(stock_data)
    model, y_pred = train_model(X_train, X_test, y_train, y_test)
    plot_results(y_test, y_pred, ticker)

if __name__ == "__main__":
    main()