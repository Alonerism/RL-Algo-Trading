# Reinforcement Learning Trading Algorithm with Data Enhancement Tools

This repository contains a set of advanced trading algorithms and data analysis tools. It leverages reinforcement learning for stock trading decisions using a custom environment, as well as traditional machine learning models for forecasting. The repository includes a small dataset for testing.

## 1. **Trader (A2C-based)**
This trading algorithm uses a custom environment built with the `gym-anytrading` library and implements an A2C (Advantage Actor Critic) model from `stable-baselines3`. The agent trades based on technical indicators such as SMA, RSI, MACD, and OBV. 

### Key Features:
- **Custom Gym Environment**: Integrates additional technical indicators for trading decisions.
- **A2C Model**: Trains with GPU support for faster learning.
- **Money Tracking**: Monitors portfolio value over time during testing.

Usage: You can train the model using the A2C algorithm and run simulations using the provided dataset.

---

## 2. **getData.py**
Fetches and preprocesses historical stock data from Alpaca's API. It retrieves minute-level data, resamples it to 30-minute intervals, and exports it as a CSV for further analysis.

### Key Features:
- **Alpaca API Integration**: Downloads historical data for selected symbols.
- **Resampling**: Converts high-frequency minute data to lower-frequency intervals.

Usage: API credentials must be provided. You can adjust the date range and symbol as needed.

---

## 3. **ARIMA.py**
This script enhances the data obtained from `getData.py` by fitting an ARIMA model. It performs differencing and grid search over ARIMA parameters for forecasting future stock prices.

### Key Features:
- **Differencing**: Automatic differencing based on ADF test results.
- **Model Selection**: Auto ARIMA model selection based on grid search.
- **Residual Analysis**: Includes stationarity checks and residual diagnostic plots.

Usage: Specify the dataset path, and the script will output model parameters, forecasts, and residual plots.

---

## 4. **LSTM.py**
This script trains an LSTM model on historical stock price data, predicting future prices over multiple time horizons (e.g., 1, 2, 5, 10 days). It includes hyperparameter tuning and evaluates the model using RMSE.

### Key Features:
- **Multi-step Forecasting**: Predicts prices for multiple future time steps.
- **Grid Search**: Optimizes LSTM architecture and learning rates.
- **Visualization**: Plots actual vs predicted prices for better interpretability.

Usage: The data should be preprocessed using `getData.py`. You can configure hyperparameters directly in the script.

---

## Sample Dataset

A small sample dataset has been included in this repository under the `Data.csv` file. You can use this dataset to try out the code without needing to fetch external data or configure an API. This allows you to test and experiment with the algorithms right away.

---

## Installation and Requirements
All scripts require Python 3.7+ and depend on the following packages:
- `pandas`
- `numpy`
- `matplotlib`
- `ta-lib`
- `torch`
- `finta`
- `stable-baselines3`
- `gym-anytrading`
- `alpaca-trade-api`
- `pmdarima`
- `statsmodels`
- `scikit-learn`

To install dependencies, run:
```bash
pip install -r requirements.txt
