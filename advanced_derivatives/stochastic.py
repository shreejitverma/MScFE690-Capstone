import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import squareform
from statsmodels.tsa.arima.model import ARIMA

np.random.seed(0)

# Define trackers for interest rates and tech stocks
class FwdIRSTrackers:
    currency_ticker_map = {
        'USD': '^TNX',
        'EUR': 'IEF',
        'JPY': 'JGBD',
        'GBP': 'GILT',
    }

class TechStockTracker:
    stock_ticker_map = {
        'AAPL': 'AAPL',
        'AMZN': 'AMZN',
        'GOOGL': 'GOOGL',
        'META': 'META',
        'MSFT': 'MSFT',
        'TSLA': 'TSLA',
        'NFLX': 'NFLX',
        'NVDA': 'NVDA',
    }

# Fetch historical data using yfinance
def fetch_data(tickers, start_date, end_date):
    ticker_strings = ' '.join(tickers)
    data = yf.download(ticker_strings, start=start_date, end=end_date)['Adj Close']
    data.index = pd.to_datetime(data.index)  # Ensure datetime index
    data = data.asfreq('B').ffill()  # Set business day frequency and forward fill missing values
    #return data.pct_change().fillna(method='ffill')  # Calculate returns
    return data.pct_change().ffill()  # Calculate returns

# Implement ARIMA Forecasting
def forecast_arima(series, order=(1, 1, 1), steps=5):
    model = ARIMA(series, order=order)
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=steps)
    return forecast

# Parameters for GBM simulation
S0 = 100       # Initial asset price
r = 0.03       # Risk-free rate, use as drift under risk-neutral measure
sigma = 0.2    # Volatility of the asset
T = 2          # Time horizon (years)
dt = 0.01      # Time step (years)

def simulate_gbm(S0, r, sigma, T, dt):
    """ Simulate asset prices under GBM with risk-neutral measure ensuring martingale properties. """
    N = int(T / dt)
    t = np.linspace(0, T, N)
    W = np.random.standard_normal(size=N)
    W = np.cumsum(W) * np.sqrt(dt)
    X = (r - 0.5 * sigma**2) * t + sigma * W
    S = S0 * np.exp(X)
    return t, S

def apply_control_strategy(S, r, T, dt):
    """ Dynamic control strategy to adjust investments over time. """
    wealth = np.empty_like(S)
    wealth[0] = 1000000  # Initial wealth
    proportion_invested = np.linspace(0.5, 0.1, len(S))  # Linearly decreasing investment in risky asset

    for i in range(1, len(S)):
        wealth[i] = wealth[i-1] * (1 + proportion_invested[i] * (S[i] / S[i-1] - 1) + (1 - proportion_invested[i]) * (r * dt))
    return wealth

# Fetch and process data
tickers = list(TechStockTracker.stock_ticker_map.values())
historical_returns = fetch_data(tickers, '2018-01-01', '2023-01-01')

# GBM simulation and applying control strategy
t, simulated_prices = simulate_gbm(S0, r, sigma, T, dt)
wealth = apply_control_strategy(simulated_prices, r, T, dt)

# Plotting the simulated asset prices and wealth evolution
plt.figure(figsize=(14, 7))
plt.subplot(1, 2, 1)
plt.plot(t, simulated_prices, label='Simulated Asset Prices')
plt.title('Asset Prices Simulation using GBM')
plt.xlabel('Time (years)')
plt.ylabel('Price')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(t, wealth, label='Wealth Over Time')
plt.title('Wealth Evolution with Dynamic Control')
plt.xlabel('Time (years)')
plt.ylabel('Wealth ($)')
plt.legend()
plt.grid(True)
plt.show()

# ARIMA forecasting as previously implemented
forecast_results = {}
for ticker in tickers:
    forecast_results[ticker] = forecast_arima(historical_returns[ticker], steps=20)

# Correct subplot setup based on the number of tickers
rows = len(tickers) // 2 if len(tickers) % 2 == 0 else len(tickers) // 2 + 1
fig, axes = plt.subplots(nrows=rows, ncols=2, figsize=(15, 5 * rows))
axes = axes.flatten()
for i, ticker in enumerate(tickers):
    axes[i].plot(historical_returns.index, historical_returns[ticker], label='Historical')
    forecast_index = pd.date_range(start=historical_returns.index[-1], periods=21, freq='B')[1:]  # excluding the first point which is the last historical point
    axes[i].plot(forecast_index, forecast_results[ticker], label='Forecast', linestyle='--')
    axes[i].set_title(f'ARIMA Forecast for {ticker}')
    axes[i].set_ylabel('Returns')
    axes[i].legend()

plt.tight_layout()
plt.show()