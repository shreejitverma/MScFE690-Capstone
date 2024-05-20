import pandas as pd
import numpy as np
import yfinance as yf
import requests
from datetime import datetime
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller, acf, pacf, coint
from pmdarima import auto_arima
from sklearn.covariance import LedoitWolf
from scipy.cluster.hierarchy import linkage, fcluster
from pypfopt import HRPOpt, expected_returns, risk_models
import seaborn as sns
import warnings

warnings.filterwarnings('ignore', message='Setting an item of incompatible dtype')

# Parameters for GBM 
S0 = 100       # Initial asset price
r = 0.03       # Risk-free rate, use as drift under risk-neutral measure
sigma = 0.2    # Volatility of the asset
T = 2          # Time horizon (years)
dt = 0.01      # Time step (years)
initial_wealth = 1000000  # Initial portfolio value

def simulate_gbm_martingale(S0, r, sigma, T, dt):
    """ Simulate asset prices under GBM ensuring martingale properties. """
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
    wealth[0] = initial_wealth  
    proportion_invested = np.linspace(0.5, 0.1, len(S))  

    for i in range(1, len(S)):
        wealth[i] = wealth[i-1] * (1 + proportion_invested[i] * (S[i] / S[i-1] - 1) + (1 - proportion_invested[i]) * (r * dt))
    return wealth

def dynamic_portfolio_optimization(initial_allocation, portfolio_returns, dt):
    """ Dynamic portfolio optimization strategy. """
    N = portfolio_returns.shape[0]
    adjusted_allocation = np.copy(initial_allocation)
    dynamic_wealth = np.zeros(N)
    dynamic_wealth[0] = initial_wealth

    for t in range(1, N):
        # Portfolio returns calculation
        current_return = np.dot(portfolio_returns[t], adjusted_allocation)
        dynamic_wealth[t] = dynamic_wealth[t-1] * (1 + current_return)
         # Rebalance based on updated risk metrics 
        var_threshold = np.percentile(portfolio_returns[t], 5)
        if current_return < var_threshold:
            adjusted_allocation = np.clip(adjusted_allocation * (1 - dt), 0, 1)
            adjusted_allocation /= np.sum(adjusted_allocation)  # Normalize weights

    return dynamic_wealth

# API Key for FRED
fred_api_key = '04c9cda8586e4d685a809f3815ed1e98'

filter_start_date = '2018-01-01'  # Date from which to start plotting
end_date = datetime.today().strftime('%Y-%m-%d')

# FRED API URL for SOFR
fred_url_sofr = f'https://api.stlouisfed.org/fred/series/observations?series_id=SOFR&api_key={fred_api_key}&file_type=json&observation_start={filter_start_date}&observation_end={end_date}'

# Local CSV file for SONIA
sonia_csv_file = 'IUDSOIA.csv'

# Fetch data from FRED (SOFR)
response_sofr = requests.get(fred_url_sofr)
data_sofr = response_sofr.json()['observations']

df_sonia = pd.read_csv(sonia_csv_file, skiprows=1)

df_sonia = df_sonia.iloc[:, [0, -1]]
df_sonia.columns = ['date', 'SONIA']
df_sonia['date'] = pd.to_datetime(df_sonia['date'], format='%Y-%m-%d')
df_sonia.set_index('date', inplace=True)

df_sonia['SONIA'] = pd.to_numeric(df_sonia['SONIA'], errors='coerce')

df_sofr = pd.DataFrame(data_sofr)
df_sofr['date'] = pd.to_datetime(df_sofr['date'])
df_sofr.set_index('date', inplace=True)
df_sofr.rename(columns={'value': 'SOFR'}, inplace=True)

# Convert SOFR to numeric, forcing errors to NaN
df_sofr['SOFR'] = pd.to_numeric(df_sofr['SOFR'], errors='coerce')

# Combine SOFR and SONIA data into a single DataFrame
df_combined = pd.concat([df_sofr, df_sonia], axis=1)

# Filter data from 2014 onwards
df_filtered = df_combined[df_combined.index >= filter_start_date]

# Drop rows with NaN values
df_filtered = df_filtered.dropna()

# Plotting SOFR and SONIA data
plt.figure(figsize=(14, 7))
plt.plot(df_filtered.index, df_filtered['SOFR'], label='SOFR', color='blue')
plt.plot(df_filtered.index, df_filtered['SONIA'], label='SONIA', color='green')
plt.xlabel('Date')
plt.ylabel('Rate')
plt.title('SOFR and SONIA Rates from 2018 to Today')
plt.legend()
plt.grid(True)
plt.show()

# Currency Risk Analysis
currencies = ['CNY=X', 'JPY=X', 'EURUSD=X', 'GBPUSD=X']
currency_data = yf.download(currencies, start=filter_start_date, end=end_date)['Adj Close']

# Handle missing values
currency_data = currency_data.dropna()

# Normalize currency data for comparison
currency_data_normalized = currency_data / currency_data.iloc[0]

# Plotting currency exchange rates
currency_data_normalized.plot(figsize=(14, 7))
plt.title('Currency Exchange Rates from 2018 to Today')
plt.xlabel('Date')
plt.ylabel('Normalized Exchange Rate')
plt.legend(currency_data.columns)
plt.grid(True)
plt.show()

# Credit Risk Analysis 
credit_ratings = {
    'Company': ['Apple', 'Microsoft', 'Amazon', 'Google', 'Meta', 'Nvidia', 'Tesla'],
    'Credit Rating': ['AA+', 'AAA', 'AA', 'AA+', 'A', 'AA-', 'BBB+'],
    'CDS Spread (bps)': [30, 20, 40, 35, 50, 45, 60]
}
df_credit = pd.DataFrame(credit_ratings)
print(df_credit)

# Market Risk Analysis
market_indices = ['^GSPC', '^DJI', '^IXIC', '^NSEI', '^FTSE']
market_data = yf.download(market_indices, start=filter_start_date, end=end_date)['Adj Close']

# Handle missing values
market_data = market_data.dropna()

# Normalize market index data for comparison
market_data_normalized = market_data / market_data.iloc[0]

# Plotting market indices
market_data_normalized.plot(figsize=(14, 7))
plt.title('Market Indices from 2018 to Today')
plt.xlabel('Date')
plt.ylabel('Normalized Index Level')
plt.legend(market_data.columns)
plt.grid(True)
plt.show()

# Correlation Analysis
correlation_matrix = market_data.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Matrix of Market Indices')
plt.show()

# Cointegration Analysis
for i in range(len(market_indices)):
    for j in range(i+1, len(market_indices)):
        score, pvalue, _ = coint(market_data.iloc[:, i], market_data.iloc[:, j])
        print(f'Cointegration test between {market_indices[i]} and {market_indices[j]}: p-value = {pvalue:.4f}')

# Hierarchical Risk Parity Clustering
linkage_matrix = linkage(correlation_matrix, method='ward')
plt.figure(figsize=(10, 7))
sns.clustermap(correlation_matrix, method='ward', cmap='coolwarm', figsize=(10, 7), linewidths=.5)
plt.title('Hierarchical Clustering of Market Indices')
plt.show()

# Mitigation Plan using Interest Rate Swap and FX Swap
hedging_strategy = {
    'Interest Rate Swap': 'Hedge 50% of SOFR and SONIA exposure',
    'FX Swap': 'Hedge 50% of exposure in CNY, JPY, EUR, GBP'
}
df_hedging = pd.DataFrame(list(hedging_strategy.items()), columns=['Instrument', 'Strategy'])
print(df_hedging)

# Portfolio Construction using HRP and GBM
# Fetch stock data for the "Magnificent 7"
magnificent_7 = ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META', 'NVDA', 'TSLA']
stock_data = yf.download(magnificent_7, start=filter_start_date, end=end_date)['Adj Close']

stock_data = stock_data.dropna()

# Calculate daily returns
returns = stock_data.pct_change().dropna()

# Expected returns and covariance matrix
mu = expected_returns.mean_historical_return(stock_data)
S = risk_models.CovarianceShrinkage(stock_data).ledoit_wolf()

# HRP Optimization with explicit dtype casting to avoid warnings
hrp = HRPOpt(returns)
hrp_weights = hrp.optimize()
hrp_weights = {k: float(v) for k, v in hrp_weights.items()}  # Cast weights to float to avoid dtype warning
print("HRP Weights:\n", hrp_weights)

# Convert HRP weights to array for simulation
hrp_weight_array = np.array(list(hrp_weights.values()))

# Simulate GBM for portfolio optimization with martingale properties
n_simulations = 1000
simulation_days = int(T / dt)  # Adjust simulation days according to T and dt
initial_price = S0

# Simulate the GBM for each stock separately
gbm_simulations = np.zeros((n_simulations, simulation_days, len(magnificent_7)))
np.random.seed(42)

for i in range(n_simulations):
    for j in range(len(magnificent_7)):
        t, gbm_simulations[i, :, j] = simulate_gbm_martingale(initial_price, r, sigma, T, dt)

# Calculate portfolio value using HRP weights
portfolio_values = np.zeros((n_simulations, simulation_days))
for i in range(n_simulations):
    portfolio_values[i, :] = initial_wealth * np.dot(gbm_simulations[i, :, :], hrp_weight_array)

# Apply control strategy
portfolio_wealth = np.zeros((n_simulations, simulation_days))
for i in range(n_simulations):
    portfolio_wealth[i] = apply_control_strategy(portfolio_values[i], r, T, dt)

# Debugging print statements
print("GBM Simulations shape:", gbm_simulations.shape)
print("Portfolio Values shape:", portfolio_values.shape)
print("Portfolio Wealth shape:", portfolio_wealth.shape)

# Plot GBM Simulations
plt.figure(figsize=(14, 7))
for i in range(n_simulations):
    plt.plot(t, portfolio_wealth[i], alpha=0.1, color='blue')
plt.title('GBM Simulations with Control Strategy for Portfolio Optimization')
plt.xlabel('Time (years)')
plt.ylabel('Portfolio Value')
plt.grid(True)
plt.show()

# Dynamic Portfolio Optimization
# Combine HRP and GBM results for dynamic optimization

# Function for dynamic rebalancing
def dynamic_rebalancing(portfolio_returns, hrp_weights, dt):
    N = portfolio_returns.shape[0]
    adjusted_weights = np.copy(hrp_weights)
    dynamic_wealth = np.zeros(N)
    dynamic_wealth[0] = initial_wealth
    
    for t in range(1, N):
        # Calculate portfolio returns
        current_return = np.dot(portfolio_returns[t], adjusted_weights)
        current_return = np.sum(current_return)  # Ensure current_return is a scalar
        dynamic_wealth[t] = dynamic_wealth[t-1] * (1 + current_return)
        
        # Rebalance based on updated risk metrics
        var_threshold = np.percentile(portfolio_returns[t], 5)
        if current_return < var_threshold:
            adjusted_weights = np.clip(adjusted_weights * (1 - dt), 0, 1)
            adjusted_weights /= np.sum(adjusted_weights)  # Normalize weights
    
    return dynamic_wealth

# Calculate daily returns for portfolio
portfolio_returns = (portfolio_values[:, 1:] / portfolio_values[:, :-1]) - 1

# Apply dynamic rebalancing
dynamic_portfolio_wealth = np.zeros((n_simulations, simulation_days - 1))
for i in range(n_simulations):
    dynamic_portfolio_wealth[i] = dynamic_rebalancing(portfolio_returns[i], hrp_weight_array, dt)

# Plot Dynamic Portfolio Wealth
plt.figure(figsize=(14, 7))
for i in range(n_simulations):
    plt.plot(t[1:], dynamic_portfolio_wealth[i], alpha=0.1, color='blue')
plt.title('Dynamic Portfolio Optimization with HRP and GBM')
plt.xlabel('Time (years)')
plt.ylabel('Portfolio Value')
plt.grid(True)
plt.show()

# ARIMA Forecasting for Magnificent 7 Stocks
plt.figure(figsize=(14, 7))
for stock in magnificent_7:
    stock_data_series = stock_data[stock]
    stock_data_series.index = pd.DatetimeIndex(stock_data_series.index).to_period('D')
    arima_model = auto_arima(stock_data_series, seasonal=False, trace=True, error_action='ignore', suppress_warnings=True)
    forecast = arima_model.predict(n_periods=30)
    forecast_index = pd.date_range(start=stock_data_series.index[-1].to_timestamp(), periods=30, freq='D')
    plt.plot(stock_data_series.index.to_timestamp(), stock_data_series, label=f'Actual {stock}')
    plt.plot(forecast_index, forecast, label=f'Forecast {stock}', linestyle='--')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.title('ARIMA Forecast for Magnificent 7 Stocks')
plt.legend()
plt.grid(True)
plt.show()

# ARIMA Forecasting for Market Indices
plt.figure(figsize=(14, 7))
for index in market_indices:
    index_data_series = market_data[index]
    index_data_series.index = pd.DatetimeIndex(index_data_series.index).to_period('D')
    arima_model = auto_arima(index_data_series, seasonal=False, trace=True, error_action='ignore', suppress_warnings=True)
    forecast = arima_model.predict(n_periods=30)
    forecast_index = pd.date_range(start=index_data_series.index[-1].to_timestamp(), periods=30, freq='D')
    plt.plot(index_data_series.index.to_timestamp(), index_data_series, label=f'Actual {index}')
    plt.plot(forecast_index, forecast, label=f'Forecast {index}', linestyle='--')
plt.xlabel('Date')
plt.ylabel('Index Level')
plt.title('ARIMA Forecast for Market Indices')
plt.legend()
plt.grid(True)
plt.show()

# Plotting returns for Magnificent 7 Stocks
returns_magnificent_7 = stock_data.pct_change().dropna()
plt.figure(figsize=(14, 7))
for stock in magnificent_7:
    plt.plot(returns_magnificent_7.index, returns_magnificent_7[stock], label=stock)
plt.xlabel('Date')
plt.ylabel('Daily Return')
plt.title('Daily Returns for Magnificent 7 Stocks')
plt.legend()
plt.grid(True)
plt.show()

# Plotting returns for Market Indices
returns_market_indices = market_data.pct_change().dropna()
plt.figure(figsize=(14, 7))
for index in market_indices:
    plt.plot(returns_market_indices.index, returns_market_indices[index], label=index)
plt.xlabel('Date')
plt.ylabel('Daily Return')
plt.title('Daily Returns for Market Indices')
plt.legend()
plt.grid(True)
plt.show()