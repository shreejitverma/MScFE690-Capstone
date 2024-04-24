import yfinance as yf
import matplotlib.pyplot as plt
from portfolio.backtesting import FHSignalBasedWeights
import pandas as pd
import numpy as np

# Define start and end dates
start_date = '2000-03-30'
end_date = pd.Timestamp.today()

# Fetch data using yfinance
df = yf.download("^GSPC", start=start_date, end=end_date)['Adj Close']
df2 = yf.download("^VIX", start=start_date, end=end_date)['Adj Close']
df3 = yf.download("AUDUSD=X", start=start_date, end=end_date)['Adj Close']
df4 = yf.download("^TNX", start=start_date, end=end_date)['Adj Close']

# Concatenate dataframes
ts = pd.concat([df, df2, df3, df4], axis=1, sort=True).dropna(how='all').astype(float)

# Calculate momentum signals
mom_signals = np.log(ts).diff(252).dropna(how='all')

# Initialize FHSignalBasedWeights object
b = FHSignalBasedWeights(ts, mom_signals, rebalance='Y')

# Run backtest and plot
b.run_backtest('mom_backtest').plot()
plt.show()
