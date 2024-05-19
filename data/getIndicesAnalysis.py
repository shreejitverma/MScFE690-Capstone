import numpy as np  
import pandas as pd 
import pickle
import matplotlib.pyplot as plt
from dateutil import relativedelta
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

def getRegion(ticker):
    for k in region_idx.keys():
        if ticker in region_idx[k]:
            return k


def nearest(dates, dateRef):

    dts = pd.to_datetime(dates)
    drf = pd.to_datetime(dateRef)

    prevDate = dts[dts < drf]
    return prevDate[-1]


def getReturn(period, number, ticker, dt, val):

    df = msi.loc[msi.ticker == ticker].reset_index()
    existingDates = df["Date"].unique()

    if period == "Y":
        dtp = pd.Timestamp(dt) - pd.DateOffset(years=number)
    elif period == "M":
        dtp = pd.Timestamp(dt) - pd.DateOffset(months=number)
    elif period == "W":
        dtp = pd.Timestamp(dt) - pd.DateOffset(weeks=number)
    elif period == "D":
        dtp = pd.Timestamp(dt) - pd.DateOffset(days=number)

    df["Date_pd"] = pd.to_datetime(df["Date"])
    if dtp in existingDates:
        return (val / df.loc[df.Date_pd == dtp, "Close"].values[0] - 1) * 100
    else:
        closestDate = nearest(existingDates, dtp)
        return (val / df.loc[df.Date_pd == closestDate, "Close"].values[0] - 1) * 100


def retBegin(ticker, val):
    start_val = begRef.loc[begRef.ticker == ticker, "Close"].values[0]
    return (val / start_val - 1) * 100


def checkPrice(ticker, date):
    df_t = msi.loc[msi.ticker == ticker].reset_index()
    existingDates = df_t["Date"].unique()
    closestDate = nearest(existingDates, date)
    return df_t.loc[df_t.Date == closestDate, "Close"]


msi = pd.read_csv(
    "/Users/shreejitverma/Documents/GitHub/MScFE690-Capstone/data/majorStockIndices.csv"
).reset_index()
no = ["^NYA", "^XAX", "^BUK100P", "^VIX", "IMOEX.ME", "^AORD", "^MERV", "^JN0U.JO"]
msi = msi.loc[msi.ticker.isin(set(msi.ticker) - set(no))]

region_idx = {
    "US & Canada": ["^GSPC", "^DJI", "^IXIC", "^RUT", "^GSPTSE"],
    "Latin America": [
        "^BVSP",
        "^MXX",
        "^IPSA",
    ],
    "East Asia": ["^N225", "^HSI", "000001.SS", "399001.SZ", "^TWII", "^KS11"],
    "ASEAN & Oceania": ["^STI", "^JKSE", "^KLSE", "^AXJO", "^NZ50"],
    "South & West Asia": ["^BSESN", "^TA125.TA"],
    "Europe": ["^FTSE", "^GDAXI", "^FCHI", "^STOXX50E", "^N100", "^BFX"],
}

ticker = {
    "^GSPC": "S&P 500",
    "^DJI": "Dow 30",
    "^IXIC": "NASDAQ",
    "^RUT": "Russell 2000",
    "^GSPTSE": "S&P/TSX",
    "^BVSP": "IBOVESPA",
    "^MXX": "IPC MEXICO",
    "^IPSA": "S&P/CLX IPSA",
    "^N225": "Nikkei 225",
    "^HSI": "Hang Seng Index",
    "000001.SS": "SSE Composite Index",
    "399001.SZ": "Shenzen Component",
    "^TWII": "TSEC Weighted Index",
    "^KS11": "KOSPI Composite Index",
    "^STI": "STI Index",
    "^JKSE": "Jakarta Composite Index",
    "^KLSE": "FTSE Bursa Malaysia KLCI",
    "^AXJO": "S&P/ASX 200",
    "^NZ50": "S&P/NZX 50 INDEX GROSS",
    "^BSESN": "S&P BSE SENSEX",
    "^TA125.TA": "TA-125",
    "^FTSE": "FTSE 100",
    "^GDAXI": "DAX PERFORMANCE-INDEX",
    "^FCHI": "CAC 40",
    "^STOXX50E": "ESTX 50 PR.EUR",
    "^N100": "EURONEXT 100",
    "^BFX": "BEL 20",
    "^NYA": "NYSE COMPOSITE (DJ)",
    "^XAX": "NYSE AMEX COMPOSITE INDEX",
    "^BUK100P": "Cboe UK 100 Price Return",
    "^VIX": "CBOE Volatility Index",
    "IMOEX.ME": "MOEX Russia Index",
    "^AORD": "ALL ORDINARIES",
    "^MERV": "MERVAL",
    "^JN0U.JO": "Top 40 USD Net TRI Index",
}
pagoda = ["#965757", "#D67469", "#4E5A44", "#A1B482", "#EFE482", "#99BFCF"]

msi["region"] = msi.ticker.apply(lambda x: getRegion(x))
lastDate = msi.loc[msi.Date == "2020-09-30"].reset_index().drop(["index"], axis=1)
cols = [
    "Date",
    "ticker",
    "region",
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",
    "Dividends",
    "Stock Splits",
    "Adj Close",
]
lastDate = lastDate[cols]
lastDate["1WR"] = lastDate.apply(
    lambda r: getReturn("W", 1, r.ticker, r.Date, r.Close), axis=1
)
lastDate["1DR"] = lastDate.apply(
    lambda r: getReturn("D", 1, r.ticker, r.Date, r.Close), axis=1
)
lastDate["1WR"] = lastDate.apply(
    lambda r: getReturn("W", 1, r.ticker, r.Date, r.Close), axis=1
)
lastDate["1MR"] = lastDate.apply(
    lambda r: getReturn("M", 1, r.ticker, r.Date, r.Close), axis=1
)
lastDate["3MR"] = lastDate.apply(
    lambda r: getReturn("M", 3, r.ticker, r.Date, r.Close), axis=1
)
lastDate["6MR"] = lastDate.apply(
    lambda r: getReturn("M", 6, r.ticker, r.Date, r.Close), axis=1
)
lastDate["1YR"] = lastDate.apply(
    lambda r: getReturn("Y", 1, r.ticker, r.Date, r.Close), axis=1
)
lastDate["3YR"] = lastDate.apply(
    lambda r: getReturn("Y", 3, r.ticker, r.Date, r.Close), axis=1
)
lastDate["5YR"] = lastDate.apply(
    lambda r: getReturn("Y", 5, r.ticker, r.Date, r.Close), axis=1
)
lastDate["10YR"] = lastDate.apply(
    lambda r: getReturn("Y", 10, r.ticker, r.Date, r.Close), axis=1
)

fig, axes = plt.subplots(1, 5, figsize=(20, 10), sharey=True)
width = 0.75
cols = ["6MR", "1YR", "3YR", "5YR", "10YR"]
for i, j in enumerate(cols):
    ax = axes[i]
    tick = lastDate.ticker.apply(lambda t: ticker[t])
    ax.barh(tick, lastDate[j], width, color=pagoda[i])
    ax.set_title(j, fontweight="bold")
    ax.invert_yaxis()

fig.text(0.5, 0, "Return (%)", ha="center", va="center", fontweight="bold")
fig.text(
    0, 0.5, "Stock Indices", ha="center", va="center", rotation=90, fontweight="bold"
)
fig.suptitle(
    "Returns for Major Stock Indices based on 30 September",
    fontweight="bold",
    y=1.03,
    fontsize=14,
)
fig.tight_layout()
# Save the figure
fig.savefig("stock_indices_returns.png", bbox_inches="tight")
# Price Change to 4 January 2010


begRef = msi.loc[msi.Date == "2010-01-04"]

msi["chBegin"] = msi.apply(lambda x: retBegin(x.ticker, x.Close), axis=1)

chBegin = msi.groupby(["Date", "ticker"])["chBegin"].first().unstack()
chBegin = chBegin.fillna(method="bfill")


fig, axes = plt.subplots(3, 2, figsize=(12, 8), sharex=True)

for i, k in enumerate(region_idx.keys()):
    ax = axes[int(i / 2), int(i % 2)]
    for j, t in enumerate(region_idx[k]):
        ax.plot(chBegin.index, chBegin[t], marker="", linewidth=1, color=pagoda[j])
        ax.legend([ticker[t] for t in region_idx[k]], loc="upper left", fontsize=7)
        ax.set_title(k, fontweight="bold")

fig.text(0.5, 0, "Year", ha="center", va="center", fontweight="bold")
fig.text(
    0,
    0.5,
    "Price Change/Return (%)",
    ha="center",
    va="center",
    rotation=90,
    fontweight="bold",
)
fig.suptitle(
    "Price Change/Return for Major Stock Indices based on 2010",
    fontweight="bold",
    y=1.05,
    fontsize=14,
)
fig.tight_layout()


g = msi.loc[msi.ticker == "^GSPC"]
first = g.Close.values[0]
last = g.Close.values[-1]


# Price to Earning Ratio
bb_df = pd.read_excel(
    "/Users/shreejitverma/Documents/GitHub/MScFE690-Capstone/data/Summary.xlsx"
)
bb_df["region"] = bb_df.YF_Ticker.apply(lambda x: getRegion(x))
per_med = bb_df[
    [
        "YF_Ticker",
        "PER_5Y_MED",
        "PER_4Y_MED",
        "PER_3Y_MED",
        "PER_2Y_MED",
        "PER_1Y_MED",
        "PER_Cur_MED",
    ]
].transpose()
per_med.index = ["YF_Ticker", 2015, 2016, 2017, 2018, 2019, 2020]
per_med.columns = per_med.loc["YF_Ticker"]
per_med.drop(["YF_Ticker"], inplace=True)
fig, axes = plt.subplots(3, 2, figsize=(12, 8), sharex=True)

for i, j in enumerate(region_idx.keys()):
    ax = axes[int(i / 2), int(i % 2)]
    for k, t in enumerate(region_idx[j]):
        ax.plot(per_med.index, per_med[t], marker="", linewidth=2, color=pagoda[k])
    ax.legend([ticker[t] for t in region_idx[j]], loc="upper left", fontsize=7)
    ax.set_title(j, fontweight="bold")

fig.text(0.5, 0, "Year", ha="center", va="center", fontweight="bold")
fig.text(
    0,
    0.5,
    "Price to Earning Ratio (x)",
    ha="center",
    va="center",
    rotation=90,
    fontweight="bold",
)
fig.suptitle(
    "Price to Earning Ratio for Major Stock Indices for The Last 5 Years (at 8 October)",
    fontweight="bold",
    y=1.05,
)
fig.tight_layout()
# Save the figure
fig.savefig("price_change_return.png", bbox_inches="tight")

tickers = msi.ticker.unique()
price_2020 = [checkPrice(x, np.datetime64("2020-10-08")).values[0] for x in tickers]
price_2019 = [checkPrice(x, np.datetime64("2019-10-08")).values[0] for x in tickers]
price_2018 = [checkPrice(x, np.datetime64("2018-10-08")).values[0] for x in tickers]
price_2017 = [checkPrice(x, np.datetime64("2017-10-08")).values[0] for x in tickers]
price_2016 = [checkPrice(x, np.datetime64("2016-10-08")).values[0] for x in tickers]
price_2015 = [checkPrice(x, np.datetime64("2015-10-08")).values[0] for x in tickers]


price_df = (
    pd.DataFrame(
        data={
            "ticker": tickers,
            "price_15": price_2015,
            "price_16": price_2016,
            "price_17": price_2017,
            "price_18": price_2018,
            "price_19": price_2019,
            "price_20": price_2020,
        }
    )
    .reset_index()
    .transpose()
)
price_df.columns = price_df.loc["ticker"]
price_df.drop(["index", "ticker"], inplace=True)
price_df.index = [2015, 2016, 2017, 2018, 2019, 2020]


fig, axes = plt.subplots(3, 2, figsize=(12, 8))
idx = np.arange(2015, 2021)
for i, j in enumerate(region_idx.keys()):
    ax = axes[int(i / 2), int(i % 2)]
    for k, t in enumerate(region_idx[j]):
        ax.plot(idx, price_df[t], marker="", linewidth=2, color=pagoda[k])
    ax.legend([ticker[t] for t in region_idx[j]], loc="upper left", fontsize=7)
    ax.set_title(j, fontweight="bold")

fig.text(0.5, 0, "Year", ha="center", va="center", fontweight="bold")
fig.text(0, 0.5, "Price", ha="center", va="center", rotation=90, fontweight="bold")
fig.suptitle(
    "Price at 8 October for Major Stock Indices in The Last 5 Years",
    fontweight="bold",
    y=1.05,
)
fig.tight_layout()
