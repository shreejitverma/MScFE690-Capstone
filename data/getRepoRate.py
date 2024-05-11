import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader.data as web
import datetime


def get_repo_rate_data(country, start_date, end_date):
    """
    Fetch repo rate data for the specified country from Yahoo Finance.

    Args:
    country: str, the country for which to fetch repo rate data.
    start_date: str, the start date of the data (YYYY-MM-DD).
    end_date: str, the end date of the data (YYYY-MM-DD).

    Returns:
    pd.DataFrame, repo rate data for the specified country.
    """
    ticker = country + "REPOUSD=X"  # Yahoo Finance format for repo rates
    repo_data = yf.download(ticker, start=start_date, end=end_date)
    return repo_data


def fetch_repo_rate_data(country, start_date, end_date):
    """
    Fetch repo rate data for the specified country.

    Args:
    country: str, the country for which to fetch repo rate data (e.g., 'US', 'China', 'India', 'UK', 'Japan').
    start_date: str, the start date of the data (YYYY-MM-DD).
    end_date: str, the end date of the data (YYYY-MM-DD).

    Returns:
    pd.DataFrame, repo rate data for the specified country.
    """
    if country == "US":
        # Fetch data from FRED (Federal Reserve Economic Data)
        repo_data = web.DataReader("DGS1MO", "fred", start_date, end_date)
    elif country == "China":
        # Fetch data from a reliable financial data provider for China
        # Example: repo_data = web.DataReader('CHN_REPORATE', 'example_source', start_date, end_date)
        pass  # Replace 'example_source' with the actual source for China data
    elif country == "India":
        # Fetch data from a reliable financial data provider for India
        # Example: repo_data = web.DataReader('IND_REPORATE', 'example_source', start_date, end_date)
        pass  # Replace 'example_source' with the actual source for India data
    elif country == "UK":
        # Fetch data from a reliable financial data provider for the UK
        # Example: repo_data = web.DataReader('UK_REPORATE', 'example_source', start_date, end_date)
        pass  # Replace 'example_source' with the actual source for UK data
    elif country == "Japan":
        # Fetch data from a reliable financial data provider for Japan
        # Example: repo_data = web.DataReader('JPN_REPORATE', 'example_source', start_date, end_date)
        pass  # Replace 'example_source' with the actual source for Japan data
    else:
        raise ValueError("Invalid country code. Please provide a valid country code.")

    return repo_data


def save_repo_rate_data_to_csv(repo_data, country):
    """
    Save repo rate data to CSV file.

    Args:
    repo_data: pd.DataFrame, repo rate data for a country.
    country: str, the country for which the data belongs.
    """
    repo_data.to_csv(country + "_repo_rate_data.csv", index=True)


def plot_repo_rate_time_series(repo_data, country):
    """
    Plot the repo rate time series.

    Args:
    repo_data: pd.DataFrame, repo rate data for a country.
    country: str, the country for which the data belongs.
    """
    plt.figure(figsize=(10, 5))
    if country == "US":
        plt.plot(repo_data.index, repo_data["DGS1MO"])
    plt.title(country + " Repo Rate Time Series (2013-2023)")
    plt.xlabel("Date")
    plt.ylabel("Repo Rate")
    plt.grid(True)
    plt.show()


# Define parameters
countries = ["US", "China", "India", "UK", "Japan"]
start_date = "2013-01-01"
end_date = "2023-01-01"

# Fetch repo rate data, save to CSV, and plot time series for each country
for country in countries:
    repo_data = fetch_repo_rate_data(country, start_date, end_date)
    # repo_data = get_repo_rate_data(country, start_date, end_date)
    save_repo_rate_data_to_csv(repo_data, country)
    plot_repo_rate_time_series(repo_data, country)
