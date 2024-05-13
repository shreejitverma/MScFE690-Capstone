import pandas as pd

class FredAPI:
    """
    Wrapper for the data API of the Federal Reserve Economic Data (FRED)
    """

    def fetch_series(self, series_ids, start_date=None, end_date=None):
        """
        Fetches series data from the FRED website and returns a pandas DataFrame.

        :param series_ids: str, list of str, or dict with series IDs or keys
        :param start_date: str in 'yyyy-mm-dd' format (optional)
        :param end_date: str in 'yyyy-mm-dd' format (optional)
        :return: pandas DataFrame with the requested series. If a dict is passed as series IDs, the dict values are used
                 as column names.
        """

        if isinstance(series_ids, list):

            dtf = pd.DataFrame()

            for series_id in series_ids:
                single_series = self._fetch_single_series(series_id)
                dtf = pd.concat([dtf, single_series], axis=1)

            dtf.sort_index(inplace=True)

        elif isinstance(series_ids, dict):

            dtf = pd.DataFrame()

            for series_id, column_name in series_ids.items():
                single_series = self._fetch_single_series(series_id)
                dtf = pd.concat([dtf, single_series], axis=1)

            dtf.columns = series_ids.values()

        else:

            dtf = self._fetch_single_series(series_ids)

        dtf = self._adjust_dates(dtf, start_date, end_date)

        return dtf

    @staticmethod
    def _fetch_single_series(series_id):
        """
        Fetches data for a single series from FRED.

        :param series_id: str, Series ID
        :return: pandas DataFrame with the series data
        """

        url = f'https://fred.stlouisfed.org/data/{series_id}.txt'
        dtf = pd.read_csv(url, sep="/n")
        series_start = dtf[dtf[dtf.columns[0]].str.contains('DATE\s+VALUE')].index[0] + 1
        dtf = dtf.loc[series_start:]
        dtf = dtf[dtf.columns[0]].str.split('\s+', expand=True)
        dtf = dtf[~(dtf[1] == '.')]
        dtf = pd.DataFrame(data=dtf[1].values.astype(float),
                          index=pd.to_datetime(dtf[0]),
                          columns=[series_id])
        dtf.index.rename('Date', inplace=True)

        return dtf

    @staticmethod
    def _adjust_dates(dtf, start_date, end_date):
        """
        Adjusts DataFrame to given start and end dates.

        :param dtf: pandas DataFrame, Input DataFrame
        :param start_date: str, Start date in 'yyyy-mm-dd' format
        :param end_date: str, End date in 'yyyy-mm-dd' format
        :return: pandas DataFrame, DataFrame filtered by start and end dates
        """

        if start_date is not None:
            start_date = pd.to_datetime(start_date)
            dtf = dtf[dtf.index >= start_date]

        if end_date is not None:
            end_date = pd.to_datetime(end_date)
            dtf = dtf[dtf.index <= end_date]

        return dtf
