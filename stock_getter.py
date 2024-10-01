import yfinance as yf


class Stock:

    def __init__(self, name: str, StartDate: int, EndDate: int, interval :str = "1d"):
        """
        This Python function initializes an object with a name, start date, end date, and optional
        interval, downloads stock data using Yahoo Finance API, and extracts the closing prices.
        
        :param name: The name of the stock or financial
        instrument for which you want to download data
        :param StartDate: The tart date for downloading stock data. 
        :param EndDate: The end date for the data you want to download.
        :param interval: The interval you want
        """
        self.name = name
        self.start_date = StartDate 
        self.end_date = EndDate

        # Avoiding chosing interval of 1h
        if interval == "1h":
            self.interval = "60m"
        else:
            self.interval = interval

        self.Stock = yf.download(self.name, self.start_date, self.end_date, interval=self.interval)

    def get_data(self):
        """f
        The method returns the closing stock prices from the `Stock` data.
        :return: The closing data
        """
        StockData = self.Stock['Close']
        return StockData

    def meanReturns_and_std(self):
        """
        This function calculates the mean returns and standard deviation of returns for a given dataset.
        :return: The function `meanReturns_and_std` is returning the mean returns and the standard
        deviation of the returns for the given interval.
        """
        data = self.Stock['Close']
        Returns = data.pct_change()
        Standart_deviation = Returns.std()
        meanReturns = Returns.mean()
        return meanReturns, Standart_deviation
    