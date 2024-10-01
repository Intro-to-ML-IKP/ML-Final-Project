import yfinance as yf

class Stock:
    def __init__(self, name: str, startDate: str, endDate: str, interval: str = "1d"):
        """This Python function initializes an object with a name, start date, end date, and optional
        interval, downloads stock data using Yahoo Finance API, and extracts the closing prices.
        
        :param name: The name of the stock or financial
        instrument for which you want to download data
        :param StartDate: The start date for downloading stock data. 
        :param EndDate: The end date for the data you want to download.
        :param interval: The interval you want
        """
        self.name = name
        self.start_date = startDate 
        self.end_date = endDate

        # Avoiding chosing interval of 1h
        if interval == "1h":
            self.interval = "60m"
        else:
            self.interval = interval

        self.stock = yf.download(self.name, self.start_date, self.end_date, interval=self.interval)

    def get_data(self):
        """The method returns the closing stock prices from the `Stock` data.
        :return: The closing data
        """
        # 'Close', 'High', 'Low', 'Open', 'T', 'Volume'
        return self.stock['Open'], self.stock['High'], self.stock['Low'], self.stock['Close']

    # def meanReturns_and_std(self):
    #     """
    #     This function calculates the mean returns and standard deviation of returns for a given dataset.
    #     :return: The function `meanReturns_and_std` is returning the mean returns and the standard
    #     deviation of the returns for the given interval.
    #     """
    #     data = self.stock['Close']
    #     returns = data.pct_change()
    #     standard_deviation = returns.std()
    #     meanReturns = returns.mean()
    #     return meanReturns, standard_deviation
    