from datetime import datetime, timedelta
from data_parser.stockGetter import Stock

class DataReader:
    def __init__(self, stockName: str, enddate: str = "2024-09-01", interval: str = "1d"):
        """Initializes dataReader class for data of the last n days
        
        Parameters:
        stockName   (string) - Code of the stock to look at
        
            
        Returns:
        None"""
        self._validate_date(enddate)
        self.stockName = stockName
        self.interval = "1d"
        self.enddate = "2024-09-01"
        self.data = None

    def getData(self, nPoints=50, nSets=100):
        """Retrieves datasets of user-specified length based on interval, ensuring sufficient data points.
        
        Parameters:
        nPoints     (int)   - Number of datapoints to download. Default=50
        nSets       (int)   - Number of sets of data to download. Default=100

        Returns:
        Stock data in format [(open, high, low, close)]
        """

        required_data_points = nPoints * nSets
        approx_total_days = int(required_data_points * (7 / 5))  # Adjust for weekends and holidays
        
        end = datetime.strptime(self.enddate, "%Y-%m-%d")
        start = end - timedelta(days=approx_total_days)
        startdate = start.strftime("%Y-%m-%d")
        
        attempts = 0
        max_attempts = 5  # Limit to prevent infinite loops
        
        while attempts < max_attempts:
            # Retrieve data
            stock = Stock(self.stockName, startdate, self.enddate, self.interval)
            open, high, low, close = stock.get_data()
            
            # Combine into OHLC format
            self.data = [(op, hi, lo, cl) for op, hi, lo, cl in zip(open, high, low, close)]
            
            # Check if we have enough data
            if len(self.data) >= required_data_points:
                # Slice to the exact number of required points
                self.data = self.data[-required_data_points:]
                return self.data
            
            # Otherwise, increase the date range and retry
            attempts += 1
            approx_total_days = int(approx_total_days * 1.5)  # Increase the time window by 50%
            start = end - timedelta(days=approx_total_days)
            startdate = start.strftime("%Y-%m-%d")
            print(f"Retry {attempts}: Extending the start date to {startdate}...")

        raise ValueError(f"Unable to retrieve sufficient data after {max_attempts} attempts.")

    
    def getLabels(self, nPoints=50, labelSize=5):
        """Get the labels, thus next labelSize candlesticks, and split them from the datapoints
        
        Parameters:
        nPoints     (int) - Number of points per input+label
        labelSize   (int) - Number of points to split off
        
        Returns:
        Data without labels
        Labels"""
        if self.data is not None:
            allData = []
            allLabels = []
            for i in range(len(self.data)//nPoints):
                data = self.data[i*nPoints:(i+1)*nPoints-labelSize]
                label = self.data[(i+1)*nPoints-labelSize:(i+1)*nPoints]
                allData.append(data)
                allLabels.append(label)
            self.data = allData
            self.labels = allLabels
            return self.data, self.labels
        else:
            print(
                "There is no data to get the Labels of."
                f"Running the getData() method first with nPoints={nPoints} and nSets=100 ..."
                )
            self.getData(nPoints, 100)
            self.getLabels(nPoints, labelSize)

    def _validate_date(self, date):
        if not isinstance(date, str):
            raise TypeError(
                "You must provide type=`str` as date in the form:"
                f" yyyy-mm-dd. You provided: type=`{type(date)}`")
        try:
            datetime.strptime(date, '%Y-%m-%d')
        except TypeError("The date must be of the form `yyyy-mm-dd`!") as e:
            raise e
