from datetime import datetime, timedelta
from data_parser.stockGetter import Stock
from pandas.core.series import Series

class DataReader:
    """
    Serves as a way to read the stock data from Yahoo's API.
    """
    def __init__(
            self,
            stock_name: str,
            end_date: str = "2024-09-01",
            interval: str = "1d"):
        """
        A way of instantiating a DataReader

        :param stock_name: the name of the stock
        :type stock_name: str
        :param end_date: the end date we want our data to be up to,
        defaults to "2024-09-01"
        :type end_date: str, optional
        :param interval: the interval we wish to collect data for,
        defaults to "1d"
        :type interval: str, optional
        """
        self._validate_date(end_date)
        self.stock_name = stock_name
        self.interval = interval
        self.end_date = end_date
        self.data: list[float]|None = None
        self.labels: list[float]|None = None

    def getData(
            self,
            number_of_points: int = 50,
            number_of_sets: int = 100
            ) -> tuple[datetime, Series, Series, Series, Series]:
        """
        Retrieves datasets of user-specified length based on interval, 
        ensuring sufficient data points.

        :param number_of_points: Number of data points to download. 
        Default is 50.
        :type number_of_points: int
        :param number_of_sets: Number of sets of data to download. 
        Default is 100.
        :type number_of_sets: int

        :return: Stock data in the format [(datetime, open, high, low, close)].
        :rtype: list[tuple[float, float, float, float]]
        """
        required_data_points = number_of_points * number_of_sets
        
        # Adjust for weekends and holidays
        approx_total_days = int(required_data_points * (7 / 5))
        
        end = datetime.strptime(self.end_date, "%Y-%m-%d")
        start = end - timedelta(days=approx_total_days)
        startdate = start.strftime("%Y-%m-%d")
        
        attempts = 0
        max_attempts = 10  # Limit to prevent infinite loops
        
        while attempts < max_attempts:
            # Retrieve data
            dates, open_, high, low, close = self._retrieve_data(startdate)
            
            # Combine into DOHLC format
            # (dates, open, high, low, close)
            self.data = [
                (dat, op, hi, lo, cl)
                for dat, op, hi, lo, cl in zip(dates, open_, high, low, close)
                ]
            
            # Check if we have enough data          
            if self._validate_data_sufficiency(required_data_points):
                return self.data
            
            # Otherwise, increase the date range and retry
            attempts += 1

            # Increase the time window by 50%
            approx_total_days = int(approx_total_days * 1.5)
            start = end - timedelta(days=approx_total_days)
            startdate = start.strftime("%Y-%m-%d")
            print(
                f"Retry {attempts}:"
                f"Extending the start date to {startdate}..."
                )

        raise ValueError(
            "Unable to retrieve sufficient data after "
            f"{max_attempts} attempts."
            )

    def _retrieve_data(
            self,
            start_date: datetime
            ) -> tuple[
                datetime,
                Series,
                Series,
                Series,
                Series
                ]:
        """
        Retrieves the date based on a starting date and ending date.

        :param start_date: the starting date
        :type start_date: datetime
        :return: a tuple in the following format:
        Date,
        Open,
        High,
        Low,
        Close.
        :rtype: tuple[
                datetime,
                Series,
                Series,
                Series,
                Series
                ]
        """
        stock = Stock(
            self.stock_name,
            start_date,
            self.end_date,
            self.interval
            )
        return stock.get_data()
    
    def _validate_data_sufficiency(
            self,
            required_data_points: int
            ) -> bool:
        """
        Validates if the data collection was sufficient. If it is
        returns True, otherwise False.

        :param required_data_points: the minimum required data points
        for the condition to be true.
        :type required_data_points: int
        :return: True if sufficient False otherwise
        :rtype: bool
        """
        if len(self.data) >= required_data_points:
            # Slice to the exact number of required points
            self.data = self.data[-required_data_points:]
            return True
    
    def getLabels(
            self,
            input_data: Series,
            number_of_points: int = 50,
            label_size: int = 5
            ) -> tuple[list[float], list[float]]:
        """
        Get the labels, thus next label_size candlesticks,
        and split them from the datapoints.

        :param input_data: the input data
        :type input_data: Series
        :param number_of_points: number of points, defaults to 50
        :type number_of_points: int, optional
        :param label_size: number of labels, defaults to 5
        :type label_size: int, optional
        :return: a tuple of the data and the labels
        :rtype: tuple[list[float], list[float]]
        """
        if input_data is not None:
            all_data = []
            all_labels = []
            for i in range(len(input_data)//number_of_points):
                # Parse input feature
                data = input_data[
                    i*number_of_points:(i+1)*number_of_points-label_size
                    ]
                
                # Parse target feature
                label = input_data[
                    (i+1)*number_of_points-label_size:(i+1)*number_of_points
                    ]
                
                # Append to respective lists
                all_data.append(data)
                all_labels.append(label)

            # Define the attributes
            self.data = all_data
            self.labels = all_labels
            return self.data, self.labels
        else:
            print(
                "There is no data to get the Labels of."
                "Running the getData() method first with"
                f"number_of_points={number_of_points}"
                "and number_of_sets=100 ..."
                )
            self.getData(number_of_points, 100)
            self.getLabels(number_of_points, label_size)

    def _validate_date(self, date: str) -> None:
        """
        Validates the date.

        :param date: the date
        :type date: str
        :raises TypeError: wrong date format
        """
        if not isinstance(date, str):
            raise TypeError(
                "You must provide type=`str` as date in the form:"
                f" yyyy-mm-dd. You provided: type=`{type(date)}`")
        try:
            datetime.strptime(date, '%Y-%m-%d')
        except TypeError("The date must be of the form `yyyy-mm-dd`!") as e:
            raise e
        