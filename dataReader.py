# Read stock data from API or folder into lists
# Apply preprocessing of data if applicable
# Apply train-test-val splits
from datetime import datetime, timedelta
from stock_getter import Stock
from sklearn.model_selection import train_test_split

class DataReader:
    def __init__(self, stockName):
        """Initializes dataReader class for data of the last n days
        
        Parameters:
        stockName   (string) - Code of the stock to look at
        
            
        Returns:
        None"""
        self.stockName = stockName
        self.interval = "1d"
        self.enddate = "2024-09-01"

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
            self.data = [(o, h, l, c) for o, h, l, c in zip(open, high, low, close)]
            
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

    def preprocess(self):
        """Preprocesses the data
        
        Returns:
        preprocessed data"""
        # Add preprocessing steps here
        return self.data
    
    
    def split_data(self, train_size=0.7, val_size=0.15):
        """Applies a train, test, validation split on the data and labels.
        
        Parameters:
        X           (list)  - The input data
        y           (list)  - The input labels
        train_size  (float) - Percentage of total data used for training
        val_size    (float) - Percentage of total data used for validation
        
        Returns:
        training data + labels
        validation data + labels
        testing data + labels"""

        X = self.data 
        y = self.labels
        test_size = 1 - train_size - val_size
        # Step 1: Split the data into train+val and test sets
        X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=test_size)

        # Step 2: Split the train+val set into training and validation sets
        val_ratio = val_size / (train_size + val_size)  # Adjust val_size proportionally
        X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=val_ratio)

        self.training_data = X_train
        self.training_labels = y_train
        self.validation_data = X_val
        self.validation_labels = y_val
        self.testing_data = X_test
        self.testing_labels = y_test
        return X_train, X_val, X_test, y_train, y_val, y_test

    

        
