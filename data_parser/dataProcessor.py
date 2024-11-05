from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from copy import deepcopy
import numpy as np
import pandas as pd
import pandas_ta as ta
import math


class DataProcessor:
    """
    Serves as a way to process stock data from Yahoo's API.
    """
    def __init__(
            self,
            data: list[tuple[str,float,float,float,float]]|None,
            ) -> None:
        """
        A way of instantating a proccessor object for stock data.

        :param data: the data as a list of tuples, where the first
        element is the date, and the rest are:
        open, high, low, and close prices.
        :type data: list[tuple[str,float,float,float,float]]
        """
        self._dates = None
        self._data: list[tuple[float, float, float, float]] = None
        if data is not None:
            self._unpack_data(data)
    
    @property
    def data(self) -> tuple[float,float,float,float]:
        """
        Retunrs a deepcopy of the stock data in OHLC tuple.
        """
        return deepcopy(self._data)

    def calculate_SMA(
            self,
            stock_data: list[tuple[float,float,float,float]],
            length: int = 3
            ) -> list[float]:
        """
        The function `_calculate_SMA` calculates the
        Simple Moving Average for a given dataset over a specified
        length of time.
        
        :param length: the length of the period to consider
        when calculating the Simple Moving Average (SMA).
        :type length: int (optional)
        :return sets_SMA: lists of floats representing
        the SMA
        :sets_SMA type: list[Float]
        """
        # Unzipping the close
        _, _, _, close = zip(*stock_data)

        # Creating a dataFrame (required for the pandas_ta module)
        close_pd = pd.DataFrame({"close": []})
        close_pd["close"] = close

        # Calculating SMA
        SMA = ta.sma(close_pd["close"], length=length)

        # Converting SMA to list and rounding it,
        # also removing the NAN value
        SMA_list = SMA.tolist()
        SMA_list = [round(x, 2) for x in SMA_list if not math.isnan(x)]

        return SMA_list
        
    def calculate_residuals(
            self,
            stock_data: list[tuple[float,float,float,float]],
            sma: list[float]
            ) -> list[float]:
        """
        Calculates the residuals by substracting the closing prices
        from a Simple Moving Average (SMA).

        :param sma: Simple Moving Average on the data.
        :sma type: list[float]
        :return residuals: the difference between SMA and the closing
        prices.
        :residuals type: list[float]
        """
        _, _, _, closing_prices = zip(*stock_data)

        nr_of_residuals = len(sma)
        closing_prices = closing_prices[-nr_of_residuals:]

        residuals = [round(a - b, 2) for a, b in zip(sma, closing_prices)]

        return residuals

    
    def extrapolate_the_SMA(
            self,
            SMA_values: list[float],
            future_periods: int,
            start: int = 0
            ) -> list[float]:
        """
        This method extrapolates a Simple Moving Average (SMA) using
        a linear regression model.

        :param SMA_values: a list of SMA values
        :SMA_values type: list[float]
        :param future_periods: how many days you wish to extrapolate.
        :future_periods type: int
        :param start: the index of the list you wish to make the
        extrapolation onwards, value of 0 is set to default witch means
        the whole list is going to be used for fitting the model.
        :start type: int
        
        :return extrapolated_SMA: the extrapolated values
        given by the LR model with a length of future_periods.
        :extrapolated_SMA type: list[Float]
        """
        # Prepare the data for linear regression
        y_coord = SMA_values[start:]  # Use the slice starting from 'start' to the end
        x_coord = np.arange(0, len(y_coord))

        # Reshape x_coord to be a 2D array
        x_coord = x_coord.reshape((-1, 1))

        # Initialise a linear regression model
        model = LinearRegression()
        model.fit(x_coord, y_coord)

        # Define the x-coordinates for future values we want to predict
        x_future = np.arange(len(SMA_values), len(SMA_values) + future_periods)
        x_future = x_future.reshape((-1, 1))

        # Do the extrapolation
        extrapolated_SMA = model.predict(x_future).round(2)

        # Convert to list
        extrapolated_SMA = extrapolated_SMA.tolist()

        # Align with the data
        align_value = y_coord[-1]

        aligned_extrapolation = self._align_extrapolation(extrapolated_SMA, align_value)

        return aligned_extrapolation
    
    def split_data(
            self,
            input_data: list[float],
            input_labels: list[float],
            train_size: float = 0.7,
            val_size: float = 0.15
            ) -> tuple[
                np.ndarray,
                np.ndarray,
                np.ndarray,
                np.ndarray,
                np.ndarray,
                np.ndarray
                ]:
        """
        Applies a train, test, and validation split on the data and labels.

        :param input_data: The input data.
        :type input_data: list
        :param input_labels: The input labels.
        :type input_labels: list
        :param train_size: Percentage of total data used for training.
        :type train_size: float
        :param val_size: Percentage of total data used for validation.
        :type val_size: float

        :return: Tuple containing:
        training data,
        validation data,
        test data,
        training labels,
        validation labels,
        test labels
        In this specific order.
        :rtype: tuple[
                np.ndarray,
                np.ndarray,
                np.ndarray,
                np.ndarray,
                np.ndarray,
                np.ndarray
                ]
        """
        test_size = 1 - train_size - val_size

        # Step 1: Split the data into train+val and test sets
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            input_data,
            input_labels,
            test_size = test_size,
            shuffle=False
            )

        # Step 2: Split the train+val set into training and validation sets
        val_ratio = val_size / (train_size + val_size) 
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val,
            y_train_val,
            test_size = val_ratio,
            shuffle=False
            )

        return (
            np.array(X_train),
            np.array(X_val),
            np.array(X_test),
            np.array(y_train),
            np.array(y_val),
            np.array(y_test)
            )
    
    def generate_labels(
            self,
            residuals_data: list[list[float]],
            label_size: int = 5
            ) -> tuple[list[list[float]], list[list[float]]]:
        """
        Generates labels for a given data based on
        label size.

        :param residuals_data: the data to generate labels on
        :type residuals_data: list[list[float]]
        :param label_size: the number of labels (size), defaults to 5
        :type label_size: int, optional
        :return: tuple of data and lebels to be used for test, train,
        validation split.
        :rtype: tuple[list[list[float]], list[list[float]]]
        """
        allData = []
        allLabels = []
        for set in residuals_data:
            allData.append(set[:-label_size])
            allLabels.append(set[-label_size:])
        return allData, allLabels
    
    def generate_sets(
            self,
            pointsPerSet: int
            ) -> list[list[float]]:
        """
        Generates sets from the Stock data to be used in training.
        Usually this is used to compute SME and get the residuals
        in order to train a FFNN.

        :param pointsPerSet: the points per data set
        :pointsPerSet type: int
        """
        allData = []
        for i in range(len(self._data)//pointsPerSet):
            data = self._data[i*pointsPerSet:(i+1)*pointsPerSet]
            allData.append(data)
        return allData

    def _unpack_data(
            self,
            data: list[tuple[str,float,float,float,float]]
            ) -> None:
        """
        Unpacks the data and separates Date from the Stock Data.
        Used in the instantiation of the Class

        :param data: stock data containing (date, open, high, low, close) data.
        :type data: list[tuple[str,float,float,float,float]]
        """
        dates, open_, high, low, close = zip(*data)
        self._dates = dates
        data = [(op, hi, lo, cl) for op, hi, lo, cl in zip(open_, high, low, close)]
        rounded_data = self._round_data(data)
        self._data = rounded_data
    
    def _align_extrapolation(
            self,
            extrapolation: list[float],
            align_value: float
            ) -> list[float]:
        """
        Aligns the extrapolation of SMA with the last closing price.

        :param extrapolation: the extrapolation
        :extrapolation type: list[float]
        :param align_value: the last value of the SMA
        :align_value type: float
        :return aligned_extrapolation: the aligned extrapolation
        :aligned_extrapolation type: list[float]
        """
        # Get the first extrapolated value
        first_extrapolation_val = extrapolation[0]

        # Compute the difference between the alignment value and the extrap.
        delta = align_value - first_extrapolation_val

        # Align the list
        aligned_extrapolation = [x + delta for x in extrapolation]

        return aligned_extrapolation
    
    def _round_data(
            self,
            data: list[tuple[float, float, float, float]]
            ) -> list[tuple[float, float, float, float]]:
        """
        Rounds a Stock Data to two decimals.

        :param data: the data as given by the getData method
        from the dataReader class.
        :data type: list[tuple[float, float, float, float]
        :return: the rounded data
        :return type: list[tuple[float, float, float, float]
        """
        rounded_data = []
        for tup in data:
            # Round each value in the tuple
            rounded_tup = tuple(round(value, 2) for value in tup)
            rounded_data.append(rounded_tup)
        return rounded_data
    