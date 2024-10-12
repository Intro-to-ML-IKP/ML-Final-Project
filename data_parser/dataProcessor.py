from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd
import pandas_ta as ta
import math


class DataProcessor:
    def __init__(self, data: list[tuple[float,float,float,float]], labels):
        self._data = data
        self._labels = labels
        self._data_sets = self._calculate_data_sets()

    @property
    def closing_prices(self):
        """
        Gets the closing prices.
        """
        return self._data_sets[0]["Close"]

    def _calculate_data_sets(self) -> list[pd.DataFrame]:
        """
        Unzips the data and puts it in a panda data frame. It also rounds
        the values for future computation.

        :return: a list of panda dataFrame with the data
        :return type: list[pd.dataFrame]
        """
        # Round the data to 2 decimal places (helps with computational speed)
        data = self._round_data(self._data)

        # Get the HLOC form of the data
        high, low, open_price, close_price = list(zip(*data))

        # Create a data frameworks
        data_frame = {
            "Open": open_price,
            "High": high,
            "Low": low,
            "Close": close_price
        }
        df = pd.DataFrame(data_frame)

        return [df]


    def calculate_SMA(
            self,
            length: int = 3
            ) -> list[list[float]]:
        """
        The function `_calculate_SMA` calculates the Simple Moving Average for a given dataset over a
        specified length of time.
        
        :param length: the length of the period to consider when calculating the
        Simple Moving Average (SMA).
        :type length: int (optional)
        :return sets_SMA: list of lists of floats representing the SMA for each set in the sets of data provided
        in the instantiation of this class
        :sets_SMA type: list[Float]
        """
        # Initialise a list that would contain the data frameworks
        sets_SMA = []

        try:
            # Loop throught all the data sets
            for data_set in self._data_sets:
                # Calculate the SMA
                SMA = ta.sma(data_set['Close'], length=length)

                SMA_list = SMA.tolist()

                SMA_list = [x for x in SMA_list if not math.isnan(x)]
                # Append the SMA to the list of SMA
                sets_SMA.append(SMA_list)

            return sets_SMA
        except TypeError(
            "There weren't any data sets found!\n"
            "Value for self._data_sets not itterable!\n"
            f"The value is of type: {type(self._data_sets)}"
            "Expected type: list[pd.dataFrame]"
            ) as e:
            raise e
        
    def calculate_residuals(self, sma: list[float]) -> list[float]:
        """
        Calculates the residuals by substracting the closing prices
        from a Simple Moving Average (SMA).

        :param sma: Simple Moving Average on the data.
        :sma type: list[float]
        :return residuals: the difference between SMA and the closing
        prices.
        :residuals type: list[float]
        """
        closing_prices = self.closing_prices

        nr_of_residuals = len(sma)
        closing_prices = closing_prices[-nr_of_residuals:]

        residuals = [a - b for a, b in zip(sma, closing_prices)]

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
        :param start: the index of the list you wish to make the extrapolation
        onwards, value of 0 is set to default witch means the whole list is going
        to be used for fitting the model.
        :start type: int
        
        :return extrapolated_SMA: the extrapolated values given by the LR model
        with a length of future_periods.
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
    
    def _align_extrapolation(self, extrapolation: list[float], align_value: float) -> list[float]:
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
            self, data: list[tuple[float, float, float, float]]
        ) -> list[tuple[float, float, float, float]]:
        """
        Rounds a Stock Data to two decimals.

        :param data: the data as given by the getData method from the dataReader class.
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

    
    def split_data(self, X, y, train_size=0.7, val_size=0.15):
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
        return np.array(X_train), np.array(X_val), np.array(X_test), np.array(y_train), np.array(y_val), np.array(y_test)
