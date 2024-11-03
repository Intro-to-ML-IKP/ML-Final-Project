from typing import Any
import numpy as np

from data_parser.dataReader import DataReader
from data_parser.dataProcessor import DataProcessor

class StockDataFactory:
    """
    In accordance with the design pattern "Factory Method" this
    class is used to generate stock data that is used for training
    a neural netowork.
    """
    def __init__(
            self,
            stock_name: str,
            points_per_set: int,
            num_of_sets: int,
            labels_per_set: int,
            testing_percentage: float,
            validation_percentage: float
            ) -> None:
        """
        A way of initialising a StockDataFactory.

        :param stock_name: the name of the stock
        :type stock_name: str
        :param points_per_set: the data points per set
        :type points_per_set: int
        :param num_of_sets: the number of sets
        :type num_of_sets: int
        :param labels_per_set: the labels per set
        :type labels_per_set: int
        :param testing_percentage: the testing percentage.
        Used to generate test data.
        :type testing_percentage: float
        :param validation_percentage: the validation percentage.
        Used to generate validation data.
        :type validation_percentage: float
        """
        self._stock_name = stock_name
        self._num_of_sets = num_of_sets
        self._points_per_set = points_per_set
        self._labels_per_set = labels_per_set
        self._testing_percentage = testing_percentage
        self._validation_percentage = validation_percentage

        self._data_reader: DataReader|None = None
        self._data_processor: DataProcessor|None = None
        
    def get_stock_data(
            self
            ) -> tuple[
                np.ndarray,
                np.ndarray,
                np.ndarray,
                np.ndarray,
                np.ndarray,
                np.ndarray
                ]:
        """
        This method is used to get the required data for the training
        of a Neural Network.

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
        # Generate the sets 
        sets = self._generate_sets()

        # Calculate the residuals
        residuals = self._calculate_residuals(sets)

        # Generate labels from the data
        data, labels = self._get_labeled_data(residuals)

        # Apply a train, test, validation split on the data
        (
            training_data,
            validation_data,
            testing_data,
            training_labels,
            validation_labels,
            testing_labels
            ) = self._data_processor.split_data(
                data,
                labels,
                self._testing_percentage,
                self._validation_percentage
                )
        
        return (
            training_data,
            validation_data,
            testing_data,
            training_labels,
            validation_labels,
            testing_labels
            )
    
    def get_raw_data(
            self,
            number_of_points: int
            ) -> list[tuple[str,float,float,float,float]]:
        """
        A way of getting a number of raw datapoints.

        :param number_of_points: the number of datapoints
        :type number_of_points: int
        :return: raw data from the DataReader
        :rtype: list[tuple[str,float,float,float,float]]
        """
        return DataReader(
            self._stock_name
            ).getData(number_of_points = number_of_points, number_of_sets = 1)

    def get_sma(
            self,
            data: list[tuple[str,float,float,float,float]],
            sma_lookback_period: int
            ) -> list[float]:
        """
        A way of getting the simple moving average of raw data.

        :param data: the data you want to get the simple moving
        average of.
        :type data: list[tuple[str,float,float,float,float]]
        :param sma_lookback_period: the lookback period for the
        calculation of the SME average. This is the number of datapoints
        used to calculate the SME. Example:
        if sma_lookback_period = 3:
            take: mean(last 3 points)
        :type sma_lookback_period: int
        :return: returns: SMA
        :rtype: list[float]
        """
        stock_data = DataProcessor(data).data
        return DataProcessor(None).\
            calculate_SMA(stock_data, length = sma_lookback_period)
    
    def get_extrapolated_sma(
            self,
            sma_values: list[float],
            number_of_predictions: int,
            regression_window: int|None = None
            ) -> list[float]:
        """
        Makes an extrapolation of the SMA using linear regression,
        for a specified period of time.

        :param sma_values: the SMA to be extraplated
        :type sma_values: list[float]
        :param number_of_predictions: the number of points that need
        extrapolation
        :type number_of_predictions: int
        :param regression_window: specifies the regression window for the
        linear regression, defaults to None
        :type regression_window: int | None, optional
        :return: a list of extapolated SMA
        :rtype: list[float]
        """
        start = regression_window
        if regression_window is None:
            if number_of_predictions > len(sma_values):
                start = 0
            else:
                start = number_of_predictions
        return DataProcessor(None).\
            extrapolate_the_SMA(
                SMA_values = sma_values,
                future_periods = number_of_predictions,
                start = -start)
    
    def get_residuals_data(
            self,
            raw_data: list[tuple[str, float, float, float, float]],
            sma: list[float]
            ) -> list[float]:
        """
        A way of getting the residuals of the SMA and the
        closing prices.

        :param raw_data: the raw data
        :type raw_data: _type_
        :param sma: the SMA of the raw data
        :type sma: _type_
        :return: the residuals
        :rtype: list[float]
        """
        stock_data = DataProcessor(raw_data).data
        return DataProcessor(None).\
            calculate_residuals(stock_data, sma)
    
    def _generate_sets(self) -> list[list[float]]:
        """
        This method is used to generate sets from stock data.

        :return: a list of sets of stock data
        :rtype: list[list[float]]
        """
        # Get data
        self._data_reader = DataReader(self._stock_name)
        stock_data = self._data_reader.getData(
            self._points_per_set+2,
            self._num_of_sets
            )
        
        # Generate sets
        self._data_processor = DataProcessor(stock_data)
        sets = self._data_processor.generate_sets(
            self._points_per_set+2)
        return sets
    
    def _calculate_residuals(
            self,
            sets: list[list[float]]
            ) -> list[list[float]]:
        """
        This method is used to calculate the residuals
        of the different sets.

        :param sets: a list of sets of stock data
        :type sets: list[list[float]]
        :return: a list of residuals for said sets
        :rtype: list[list[float]]
        """
        residuals = []
        for set_ in sets:
            simple_moving_average = self._data_processor.calculate_SMA(set_)
            residual = self._data_processor.calculate_residuals(
                set_,
                simple_moving_average
                )
            residuals.append(residual)
        return residuals
    
    def _get_labeled_data(
            self,
            residuals: Any
            ) -> tuple[list[list[float]], list[list[float]]]:
        """
        Labels the data to prepare it for train, test, validation split

        :param residuals: _description_
        :type residuals: Any
        :return: _description_
        :rtype: tuple[list[list[float]], list[list[float]]]
        """
        data, labels = self._data_processor.generate_labels(
            residuals,
            self._labels_per_set
            )
        return data, labels
    