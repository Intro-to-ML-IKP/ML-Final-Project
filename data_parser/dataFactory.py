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
        print(data)

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
        print(residuals)
        data, labels = self._data_processor.generate_labels(
            residuals,
            self._labels_per_set
            )
        return data, labels
    