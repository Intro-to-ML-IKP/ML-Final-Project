import numpy as np
from sklearn.metrics import mean_absolute_error

from forecast.forcastFactory import ForcastFactory
from data_parser.dataFactory import StockDataFactory
from data_parser.dataProcessor import DataProcessor
from forecast.ensembleModel import EnsembleModel

import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.metrics import mean_squared_error
from typing import Any
from copy import deepcopy

from network.networkFactory import NetworkFactory
from data_parser.dataFactory import StockDataFactory
from visualisation.visualize import PlotStocks, PlotForcastComparison


class ForcastFactoryEnsemble(ForcastFactory):
    def __init__(
            self,
            stock_name: str,
            residual_model: str, residual_model_folder: str, trend_model: str, trend_model_folder: str,
            datafactory_param_dict: dict
            ) -> None:
        """
        A way of initializing a ForcastFactory. There is
        a special class called ForecastFactoryInitializer that
        generates the model_param_dict and
        datafactory_param_dict for the instance of this object.

        :param stock_name: the name of the financial asset
        :type stock_name: str
        :param model_param_dict: the model dict as generated by
        ForecastFactoryInitializer
        :type model_param_dict: dict
        :param datafactory_param_dict: the datafactory dict as
        generated by ForecastFactoryInitializer
        :type datafactory_param_dict: dict
        """
        self._stock_name = stock_name
        
        # # Unpack model parameters
        # for key, value in model_param_dict.items():
        #     setattr(self, f"_{key}", value)

        # Unpack datafactory parameters
        for key, value in datafactory_param_dict.items():
            setattr(self, f"_{key}", value)    

        # # Define the model's activation functions
        # activations = ["relu" for _ in range(len(self._architecture))]    
        # activations.append("linear")

        # self._input_shape = self._points_per_set - self._labels_per_set
        # output_shape = self._labels_per_set

        # model_shape = self._architecture

        # # Initialize a model
        # self._model = NetworkFactory(
        #     model_shape = model_shape,
        #     activations = activations,
        #     input_shape = self._input_shape,
        #     output_shape = output_shape
        # )
        
        # Initialize Ensemble Model
        self._ensemble_model = EnsembleModel(
            residual_model, residual_model_folder, trend_model, trend_model_folder
        )

        # Initialize a StockDataFactory
        self._data_factory = StockDataFactory(
            self._stock_name,
            self._points_per_set,
            self._num_sets,
            self._labels_per_set,
            self._training_percentage,
            self._validation_percentage
            )
        
        # Values being infered
        self._end_date: str|None = None
        self._interval: str|None = None

        # Values that are being calculated
        self._raw_data: list[
            tuple[str, float, float, float, float]
            ]|None = None
        self._sma: list[float]|None = None
        self._residuals: list[float]|None = None

        # Predictions
        self._predicted_residuals: list[float]|None = None
        self._predicted_closing_prices: list[float]|None = None
        self._extrapolated_sma: list[float]|None = None

        # Observed data
        self._observed_raw_data: list[
            tuple[str, float, float, float, float]
            ]|None = None
        self._observed_sma: list[float]|None = None
        self._observed_residuals: list[float]|None = None
        self._observed_closing_prices: list[float]|None = None

    def predict(
            self,
            raw_data_amount: int = 50,
            sma_lookback_period: int = 3,
            regression_window: int|None = None,
            end_date: str = "2024-09-01",
            interval: str = "1d"
            ) -> None:
        """
        Predicts the closing prices and residuals.

        :param number_of_predictions: the number of predictions
        :type number_of_predictions: int
        :param raw_data_amount: the amount of raw data to be generated,
        used for the plotting faculties of this factory as well, defaults to 50
        :type raw_data_amount: int, optional
        :param sma_lookback_period: the lookback time used to calculate
        the simple moving average (this is a hyperparameter for the ML model),
        defaults to 3
        :type sma_lookback_period: int, optional
        :param regression_window: this is the window used by the
        linear regression extrapolation of the SMA. When left to None the
        window is set to be equal to the number_of_predictions, defaults to None
        :type regression_window: int | None, optional
        :param end_date: the end date of the raw data retrieval,
        defaults to "2024-09-01"
        :type end_date: str, optional
        :param interval: the scale of the candles, defaults to "1d"
        :type interval: str, optional
        """
        self._get_raw_data(raw_data_amount, end_date, interval)

        self._predict_residuals(sma_lookback_period)

        self._extrapolate_sma()

        self._predicted_closing_prices = self._calculate_closing_prices()

    ###############################################################
    ########### MOST LIKELY HAVE TO FIX THIS HERE #################
    ###############################################################
    def compare_predictions_with_observations(self) -> float:
        """
        Compare the model's predictions with the actual observed data.

        Note:
            Be cautious when performing this comparison. In some cases, 
            there may not be enough data to compare predictions with actual
            values. This situation can arise if the selected end date and
            prediction window result in a period where:
            
                (current date) < (end date + prediction window)

            Ensure there is sufficient observed data covering the
            entire prediction period before attempting the comparison.

        :return: the mean squared error of the observed closing prices
        and the predicted closing prices
        :rtype: float
        """
        self._validate_predictions(self._predicted_closing_prices)

        actual_closing_prices = self._calculate_actual_closing_prices()

        predicted_closing_prices = [float(x) for x in self._predicted_closing_prices]
        mae = mean_absolute_error(
            np.array(actual_closing_prices),
            np.array(predicted_closing_prices)
            )
        
        return mae
    
    def _calculate_actual_closing_prices(self) -> list[float]:
        """
        Calculates the closing prices from the extrapolated SMA
        and the predicted residuals.

        :return: a list of the predicted closing prices
        :rtype: list[float]
        """
        return [round(sum(x+y),2) for x, y in zip(self._actual_residuals,self._actual_sma)]

    def _predict_residuals(self, sma_lookback_period: int) -> None:
        """
        Used to predict the residuals. First it calculates the SMA
        of the raw data, get's their residuals, preprocess them,
        and does the prediction.

        :param sma_lookback_period: the lookback period used to
        compute the SMA
        :type sma_lookback_period: int
        """
        # Calculate the SMA
        self._sma_lookback_period = sma_lookback_period
        self._sma = self._data_factory.get_sma(
            self._raw_data,
            self._sma_lookback_period
            )
        
        # Calculate the residuals
        self._residuals = self._data_factory.get_residuals_data(
            self._raw_data,
            self._sma
            )

        test_data, test_labels = self._preprocess_residuals()

        self._predicted_residuals = self._ensemble_model.\
            predict_residuals(test_data)
        
        self._actual_residuals = test_labels

    def _extrapolate_sma(
            self
            ) -> None:
        """
        Extrapolates the SMA using the DataFactory.

        :param regression_window: the regression window if specified.
        Adviced to be left to None.
        :type regression_window: int | None
        """
        sma_data = self._data_factory.get_sma(self._raw_data, 3)

        processor = DataProcessor(sma_data, unpack=False)

        sets = processor.generate_sets(10)

        test_data, test_labels = processor.generate_labels(sets, 1)

        self._extrapolated_sma = self._ensemble_model.\
            predict_sma(test_data)#(--------args-------)
        
        self._actual_sma = test_labels
        
    def _preprocess_residuals(self) -> tf.Tensor:
        """
        This method ensures that there are enough datapoints to fit
        the input shape of the model. After that it reduces the number
        of residuals to fit the input shape. Utilises tensorflow's
        convert_to_tensor method to create a tensor in the right shape
        for the prediciction method of the Model.

        :raises ValueError: if the number of datapoints is smaller
        than the input shape
        :return: resiudals in the form of a tensor object ready to
        be used as an input of a NN Model.
        :rtype: tf.Tensor
        """
        processor = DataProcessor(self._residuals, unpack=False)

        sets = processor.generate_sets(10)

        test_data, test_labels = processor.generate_labels(sets, 1)
        
        return test_data, test_labels

    def plot_predictions(self) -> None:
        raise NotImplementedError("This is not implimented!")

    def plot_comparison(self) -> None:
        raise NotImplementedError("This is not implimented!")
    
    def _train_model(self) -> None:
        raise NotImplementedError("This is not implimented!")