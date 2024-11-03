import numpy as np
import tensorflow as tf

from network.networkFactory import NetworkFactory
from data_parser.dataFactory import StockDataFactory
from visualisation.visualize import PlotStocks


class ForcastFactory:
    def __init__(self, stock_name: str, model_param_dict: dict, datafactory_param_dict: dict):
        self._stock_name = stock_name
        
        # Unpack relevant parameters for the model
        self._architecture = model_param_dict["architecture"]
        self._learning_rate = model_param_dict["learning_rate"]
        self._loss_function = model_param_dict["loss_function"]
        self._metrics = model_param_dict["metrics"]
        self._epochs = model_param_dict["epochs"]
        self._batch_size = model_param_dict["batch_size"]

        # Unpacking the parameters for the datafacory
        self._points_per_set = datafactory_param_dict["points_per_set"] # StockDataFactory
        self._num_sets = datafactory_param_dict["num_sets"] # StockDataFactory
        self._labels_per_set = datafactory_param_dict["labels_per_set"] # StockDataFactory
        self._testing_percentage = datafactory_param_dict["testing_percentage"] # StockDataFactory
        self._validation_percentage = datafactory_param_dict["validation_percentage"] # StockDataFactory

        # Define the model's parameters
        activations = ["relu" for _ in range(len(self._architecture))]    
        activations.append("linear")

        self._input_shape = self._points_per_set - self._labels_per_set
        output_shape = self._labels_per_set

        model_shape = self._architecture

        # Initialize a model
        self._model = NetworkFactory(
            model_shape = model_shape,
            activations = activations,
            input_shape = self._input_shape,
            output_shape = output_shape
        )

        # Initialize a StockDataFactory
        self._data_factory = StockDataFactory(
            self._stock_name,
            self._points_per_set,
            self._num_sets,
            self._labels_per_set,
            self._testing_percentage,
            self._validation_percentage
            )
        
        # Values that are being calculated
        self._raw_data: list[tuple[str, float, float, float, float]]|None = None
        self._sma: list[float]|None = None
        self._predicted_residuals: list[float]|None = None
        self._residuals: list[float]|None = None
        self._extrapolated_sma: list[float]|None = None
        self._closing_prices: list[float]|None = None
        self._predicted_residuals: list[float]|None = None

    def predict(self, number_of_predictions: int, raw_data_amount: int = 50, sma_lookback_period: int = 3, regression_window: int|None = None):
        self._train_model()

        self._get_raw_data(raw_data_amount)

        self._predicted_residuals = self._predict_residuals(number_of_predictions, sma_lookback_period)

        self._extrapolate_sma(number_of_predictions, regression_window)

        self._closing_prices = self._calculate_closing_prices()

        return self._closing_prices, self._raw_data, self._predicted_residuals, self._residuals
    
    def plot_predictions(self):
        if self._closing_prices is None:
            raise ValueError(
                "You need to make predictions first!"
                "Please first use the `predict` method.")
        
        # Initialize the plotting class
        plotter = PlotStocks(
            self._raw_data,
            self._sma,
            self._extrapolated_sma,
            self._residuals,
            self._closing_prices,
            self._predicted_residuals
            )
        
        plotter.masterPlot(simpleMovingAverage = True, predictedClosingPrices = True, predictedResiduals = True)
        

    def _train_model(self):
        # Get training data 
        (
            training_data,
            validation_data,
            _,
            training_labels,
            validation_labels,
            _
            ) = self._data_factory.get_stock_data()
        
        # Train the model
        self._model.train(
            training_data,
            training_labels,
            validation_data,
            validation_labels,
            self._learning_rate,
            self._loss_function,
            self._metrics,
            self._epochs,
            self._batch_size
            )
        
    def _get_raw_data(self, raw_data_amount):
        self._raw_data = self._data_factory.get_raw_data(raw_data_amount)
        
    def _predict_residuals(self, number_of_predictions: int, sma_lookback_period: int):
        self._sma = self._data_factory.get_sma(self._raw_data, sma_lookback_period = sma_lookback_period)
        self._residuals = self._data_factory.get_residuals_data(self._raw_data, self._sma)

        preprocessed_residuals = self._preprocess_residuals()

        self._predicted_residuals = self._model.predict(preprocessed_residuals, number_of_predictions)

        return self._predicted_residuals
    
    def _preprocess_residuals(self) -> tf.Tensor:
        if len(self._residuals) < self._input_shape:
            raise ValueError(
                f"The number of residuals {len(self._residuals)} is smaller\n"
                f"than the input shape {self._input_shape}.\n"
                "You can not perform a prediction!\n"
                "You can fix that by increasing 'raw_data_amount' in the 'predict' method")
        
        reduced_residuals = self._residuals[-self._input_shape:]

        # Convert to tensors in the correct shape
        residuals_tensor = tf.convert_to_tensor(reduced_residuals)
        residuals_tensor = tf.reshape(residuals_tensor, (-1, self._input_shape))
        return residuals_tensor

    def _extrapolate_sma(self, number_of_predictions: int, regression_window: int|None):
        self._extrapolated_sma = self._data_factory.get_extrapolated_sma(self._sma, number_of_predictions, regression_window)
    
    def _calculate_closing_prices(self) -> list[float]:
        return [sum(x) for x in zip(self._extrapolated_sma, self._predicted_residuals)]
