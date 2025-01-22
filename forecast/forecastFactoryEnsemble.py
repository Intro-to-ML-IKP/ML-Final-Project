from sklearn.metrics import mean_absolute_error
import tensorflow as tf

from forecast.forcastFactory import ForcastFactory
from data_parser.dataFactory import StockDataFactory
from data_parser.dataProcessor import DataProcessor
from forecast.ensembleModel import EnsembleModel


class ForcastFactoryEnsemble(ForcastFactory):
    """
    This class provides a way to test the forcasting abilities of an
    Ensemble model.
    """
    def __init__(
            self,
            stock_name: str,
            residual_model: str,
            residual_model_folder: str,
            trend_model: str,
            trend_model_folder: str,
            datafactory_param_dict: dict
            ) -> None:
        """
        A way of instantiating ForcastFactoryEnsemble.

        :param stock_name: the stock code.
        :type stock_name: str
        :param residual_model: the residual model's name
        :type residual_model: str
        :param residual_model_folder: the residual model's folder
        :type residual_model_folder: str
        :param trend_model: the trend model's name
        :type trend_model: str
        :param trend_model_folder: the trend model's folder
        :type trend_model_folder: str
        :param datafactory_param_dict: the parameters of the DataFactory.
        :type datafactory_param_dict: dict
        """
        self._stock_name = stock_name

        # Unpack datafactory parameters
        for key, value in datafactory_param_dict.items():
            setattr(self, f"_{key}", value)    
        
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

    def predict(
            self,
            raw_data_amount: int = 50,
            sma_lookback_period: int = 3,
            end_date: str = "2024-09-01",
            interval: str = "1d"
            ) -> None:
        """
        Predicts the closing prices and residuals.

        :param raw_data_amount: the amount of raw data to be generated,
        used for the plotting faculties of this factory as well, defaults to 50
        :type raw_data_amount: int, optional
        :param sma_lookback_period: the lookback time used to calculate
        the simple moving average (this is a hyperparameter for the ML model),
        defaults to 3
        :type sma_lookback_period: int, optional
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

    def compare_predictions_with_observations(self) -> float:
        """
        Compare the model's predictions with the actual observed data.

        :return: the mean absolute error of the observed closing prices
        and the predicted closing prices
        :rtype: float
        """
        self._validate_predictions(self._predicted_closing_prices)

        actual_closing_prices = self._calculate_actual_closing_prices()

        predicted_closing_prices = self._predicted_closing_prices
        mae = mean_absolute_error(
            actual_closing_prices,
            predicted_closing_prices
            )
        
        return mae
    
    def _calculate_actual_closing_prices(self) -> list[float]:
        """
        Calculates the actual closing prices.

        :return: a list of actual closing prices.
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

        # Makes labels
        test_data, test_labels = self._preprocess_residuals()

        # Predict the residuals
        self._predicted_residuals = self._ensemble_model.\
            predict_residuals(test_data)
        
        # Set the actual residuals to the test labels
        self._actual_residuals = test_labels

    def _extrapolate_sma(
            self
            ) -> None:
        """
        Extrapolates the SMA using the trend model.
        """
        # Gets the SMA data
        sma_data = self._data_factory.get_sma(self._raw_data, 3)

        # Generates sets
        processor = DataProcessor(sma_data, unpack=False)
        sets = processor.generate_sets(10)

        # Creates labels
        test_data, test_labels = processor.generate_labels(sets, 1)

        # Predict the SMA
        self._extrapolated_sma = self._ensemble_model.\
            predict_sma(test_data)#(--------args-------)
        
        # Set the actual SMA to the test labels
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
        # Generates sets
        processor = DataProcessor(self._residuals, unpack=False)
        sets = processor.generate_sets(10)

        # Generates labels
        test_data, test_labels = processor.generate_labels(sets, 1)
        
        return test_data, test_labels

    def plot_predictions(self) -> None:
        """
        Not needed, hence overwritten
        """
        raise NotImplementedError("This is not implimented!")

    def plot_comparison(self) -> None:
        """
        Not needed, hence overwritten
        """
        raise NotImplementedError("This is not implimented!")
    
    def _train_model(self) -> None:
        """
        Not needed, hence overwritten
        """
        raise NotImplementedError("This is not implimented!")
    