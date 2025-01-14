from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.layers.recurrent import LSTM
from tensorflow.python.keras.models import Sequential
from typing_extensions import override
from network.network import Model
from tensorflow.keras.optimizers import Adam # type: ignore



# TO DO:
### Research:
# ARIMA
# What data format for the model -> normalization?
# syntax (plotting & model)
### Code:
# begin Model class (template at network\network)
## attribute model = None
## method create_model...
## method train -> returns loss/error (check template)
## method fit
## method predict -> returns loss/error
## method plotting metrics -> RMSE


class LstmModel(Model):
    """
    The base class for all models.
    """
    #_model = None

    '''def create_ARIMA_model(self, p, d, q, series):
        self._model = ARIMA(series, p, d, q)'''

    @override
    def create_sequential_model(
            self,
            look_back: int,
            model_shape: int,
            # activations: list[str],
            input_shape: list[int],
            output_size: int
    ) -> None:
        """
        Creates the model architecture and assigns it to the model attribute.

        :param model_shape: Number of neurons in each layer.
        :type model_shape: list[float]
        :param activations: Activation function for each layer.
        :type activations: list[str]
        :param input_shape: Number of data points used for input.
        :type input_shape: int
        :param output_size: Number of neurons in the output layer.
        :type output_size: int
        """

        model = Sequential()

        # Defining LSTM model
        model.add(
            LSTM(
                model_shape, batch_input_shape=(
                    input_shape[0], input_shape[1], look_back)))
        # output layer
        model.add(Dense(output_size))

        self.model = model

    # def model(self) -> Sequential:
    #     return self._model

    @override
    def compileModel(
            self,
            learning_rate: float,
            lossFunc: str,
            metrics: list[str]
            ) -> None:
        """
        Compiles the model to prepare it for training.

        :param learning_rate: The step size used for training.
        :type learning_rate: float
        :param lossFunc: The function used to calculate the training loss.
        :type lossFunc: str
        :param metrics: A list of metrics to track during training.
        :type metrics: list[str]
        """
        self._model_validator()
        # self.model.compile(loss='mean_squared_error', optimizer='adam', metrics=metrics)
        self.model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss=lossFunc,
            metrics=metrics
            )

