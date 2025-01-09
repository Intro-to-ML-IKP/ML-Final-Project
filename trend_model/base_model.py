from abc import abstractmethod, ABC
import numpy as np
from keras.src.layers import BatchNormalization, LSTM
from keras.src.optimizers import Adam
from tensorflow.keras.models import Sequential
#from statsmodels.tsa.arima.model import ARIMA
from matplotlib import pyplot
from tensorflow.python.keras import Input
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.regularizers import l2
from typing_extensions import override

from network.network import Model


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


class TrendModel(Model):
    """
    The base class for all models.
    """
    _model = None

    '''def create_ARIMA_model(self, p, d, q, series):
        self._model = ARIMA(series, p, d, q)'''

    @override
    def create_sequential_model(
            self,
            lstm_neurons: int,
            model_shape: list[float],
            activations: list[str],
            input_shape: int,
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

        # Define the input layer
        #model.add(Input(shape=(input_shape,)))
        model.add(LSTM(lstm_neurons, batch_input_shape=input_shape, stateful=True))

        # Add all layers, including hidden layers and the output layer
        for number_of_neurons, activation in zip(model_shape, activations[:-1]):
            model.add(Dense(number_of_neurons, activation=activation, kernel_regularizer=l2(0.01)),
                      BatchNormalization())

        model.compile(loss='mean_squared_error', optimizer='adam')


        # Add the output layer
        model.add(Dense(output_size, activation=activations[-1], kernel_regularizer=l2(0.01)), BatchNormalization())

        self._model = model




