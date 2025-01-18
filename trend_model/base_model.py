import numpy as np
from keras.src.layers import ReLU
from tensorflow.keras.layers import Dense  # type: ignore
from tensorflow.keras.layers import LSTM  # type: ignore
from tensorflow.keras.models import Sequential  # type: ignore
from network.network import Model
from tensorflow.keras.optimizers import Adam  # type: ignore

from trend_model.model_factory import DataNormalizer


class LstmModel(Model):
    """
    The base class for LSTM models.
    """

    def __init__(self):
        super().__init__()
        self.scaler = DataNormalizer()


    def create_sequential_model(
            self,
            architecture: list[int],
            activations: list[str],
            input_shape: int,
            output_size: int
    ) -> None:
        """
        Creates the model architecture and assigns it to the model attribute.

        :param architecture: Holds parameters look_back (0) - the number of data points the LSTM layer uses
        for predictions and model_shape (1) - the number of neurons in the LSTM layer.
        :type architecture: list[int]
        :param activations: Activation function for each layer.
        :type activations: list[str]
        :param input_shape: Number of data points used for input.
        :type input_shape: int
        :param output_size: Number of neurons in the output layer.
        :type output_size: int
        """
        look_back = int(architecture[0])
        model_shape = int(architecture[1])

        model = Sequential([
            LSTM(units=model_shape, input_shape=(input_shape, look_back)),
            ReLU(),
            Dense(units=output_size)
        ])

        self.model = model

    def trainModel(
            self,
            training_data: np.ndarray,
            training_labels: np.ndarray,
            validation_data: np.ndarray,
            validation_labels: np.ndarray,
            epochs: int,
            batch_size: int
    ) -> None:
        """
        Trains the model using specified training and validation data.

        :param training_data: Data for training, matching the input shape of
        the model.
        :type training_data: np.ndarray
        :param training_labels: Labels for training data, matching the output
        shape of the model.
        :type training_labels: np.ndarray
        :param validation_data: Data for validation during training to prevent
        overfitting.
        :type validation_data: np.ndarray
        :param validation_labels: Labels for validation data.
        :type validation_labels: np.ndarray
        :param epochs: Number of training iterations.
        :type epochs: int
        :param batch_size: Data points per batch, the number processed before
        updating weights.
        :type batch_size: int
        """

        x_data = [training_data, validation_data]
        y_data = [training_labels, validation_labels]

        x_data = [self.scaler.scale_data(dt) for dt in x_data]
        x_data = [self.scaler.reshape_input(dt) for dt in x_data]

        y_data = [self.scaler.scale_data(dt) for dt in y_data]

        return super().trainModel(x_data[0], y_data[0], x_data[1], y_data[1], epochs, batch_size)

    def predict(
            self,
            data: np.ndarray
    ) -> np.ndarray:
        """
        Makes a prediction on the specified data using the trained model

        :param data: the data you want to predict
        :type data: np.ndarray
        :return: the predictions
        :type return: np.ndarray
        """
        x_test = data
        x_test = self.scaler.scale_data(x_test)
        x_test = self.scaler.reshape_input(x_test)

        predictions = super().predict(x_test)

        padded_predictions = np.zeros((predictions.shape[0], 9))    # dummy values to avoid input-output shape mismatch
        padded_predictions[:, 0] = predictions[:, 0]

        y_predictions = self.scaler.inverse_scaled_data(padded_predictions)
        return y_predictions[:, 0].reshape(-1, 1)
