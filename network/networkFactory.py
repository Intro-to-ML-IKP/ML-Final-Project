import numpy as np
import tensorflow as tf
from network.network import Model


class NetworkFactory:
    def __init__(
            self,
            model_shape : list[float],
            activations: list[str],
            input_shape: int,
            output_shape: int = 1
            ) -> None:
        """
        Instanitates a Network factory that is used to
        construct a Neural Network.

        :param model_shape: the shape iof the model
        :type model_shape: list[float]
        :param activations: the activation functions
        :type activations: list[str]
        :param input_shape: the input shape
        :type input_shape: int
        :param output_shape: the output shape,
        defaults to 1
        :type output_shape: int
        """
        self._model_shape = model_shape
        self._activations = activations
        self._input_shape = input_shape
        self._output_shape = output_shape
        self._model = Model()

    def train(
            self,
            training_data: np.ndarray,
            training_labels: np.ndarray,
            validation_data: np.ndarray,
            validation_labels: np.ndarray,
            learning_rate: float,
            lossFunc: str,
            metrics: list[str],
            epochs: int,
            batch_size: int
            ) -> None:
        # Create Sequential model
        self._model.create_sequential_model(self._model_shape, self._activations, self._input_shape, self._output_shape)

        # Compile the model
        self._model.compileModel(learning_rate, lossFunc, metrics)

        # Train the model
        self._model.trainModel(training_data, training_labels, validation_data, validation_labels, epochs, batch_size)

    def predict(self, data: tf.Tensor, number_of_predictions: int) -> list[float]:
        # Check if the input is a tensor
        if not isinstance(data, tf.Tensor):
            raise ValueError("To predict use data of type <class 'tf.Tensor'>! "
                            f"You are trying to use {type(data)}")

        sliding_data = data
        for current_prediction in range(number_of_predictions):
            # Slice the data
            sliced_data = sliding_data[0][current_prediction:]
            preprocessed_data = tf.reshape(sliced_data, (-1, 9))

            # Make the prediction
            prediction = self._model.predict(preprocessed_data)

            # Adjust the shape of the prediction
            prediction = tf.tile(prediction, [1, sliding_data.shape[1]])

            preprocessed_prediction = tf.reshape(prediction[0][0], (-1,1))

            # Add the prediction to the sliding data
            sliding_data = tf.concat([sliding_data, preprocessed_prediction], axis=1)

        # Separate the predictions from the input data and convert to list
        predictions = sliding_data[0][len(data[0]):].numpy().tolist()
        return predictions
