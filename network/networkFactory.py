import numpy as np
import tensorflow as tf
from network.network import Model


class NetworkFactory:
    """
    Serves as an way of creating a Neural Network model, that can be trained
    to predict.
    """
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
        """
        Trains the model using specified training and validation data, with the 
        given configurations for learning rate, loss function, metrics, epochs, 
        and batch size.

        :param training_data: Input data for training.
        :type training_data: np.ndarray
        :param training_labels: Target labels for training data.
        :type training_labels: np.ndarray
        :param validation_data: Data used to validate model during training.
        :type validation_data: np.ndarray
        :param validation_labels: Labels for validation data.
        :type validation_labels: np.ndarray
        :param learning_rate: Learning rate for the optimizer.
        :type learning_rate: float
        :param lossFunc: Loss function to be minimized.
        :type lossFunc: str
        :param metrics: List of metrics to evaluate during training.
        :type metrics: list[str]
        :param epochs: Number of training epochs.
        :type epochs: int
        :param batch_size: Batch size for training.
        :type batch_size: int

        :return: None
        """
        # Create Sequential model
        self._model.create_sequential_model(self._model_shape, self._activations, self._input_shape, self._output_shape)

        # Compile the model
        self._model.compileModel(learning_rate, lossFunc, metrics)

        # Train the model
        self._model.trainModel(training_data, training_labels, validation_data, validation_labels, epochs, batch_size)

    def predict(self, data: tf.Tensor, number_of_predictions: int) -> list[float]:
        """
        Generates a specified number of predictions based on input data using a 
        sliding window approach. Appends each new prediction to the input data 
        to iteratively make future predictions.

        :param data: Input data in the form of a Tensor.
        :type data: tf.Tensor
        :param number_of_predictions: Number of future predictions to generate.
        :type number_of_predictions: int

        :raises ValueError: If `data` is not of type `tf.Tensor`.

        :return: A list containing the generated predictions.
        :rtype: list[float]
        """
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
