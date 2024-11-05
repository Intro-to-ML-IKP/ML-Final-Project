import numpy as np
from typing import Any
from tensorflow import keras
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense,Input # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore
from tensorflow.keras.callbacks import EarlyStopping # type: ignore
from sklearn.metrics import mean_absolute_error

class Model:
    """
    Used as an interface between keras' sequantial model
    (that is used to create an MLP).
    """
    def __init__(self):
        """
        Instantiates a model.
        """
        self.model = None

    def create_sequential_model(
            self,
            model_shape : list[float],
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
        model.add(Input(shape=(input_shape,)))

        # Add all layers, including hidden layers and the output layer
        for number_of_neurons, activation in zip(model_shape, activations[:-1]):
            model.add(Dense(number_of_neurons, activation=activation))

        # Add the output layer
        model.add(Dense(output_size, activation=activations[-1]))

        self.model = model

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
        self.model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss=lossFunc,
            metrics=metrics
            )

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

        self._model_validator()

        # Stops training when validation performance stops improving
        early_stopping = EarlyStopping(monitor='val_loss', patience=4)
        
        self.model.fit(
            training_data,
            training_labels,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(validation_data, validation_labels), 
            callbacks=[early_stopping],
            verbose=0
        )

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
        self._model_validator()
        predictions = self.model.predict(data)
        return predictions
    
    def compute_mae(
            self,
            testing_data: np.ndarray,
            testing_labels: np.ndarray
            ) -> float:
        """
        Computes the mean absolute error of the model.

        :param testing_data: the testing data
        :type testing_data: np.ndarray
        :param testing_labels: the testing target data
        :type testing_labels: np.ndarray
        :return: the mean absolute error
        :rtype: float
        """
        predictions = self.predict(testing_data)
        mae = mean_absolute_error(testing_labels, predictions)
        return mae
    
    def model_summary(self) -> Any:
        """
        Returns the summary of the model.
        """
        self._model_validator()
        return self.model.summary()

    def save_model(
            self,
            stockName: str
            ) -> None:
        """
        Saves the model.

        :param stockName: the name of the file to be saved.
        The name is created and is going to look like:
        'stockName_model.keras'
        and is going to be stored in the models folder.
        """
        self._model_validator()
        self.model.\
            save(f"models/{stockName}_model.keras")

    def load_model(
            self,
            stockName: str
            ) -> None:
        """
        Loads a model from the models folder.

        :param stockName: the name of the file to be loaded.
        The name of the file by convetion is: 'stockName_model.keras'
        and is going to be loaded from the models folder.
        If the file doesn't exists it will raise an exception.
        """
        try:
            self.model = keras.models.\
                load_model(f"models/{stockName}_model.keras")
        except FileExistsError(
            f"No such Model named '{stockName}_model.keras'"
            "exists in the 'models' folder!"
            ) as e:
            raise e

    def _model_validator(self)  -> None:
        """
        Validates if there is a model instantiated.
        """
        if self.model is None:
            raise AttributeError("There is no Model!")
        