import numpy as np
from typing import Any
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore
from tensorflow.keras.callbacks import EarlyStopping # type: ignore
from sklearn.metrics import mean_absolute_error

class Model:
    def __init__(self):
        """
        Instantiates a model.
        """
        self.model = None

    def create_sequential_model(self, model_shape, activations, input_shape, output_size):
        """Create the model architecture and sets the model attribute to that model.

        Parameters:
        model_shape     (list[float])   - The number of neurons in each layer
        activations     (list[string])  - The activation used for each layer
        input_shape     (int)           - The number of datapoints used for input
        output_size     (int)           - The number of neurons in output
        
        Returns:
        None"""
        model = Sequential()

        model.add(Dense(model_shape[0], input_dim=input_shape, activation=activations[0]))  # Input layer (n=7), hidden layer with 64 units

        for nNeurons, activ in zip(model_shape[1:], activations[1:-1]):
            model.add(Dense(nNeurons, activation=activ))               # Second hidden layer with 32 units
            
        model.add(Dense(output_size, activation=activations[-1]))
        
        self.model = model

    def compileModel(self, learning_rate, lossFunc, metrics):
        """Compiles the model so it can be trained.
        
        Parameters:
        learning_rate   (float)         - The step size used for training
        lossFunc        (string)        - Function used to determine cost
        metrics         (list[string])  - List of metrics to track during training
        
        Returns:
        None"""
        self._model_validator()
        self.model.compile(optimizer=Adam(learning_rate=learning_rate), loss=lossFunc, metrics=metrics)

    def trainModel(self, training_data, training_labels, validation_data, validation_labels, epochs, batch_size):
        """Trains the model using specified training data and validation data.
        
        Parameters:
        training_data       - Data used for training. Must match input shape of the model
        training_labels     - Labels corresponding to training data. Shape must match output shape of the model
        validation_data     - Data used in training to prevent overfitting
        validation_labels   - Labels corresponding to validation data
        epochs (int)        - Number of iterations of training
        batch_size  (int)   - Number of datapoints used per batch, thus the number of processed datapoints before updating weight matrix
        
        Returns:
        None"""
        self._model_validator()
        #early_stopping = EarlyStopping(monitor='val_loss', patience=4)  # Stops training when validation performance stops improving and thus prevents overfitting
        self.model.fit(
            training_data,
            training_labels,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(validation_data, validation_labels))#,
            #callbacks=[early_stopping]
            #)

    def predict(self, data):
        """Makes a prediction on the specified data using the trained model
        
        Parameters:
        data    - Input data to predict, same shape as data used for training
        
        Returns:
        predictions - Same shape of labels used for training, values based on input data"""
        self._model_validator()
        predictions = self.model.predict(data)
        return predictions
    
    def compute_mae(self, testing_data, testing_labels):
        predictions = self.predict(testing_data)
        mae = mean_absolute_error(testing_labels, predictions)
        return mae
    
    def model_summary(self) -> Any:
        """
        Returns the summary of the model.
        """
        self._model_validator()
        return self.model.summary()

    def save_model(self, stockName: str):
        """
        Saves the model.

        :param stockName: the name of the file to be saved.
        The name is created and is going to look like: 'stockName_model.keras'
        and is going to be stored in the models folder.
        """
        self._model_validator()
        self.model.save(f"models/{stockName}_model.keras")

    def load_model(self, stockName: str):
        """
        Loads a model from the models folder.

        :param stockName: the name of the file to be loaded.
        The name of the file by convetion is: 'stockName_model.keras'
        and is going to be loaded from the models folder.
        If the file doesn't exists it will raise an exception.
        """
        try:
            self.model = keras.models.load_model(f"models/{stockName}_model.keras")
        except FileExistsError(f"No such Model named '{stockName}_model.keras' exists in the 'models' folder!") as e:
            raise e

    def _model_validator(self):
        """
        Validates if there is a model instantiated.
        """
        if self.model is None:
            raise AttributeError("There is no Model!")
        