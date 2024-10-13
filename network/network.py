import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore
from tensorflow.keras.callbacks import EarlyStopping # type: ignore

class Model:
    def __init__(self, model_shape, activations, input_shape):
        """Create the model architecture

        Parameters:
        Model shape     (list[float])   - The number of neurons in each layer
        activations     (list[string])  - The activation used for each layer
        input_shape     (int)           - The number of datapoints used for input
        
        Returns:
        None"""
        model = Sequential()

        model.add(Dense(model_shape[0], input_dim=input_shape, activation=activations[0]))  # Input layer (n=7), hidden layer with 64 units

        for nNeurons, activ in zip(model_shape[1:], activations[1:]):
            model.add(Dense(nNeurons, activation=activ))               # Second hidden layer with 32 units

        self.model = model

    def compileModel(self, learning_rate, lossFunc, metrics):
        """Compiles the model so it can be trained.
        
        Parameters:
        learning_rate   (float)         - The step size used for training
        lossFunc        (string)        - Function used to determine cost
        metrics         (list[string])  - List of metrics to track during training
        
        Returns:
        None"""
        self.model.compile(optimizer=Adam(learning_rate=learning_rate), loss=lossFunc, metrics=['mae'])

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
        early_stopping = EarlyStopping(monitor='val_loss', patience=4)     # Stops training when validation performance stops improving and thus prevents overfitting
        self.model.fit(training_data, training_labels, epochs=epochs, batch_size=batch_size, validation_data=(validation_data, validation_labels), callbacks=[early_stopping])

    def predict(self, data):
        """Makes a prediction on the specified data using the trained model
        
        Parameters:
        data    - Input data to predict, same shape as data used for training
        
        Returns:
        predictions - Same shape of labels used for training, values based on input data"""
        predictions = self.model.predict(data)
        return predictions

