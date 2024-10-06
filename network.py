import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense # type: ignore
from tensorflow.keras.callbacks import ModelCheckpoint # type: ignore
import os

# Function to build and train the model
def train_stock_predictor(X_train, y_train, X_val, y_val, 
                          n, k, 
                          hidden_layers=[128, 64, 32], 
                          activation="mse", 
                          loss_function="mse",
                          learning_rate=0.001, 
                          batch_size=32, 
                          epochs=50, 
                          model_save_path='stock_model.h5'):
    """
    Trains a feedforward neural network to predict stock prices.
    
    Parameters:
    X_train, y_train : Training data and labels (closing prices)
    X_val, y_val     : Validation data and labels (closing prices)
    n                : Number of candlesticks (each with 4 features - open, high, low, close)
    k                : Number of next closing prices to predict
    hidden_layers    : List of neurons for each hidden layer (default=[128, 64, 32])
    activation       : Activation function for hidden layers (default="relu")
    loss_function    : Loss function to calculate cost (default="mse")
    learning_rate    : Learning rate for the optimizer (default=0.001)
    batch_size       : Batch size for training (default=32)
    epochs           : Number of epochs for training (default=50)
    model_save_path  : Path to save the trained model (default="stock_model.h5")
    
    Returns:
    model            : Trained Keras model
    history          : Training history object (for plotting or further analysis)
    """

    # Input shape is (4n) where 4 is for OHLC and n is the number of candlesticks
    input_shape = 4 * n  


    model = Sequential()
    
    # Input layer and hidden layers
    model.add(Dense(hidden_layers[0], activation=activation, input_dim=input_shape))  # First hidden layer
    
    # Add additional hidden layers
    for units in hidden_layers[1:]:
        model.add(Dense(units, activation=activation))  # More hidden layers

    # Output layer, with 'k' outputs corresponding to next k closing prices
    model.add(Dense(k))

    # Compile the model using the specified learning rate and optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)       # TODO figure out more on how this optimizer works
    model.compile(optimizer=optimizer, loss=loss_function)

    # Set up model checkpointing to save the best model during training
    checkpoint = ModelCheckpoint(model_save_path, save_best_only=True, monitor='val_loss', mode='min')

    # Train the model and store training history
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val), 
                        epochs=epochs, batch_size=batch_size, 
                        callbacks=[checkpoint])

    return model, history


# Function to load a saved model
def load_stock_predictor(model_load_path='stock_model.h5'):
    """
    Loads a trained model from the specified file.
    
    Parameters:
    model_load_path : Path to the saved model (default='stock_model.h5')
    
    Returns:
    model           : Loaded Keras model
    """
    if os.path.exists(model_load_path):
        model = tf.keras.models.load_model(model_load_path)
        print(f"Model loaded from {model_load_path}")
        return model
    else:
        raise FileNotFoundError(f"No model found at {model_load_path}")


# Function to test the model on unseen data (e.g., test set)
def test_stock_predictor(model, X_test, y_test):
    """
    Tests the model on test data and returns predictions and test loss.
    
    Parameters:
    model   : Trained Keras model
    X_test  : Test data (input candlesticks)
    y_test  : True labels (next closing prices)
    
    Returns:
    predictions : Predicted closing prices by the model
    test_loss   : Loss on the test set (MSE)
    """
    # Evaluate the model on the test set
    test_loss = model.evaluate(X_test, y_test)

    # Predict the closing prices for the test data
    predictions = model.predict(X_test)

    return predictions, test_loss


# Example function to analyze trends in predictions
def analyze_trend(predictions):
    """
    Analyzes the trend in predicted prices to determine whether they are increasing, 
    decreasing, or stable.
    
    Parameters:
    predictions : Predicted closing prices (numpy array)
    
    Returns:
    trends      : List of trend analysis for each set of predicted prices
    """
    trends = []
    for pred in predictions:
        if len(pred) > 1:
            trend = np.sign(np.diff(pred))  # Get the difference between consecutive predictions
            trends.append(trend)
        else:
            trends.append(None)
    return trends



