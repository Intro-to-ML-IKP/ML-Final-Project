# Main program
from data_parser.dataReader import DataReader
from neuralNetwork.network import train_stock_predictor, load_stock_predictor, test_stock_predictor, analyze_trend
from visualisation.visualize import plot_candlestick
import numpy as np

def reshape_data(data):
    return [np.array(candlestick).flatten() for candlestick in data]

def main(stockCode, numSets, pointsPerSet, labelsPerSet, testingPercentage, validationPercentage, networkStructure, activationFunction, learning_rate, batch_size, epochs):
    """Trains a model on a specified stock to predict the next prices
    
    Parameters:
    stockCode               - The code of the desired stock on the market
    numSets                 - The number of total datasets to generate
    pointsPerSet            - The number of datapoints + labels per set
    labelsPerSet            - The number of labels per set to be subtracted from pointsPerSet. Will also be the amount of points predicted
    testingPercentage       - The percentage of data chunks to be used for testing
    validationPercentage    - The percentage of data chunks to be used for validation
    networkStructure        - List containing the amount of neurons in hidden layers
    activationFunction      - Activation function for hidden layers
    learning_rate           - Learning rate for the optimizer
    batch_size              - Number of chunks of data used for training
    epochs                  - Number of iterations of training performed
    
    Returns:
    None"""
    dR = DataReader(stockCode)                                  # Initialize for AAPL stock
    stock_data = dR.getData(pointsPerSet, numSets)              # Download sufficient data for [numSets] sets of [pointsPerSet] datapoints
    data, labels = dR.getLabels(pointsPerSet, labelsPerSet)     # Retrieve the labels of the next [labelsPerSet] datapoints for each set of [pointsPerSet] points, splitting data from labels
    data = dR.preprocess()                                      # Apply preproccesing on data if applicable

    # Apply a train, test, validation split on the data
    training_data, training_labels, validation_data, validation_labels, testing_data, testing_labels = dR.split_data(testingPercentage, validationPercentage)

    # Set parameters
    model_save_path = f"{stockCode}_model.keras"

    xTrain = reshape_data(training_data)
    print(len(xTrain))
    print(xTrain)

    yTrain = training_labels
    print(len(yTrain))
    print(yTrain)

    return

    # Train the model
    model, history = train_stock_predictor(
        X_train=list(reshape_data(training_data)), y_train=training_labels, 
        X_val=reshape_data(validation_data), y_val=validation_labels, 
        n=(pointsPerSet - labelsPerSet), 
        k=labelsPerSet, 
        hidden_layers=networkStructure, # Adjust hidden layers if necessary
        activation=activationFunction,              # Choose the activation function
        learning_rate=learning_rate,            # Adjust learning rate
        batch_size=batch_size,                  # Adjust batch size
        epochs=epochs,                      # Number of epochs for training
        model_save_path=model_save_path
    )

    # Test the model on unseen data
    model = load_stock_predictor(model_save_path)
    predictions, test_loss = test_stock_predictor(model, reshape_data(testing_data), testing_labels)

    # Analyze the trend in predictions
    predicted_trends = analyze_trend(predictions)

    print(f"Test Loss: {test_loss}")
    print(f"Predicted trends: {predicted_trends[:5]}")  # Print the first few trends as a preview

    # `predictions` contains the predicted closing prices, and `predicted_trends` 
    # contains the trend analysis based on the predictions.
    
    

if __name__ == "__main__":
    #main("AAPL", 5, 10, 3, 0.8, 0.1, [128, 64, 32], "relu", 0.001, 32, 50)
    date = "2000-01-24"
    print(date.split("-")[0])