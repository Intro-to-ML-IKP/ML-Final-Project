# Main program
from data_parser.dataReader import DataReader
from data_parser.dataProcessor import DataProcessor
from visualisation.visualize import PlotStocks  #plot_candlestick, plot_residuals
import numpy as np
from sklearn.model_selection import train_test_split

# Import NN tools
from neuralNetwork.losses import *
from neuralNetwork.network import Network
from neuralNetwork.dense import Dense
from neuralNetwork.activations import Tanh, Sigmoid

def reshape_data(data):
    return [np.array(candlestick).flatten() for candlestick in data]

def testing_SMA():
    dR = DataReader("AAPL")                                  # Initialize for AAPL stock
    stock_data = dR.getData(10, 5)              # Download sufficient data for [numSets] sets of [pointsPerSet] datapoints
    #data, labels = dR.getLabels(10, 3)     # Retrieve the labels of the next [labelsPerSet] datapoints for each set of [pointsPerSet] points, splitting data from labels
    high, low, open_, close = list(zip(*stock_data))

    processor = DataProcessor(stock_data, None)

    sets_SMA = processor.calculate_SMA()

    SMA = sets_SMA[0]       # SMA of closing price

    extrapolation_SMA = processor.extrapolate_the_SMA(SMA, 10, 0)  # Extrapolates Simple Moving Average 10 points into the future, starting from index -2

    residuals = processor.calculate_residuals(SMA)                  # Subtracts the SMA from the closing prices

    plotter = PlotStocks(stock_data, SMA, extrapolation_SMA, residuals)

    # # plotter.plot_candlestick(simpleMovingAverage=True)    # If you want to plot only the candlesticks

    plotter.plot_residuals()                              # If you want to plot only the residuals

    # # Master Plot
    plotter.masterPlot()


def main(stockCode, numSets, pointsPerSet, labelsPerSet, testingPercentage, validationPercentage, learning_rate, epochs):
    """Trains a model on a specified stock to predict the next prices
    
    Parameters:
    stockCode               - The code of the desired stock on the market
    numSets                 - The number of total datasets to generate
    pointsPerSet            - The number of datapoints + labels per set
    labelsPerSet            - The number of labels per set to be subtracted from pointsPerSet. Will also be the amount of points predicted
    testingPercentage       - The percentage of data chunks to be used for testing
    validationPercentage    - The percentage of data chunks to be used for validation
    activationFunction      - Activation function for hidden layers
    learning_rate           - Learning rate for the optimizer
    epochs                  - Number of iterations of training performed
    
    Returns:
    None"""
    dR = DataReader(stockCode)                                  # Initialize for AAPL stock
    stock_data = dR.getData(pointsPerSet+2, numSets)            # Download sufficient data for [numSets] sets of [pointsPerSet] datapoints. +2 because we will lose those with the SMAs
    sets = dR.splitSets(stock_data, pointsPerSet+2)               # Splits the stock data into sets for the processor

    # Apply preprocessing to get SMA and residuals
    allResiduals = []
    allExtrapolations = []
    for sD in sets:
        processor = DataProcessor(sD, None)
        closing_SMA = processor.calculate_SMA()[0]              # We lose 2 values here
        residuals = processor.calculate_residuals(closing_SMA)                  # Subtracts the SMA from the closing prices, this will be used in the network
        extrapolation_SMA = processor.extrapolate_the_SMA(closing_SMA, labelsPerSet, 0)  # Extrapolates Simple Moving Average [labelsPerSet] points into the future
        allResiduals.append(residuals)
        allExtrapolations.append(extrapolation_SMA)

    # Split the data from the labels
    data, labels = dR.splitLabels(allResiduals, labelsPerSet)

    # Apply a train, test, validation split on the data
    training_data, training_labels, validation_data, validation_labels, testing_data, testing_labels = processor.split_data(data, labels, testingPercentage, validationPercentage)


    # Train the model on training data
    networkStructure = [                    # TODO add more dynamic network structure
        Dense(len(training_data[0]), 8),
        Tanh(),
        Dense(8, labelsPerSet),
        Tanh()
    ]
    network = Network(networkStructure, learning_rate=learning_rate)
    errors = network.train(mse, mse_prime, training_data, training_labels, epochs=epochs, verbose = True)       # TODO add validation data in the network to avoid overfitting
    network.saveNetwork("residualsv1")

    # Test the model on testing data
    for x, y in zip(testing_data, testing_labels):
        output = network.predict(x)
        print(f"Prediction: {output}\nLabels: {y}")

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
    #main("AAPL", 5, 10, 3, 0.8, 0.1, 0.001, 50)
    testing_SMA()
    