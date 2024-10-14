# Main program
from data_parser.dataReader import DataReader
from data_parser.dataProcessor import DataProcessor
from visualisation.visualize import PlotStocks  #plot_candlestick, plot_residuals
from network.network import Model

def testing_SMA():
    dR = DataReader("AAPL")                                  # Initialize for AAPL stock
    stock_data = dR.getData(10, 5)              # Download sufficient data for [numSets] sets of [pointsPerSet] datapoints
    #data, labels = dR.getLabels(10, 3)     # Retrieve the labels of the next [labelsPerSet] datapoints for each set of [pointsPerSet] points, splitting data from labels
    # stock = Stock("AAPL", "2024-09-01", "2024-09-01")
    # high, low, open_, close = stock.get_data()# list(zip(*stock_data))
    
    # stockData = [(op, hi, lo, cl) for op, hi, lo, cl in zip(high, low, open_, close)]

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

def main(stockCode, numSets, pointsPerSet, labelsPerSet, testingPercentage, validationPercentage, learning_rate, epochs, batch_size):
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
    batch_size              - Number of data chunks used before updating the weights
    
    Returns:
    None"""
    dR = DataReader(stockCode)                                      # Initialize for AAPL stock
    stock_data = dR.getData(pointsPerSet+2, numSets)                # Download sufficient data for [numSets] sets of [pointsPerSet] datapoints. +2 because we will lose those with the SMAs                
    processor = DataProcessor(stock_data, None)
    sets = processor.splitSets(stock_data, pointsPerSet+2)          # Splits the stock data into sets for the processor

    # Apply preprocessing to get SMA and residuals
    allResiduals = []
    allExtrapolations = []
    for sD in sets:
        processor = DataProcessor(sD, None)
        closing_SMA = processor.calculate_SMA()[0]                                          # We lose 2 values here
        residuals = processor.calculate_residuals(closing_SMA)                              # Subtracts the SMA from the closing prices, this will be used in the network
        extrapolation_SMA = processor.extrapolate_the_SMA(closing_SMA, labelsPerSet, 0)     # Extrapolates Simple Moving Average [labelsPerSet] points into the future
        allResiduals.append(residuals)
        allExtrapolations.append(extrapolation_SMA)

    # Split the data from the labels
    data, labels = processor.splitLabels(allResiduals, labelsPerSet)

    # Apply a train, test, validation split on the data
    training_data, validation_data, testing_data, training_labels, validation_labels, testing_labels = processor.split_data(data, labels, testingPercentage, validationPercentage)

    # Make and train the model
    model = Model()
    model.create_sequential_model([27, labelsPerSet], ["relu", "linear"], pointsPerSet-labelsPerSet)
    model.compileModel(learning_rate, "mse", ["mae"])
    model.trainModel(training_data, training_labels, validation_data, validation_labels, epochs, batch_size)

    # Evaluate on testing data
    mae = model.compute_mae(testing_data, testing_labels)
    
    print(mae)
    # Save model
    #model.save_model(stockCode)

if __name__ == "__main__":
    main("AAPL", 5, 10, 3, 0.8, 0.1, 0.001, 50, 1)
    #testing_SMA()
    #model = Model()
    #model.load_model("AAPL")
    #print(model.model_summary())
    