# Main program
from data_parser.dataReader import DataReader
from data_parser.dataProcessor import DataProcessor
from visualisation.visualize import PlotStocks  #plot_candlestick, plot_residuals
from network.parameterConstructor import ParameterConstructor
from network.network_constructor import NetworkConstructor, NetworksDict
from results.result_handler import ResultsHandler


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

def get_Data(stockCode: str, pointsPerSet: int, numSets: int, labelsPerSet: int, testingPercentage: float, validationPercentage: float):
    dR = DataReader(stockCode)                                      # Initialize for AAPL stock
    stock_data = dR.getData(pointsPerSet+2, numSets)                # Download sufficient data for [numSets] sets of [pointsPerSet] datapoints. +2 because we will lose those with the SMAs                
    processor = DataProcessor(stock_data, None)
    sets = processor.generate_sets(pointsPerSet+2)          # Splits the stock data into sets for the processor

    # Get SMA and residuals
    allResiduals = []
    allExtrapolations = []
    for sD in sets:
        SMA = processor.calculate_SMA(sD)                                          # We lose 2 values here
        residuals = processor.calculate_residuals(sD, SMA)                              # Subtracts the SMA from the closing prices, this will be used in the network
        extrapolation_SMA = processor.extrapolate_the_SMA(SMA, labelsPerSet)     # Extrapolates Simple Moving Average [labelsPerSet] points into the future
        allResiduals.append(residuals)
        allExtrapolations.append(extrapolation_SMA)

    # Split the data from the labels
    data, labels = processor.generate_labels(allResiduals, labelsPerSet)

    # Apply a train, test, validation split on the data
    training_data, validation_data, testing_data, training_labels, validation_labels, testing_labels = processor.split_data(data, labels, testingPercentage, validationPercentage)
    return training_data, validation_data, testing_data, training_labels, validation_labels, testing_labels

def testNetworkConstructor(stockCode, pointsPerSet, numSets, labelsPerSet, testingPercentage, validationPercentage, maxEpochs):
    # Generate arbitrary list of parameters
    pConst = ParameterConstructor()
    pConst.calcNetworkArchitectures(
        maxLayers = 2,
        minNeurons = 16,
        maxNeurons = 32,
        dNeurons = 4
    )   # Just some sample numbers, check the code to find out what it does
    pConst.calcLearningRates(0.0005, 0.01, 0.0005)
    pConst.calcBatchSize(1, 8, 1)

    # Less realistic values but this is for testing baby, relax
    # pConst.calcNetworkArchitectures(2, 2, 31, 1) 
    # pConst.calcLearningRates(0.001, 0.1, 0.1)
    # pConst.calcBatchSize(1,5,1)
    pConst.calcParamList()  # 12 different parameter sets
    print(len(pConst.paramList))

    # Evaluate the model 
    training_data, validation_data, testing_data, training_labels, validation_labels, testing_labels = get_Data(stockCode, pointsPerSet, numSets, labelsPerSet, testingPercentage, validationPercentage)
    netConst = NetworkConstructor(len(training_data[0]), len(training_labels[0]), maxEpochs)
    netConst.explore_different_architectures(training_data, training_labels, validation_data, validation_labels, testing_data, testing_labels, pConst.paramList)
    maes = NetworksDict()

    results_handler = ResultsHandler(maes)
    results_handler.save_results("NN_results_final")

def test_statistical_analysis():
    list_ = ["NN_results_2000", "NN_results_3000", "NN_results_4000"]
    results = ResultsHandler()
    results.load_multiple_results(list_)
    print("Correlation Coeficients:")
    mae_correlations, p_values = results.calculate_correlation_coefficients()
    print(mae_correlations)
    print("P-values:")
    print(p_values)
    print("Regression Analysis:")
    print(results.perform_regression_analysis())
    results.create_scatterplot_matrix()
    results.create_correlation_heatmap()
    param_ranges = results.get_parmeter_ranges()
    print(param_ranges)
    
if __name__ == "__main__":
    # main("AAPL", 5, 10, 3, 0.8, 0.1, 0.001, 50, 1)
    #testing_SMA()
    #model = Model()
    #model.load_model("AAPL")
    #print(model.model_summary())
    testNetworkConstructor(
        stockCode="AAPL",
        pointsPerSet=20,
        numSets=500,
        labelsPerSet=1,
        testingPercentage=0.8,
        validationPercentage=0.1,
        maxEpochs=50)
    # testNetworkConstructor("AAPL", 10, 5, 3, 0.8, 0.1, 50)
    # test_statistical_analysis()
